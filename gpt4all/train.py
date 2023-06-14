import os
import transformers
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, LlamaForCausalLM, DefaultDataCollator
import torch
import glob
from datasets import load_dataset, Dataset
from torch.optim import AdamW
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptim, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics import MeanMetric
from tqdm import tqdm
import wandb
from read import read_config
from torch.utils.data import DataLoader
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'




torch.backends.cuda.matmul.allow_tf32 = True
def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log


def evaluate(model, val_dataloader):
    model.eval()
    val_loss = MeanMetric(nan_strategy="error").to(model.device)

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss = model(**batch).loss

            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})

            val_loss.update(loss_values["loss"])

    return val_loss


def tokenize_inputs(config, tokenizer, examples):
    max_length = config["max_length"]

    # hacky backward compatible
    different_eos = tokenizer.eos_token != "</s>"
    out = {"labels": [], "input_ids": []}
    for prompt, response in zip(examples["prompt"], examples["response"]):
        if different_eos:
            if response.count("</s> \n") > 0:
                response = response.replace("</s> \n", f"{tokenizer.eos_token} \n") 

        prompt_len = len(tokenizer(prompt + "\n", return_tensors="pt")["input_ids"][0])

        # hack if our prompt is super long
        # we need to include some labels so we arbitrarily trunacate at max_length // 2
        # if the length is too long
        if prompt_len >= max_length // 2:
            # if prompt is too long, truncate
            # but make sure to truncate to at max 1024 tokens
            new_len = min(max_length // 2, len(prompt) // 2)
            prompt = prompt[:new_len]
            # get new prompt length
            prompt_len = tokenizer(prompt + "\n", return_tensors="pt", max_length=max_length // 2, truncation=True).input_ids.ne(tokenizer.pad_token_id).sum().item()

        assert prompt_len <= max_length // 2, f"prompt length {prompt_len} exceeds max length {max_length}"

        input_tokens = tokenizer(prompt + "\n" + response + tokenizer.eos_token,
                                 truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].squeeze()

        labels = input_tokens.clone()
        labels[:prompt_len] = -100
        if len(labels) < max_length:
            # pad to max_length with -100
            labels = torch.cat([labels, torch.full((max_length - len(labels),), -100)])

        assert (labels == -100).sum() < len(labels), f"Labels are all -100, something wrong. prompt length {prompt_len} exceeds max length {max_length}" 
        
        if (labels == -100).sum() == len(labels) - 1:
            print(prompt)
            print(response)
            raise

        input_tokens = tokenizer.pad({"input_ids": input_tokens}, padding="max_length", max_length=max_length)["input_ids"]
        out["labels"].append(labels)
        out["input_ids"].append(input_tokens)

    out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}

    return out


def load_data(config, tokenizer, wandb_url):
    
    api = wandb.Api()
    run = api.run(wandb_url)
    #/opentensor/opentensor-validator/runs/kltiefxf
    #Show history data:
    historyData = run.history()
    historyData.loc[historyData.best_answer.str.len() == 0, "best_answer"] = '---'


    data1 = historyData[["base_prompt", "best_followup"]].copy()
    data1.columns = ['prompt', 'response']
    data1["source"] = ""

    data2 = historyData[["answer_prompt", "best_answer"]].copy()
    data2.columns = ['prompt', 'response']
    data2["source"] = ""

    data = pd.concat([data1, data2])


    reseted_data = data.reset_index(drop=True)
    dataset = Dataset.from_pandas(reseted_data)
    dataset.to_csv('out_1.csv', index=False)  
    
    dataset = dataset.train_test_split(test_size=.05, seed=30)

    train_dataset, val_dataset = dataset["train"], dataset["test"]

    if config["streaming"] is False:
        kwargs = {"num_proc": config["num_proc"]}
    else:
        kwargs = {}

    # tokenize inputs and return labels and attention mask
    train_dataset = train_dataset.map(
        lambda ele: tokenize_inputs(config, tokenizer, ele),
        batched=True,
        remove_columns=["source", "prompt"],
        **kwargs
    )
    val_dataset = val_dataset.map(
        lambda ele: tokenize_inputs(config, tokenizer, ele),
        batched=True,
        remove_columns=["source", "prompt"],
        **kwargs
    )

    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")

    # create dataloader with default data collator since we already have labels

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=DefaultDataCollator(),
        batch_size=config["batch_size"],
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=DefaultDataCollator(),
        batch_size=config["batch_size"],
    )

    return train_dataloader, val_dataloader

def train(accelerate, config, wandb_url):
  accelerator.print(f"Using {accelerator.num_processes} GPUs")
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], model_max_length=config['max_length'])
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  with accelerator.main_process_first():
    train_dataloader, val_dataloader = load_data(config, tokenizer, wandb_url=wandb_url)
    print("train_dataloader", train_dataloader)
  checkpoint = config["gradient_checkpointing"]
  print("----------------------------------------------")
  model = AutoModelForCausalLM.from_pretrained(config["model_name"], use_cache=False if checkpoint else True, trust_remote_code=True)
  print("----model loaded-------------------------------")
  if checkpoint:
    model.gradient_checkpointing_enable()

  if config["lora"]:
    peft_config = LoraConfig(
        # should R be configurable?
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

  optimizer_cls = (
    AdamW
    if accelerator.state.deepspeed_plugin is None
    or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    else DummyOptim
  )
  gradient_accumulation_steps = 1
  optimizer = optimizer_cls(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

  if accelerator.state.deepspeed_plugin is not None:
    gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
        "gradient_accumulation_steps"
    ]

  # decay to min_lr instead of 0
  lr_ratio = config["min_lr"] / config["lr"]
  accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")
  total_num_steps = (len(train_dataloader) / gradient_accumulation_steps) * config["num_epochs"]
  # instead of decaying to zero, decay to ratio of min_lr / lr
  total_num_steps += int(total_num_steps * lr_ratio) + config["warmup_steps"]
  accelerator.print(f"Total training steps: {total_num_steps}")
  if (
    accelerator.state.deepspeed_plugin is None
    or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
  ):
    scheduler = get_scheduler(
      name="cosine",
      optimizer=optimizer,
      num_warmup_steps=config["warmup_steps"] * accelerator.num_processes,
      num_training_steps=total_num_steps,
      )
  else:
    scheduler = DummyScheduler(
      optimizer, total_num_steps=config["warmup_steps"], warmup_num_steps=config["warmup_steps"]
    )
  model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # setup for saving training states in case preemption
  accelerator.register_for_checkpointing(scheduler)

  if config["checkpoint"]:
    accelerator.load_state(config["checkpoint"])
    accelerator.print(f"Resumed from checkpoint: {config['checkpoint']}")
    path = os.path.basename(config["train_args"]["resume_from_checkpoint"])
    training_difference = os.path.splitext(path)[0]
    resume_step = int(training_difference.replace("step_", ""))
    accelerator.skip_first_batches(train_dataloader, resume_step)
    accelerator.print(f"Resuming from step {resume_step}")


    # log gradients
  if accelerator.is_main_process and config["wandb"]:
    wandb.watch(model, log_freq=config["log_grads_every"], log="all")
  print("started training...")
  for epoch in range(config["num_epochs"]):
    print(epoch)
    train_loss = MeanMetric(nan_strategy="error").to(model.device)
    for step, batch in enumerate(tqdm(train_dataloader)):
      model.train()
      outputs = model(**batch)
      loss = outputs.loss

      # gather loss before backprop in case of gradient accumulation
      loss_values = accelerator.gather_for_metrics({"loss": loss.detach().float()})
      train_loss.update(loss_values["loss"])

      loss = loss / gradient_accumulation_steps
      accelerator.backward(loss)
      # get gradient norm of all params

      # log LR in case something weird happens 
      if step > 0 and step % (config["eval_every"] // 10) == 0:
        if config["wandb"]:
          curr_step = step + epoch * len(train_dataloader)
          accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)

      if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


      if step > 0 and step % config["save_every"] == 0:
        curr_step = step + epoch * len(train_dataloader)
        accelerator.save_state(f"{config['output_dir']}/step_{curr_step}")

      if step > 0 and (step % config["eval_every"] == 0 or step == len(train_dataloader) - 1):
        val_loss = evaluate(model, val_dataloader)

        log_train = {
          "train_loss": train_loss.compute()
        }
        log_val = {
          "val_loss": val_loss.compute()
        }

        if config["wandb"]:
          curr_step = step + epoch * len(train_dataloader)
          accelerator.log({**log_train, **log_val}, step=curr_step)

  accelerator.print(f"Current LR: {scheduler.get_last_lr()[0]}")
  accelerator.print(format_metrics(log_train, "train", f" step {step} "))
  accelerator.print(format_metrics(log_val, "val", f" step {step} "))

  train_loss.reset()

  accelerator.print(f"Epoch {epoch} finished")
  accelerator.print(f"Pushing to HF hub")
  accelerator.wait_for_everyone()
  unwrapped_model = accelerator.unwrap_model(model)
  try:
    if accelerator.is_main_process:
      unwrapped_model.push_to_hub(config["save_name"] + f"-epoch_{epoch}", private=True)

  except Exception as e:
    accelerator.print(e)
    accelerator.print(f"Failed to push to hub")

  unwrapped_model.save_pretrained(
    f"{config['output_dir']}/epoch_{epoch}",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(model),
  )
            
  accelerator.wait_for_everyone()
  unwrapped_model = accelerator.unwrap_model(model)
  unwrapped_model.save_pretrained(
    f"{config['output_dir']}/final",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(model),
  )

  accelerator.end_training()



if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--datapath", type=str, default="/opentensor/opentensor-validator/runs/kltiefxf")
  args = parser.parse_args()
  config = read_config("config.yaml")
  accelerator = Accelerator()
  train(accelerator, config = config, wandb_url= args.datapath)



#accelerate launch --dynamo_backend=inductor --num_processes=8 --num_machines=1 --machine_rank=0 --deepspeed_multinode_launcher standard --mixed_precision=bf16  --use_deepspeed --deepspeed_config_file=ds_config.json train.py --config config.yaml
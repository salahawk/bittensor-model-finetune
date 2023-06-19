#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import logging
import math
import os
import random
from itertools import chain
import wandb
import gc
import datasets
import hydra
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, LlamaForCausalLM, DefaultDataCollator
import pandas as pd
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

import bittensor


def check_cfg_and_load_defaults(cfg: DictConfig) -> DictConfig:
    subtensor = bittensor.subtensor(network=cfg.bittensor.network)
    if cfg.dataset.block_size is None:
        cfg.dataset.block_size = subtensor.validator_sequence_length(netuid=cfg.bittensor.netuid)
    if cfg.training.train_batch_size is None:
        cfg.training.train_batch_size = subtensor.validator_batch_size(netuid=cfg.bittensor.netuid)
    if cfg.training.eval_batch_size is None:
        cfg.training.eval_batch_size = subtensor.validator_batch_size(netuid=cfg.bittensor.netuid)

    return cfg


def create_accelerator(cfg: DictConfig) -> Accelerator:
    accelerator = (
        Accelerator(log_with=cfg.tracking.report_to, fp16=True, num_processes=2, logging_dir=cfg.output_dir)
        if cfg.tracking.enabled
        else Accelerator()
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return accelerator


def load_raw_datasets(cfg: DictConfig) -> DatasetDict:
    if cfg.dataset.wandb == 1:
        wandbUrl = cfg.dataset.name
        api = wandb.Api()
        run = api.run(wandbUrl)
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
        return dataset
    if cfg.dataset.name == "bittensor":

        dataset = bittensor.dataset(
            no_tokenizer=True,
            batch_size=cfg.training.train_batch_size,
            block_size=cfg.dataset.block_size,
        )
        dataloader = dataset.dataloader(cfg.dataset.num_batches)
        bittensor_dataset = {"text": []}
        for batch in tqdm(dataloader, desc="Loading data from bittensor IPFS"):
            bittensor_dataset["text"].extend(batch)
        raw_datasets = Dataset.from_dict(bittensor_dataset)

        dataset.close()  # Avoid leaving threadqueue running.
        return raw_datasets

    if os.path.exists(cfg.dataset.name):
        data_files = {"text": cfg.dataset.name}
        dataset_args = {}

        extension = os.path.splitext(cfg.dataset.name)[-1].lstrip(".")

        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = cfg.dataset.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        raw_datasets = raw_datasets["text"]
    else:
        raw_datasets = load_dataset(cfg.dataset.name, cfg.dataset.config_name)

    return raw_datasets


def load_model_and_tokenizer(cfg: DictConfig):
    
    if cfg.model.config_name is not None:
        config = AutoConfig.from_pretrained(cfg.model.config_name)
    else:
        config = AutoConfig.from_pretrained(cfg.model.name)

    if cfg.tokenizer.name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.name, use_fast=cfg.tokenizer.use_fast
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name, use_fast=cfg.tokenizer.use_fast
        )
    # tokenizer.pad_token = cfg.tokenizer.pad_token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        from_tf=bool(".ckpt" in cfg.model.name),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def create_optimizer(cfg, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters, lr=cfg.training.learning_rate
    )


def tokenize_inputs(tokenizer, examples):
    max_length = 1024

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
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")
def preprocess(cfg, accelerator, tokenizer, raw_datasets):

    raw_datasets = raw_datasets.train_test_split(test_size=.05, seed=30)
    train_dataset, val_dataset = raw_datasets["train"], raw_datasets["test"]

    with accelerator.main_process_first():
        kwargs = {}
        train_dataset = train_dataset.map(
        lambda ele: tokenize_inputs(tokenizer, ele),
        batched=True,
        remove_columns=["source", "prompt"],
        **kwargs
        )
        val_dataset = val_dataset.map(
            lambda ele: tokenize_inputs( tokenizer, ele),
            batched=True,
            remove_columns=["source", "prompt"],
            **kwargs
        )
        train_dataset = train_dataset.with_format("torch")
        val_dataset = val_dataset.with_format("torch")
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=DefaultDataCollator(),
            batch_size=cfg.training.train_batch_size,
        )

        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=DefaultDataCollator(),
            batch_size=cfg.training.eval_batch_size,
        )

    return train_dataloader, val_dataloader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = check_cfg_and_load_defaults(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = create_accelerator(cfg)
    accelerator.wait_for_everyone()

    if cfg.training.seed is not None:
        logger.info(f"Setting random seed to {cfg.training.seed}")
        set_seed(cfg.training.seed)

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))

    tokenizer, model = load_model_and_tokenizer(cfg)
    optimizer = create_optimizer(cfg, model)

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=cfg.training.max_train_steps,
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Load data
    raw_datasets = load_raw_datasets(cfg)


    # DataLoaders creation:\
    train_dataloader, eval_dataloader = preprocess(cfg, accelerator, tokenizer, raw_datasets)

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = (
                cfg.training.num_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader
    # may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        cfg.training.max_train_steps = (
                cfg.training.num_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_epochs = math.ceil(
        cfg.training.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if cfg.tracking.enabled is True and accelerator.is_main_process:
        experiment_config = vars(cfg)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("finetune_using_clm", experiment_config)

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_epochs}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(cfg.training.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.training.checkpoint.resume_from_checkpoint > 0:
        accelerator.print(
            f"Resumed from checkpoint: {cfg.training.checkpoint.resume_from_checkpoint}"
        )
        accelerator.load_state(cfg.training.checkpoint.resume_from_checkpoint)
        path = os.path.basename(cfg.training.checkpoint.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, cfg.training.num_epochs):
        model.train()
        if cfg.tracking.enabled is True:
            total_loss = 0
        train_losses = []
        for step, batch in enumerate(train_dataloader):
            torch.cuda.memory_summary(device=None, abbreviated=False)
            min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB
            clear_gpu_memory()
            wait_until_enough_gpu_memory(min_memory_available)
            # We need to skip steps until we reach the resumed step
            if (
                    cfg.training.checkpoint.resume_from_checkpoint
                    and epoch == starting_epoch
            ):
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            outputs = model(**batch)
            loss = outputs.loss
            train_losses.append(
                accelerator.gather(loss.repeat(cfg.training.train_batch_size))
            )
            # We keep track of the loss at each epoch
            if cfg.tracking.enabled is True:
                total_loss += loss.detach().float()
            loss = loss / cfg.training.gradient_accumulation_steps
            print(loss)
            accelerator.backward(loss)

            if (
                    step % cfg.training.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step % cfg.training.eval_every == 0:
                train_losses_tensor = torch.cat(train_losses)
                train_loss = torch.mean(train_losses_tensor)
                model.eval()
                eval_losses = []
                for _eval_step, eval_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**eval_batch)

                    loss = outputs.loss
                    eval_losses.append(
                        accelerator.gather(loss.repeat(cfg.training.eval_batch_size))
                    )

                losses = torch.cat(eval_losses)
                losses = losses[: len(eval_dataset)]
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(
                    f"epoch {epoch}: perplexity: {perplexity} train_loss: {train_loss} eval_loss: {eval_loss}"
                )

                epoch_dir = f"epoch_{epoch}_most_recent"
                if cfg.output_dir is not None:
                    output_dir = os.path.join(cfg.output_dir, epoch_dir)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)

        if cfg.tracking.enabled is True:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        logger.info(f"done epoch {epoch}")

    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            cfg.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()

# GPT4ALL Model Fine-Tuning by validator's result.

**Note**: This script was adapted from gpt4all language-modeling code.

## Prerequisites
- Python 3.8+

## Installation
1. Clone the repository
2. All following commands assume you are working from this folder, i.e. you must `cd` into the directory
created by the previous step.
3. Run ```pip install -r requirements.txt``` to install the additional packages required by this script.

## Configuration

All configurable parameters are visible and documented in `config.yaml` and `ds_config.json`. 
The defaults are chosen for quick tuning; you will need to experiment and adjust 
these.
1. You can configure some parameters for training such as model name and batch size in `config.yaml`
```bash
model_name: "nomic-ai/gpt4all-j"
tokenizer_name: "nomic-ai/gpt4all-j"
```
2. You can configure parameters for deepspeed in `ds_config.json`
```bash
"optimizer": {
    "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [
            0.9,
            0.999
            ],
            "eps": 1e-08
        }
    }
}
```

## Running

```bash
accelerate launch --dynamo_backend=inductor --num_processes=8 --num_machines=1 --machine_rank=0 --deepspeed_multinode_launcher standard --mixed_precision=bf16  --use_deepspeed --deepspeed_config_file=ds_config.json train.py --datapath /opentensor/opentensor-validator/runs/kltiefxf
```

**Note**
This process requires WandB api key when it's running.

### Main optional parameters in CLI
---
    --datapath
        This parameter specifies the WandB url that saves the validator's result.
    --dynamo_backend
        This parameter specifies the backend for the Dynamo library being used.
    --num_processes 
        This parameter sets the number of processes to be used in the execution.
    --num_machines
        This parameter specifies the number of machines or nodes being used.
    --machine_rank
        This parameter sets the rank or identifier for the current machine. 
    --deepspeed_multinode_launcher
        Configures the launch mode for DeepSpeed, a library for distributed training.
    --mixed_precision
        This parameter enables mixed precision training, specifically using the bf16 (bfloat16) data type. Mixed precision training can help improve training speed and memory usage.
    --use_deepspeed
        This parameter enables the use of DeepSpeed, a library that optimizes and accelerates training on distributed systems.
    --deepspeed_config_file
        This parameter specifies the configuration file for DeepSpeed. 
    
---

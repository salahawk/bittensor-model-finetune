# GPT4ALL Model Fine-Tuning by validator's result.


**Note**: This script was adapted from gpt4all language-modeling code.

## To start:

All following commands assume you are working from this folder, i.e. you must `cd` into the directory
created by the previous step.

Run ```pip install -r requirements.txt``` to install the additional packages required by this script.

## With Wandb url

You have to override `dataset.name` with your wandb url:

```bash
accelerate launch --dynamo_backend=inductor --num_processes=8 --num_machines=1 --machine_rank=0 --deepspeed_multinode_launcher standard --mixed_precision=bf16  --use_deepspeed --deepspeed_config_file=ds_config.json train.py --config config.yaml --datapath /opentensor/opentensor-validator/runs/kltiefxf
```

## Configuring training parameters

All configurable parameters are visible and documented in `config.yaml` and `ds_config.json`. 
The defaults are chosen for quick tuning; you will need to experiment and adjust 
these.


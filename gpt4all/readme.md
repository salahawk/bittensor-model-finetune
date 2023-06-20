# GPT4ALL Model Tuning with dataset from validators

<!---
Copyright 2020 The OpenTensor Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

**Note**: This script was adapted from HuggingFace's Transformers/language-modeling code.

## Installation & Requirements
This code assumes you have `bittensor` already installed on your machine, and is meant to be
run entirely separately. Some basic linux commandline knowledge is assumed, but 
[this guide](https://ubuntu.com/tutorials/command-line-for-beginners) should provide a good starting
point to navigate and move around files, directories, etc.

To start, clone this repository:
```commandline
git clone https://github.com/salahawk/bittensor-model-finetune/gpt4all 
```

All following commands assume you are working from this folder, i.e. you must `cd` into the directory
created by the previous step.

Run ```pip install -r requirements.txt``` to install the additional packages required by this script.


## Fine-tuning with the wandb url

If you have a wandb link from the validator, you can override `dataset.name` as above and set `wandb` as 1:
```bash
python gpt4all_finetuning.py dataset.wandb=1 dataset.name=/opentensor/opentensor-validator/runs/kltiefxf
```


## Configuring training parameters

And you have to login to wandb with wandb api key for loading data from wandb.
Please type `wandb init` on cmd and input api key.

All configurable parameters are visible and documented in `conf/config.yaml`. 
The defaults are chosen for quick training and not tuned; you will need to experiment and adjust 
these.

**Note**: The above parameters are the *only* commands you can override with this script. That is,
you may not pass flags you would normally use when running `btcli` (i.e. `--neuron.device` will *not* work).
If there is a flag you wish to modify feel free to submit a feature request.

To view the changeable parameters, open `conf/config.yaml` in whatever text editor you prefer, or
use `cat conf/config.yaml` to view them.

You do not need to edit this file to change the parameters; they may be overridden when you call this
script. e.g., if you wish to change the model to `nomic-ai/gpt4all-j`, and the output directory to `nomic-ai/gpt4all-j-tuned`, you would run:

```commandline
python gpt4all_finetuning.py model.name=nomic-ai/gpt4all-j output_dir=nomic-ai/gpt4all-j-tuned dataset.wandb=1 dataset.name=/opentensor/opentensor-validator/runs/kltiefxf
```

Note the nested structure in the config, since `model` is above `name` in `conf.yaml`, you must override
`model.name` when invoking the command.


## Serving custom models on bittensor

To serve your tuned model on bittensor, just override `neuron.model_name` with the path to your 
tuned model:
```bash
btcli run ..... --neuron.model_name=/home/{YOUR_USENAME}/gpt4all_model_tuning/tuned-model
```

## Limitations & Warnings

Early stopping is not yet supported. Many features are implemented but not thoroughly tested, if
you encounter an issue, reach out on discord or (preferably) create an issue on this github page.

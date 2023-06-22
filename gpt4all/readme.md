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

For developing server, if you're using Runpod online gpus, I recommend if you use pytorch@1.13 template.

To start, clone this repository:
```commandline
git clone https://github.com/salahawk/bittensor-model-finetune
```

All following commands assume you are working from this folder, i.e. you must `cd` into the directory
created by the previous step.

Run ```pip install -r requirements.txt``` to install the additional packages required by this script.

**Note**:If you got some issues with Deepspeed and cpu_adam, you can uninstall deepspeed and reinstall by git repo.(https://huggingface.co/docs/transformers/main/main_classes/deepspeed)
Example:
```
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 pip install .  --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
```
And also libaio library is needed.
You can install libaio library using `apt install libaio-dev` cli.


## Configuring training parameters
First of all, you have to configure accelerate with `accelerate config` cli.
You can check your accelerate configuration by `accelerate env` cli.
Here's example of env result of accelerate.
```
Accelerate version: 0.20.3
Platform: Linux-5.4.0-148-generic-x86_64-with-glibc2.31
Python version: 3.10.11
Numpy version: 1.21.6
PyTorch version (GPU?): 1.13.1+cu116 (True)
PyTorch XPU available: False
System RAM: 1007.74 GB
GPU type: NVIDIA A100-SXM4-80GB
Accelerate default config:
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
use_cpu: False
num_processes: 4
machine_rank: 0
num_machines: 1
rdzv_backend: static
same_network: True
main_training_function: main
deepspeed_config: {'deepspeed_config_file': 'dsCpuConfig.json', 'zero3_init_flag': True}
downcast_bf16: no
tpu_use_cluster: False
tpu_use_sudo: False
tpu_env: []
```

And you have to login to wandb with wandb api key for loading data from wandb.
Please type `wandb init` on cmd and input api key.

All configurable parameters for training are visible and documented in `conf/config.yaml`. 
The defaults are chosen for quick training and not tuned; you will need to experiment and adjust 
these.

And also you can configure deepspeed plugin with `dsCpuConfig.json`(for cpu parameter offloading) and `dsNVMEconfig.json`(for NVME parameter offloading).


You do not need to edit this file to change the parameters; they may be overridden when you call this
script. e.g., if you wish to change the model to `nomic-ai/gpt4all-j`, and the output directory to `nomic-ai/gpt4all-j-tuned`, you would run:

```commandline
accelerate launch gpt4all_finetuning.py model.name=nomic-ai/gpt4all-j output_dir=nomic-ai/gpt4all-j-tuned dataset.wandb=1 dataset.name=/opentensor/opentensor-validator/runs/kltiefxf
```

Note the nested structure in the config, since `model` is above `name` in `conf.yaml`, you must override
`model.name` when invoking the command.


## Fine-tuning with the wandb url

-When you're running script,if you didn't run `wandb init`, it will require wandb key and some informations.
Please copy/paste your wandb api key and feel free input the informations manually.

If you have a wandb link from the validator, you can override `dataset.name` as above and set `wandb` as 1:
```bash
accelerate launch gpt4all_finetuning.py dataset.wandb=1 dataset.name=/opentensor/opentensor-validator/runs/kltiefxf
```
or
```bash
accelerate launch gpt4all_finetuning.py
```


## Limitations & Warnings

Early stopping is not yet supported. Many features are implemented but not thoroughly tested, if
you encounter an issue, reach out on discord or (preferably) create an issue on this github page.
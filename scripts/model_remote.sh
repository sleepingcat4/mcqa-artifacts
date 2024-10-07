#!/bin/bash

model_name="phi 2" # model nickname (for saving in folders)
model_name_hf="microsoft/phi-2" # huggingface directory

# list of experiments
# see all possible experiments in: /mcqa-artifacts/model/data_loader.py
experiments=("normal" "artifact_choices_cot")

# list of datasets to test
# see all possible datasets in: /mcqa-artifacts/model/data_loader.py
datasets=("ARC")
train_dataset_split="train"
eval_dataset_split="test"
hf_dataset_name="nbalepur/mcqa_artifacts"

# what partition of the dataset to run
# can be "full" or in halves (e.g. "first_half"), quarters (e.g. "first_quarter"), or eigths (e.g. "first_eighth")
partition="full" 

hf_token=... # huggingface token (for downloading gated models)
load_in_8bit="False" # load the model in 8bit? ("False" or "True")
load_in_4bit="False" # load the model in 4bit? ("False" or "True")
use_20_fewshot="False" # use a 20-shot prompt in ARC? ("False" or "True") => we set this to "True" for Falcon 

res_dir="/mcqa-artifacts/results/" # Results folder directory
prompt_dir="/mcqa-artifacts/prompts/" # Prompt folder directory
cache_dir="/mcqa-artifacts/cache/" # Cache directory to save the model

datasets_str=$(IFS=" "; echo "${datasets[*]}")
experiments_str=$(IFS=" "; echo "${experiments[*]}")

# add the correct file below
python3 /mcqa-artifacts/model/run_hf_remote.py \
--model_name="$model_name" \
--model_name_hf="$model_name_hf" \
--dataset_name="$datasets_str" \
--train_dataset_split="$train_dataset_split" \
--eval_dataset_split="$eval_dataset_split" \
--hf_dataset_name="$hf_dataset_name" \
--hf_token="$hf_token" \
--load_in_4bit="$load_in_4bit" \
--load_in_8bit="$load_in_8bit" \
--partition="$partition" \
--prompt_types="$experiments_str" \
--use_20_fewshot="$use_20_fewshot" \
--res_dir="$res_dir" \
--prompt_dir="$prompt_dir" \
--cache_dir="$cache_dir"
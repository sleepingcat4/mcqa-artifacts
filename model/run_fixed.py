# imports and directory setup
from data_loader import create_data, DatasetName, PromptType

import pickle
import datasets
import json
from transformers import pipeline
import torch
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
import copy
from transformers import AutoTokenizer
from huggingface_hub.hf_api import HfFolder

# =========================================== Argument Setup ===========================================

def setup():
    def enum_type(enum):
        enum_members = {e.name: e for e in enum}

        def converter(input):
            out = []
            for x in input.split():
                if x in enum_members:
                    out.append(enum_members[x])
                else:
                    raise argparse.ArgumentTypeError(f"You used {x}, but value must be one of {', '.join(enum_members.keys())}")
            return out

        return converter

    # hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        '-m',
        type=str,
        help="(Nick)name of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--model_name_hf",
        type=str,
        help="Name of the model on hugging face",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--dataset_name",
        nargs='*',
        type=enum_type(DatasetName),
        help="Name of the dataset (in dataset_name column)",
        default=[],
    )
    parser.add_argument(
        "--train_dataset_split",
        nargs='*',
        type=str,
        help="Training dataset split",
        default="",
    )
    parser.add_argument(
        "--eval_dataset_split",
        nargs='*',
        type=str,
        help="Evaluation dataset split",
        default="",
    )
    parser.add_argument(
        "--load_in_8bit",
        type=str,
        help="Should we load the model in 8 bit?",
        default="False",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=str,
        help="Should we load the model in 4 bit?",
        default="False",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Huggingface token for access to the model",
        default="",
    )
    parser.add_argument(
        '--prompt_types', 
        nargs='*', 
        type=enum_type(PromptType), 
        help='Prompt types/experiments to run', 
        default=[]
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Which partition should be done",
        default="none",
    )
    parser.add_argument(
        "--use_20_fewshot",
        type=str,
        help="Should we use 20 fewshot examples? (for smaller models)",
        default="False",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        help="Absolute directory of the prompt folder",
        default="False",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Absolute directory of the cache folder for models",
        default="False",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Absolute directory of the output results folder",
        default="False",
    )

    args = parser.parse_args()
    print(args)
    load_in_4bit = (args.load_in_4bit == 'True')
    load_in_8bit = (args.load_in_8bit == 'True')
    use_20_fewshot = (args.use_20_fewshot == 'True')

    assert(not (load_in_4bit and load_in_8bit))

    dataset_names = args.dataset_name
    dataset_split = (args.train_dataset_split, args.eval_dataset_split)
    prompt_types = args.prompt_types
    model_name = args.model_name
    hf_model_name = args.model_name_hf
    partition = args.partition

    hf_token = args.hf_token
    HfFolder.save_token(hf_token)

    return dataset_names, dataset_split, prompt_types, model_name, hf_model_name, load_in_4bit, load_in_8bit, use_20_fewshot, partition, args.prompt_dir, args.res_dir, args.cache_dir

# =========================================== Load Model ===========================================

def load_model(hf_model_name, load_in_4bit, load_in_8bit, cache_dir):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir = cache_dir)

    # set up pipeline
    dtype = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "auto": "auto",
    }['auto']
    pipe = pipeline(
        model=hf_model_name,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=dtype,
        min_new_tokens=5,
        max_new_tokens=200,
        model_kwargs={"cache_dir": cache_dir, "do_sample": False, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
    )
    return pipe, tokenizer

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens = [], prompt_len = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist)+i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen

def generate_text(prompt, stop_token):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:], prompt_len=input_ids.shape[1])])
    return pipe(prompt, 
    stopping_criteria=stopping_criteria,
    return_full_text=False)[0]['generated_text'][:-len(stop_token)].strip()

def run_inference(dataset_names, dataset_split, prompt_types, model_name, partition, use_20_fewshot, pipe, tokenizer, prompt_dir, res_dir):

    # Hardcoded dataset name
    hf_dataset_name = "nbalepur/mcqa_artifacts"

    # load data
    ds = datasets.load_dataset(hf_dataset_name, split=dataset_split)

    for dataset_name in dataset_names[0]:

        # results directory setup
        results_dir = f'{res_dir}{dataset_name.value}/{model_name}'

        for pt in prompt_types[0]:
            data = create_data(ds, dataset_name, dataset_split, pt, prompt_dir, use_20_fewshot=use_20_fewshot)
            input_prompts, output_letters, stop_token = data['input'], data['output'], data['stop_token']

            # run generation
            answers = {'raw_text': [], 'prompt': [], 'answer': output_letters}

            partition_map = {'full': (0, len(input_prompts)),
                             'first_half': (0, int(0.5 * len(input_prompts))),
                             'second_half': (int(0.5 * len(input_prompts)), len(input_prompts)),
                             'first_quarter': (0, int(0.25 * len(input_prompts))),
                             'second_quarter': (int(0.25 * len(input_prompts)), int(0.5 * len(input_prompts))),
                             'third_quarter': (int(0.5 * len(input_prompts)), int(0.75 * len(input_prompts))),
                             'fourth_quarter': (int(0.75 * len(input_prompts)), len(input_prompts)),
                             'first_eighth': (0, int(0.125 * len(input_prompts))),
                             'second_eighth': (int(0.125 * len(input_prompts)), int(2*0.125 * len(input_prompts))),
                             'third_eighth': (int(2*0.125 * len(input_prompts)), int(3*0.125 * len(input_prompts))),
                             'fourth_eighth': (int(3*0.125 * len(input_prompts)), int(4*0.125 * len(input_prompts))),
                             'fifth_eighth': (int(4*0.125 * len(input_prompts)), int(5*0.125 * len(input_prompts))),
                             'sixth_eighth': (int(5*0.125 * len(input_prompts)), int(6*0.125 * len(input_prompts))),
                             'seventh_eighth': (int(6*0.125 * len(input_prompts)), int(7*0.125 * len(input_prompts))),
                             'eighth_eighth': (int(7*0.125 * len(input_prompts)), len(input_prompts)),
                             }
            start, end = partition_map[partition]
            input_prompts = input_prompts[start:end]
            output_letters = output_letters[start:end]

            for i, (ipt, opt) in tqdm.tqdm(enumerate(zip(input_prompts, output_letters))):
                text = generate_text(ipt, stop_token)
                answers['raw_text'].append(text)
                answers['prompt'].append(ipt)

            # save results
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            with open(f'{results_dir}/{dataset_split[1]}_{pt.value}_{partition}.json', 'w') as f:
                json.dump(answers, f)

if __name__ == "__main__":
    dataset_names, dataset_split, prompt_types, model_name, hf_model_name, load_in_4bit, load_in_8bit, use_20_fewshot, partition, prompt_dir, res_dir, cache_dir = setup()
    pipe, tokenizer = load_model(hf_model_name, load_in_4bit, load_in_8bit, cache_dir)
    run_inference(dataset_names, dataset_split, prompt_types, model_name, partition, use_20_fewshot, pipe, tokenizer, prompt_dir, res_dir)

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
import warnings
from accelerate.utils import InitProcessGroupKwargs
warnings.filterwarnings("ignore")
from data.dataset_loader import get_datasets
import random

def generate_vllm(config):
    model_path = config["model_name"]
    world_size = config["world_size"]
    data_frac = config["data_frac"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(model=model_path, tensor_parallel_size=world_size)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

    # load data
    train_data, val_data, test_data, prompt = get_datasets(config['seed'], config['input_dir'])
    if config["split"] == 'test':
        data = test_data
    elif config["split"] == 'val':
        data = val_data
    else:
        data = train_data
    # random.seed(seed)
    random.shuffle(data)
    if config['frac_len'] > 0:
        sub_len = config['frac_len']
        if sub_len * (data_frac + 1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    else:
        data = data[:]
    
    prompts_all = ["Question: " + data[idx]['question'] + "\n\nAnswer: " for idx in range(len(data))]
    prompts_old = [data[idx]['question'] for idx in range(len(data))]
    corrects_all = [data[idx]['answer'] for idx in range(len(data))]

    start = time.time()

    # run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, llm.generate(prompts_all, sampling_params)))
    results = [r.replace(tokenizer.eos_token, "").lstrip() for r in results_gathered]

    timediff = time.time() - start
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in tqdm(range(len(corrects_all))):
        d = {"question": prompts_old[idx], "answer": corrects_all[idx], "generation": results[idx]}
        if config["split"] == 'test':
            file_name = f"{config['output_dir']}/{config['data_frac']}_test.jsonl"
        else:
            file_name = f"{config['output_dir']}/{config['data_frac']}.jsonl"
        with open(file_name, 'a') as f:
            json.dump(d, f)
            f.write('\n')

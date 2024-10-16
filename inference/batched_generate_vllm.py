import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from vllm import LLM, SamplingParams
from data.prompt_loader import load_prompts
import json

def run_process_on_gpu(config, gpu_queue, data_frac):
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running on GPU: {gpu_id}")
    # Assuming the existence of a function that handles te generation process for a single GPU
    generate_on_single_gpu(config, data_frac)
    gpu_queue.put(gpu_id)

def generate_on_single_gpu(config, data_frac):
    output_dir = config["output_dir"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating on GPU with data fraction: {data_frac}...")
    # load the base model and tokenizer
    model_path = config["model_name"]
    tokenizer_path = config["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    world_size = config["tp_per_worker"]
    llm = LLM(model=model_path, tokenizer=tokenizer_path, tensor_parallel_size=world_size)
    # sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)
    sampling_params = SamplingParams(
        temperature=config["sampling_params"]["temperature"],
        top_p=config["sampling_params"]["top_p"],
        max_tokens=config["sampling_params"]["max_tokens"],
        n=config["sampling_params"]["n"],
    )

    # load data
    prompt = load_prompts(config["input_dir"])
    if 'split' in config and 'subset' in config:
        data = load_dataset(config["input_dir"], config["subset"], split=config["split"])
    elif 'subset' not in config:
        data = load_dataset(config["input_dir"], split=config["split"])
    seed = config["seed"]
    data = data.shuffle(seed=seed)

    if config["frac_len"] > 0:
        sub_len = config["frac_len"]
        if sub_len * (data_frac + 1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    
    if prompt is not None:
        prompts_all = ['<s> [INST] ' + prompt + "\n\nQ: " + data[idx]['question'] + "[/INST] \nA: " for idx in range(len(data))]
    else:
        prompts_all = ["<s> [INST] Q: " + data[idx]['question'] + "[/INST] \nA: " for idx in range(len(data))]
    prompts_old = [data[idx]['question'] for idx in range(len(data))]
    corrects_all = [data[idx]['answer'] for idx in range(len(data))]

    start_time = time.time()

    # run vllm
    if config["sampling_params"]["n"] == 1:
        results_gathered = list(
            map(lambda x: x.outputs[0].text, llm.generate(prompts_all, sampling_params))
        )
    else:
        # flatten x.outputs
        results_gathered = list(
            map(lambda x: [y.text for y in x.outputs], llm.generate(prompts_all, sampling_params))
        )
        results_gathered = [item for sublist in results_gathered for item in sublist]
    # print(results_gathered)
    # input()
    results = [r.replace("</s>", "").lstrip() for r in results_gathered]
    print(len(results))
    timediff = time.time() - start_time
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in range(len(corrects_all)):
        d = {"question": prompts_old[idx], "answer": corrects_all[idx], "generation": results[idx]}
        if config["split"] == 'test':
            file_name = f"{config['output_dir']}/{config['data_frac']}_test.jsonl"
        else:
            file_name = f"{config['output_dir']}/{config['data_frac']}.jsonl"
        with open(file_name, 'a') as f:
            json.dump(d, f)
            f.write('\n')

def generate_on_multiple_gpus(config):
    start = time.time()
    mp.set_start_method("spawn", force=True)
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Check if the model is already downloaded
    model_path = config["model_name"]
    tokenizer_path = config["tokenizer_name"]
    if not model_path.startswith("/"): # hub_path
        filepath = try_to_load_from_cache(model_path, "config.json")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_directory = cache_dir / f"models--{model_path.replace('/', '--')}"

        print(f"checking cache results: {filepath}")
        if isinstance(filepath, str):
            print(f"Model {model_path} is alread downloaded.")
        else:
            print(f"Model {model_path} is not downloaded yet, will download now.")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"Model {model_path} downloaded.")
            del tokenizer
            del model
    else:
        model_directory = model_path
    print(f"Model directory: {model_directory}")

    # create a pool of processes. Each process will run on a seperate GPU
    with mp.Manager() as manager:
        gpu_queue = manager.Queue()
        # Add the gpu_id to the queue
        for i in range(num_gpus):
            gpu_queue.put(i)
        
        with mp.Pool(processes=num_gpus) as pool:
            # Partial function with all arguments except the one that changes per process (data_frac)
            func = partial(
                run_process_on_gpu,
                config,
            )

            # for each data_frac, scheduling one task
            res_futs = []
            for data_frac in range(config["num_data_frac"]):
                res_futs.append(
                    pool.apply_async(
                        func,
                        (
                            gpu_queue,
                            data_frac,
                        )
                    )
                )
            for res in res_futs:
                res.get()
    print(f"Total time taken: {time.time() - start}")

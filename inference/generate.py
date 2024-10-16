from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import warnings
warnings.filterwarnings("ignore")

from data.dataset_loader import get_datasets
from utils.util import load_config
from data.prompt_loader import load_prompts
import random
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def prepare_prompts(prompts, tokenizer, batch_size=4):
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding='longest',
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False
            ).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok

def generate(config):
    model_path = config["model_name"]
    tokenizer_path = config["tokenizer_name"]
    data_frac = config["data_frac"]
    batch_size = config["batch_size"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # credentials
    token = config["token"]

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # load_data
    prompt = load_prompts(config["input_dir"])
    seed = config["seed"]
    train_dataset, val_dataset, test_dataset, prompt = get_datasets(seed, config["input_dir"])
    if config["split"] == 'train':
        data = train_dataset
    elif config["split"] == 'val':
        data = val_dataset
    elif config["split"] == 'test':
        data = test_dataset
    random.seed(seed)
    random.shuffle(data)
    if config["frac_len"] > 0:
        sub_len = config["frac_len"]
        if sub_len * (data_frac + 1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    
    print(data[0])

    # modification here
    if prompt is not None:
        prompts_all = ["<s> [INST] " + prompt + "\n\nQ: " + data[idx]['question'] + " [/INST] \nA: " for idx in range(len(data))]
    else:
        prompts_all = ["<s> [INST] Q: " + data[idx]['question'] + " [/INST] \nA: " for idx in range(len(data))]
    prompts_old = [data[int(idx/config["num_return_sequences"])]['question'] for idx in range(config["num_return_sequences"]*len(data))]
    corrects_all = [data[int(idx/config["num_return_sequences"])]['answer'] for idx in range(config["num_return_sequences"]*len(data))]

    print(len(prompts_old), len(corrects_all))
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the avilable GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        prompt_batches = prepare_prompts(prompts, tokenizer, batch_size)
        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=config["max_length"], pad_token_id=tokenizer.eos_token_id, num_return_sequences=config["num_return_sequences"])
            inputs_tokenized = prompts_tokenized["input_ids"].repeat_interleave(config["num_return_sequences"], dim=0)

            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):] for tok_in, tok_out in zip(inputs_tokenized, outputs_tokenized)]
            # decode the generated tokens
            outputs = tokenizer.batch_decode(outputs_tokenized)
            # print(outputs.shape)
            results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered = gather_object(results)
    results = [r.replace(tokenizer.eos_token, "").lstrip() for r in results_gathered]
    print(len(results))
    # input()
    if accelerator.is_local_main_process:
        timediff = time.time() - start
        print(f"Time elapsed: {timediff}")

        # collecting data
        for idx in range(len(corrects_all)):
            d = {"question": prompts_old[idx], "answer": corrects_all[idx], "generation": results[idx]}
            if config["split"] == 'test':
                if config['num_return_sequences'] == 1:
                    file_name = f"{config['output_dir']}/{config['data_frac']}_test.jsonl"
                else:
                    file_name = f"{config['output_dir']}/{config['data_frac']}_test_candidates.jsonl"
            else:
                file_name = f"{config['output_dir']}/{config['data_frac']}.jsonl"
            with open(file_name, 'a') as f:
                json.dump(d, f)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/gsm8k.yaml', type=str, help='Path to the config file')
    args = parser.parse_args()

    config_path = args.config
    assert os.path.isfile(config_path), f"Invalid config path: {config_path}"

    config = load_config(config_path)
    generate(config)
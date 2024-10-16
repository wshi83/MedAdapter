import os
from openai import OpenAI, AzureOpenAI
import numpy as np
from utils.credentials import api_key_list
from tenacity import wait_random_exponential, stop_after_attempt, retry, RetryError
from data.dataset_loader import get_datasets
import json
from pathlib import Path
from tqdm import tqdm

class generate_openai():
    def __init__(self, config=None):
        self.api_key_list = api_key_list(config["generator"]["openai_credentials"])
        self.api_idx = 0
        self.client = AzureOpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"], 
            api_version=self.api_key_list[self.api_idx]["api_version"],
            azure_endpoint=self.api_key_list[self.api_idx]["azure_endpoint"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
        self.config = config
        self.token_usage = {"input": 0, "output": 0}

    def switch_api_key(self):
        self.api_idx = (self.api_idx + 1) % len(self.api_key_list)
        self.client = AzureOpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"],
            api_version=self.api_key_list[self.api_idx]["api_version"],
            azure_endpoint=self.api_key_list[self.api_idx]["azure_endpoint"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
    
    def query(self, prompt, temp=None, n=None, stop=None, max_tokens=None,):
        prompt_chat = [
            {"role": "user", "content": prompt}
        ]
        flag = False
        num_trials = 0
        while not flag:
            try:
                raw_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt_chat,
                    max_tokens=self.config['generator']['max_length'] if max_tokens is None else max_tokens,
                    temperature=self.config['generator']['temperature'] if temp is None else temp,
                    frequency_penalty=self.config['generator']['frequency_penalty'],
                    presence_penalty=self.config['generator']['presence_penalty'],
                    stop=self.config['generator']['stop'] if stop is None else stop,
                    n=self.config['generator']['num_return_sequences'] if n is None else n,
                )
                self.token_usage["input"] += raw_response.usage.prompt_tokens
                self.token_usage["output"] += raw_response.usage.completion_tokens

                contents = [choice.message.content.strip() for choice in raw_response.choices]
                flag = True
                if len(contents) == 0:
                    flag = False
                    raise RuntimeError("No response from the API")
            except:
                self.switch_api_key()
                flag = False
                num_trials += 1
            if num_trials > 3:
                flag = True
                contents = None
        return contents
    
    def generate(self):
        train_data, val_data, test_data, prompt = get_datasets(self.config['seed'], self.config['generator']['input_dir'])
        if self.config['generator']["split"] == 'test':
            data = test_data
        elif self.config['generator']["split"] == 'val':
            data = val_data
        else:
            data = train_data
        generation = []
        template = "{prompt}\nQ: {question}\nA:"
        for idx in tqdm(range(len(data))):
            prompt_msg = template.format(prompt=prompt, question=data[idx]['question'], answer=data[idx]['answer'])
            responses = self.query(prompt_msg)
            if responses != None:
                for res in responses:
                    generation.append({"question": data[idx]['question'], "answer": data[idx]['answer'], "generation": res})

        output_dir = Path(self.config['generator']["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for gen in generation:
            if self.config['generator']["split"] == 'test':
                file_name = f"{self.config['generator']['output_dir']}/{self.config['generator']['data_frac']}_test.jsonl"
            else:
                file_name = f"{self.config['generator']['output_dir']}/{self.config['generator']['data_frac']}.jsonl"
            with open(file_name, 'a') as f:
                json.dump(gen, f)
                f.write('\n')

class generate_vllm_openai():
    def __init__(self, config=None):
        self.api_key_list = api_key_list(config["generator"]["openai_credentials"])
        self.api_idx = 0
        self.client = OpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"], 
            base_url=self.api_key_list[self.api_idx]["base_url"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
        self.config = config
        self.token_usage = {"input": 0, "output": 0}

    def switch_api_key(self):
        self.api_idx = (self.api_idx + 1) % len(self.api_key_list)
        self.client = OpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"], 
            base_url=self.api_key_list[self.api_idx]["base_url"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
    
    def query(self, prompt, temp=None, n=None, stop=None, max_tokens=None,):
        prompt_chat = [
            {"role": "user", "content": prompt}
        ]
        flag = False
        num_trials = 0
        while not flag:
            try:
                raw_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt_chat,
                    max_tokens=self.config['generator']['max_length'] if max_tokens is None else max_tokens,
                    temperature=self.config['generator']['temperature'] if temp is None else temp,
                    frequency_penalty=self.config['generator']['frequency_penalty'],
                    presence_penalty=self.config['generator']['presence_penalty'],
                    stop=self.config['generator']['stop'] if stop is None else stop,
                    n=self.config['generator']['num_return_sequences'] if n is None else n,
                )
                self.token_usage["input"] += raw_response.usage.prompt_tokens
                self.token_usage["output"] += raw_response.usage.completion_tokens

                contents = [choice.message.content.strip() for choice in raw_response.choices]
                flag = True
                if len(contents) == 0:
                    flag = False
                    raise RuntimeError("No response from the API")
            except:
                self.switch_api_key()
                flag = False
                num_trials += 1
            if num_trials > 3:
                flag = True
                contents = None
        return contents
    
    def generate(self):
        train_data, val_data, test_data, prompt = get_datasets(self.config['seed'], self.config['generator']['input_dir'])
        if self.config['generator']["split"] == 'test':
            data = test_data
        elif self.config['generator']["split"] == 'val':
            data = val_data
        else:
            data = train_data
        generation = []
        template = "{prompt}\nQ: {question}\nA:"
        for idx in tqdm(range(len(data))):
            prompt_msg = template.format(prompt=prompt, question=data[idx]['question'], answer=data[idx]['answer'])
            responses = self.query(prompt_msg)
            if responses != None:
                for res in responses:
                    generation.append({"question": data[idx]['question'], "answer": data[idx]['answer'], "generation": res})

        output_dir = Path(self.config['generator']["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for gen in generation:
            if self.config['generator']["split"] == 'test':
                file_name = f"{self.config['generator']['output_dir']}/{self.config['generator']['data_frac']}_test.jsonl"
            else:
                file_name = f"{self.config['generator']['output_dir']}/{self.config['generator']['data_frac']}.jsonl"
            with open(file_name, 'a') as f:
                json.dump(gen, f)
                f.write('\n')
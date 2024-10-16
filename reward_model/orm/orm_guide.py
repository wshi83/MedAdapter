import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from inference.generate import generate
from reward_model.prm.prm_data import decompose_samples

class orm_guided_generation():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'orm-classification' in config["reward_model"]["type"]:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(config["reward_model"]["model_name"], num_labels=2)
            self.reward_tokenizer = AutoTokenizer.from_pretrained(config["reward_model"]["tokenizer_name"])
        self.reward_model = self.reward_model.to(self.device)
        self.reward_tokenizer = self.reward_tokenizer
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

    def generate_solutions(self):
        # self.config["generator"]["output_dir"] = self.config["generator"]["output_dir"] + "_temp_candidates"
        generate(self.config["generator"])
        if self.config["generator"]["split"] == 'test':
            if self.config['generator']['num_return_sequences'] == 1:
                solution_file = f"{self.config['generator']['output_dir']}/{self.config['generator']['data_frac']}_test.jsonl"
            else:
                solution_file = f'{self.config["generator"]["output_dir"]}/{self.config["generator"]["data_frac"]}_test_candidates.jsonl'
        else:
            solution_file = f'{self.config["generator"]["output_dir"]}/{self.config["generator"]["data_frac"]}.jsonl'
        # use json to read the list of dictionaries from solution_file
        self.solutions = load_dataset('json', data_files=solution_file, split='train')

    def read_solutions(self):
        solution_file = self.config["generator"]["input_dir"]
        # use json to read the list of dictionaries from solution_file
        self.solutions = load_dataset('json', data_files=solution_file, split='train')
    
    def tokenizer_dataset(self, data):
        return self.reward_tokenizer(data["text"], truncation=False)
    
    def process_solutions(self):
        # pass them through the reward model
        samples = []
        idxes = []
        template = "Q: {question}\nA: {answer}"
        for idx in tqdm(range(len(self.solutions))):
            sample = self.solutions[idx]
            prediction = sample['generation'].strip().split('\n#### ')
            if len(prediction) != 1:
                pred_answer = prediction[1].split('\n')[0]
                prediction = prediction[0] + '\n#### ' + pred_answer
            else:
                prediction = prediction[0]
            sample_text = template.format(question=sample['question'], answer=prediction)
            samples.append(sample_text)
            idxes.append(idx)
        
        # convert samples into huggingface dataset with Dataset.from_dict
        self.dataset = Dataset.from_dict({"idxes": idxes, "text": samples}).with_format("torch")
        data_collator = DataCollatorWithPadding(tokenizer=self.reward_tokenizer)
        def tokenized_dataset(data):
            return self.reward_tokenizer(data["text"], truncation=True)
        self.tokenized_dataset = self.dataset.map(tokenized_dataset, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(["idxes", "text"])
        # self.tokenized_dataset = self.tokenized_dataset.set_format("torch")
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=self.config["reward_model"]["per_device_eval_batch_size"], collate_fn=data_collator, shuffle=False)
    
    def get_reward_score(self):
        # pass the dataset through the reward model
        self.solution_scores = []
        num = 0
        for batch in self.dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.reward_model(**batch)
            logits = outputs.logits.detach().cpu() # B, 2
            # apply softmax on the logits
            logits = torch.nn.functional.softmax(logits, dim=-1)
            scores = logits[:,1]
            self.solution_scores.append(scores)
        self.solution_scores = torch.cat(self.solution_scores, dim=0)
        print(len(self.solution_scores), len(self.dataset))
    
    def select_and_save(self):
        # select the solution with the highest score every num_return_sequences solutions
        selected_solutions = []
        for idx in range(0, len(self.solutions), self.config["generator"]["num_return_sequences"]):
            if idx + self.config["generator"]["num_return_sequences"] < len(self.solutions):
                max_idx = np.argmax(self.solution_scores[idx:idx+self.config["generator"]["num_return_sequences"]])
                selected_solutions.append(self.solutions[idx+int(max_idx)])
            else:
                max_idx = np.argmax(self.solution_scores[idx:])
                selected_solutions.append(self.solutions[idx+int(max_idx)])
            # for i in range(self.config["generator"]["num_return_sequences"]):
            #     print(self.solutions[idx+i], ' ---> ', self.solution_scores[idx+i])
        if not os.path.exists(self.config["generator"]["output_dir"]):
            os.mkdir(self.config["generator"]["output_dir"])
        if self.config["generator"]["split"] == 'test':
            solution_file = f'{self.config["generator"]["output_dir"]}/{self.config["generator"]["data_frac"]}_selected_test.jsonl'
        else:
            solution_file = f'{self.config["generator"]["output_dir"]}/{self.config["generator"]["data_frac"]}_selected.jsonl'
        
        for sol in selected_solutions:
            with open(solution_file, 'a') as f:
                json.dump(sol, f)
                f.write('\n')
    
    def guide_generation(self):
        # check if the 'input_dir' file exists
        if not os.path.exists(self.config["generator"]["input_dir"]):
            self.generate_solutions()
        else:
            self.read_solutions()
        self.process_solutions()
        self.get_reward_score()
        self.select_and_save()
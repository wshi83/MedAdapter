import torch
from torch.utils.data import DataLoader
from accelerate.state import PartialState
from accelerate.utils import release_memory, InitProcessGroupKwargs
import datasets
from datasets import Dataset
datasets.disable_progress_bar()
from datetime import timedelta
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_LOG_MODEL"] = "false"
from tqdm.auto import tqdm
from utils.util import get_answer_start_idx
from utils.loggers import loggers
from accelerate import Accelerator

from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_constant_schedule_with_warmup
)

torch.cuda_empty_cache()
torch.set_printoptions(threshold=10_000)

class reward_model():
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["reward_model"], truncation_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=96000))
        self.accelerator = Accelerator(
            split_batches=False,
            mixed_precision='fp16',
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            log_with='wandb' if self.config.get("log_with_wandb", False) else None,
            project_dir='logs' if self.config.get("log_with_wandb", False) else None,
            device_placement=True,
            kwargs_handlers=[kwargs]
        )
        self.mode = config["reward_mode"]
        if self.mode == 'generation':
            self.model = AutoModelForCausalLM.from_pretrained(config["reward_model"], trust_remote_code=True)
        elif self.mode == 'classification-1':
            self.model=AutoModelForSequenceClassification.from_pretrained(config["reward_model"], trust_remote_code=True, num_labels=1)
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        elif self.mode == 'classification-2':
            self.model=AutoModelForSequenceClassification.from_pretrained(config["reward_model"], trust_remote_code=True, num_labels=2)
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise NotImplementedError
        
        self.model.config.use_cache = False if "phi" in self.config["reward_model"].lower() else True
        self.model.config.pretraining_tp = 1

        if self.tokenizer.pad_token_id is None:
            self.accelerator.print("Adding pad token to the tokenizer...")
            self.tokenizer.add_special_tokens({"pad_token": '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.answer_token = self.tokenizer.encode("\nA: ", return_tensors="pt", add_special_tokens=False)[0, 1:]
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"] * self.accelerator.gradient_accumulation_steps,
            weight_decay=0.01,
        )
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
        )
        self.accelerator.print(f"Distributed: {self.accelerator.distributed_type}, Mixed precision: {self.accelerator.mixed_precision}")
    
    def build_dataset(self, positive_texts, negative_texts, save_to):
        pos_len, neg_len = len(positive_texts), len(negative_texts)
        labels = - torch.ones(pos_len + neg_len)
        labels[:pos_len] = 1

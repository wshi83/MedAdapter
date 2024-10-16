import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import InitProcessGroupKwargs, release_memory
from datetime import timedelta

class generator_model():
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["generator_model"])
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
        self.model = AutoModelForCausalLM.from_pretrained(config["generator_model"], trust_remote_code=True)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        if self.tokenizer.pad_token is None:
            self.acclerator.print("Adding pad token to the tokenizer...")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.answer_token = self.tokenizer.encode("\nA: ", return_tensors="pt", add_special_tokens=False)[0, 1:]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"] * self.accelerator.gradient_accumulation_steps,
            weight_decay=0.01
        )
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"]
        )
        self.accelerator.print(f"Distributed: {self.accelerator.distributed_type}, Mixed precision: {self.accelerator.mixed_precision}")
    
    def input_text_process(self, input_texts):
        return input_texts
    
    def initialization_step(self):
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

    def inference_step(self, input_texts):
        self.model.eval()
        input_texts = self.input_text_process(input_texts)
        inputs = self.tokenizer(
            [input_texts], 
            return_tensors="pt", 
            add_special_tokens=self.config["add_special_tokens"], 
            padding=True,
            truncation=True,
        ).to(self.accelerator.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        print(input_texts)
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.accelerator.unwrap_model(self.model),
            tokenizer=self.tokenizer,
            device=self.accelerator.device,
            # do_sample=True,
            # top_k=10,
            num_return_sequences=1,
            # eos_token_id=self.tokenizer.eos_token_id,
            max_length=500
        )
        outputs = pipeline(input_texts)
        print(outputs)
        return outputs[0]["generated_text"]
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

import argparse
from data.prompt_loader import load_prompts
from data.dataset_loader import get_datasets
from utils.util import load_config

def load_model(config):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["use_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["use_nested_quant"],
    )

    if compute_dtype == torch.float16 and config["use_4bit"]:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto",
        # quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        r=config["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

def train(config):
    model, tokenizer, peft_config = load_model(config)
    training_arguments = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim=config["optim"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        max_grad_norm=config["max_grad_norm"],
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        group_by_length=config["group_by_length"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to=config["report_to"],
    )

    # dataset = load_dataset(config["dataset_name"], config["subset"], split=config["split"])
    # prompt = load_prompts(config["dataset_name"])
    dataset, _, _, prompt = get_datasets(42, config["dataset_name"])
    
    train_text_list = []
    for i in range(len(dataset)):
        template = "<s> [INST] Q: {question} [/INST] \nA: {answer} </s>"
        question = dataset[i]['question']
        answer = dataset[i]['answer']
        if not '####' in answer:
            answer = '#### ' + answer
        if prompt != None:
            train_text_list.append(prompt + '\n\n' + template.format(question=question, answer=answer))
        else:
            train_text_list.append(template.format(question=question, answer=answer))

    temp_dataset = Dataset.from_dict({
                    "text": train_text_list,
                }).with_format("torch")


    trainer = SFTTrainer(
        model=model,
        train_dataset=temp_dataset,
        # peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config["packing"],
    )

    trainer.train()
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    trainer.model.save_pretrained(config["output_dir"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/gsm8k.yaml', type=str, help='Path to the config file')
    args = parser.parse_args()

    config_path = args.config
    assert os.path.isfile(config_path), f"Invalid config path: {config_path}"

    config = load_config(config_path)
    train(config)

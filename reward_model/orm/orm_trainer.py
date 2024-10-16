import os
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from reward_model.orm.orm_data import prepare_orm_data

def orm_classification_trainer(config):
    train_dataset = prepare_orm_data(config)
    tokenizer = AutoTokenizer.from_pretrained(config["reward_model"]["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=config["reward_model"]["output_dir"],
        learning_rate=config["reward_model"]["learning_rate"],
        per_device_train_batch_size=config["reward_model"]["per_device_train_batch_size"],
        # per_device_eval_batch_size=config["reward_model"]["per_device_eval_batch_size"],
        num_train_epochs=config["reward_model"]["num_train_epochs"],
        weight_decay=config["reward_model"]["weight_decay"],
        # evaluation_strategy=config["reward_model"]["evaluation_strategy"],
        save_strategy=config["reward_model"]["save_strategy"],
        # load_best_model_at_end=config["reward_model"]["load_best_model_at_end"],
        push_to_hub=config["reward_model"]["push_to_hub"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(config["reward_model"]["model_name"], num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    if not os.path.exists(config["reward_model"]["output_dir"]):
        os.makedirs(config["reward_model"]["output_dir"])
    trainer.model.save_pretrained(config["reward_model"]["output_dir"])
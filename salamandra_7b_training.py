from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Model to fine-tune
model_id = "BSC-LT/salamandra-7b-instruct"

# Load tokenizer and model with ROCm-compatible dtype
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",         # Automatically uses bfloat16 on ROCm
    device_map="auto"           # Utilizes all visible GPUs
)

# Load dataset (example: Wikitext-2)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize dataset properly (convert text → input_ids)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

# Remove original "text" column after tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./salamandra7b-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,
    bf16=True,                        # Use ROCm-native bfloat16
    fp16=False,
    report_to="none",                 # Disable external logging
    dataloader_pin_memory=False,      # Avoid ROCm + DDP memory pinning issues
    remove_unused_columns=False
)

# Data collator for language modeling (no masking)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

# Start training
trainer.train()

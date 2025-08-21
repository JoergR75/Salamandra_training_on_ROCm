import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset

# ------------------------------
# Model and Tokenizer Setup
# ------------------------------
model_id = "BSC-LT/salamandra-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",        # Uses bfloat16 automatically on ROCm
    device_map="auto"          # Utilizes all visible GPUs
)

# ------------------------------
# Dataset Loading and Tokenization
# ------------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ------------------------------
# Training Arguments (with DeepSpeed ZeRO-3)
# ------------------------------
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,                 # ZeRO-3: optimizer, gradients, and states partitioned
        "offload_param": {
            "device": "cpu",        # Offload parameters to CPU
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "bf16": {
        "enabled": True              # Use bfloat16
    },
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4
}

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
    bf16=True,
    fp16=False,
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    deepspeed=deepspeed_config      # Enable DeepSpeed ZeRO-3
)

# ------------------------------
# Data Collator
# ------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ------------------------------
# Performance Callback
# ------------------------------
class PerformanceCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.first_step_time = None
        self.total_tokens = 0
        self.total_steps = 0
        self.avg_tokens_per_sec = 0.0
        self.avg_time_per_step = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n[INFO] Training started... Measuring TTFT, T/S, and TOS.\n")

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time()

        # TTFT: Time to first token (after first training step)
        if self.first_step_time is None:
            self.first_step_time = step_time - self.start_time
            print(f"[METRIC] TTFT: {self.first_step_time:.3f} sec")

        # Estimate tokens processed per step
        batch_size = args.per_device_train_batch_size
        seq_len = 256  # fixed max length
        grad_accum = args.gradient_accumulation_steps
        effective_tokens = batch_size * seq_len * grad_accum

        self.total_steps += 1
        self.total_tokens += effective_tokens

        # Tokens/sec = total tokens processed / total elapsed time
        elapsed = step_time - self.start_time
        self.avg_tokens_per_sec = self.total_tokens / elapsed
        self.avg_time_per_step = elapsed / self.total_steps

        if self.total_steps % 10 == 0:
            print(f"[METRIC] Step: {self.total_steps} | "
                  f"T/S: {self.avg_tokens_per_sec:.2f} tokens/sec | "
                  f"TOS: {self.avg_time_per_step:.3f} sec/step")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n===== Final Training Performance =====")
        print(f"Total steps       : {self.total_steps}")
        print(f"Total tokens      : {self.total_tokens}")
        print(f"TTFT              : {self.first_step_time:.3f} sec")
        print(f"Avg T/S           : {self.avg_tokens_per_sec:.2f} tokens/sec")
        print(f"Avg TOS           : {self.avg_time_per_step:.3f} sec/step")
        print(f"Total training    : {total_time:.2f} sec\n")

# ------------------------------
# Trainer Instance
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    callbacks=[PerformanceCallback()]
)

# ------------------------------
# Start Training
# ------------------------------
trainer.train()

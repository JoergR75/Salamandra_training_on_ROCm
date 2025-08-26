# ROCm 6.4.3 + OCL 2.x + Pytorch 2.9.0 + Transformers Setup for Ubuntu 22.04.x and 24.04.x DT and Server build (! 20.04.x will not be supported anymore)
# ==========================================================================================================================================
# This script will automatically install ROCm 6.4.3 + Pytroch 2.9.0 (nightly build) for Ubuntu 22.04.x and 24.04.x automatically downloading
# the correct install script in non interactive mode.
#
# Requirements
# OS:                   Ubuntu Server 22.04.5 LTS (Jammy Jellyfish) or Ubuntu 24.04.2 LTS (Noble Numbat)
# Kernel:               tested: 5.15.0-144 (22.04) and 6.8.0-71 (24.04)
# Supported HW:         CDNA 2, CDNA 3, RDNA 3, RDNA 4
#
# Software
# ROCm(TM) Platform:    6.4.3
# Release:              https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html
# Driver:               https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/
# Pytorch:              2.9.0.dev20250720+rocm6.4
# Transformers:         4.55.2
# Tools:                git (version control system used for tracking changes in computer files)
#                       htop (monitoring - dynamic overview of running processes)
#                       ncdu (NCurses Disk Usage utility, which provides a text-based interface for viewing disk usage)
#                       cmake (CMake is an open-source, cross-platform family of tools designed to build, test, and package software)
#                       libmsgpack-dev (development package for MessagePack, a binary serialization format. MessagePack is designed to be efficient in both size and speed, making it a popular choice for data interchange in performance-sensitive applications)
#                       freeipmi-tools (read BMC version)
#                       git-lfs (is an extension to Git that helps manage large files (such as datasets, images, videos, binaries, etc.) efficiently by replacing them with lightweight text pointers in your Git repository)
#                       rocm-bandwidth-test (is a diagnostic utility which measures and validates memory bandwidth performance between the host (CPU) and GPU(s), as well as between multiple GPUs in a system)
#
# Author: Joerg Roskowetz
# Script process time: ~15 minutes (depending on system and internet configuration)
# Date: August 26th 2025

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
# Training Arguments
# ------------------------------
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
    bf16=True,                        # ROCm-native bfloat16
    fp16=False,
    report_to="none",                 # No external logging
    dataloader_pin_memory=False,      # Avoid ROCm DDP memory pinning issues
    remove_unused_columns=False
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

        if self.total_steps % 10 == 0:  # Print every 10 steps
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

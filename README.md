# 🦎 Salamandra-7B Training on ROCm

This repository provides a PyTorch and Hugging Face Transformers-based training setup for the **Salamandra-7B-Instruct** model on AMD GPUs using ROCm. It includes **bfloat16 training**, **gradient accumulation**, and **performance metrics logging** (TTFT, tokens/sec, time per step) during training.

---

## Features

- Supports **Salamandra-7B-Instruct** model.
- Fully compatible with **ROCm GPUs**.
- Uses **bfloat16 (BF16)** precision for improved performance on AMD GPUs.
- Gradient accumulation for large batch training on limited GPU memory.
- Automatic dataset tokenization with `wikitext-2-raw-v1`.
- Performance monitoring callback logging:
  - TTFT (Time To First Token)
  - T/S (Tokens per Second)
  - TOS (Time per Step)

---

## Requirements

- Python 3.12+
- PyTorch with ROCm support ([use the automated deployment script for ROCm 6.4.3 + Pytorch 2.9.0](https://github.com/JoergR75/ROCm-6.4.3-deployment-on-RDNA4/tree/main))
- Transformers
- Datasets
- Hugging Face Tokenizers

Install dependencies:

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.4
pip3 install transformers datasets
```

---

## Usage

Download the python script:
```bash
wget https://raw.githubusercontent.com/JoergR75/Salamandra_training_on_ROCm/refs/heads/main/salamandra_7b_training.py
```
Run the training script:
```bash
python3 salamandra_7b_training.py
```

The script will automatically:

- Load the model and tokenizer
- Tokenize the dataset
- Train the model with BF16 precision
- Log performance metrics during training

---

## Training Configuration

- Model: BSC-LT/salamandra-7b-instruct
- Dataset: wikitext-2-raw-v1
- Sequence length: 256 tokens
- Batch size: 1 per device
- Gradient accumulation steps: 4
- Learning rate: 5e-5
- Epochs: 3
- Precision: BF16 (ROCm-native)
- Logging steps: every 50 steps
- Checkpointing: every 500 steps, keeping 2 most recent

## Performance Metrics

The script logs key training metrics via a custom callback:

- TTFT (Time to First Token): Time to complete first training step
- T/S (Tokens per Second): Throughput
- TOS (Time per Step): Average time per training step

Example output:
```bash
[METRIC] TTFT: 3.512 sec
[METRIC] Step: 10 | T/S: 12345.67 tokens/sec | TOS: 1.234 sec/step
...
===== Final Training Performance =====
Total steps       : 100
Total tokens      : 256000
TTFT              : 3.512 sec
Avg T/S           : 12500.23 tokens/sec
Avg TOS           : 1.210 sec/step
Total training    : 125.34 sec
```

## Salamandra 7B Model – Download Summary

This document summarizes the **download size** and **files** fetched during the setup of the **[BSC-LT/salamandra-7b-instruct](https://huggingface.co/BSC-LT/salamandra-7b-instruct)** model when running the training test script `salamandra_7B_training.py`.

---

## 1. Tokenizer & Config Files

| File                     | Size    |
|-------------------------|---------|
| `tokenizer_config.json` | **3.80 KB** |
| `tokenizer.model`       | **4.81 MB** |
| `tokenizer.json`        | **19.1 MB** |
| `special_tokens_map.json` | **513 B** |
| `config.json`           | **730 B** |
| `generation_config.json` | **200 B** |
| `README.md`            | **10.5 KB** |

**Subtotal:** ~ **24.0 MB**

---

## 2. Model Checkpoint Files (Sharded)

| File                                   | Size    |
|--------------------------------------|---------|
| `model-00001-of-00004.safetensors`   | **4.98 GB** |
| `model-00002-of-00004.safetensors`   | **5.00 GB** |
| `model-00003-of-00004.safetensors`   | **3.46 GB** |
| `model-00004-of-00004.safetensors`   | **2.10 GB** |
| `model.safetensors.index.json`       | **23.9 KB** |

**Subtotal:** ~ **15.54 GB**

---

## 3. Dataset Files (WikiText-2)

| File                                | Size    |
|----------------------------------|---------|
| `train-00000-of-00001.parquet`   | **6.36 MB** |
| `test-00000-of-00001.parquet`    | **733 KB** |
| `validation-00000-of-00001.parquet` | **657 KB** |

**Subtotal:** ~ **7.8 MB**

---

## 4. Total Download Size

| Category           | Size       |
|-------------------|------------|
| Tokenizer & Config | ~ **24.0 MB** |
| Model Weights     | ~ **15.54 GB** |
| Dataset           | ~ **7.8 MB** |
| **Total**         | **≈ 15.57 GB** |

---
Output for the Model, Dataset, Tokenizer and Config Files:
<img width="1313" height="364" alt="{2CAA47DA-8D33-45BD-91FD-176707628830}" src="https://github.com/user-attachments/assets/43559523-44f2-482f-a328-21ae0876c08d" />


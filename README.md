# Salamandra-7B Training on ROCm

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

- Python 3.10+
- PyTorch with ROCm support
- Transformers
- Datasets
- Hugging Face Tokenizers

Install dependencies:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.5
pip3 install transformers datasets
```

# 🦎 Salamandra 7B Model – Download Summary

This document summarizes the **download size** and **files** fetched during the setup of the **[BSC-LT/salamandra-7b-instruct](https://huggingface.co/BSC-LT/salamandra-7b-instruct)** model when running the training test script `salamandra_7B_training.py`.

---

## 📦 1. Tokenizer & Config Files

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

## 🧠 2. Model Checkpoint Files (Sharded)

| File                                   | Size    |
|--------------------------------------|---------|
| `model-00001-of-00004.safetensors`   | **4.98 GB** |
| `model-00002-of-00004.safetensors`   | **5.00 GB** |
| `model-00003-of-00004.safetensors`   | **3.46 GB** |
| `model-00004-of-00004.safetensors`   | **2.10 GB** |
| `model.safetensors.index.json`       | **23.9 KB** |

**Subtotal:** ~ **15.54 GB**

---

## 📚 3. Dataset Files (WikiText-2)

| File                                | Size    |
|----------------------------------|---------|
| `train-00000-of-00001.parquet`   | **6.36 MB** |
| `test-00000-of-00001.parquet`    | **733 KB** |
| `validation-00000-of-00001.parquet` | **657 KB** |

**Subtotal:** ~ **7.8 MB**

---

## 📊 4. Total Download Size

| Category           | Size       |
|-------------------|------------|
| Tokenizer & Config | ~ **24.0 MB** |
| Model Weights     | ~ **15.54 GB** |
| Dataset           | ~ **7.8 MB** |
| **Total**         | **≈ 15.57 GB** |

---
ai1@pc1:~$ python3 salamandra_7B_training.py
tokenizer_config.json: 3.80kB [00:00, 14.4MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.81M/4.81M [00:00<00:00, 22.6MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 19.1M/19.1M [00:00<00:00, 31.6MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 513/513 [00:00<00:00, 3.50MB/s]
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 730/730 [00:00<00:00, 7.54MB/s]
model.safetensors.index.json: 23.9kB [00:00, 68.1MB/s]
model-00004-of-00004.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2.10G/2.10G [04:05<00:00, 8.56MB/s]
model-00001-of-00004.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████| 4.98G/4.98G [07:43<00:00, 10.7MB/s]
model-00003-of-00004.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3.46G/3.46G [07:59<00:00, 7.21MB/s]
model-00002-of-00004.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████| 5.00G/5.00G [08:11<00:00, 10.2MB/s]
Fetching 4 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [08:11<00:00, 122.86s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.61s/it]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 1.39MB/s]
README.md: 10.5kB [00:00, 31.3MB/s]
test-00000-of-00001.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 733k/733k [00:00<00:00, 10.9MB/s]
train-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 6.36M/6.36M [00:00<00:00, 31.8MB/s]
validation-00000-of-00001.parquet: 100%|████████████████████████████████████████████████████████████████████████████████████████| 657k/657k [00:00<00:00, 33.5MB/s]
Generating test split: 100%|████████████████████████████████████████████████████████████████████████████████████████| 4358/4358 [00:00<00:00, 268876.71 examples/s]
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:00<00:00, 1246904.76 examples/s]
Generating validation split: 100%|██████████████████████████████████████████████████████████████████████████████████| 3760/3760 [00:00<00:00, 715998.50 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 4358/4358 [00:00<00:00, 16306.99 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:02<00:00, 16372.03 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 3760/3760 [00:00<00:00, 10396.38 examples/s]

<img width="1313" height="364" alt="{2CAA47DA-8D33-45BD-91FD-176707628830}" src="https://github.com/user-attachments/assets/43559523-44f2-482f-a328-21ae0876c08d" />


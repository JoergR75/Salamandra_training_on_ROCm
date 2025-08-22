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

Output for the Model, Dataset, Tokenizer and Config Files download:
<img width="1313" height="364" alt="{2CAA47DA-8D33-45BD-91FD-176707628830}" src="https://github.com/user-attachments/assets/43559523-44f2-482f-a328-21ae0876c08d" />

---

Training example output for 2x INSTINCT MI210, gfx90a (AMD Ryzen Threadripper PRO 5955WX (16C/32T), 32GB DDR4 2133 MT/s RAM, 250GB WD blue SSD)
```bash
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.65s/it]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:02<00:00, 14072.03 examples/s]

[INFO] Training started... Measuring TTFT, T/S, and TOS.

  0%|                                                                                                                                    | 0/27540 [00:00<?, ?it/s][METRIC] TTFT: 3.023 sec
  0%|                                                                                                                          | 9/27540 [00:11<8:15:17,  1.08s/it][METRIC] Step: 10 | T/S: 823.60 tokens/sec | TOS: 1.243 sec/step
  0%|                                                                                                                         | 19/27540 [00:21<8:00:05,  1.05s/it][METRIC] Step: 20 | T/S: 894.00 tokens/sec | TOS: 1.145 sec/step
  0%|▏                                                                                                                        | 29/27540 [00:32<7:59:27,  1.05s/it][METRIC] Step: 30 | T/S: 920.54 tokens/sec | TOS: 1.112 sec/step
  0%|▏                                                                                                                        | 39/27540 [00:42<7:59:56,  1.05s/it][METRIC] Step: 40 | T/S: 934.50 tokens/sec | TOS: 1.096 sec/step
  0%|▏                                                                                                                        | 49/27540 [00:53<7:59:06,  1.05s/it][METRIC] Step: 50 | T/S: 943.12 tokens/sec | TOS: 1.086 sec/step
  0%|▎                                                                                                                        | 59/27540 [01:03<7:59:49,  1.05s/it][METRIC] Step: 60 | T/S: 948.85 tokens/sec | TOS: 1.079 sec/step
  0%|▎                                                                                                                        | 69/27540 [01:14<7:58:52,  1.05s/it][METRIC] Step: 70 | T/S: 953.02 tokens/sec | TOS: 1.074 sec/step
  0%|▎                                                                                                                        | 79/27540 [01:24<7:59:20,  1.05s/it][METRIC] Step: 80 | T/S: 956.16 tokens/sec | TOS: 1.071 sec/step
  0%|▍                                                                                                                        | 89/27540 [01:35<7:57:25,  1.04s/it][METRIC] Step: 90 | T/S: 958.54 tokens/sec | TOS: 1.068 sec/step
  0%|▍                                                                                                                        | 99/27540 [01:45<7:58:01,  1.05s/it][METRIC] Step: 100 | T/S: 960.44 tokens/sec | TOS: 1.066 sec/step
  0%|▍                                                                                                                       | 109/27540 [01:56<7:58:36,  1.05s/it][METRIC] Step: 110 | T/S: 962.25 tokens/sec | TOS: 1.064 sec/step
  0%|▌                                                                                                                       | 119/27540 [02:06<7:59:25,  1.05s/it][METRIC] Step: 120 | T/S: 963.38 tokens/sec | TOS: 1.063 sec/step
  0%|▌                                                                                                                       | 129/27540 [02:16<7:59:57,  1.05s/it][METRIC] Step: 130 | T/S: 964.55 tokens/sec | TOS: 1.062 sec/step
  1%|▌                                                                                                                       | 139/27540 [02:27<7:57:59,  1.05s/it][METRIC] Step: 140 | T/S: 965.53 tokens/sec | TOS: 1.061 sec/step
  1%|▋                                                                                                                       | 149/27540 [02:37<7:56:31,  1.04s/it][METRIC] Step: 150 | T/S: 966.45 tokens/sec | TOS: 1.060 sec/step
  1%|▋                                                                                                                       | 159/27540 [02:48<7:56:09,  1.04s/it][METRIC] Step: 160 | T/S: 967.20 tokens/sec | TOS: 1.059 sec/step
  1%|▋                                                                                                                       | 169/27540 [02:58<7:57:54,  1.05s/it][METRIC] Step: 170 | T/S: 967.76 tokens/sec | TOS: 1.058 sec/step
  1%|▊                                                                                                                       | 179/27540 [03:09<7:57:37,  1.05s/it][METRIC] Step: 180 | T/S: 968.30 tokens/sec | TOS: 1.058 sec/step
  1%|▊                                                                                                                       | 189/27540 [03:19<7:55:58,  1.04s/it][METRIC] Step: 190 | T/S: 968.80 tokens/sec | TOS: 1.057 sec/step
  1%|▊                                                                                                                       | 199/27540 [03:30<7:55:35,  1.04s/it][METRIC] Step: 200 | T/S: 969.20 tokens/sec | TOS: 1.057 sec/step
  1%|▉                                                                                                                       | 209/27540 [03:40<7:58:25,  1.05s/it][METRIC] Step: 210 | T/S: 969.50 tokens/sec | TOS: 1.056 sec/step
  1%|▉                                                                                                                       | 219/27540 [03:51<7:57:40,  1.05s/it][METRIC] Step: 220 | T/S: 970.00 tokens/sec | TOS: 1.056 sec/step
  1%|▉                                                                                                                       | 229/27540 [04:01<7:58:01,  1.05s/it][METRIC] Step: 230 | T/S: 970.27 tokens/sec | TOS: 1.055 sec/step
  1%|█                                                                                                                       | 239/27540 [04:12<7:57:48,  1.05s/it][METRIC] Step: 240 | T/S: 970.47 tokens/sec | TOS: 1.055 sec/step
  1%|█                                                                                                                       | 249/27540 [04:22<7:57:08,  1.05s/it][METRIC] Step: 250 | T/S: 970.82 tokens/sec | TOS: 1.055 sec/step
  1%|█▏                                                                                                                      | 259/27540 [04:33<7:54:12,  1.04s/it][METRIC] Step: 260 | T/S: 971.11 tokens/sec | TOS: 1.054 sec/step
  1%|█▏                                                                                                                      | 269/27540 [04:43<7:56:54,  1.05s/it][METRIC] Step: 270 | T/S: 971.33 tokens/sec | TOS: 1.054 sec/step
  1%|█▏                                                                                                                      | 279/27540 [04:54<7:56:18,  1.05s/it][METRIC] Step: 280 | T/S: 971.51 tokens/sec | TOS: 1.054 sec/step
  1%|█▎                                                                                                                      | 289/27540 [05:04<7:57:03,  1.05s/it][METRIC] Step: 290 | T/S: 971.75 tokens/sec | TOS: 1.054 sec/step
  1%|█▎                                                                                                                      | 299/27540 [05:15<7:55:57,  1.05s/it][METRIC] Step: 300 | T/S: 971.92 tokens/sec | TOS: 1.054 sec/step
  1%|█▎                                                                                                                      | 309/27540 [05:25<7:56:20,  1.05s/it][METRIC] Step: 310 | T/S: 972.15 tokens/sec | TOS: 1.053 sec/step
  1%|█▍                                                                                                                      | 319/27540 [05:35<7:54:21,  1.05s/it][METRIC] Step: 320 | T/S: 972.33 tokens/sec | TOS: 1.053 sec/step
  1%|█▍                                                                                                                      | 329/27540 [05:46<7:54:47,  1.05s/it][METRIC] Step: 330 | T/S: 972.48 tokens/sec | TOS: 1.053 sec/step
  1%|█▍                                                                                                                      | 339/27540 [05:56<7:53:23,  1.04s/it][METRIC] Step: 340 | T/S: 972.60 tokens/sec | TOS: 1.053 sec/step
  1%|█▌                                                                                                                      | 349/27540 [06:07<7:56:01,  1.05s/it][METRIC] Step: 350 | T/S: 972.78 tokens/sec | TOS: 1.053 sec/step
  1%|█▌                                                                                                                      | 359/27540 [06:17<7:55:34,  1.05s/it][METRIC] Step: 360 | T/S: 972.85 tokens/sec | TOS: 1.053 sec/step
  1%|█▌                                                                                                                      | 369/27540 [06:28<7:55:05,  1.05s/it][METRIC] Step: 370 | T/S: 972.96 tokens/sec | TOS: 1.052 sec/step
  1%|█▋                                                                                                                      | 379/27540 [06:38<7:53:44,  1.05s/it][METRIC] Step: 380 | T/S: 973.08 tokens/sec | TOS: 1.052 sec/step
  1%|█▋                                                                                                                      | 389/27540 [06:49<7:54:29,  1.05s/it][METRIC] Step: 390 | T/S: 973.23 tokens/sec | TOS: 1.052 sec/step
  1%|█▋                                                                                                                      | 399/27540 [06:59<7:56:31,  1.05s/it][METRIC] Step: 400 | T/S: 973.29 tokens/sec | TOS: 1.052 sec/step
  1%|█▊                                                                                                                      | 409/27540 [07:10<7:54:22,  1.05s/it][METRIC] Step: 410 | T/S: 973.41 tokens/sec | TOS: 1.052 sec/step
  2%|█▊                                                                                                                      | 419/27540 [07:20<7:52:42,  1.05s/it][METRIC] Step: 420 | T/S: 973.50 tokens/sec | TOS: 1.052 sec/step
  2%|█▊                                                                                                                      | 429/27540 [07:31<7:51:09,  1.04s/it][METRIC] Step: 430 | T/S: 973.62 tokens/sec | TOS: 1.052 sec/step
  2%|█▉                                                                                                                      | 439/27540 [07:41<7:53:34,  1.05s/it][METRIC] Step: 440 | T/S: 973.68 tokens/sec | TOS: 1.052 sec/step
  2%|█▉                                                                                                                      | 449/27540 [07:52<7:54:30,  1.05s/it][METRIC] Step: 450 | T/S: 973.75 tokens/sec | TOS: 1.052 sec/step
  2%|██                                                                                                                      | 459/27540 [08:02<7:53:59,  1.05s/it][METRIC] Step: 460 | T/S: 973.87 tokens/sec | TOS: 1.051 sec/step
  2%|██                                                                                                                      | 469/27540 [08:13<7:53:54,  1.05s/it][METRIC] Step: 470 | T/S: 973.95 tokens/sec | TOS: 1.051 sec/step
  2%|██                                                                                                                      | 479/27540 [08:23<7:54:06,  1.05s/it][METRIC] Step: 480 | T/S: 973.80 tokens/sec | TOS: 1.052 sec/step
  2%|██▏                                                                                                                     | 489/27540 [08:34<7:52:23,  1.05s/it][METRIC] Step: 490 | T/S: 973.89 tokens/sec | TOS: 1.051 sec/step
  2%|██▏                                                                                                                     | 499/27540 [08:44<7:52:54,  1.05s/it][METRIC] Step: 500 | T/S: 973.99 tokens/sec | TOS: 1.051 sec/step
```

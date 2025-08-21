# 🦎 Fine-Tuning Salamandra-7B-Instruct on AMD ROCm

This repository demonstrates how to **fine-tune** the [BSC-LT/salamandra-7b-instruct](https://huggingface.co/BSC-LT/salamandra-7b-instruct) model using **Hugging Face Transformers** on **AMD ROCm**-enabled GPUs.  
It is optimized for **MI300X**, **MI250**, and **RDNA3/4** GPUs and serves as a **training test** to validate your ROCm setup.

---

## 🚀 Overview

- 🏎 **Optimized for ROCm** → automatically uses **bfloat16** (`bf16`) on supported AMD GPUs.
- 🧠 Uses **Trainer API** for quick prototyping and testing.
- 📦 Example dataset: **WikiText-2**.
- ⚡ Compatible with multi-GPU environments (`device_map="auto"`).
- 🛠 Includes ROCm-specific optimizations to avoid common issues.

---

## 📌 Requirements

### 1. Install ROCm (AMD GPUs only)

For **Ubuntu 22.04 / 24.04**:

```bash
sudo apt update && sudo apt install -y rocm-dev rocm-libs

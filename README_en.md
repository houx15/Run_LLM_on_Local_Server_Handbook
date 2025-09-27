# Running Open Source Large Language Models on Local Servers (DeepSeek + A100 80GB, Pure Offline Scenario)

## Table of Contents

- [Running Open Source Large Language Models on Local Servers (DeepSeek + A100 80GB, Pure Offline Scenario)](#running-open-source-large-language-models-on-local-servers-deepseek--a100-80gb-pure-offline-scenario)
  - [Table of Contents](#table-of-contents)
  - [I. Environment Setup + Model Weight Caching](#i-environment-setup--model-weight-caching)
    - [1) Generate and Download Dependencies](#1-generate-and-download-dependencies)
    - [2) Download Model Snapshots (DeepSeek Series Example)](#2-download-model-snapshots-deepseek-series-example)
  - [II. Offline Model Loading](#ii-offline-model-loading)
    - [1) Offline Dependency Installation](#1-offline-dependency-installation)
    - [2) Load Models from Local Directory (No Network Required)](#2-load-models-from-local-directory-no-network-required)
  - [III. Inference: Full Precision and Reduced Precision](#iii-inference-full-precision-and-reduced-precision)
    - [Principle Overview](#principle-overview)
    - [Case A: Full Precision Inference (14B, FP16)](#case-a-full-precision-inference-14b-fp16)
    - [Case B: Reduced Precision Inference (70B, 4bit Quantization)](#case-b-reduced-precision-inference-70b-4bit-quantization)
  - [IV. Fine-tuning Case 1 (Unsupervised): QLoRA Adaptation with Only Large-Scale Corpora](#iv-fine-tuning-case-1-unsupervised-qlora-adaptation-with-only-large-scale-corpora)
    - [Scenario and Principle](#scenario-and-principle)
    - [Code (DeepSeek 14B, Self-supervised QLoRA, field text)](#code-deepseek-14b-self-supervised-qlora-field-text)
    - [Usage Effects and Recommendations](#usage-effects-and-recommendations)
  - [V. Fine-tuning Case 2 (Supervised): Opinion Analysis (Output -2 to 2)](#v-fine-tuning-case-2-supervised-opinion-analysis-output--2-to-2)
    - [Scenario and Principle](#scenario-and-principle-1)
    - [Data Format (JSONL)](#data-format-jsonl)
    - [Training Code (DeepSeek 14B, QLoRA)](#training-code-deepseek-14b-qlora)
    - [Inference (Output Numbers Only)](#inference-output-numbers-only)
  - [VI. Resources and Practical Points (A100 80GB)](#vi-resources-and-practical-points-a100-80gb)
  - [VII. Common Issues and Troubleshooting](#vii-common-issues-and-troubleshooting)
  - [VIII. Appendix: Complete Script Example](#viii-appendix-complete-script-example)
  - [IX. Detailed Tutorial Links](#ix-detailed-tutorial-links)

---

Hardware Prerequisites: NVIDIA A100 80GB (this guide uses this card as an example)  
Network Prerequisites: GPU environment is completely offline, but can download dependencies and models in a "networked environment" on the same machine first, then switch to "GPU offline environment" for use

Goal: Complete model caching and installation in an offline environment, with three representative case studies:
- Inference: Full precision and reduced precision (including principle explanations)
- Fine-tuning: Unsupervised QLoRA adaptation with only large-scale corpora
- Fine-tuning: Opinion Analysis (input text, output -2 to 2 numbers)

---

## I. Environment Setup + Model Weight Caching

### 1) Generate and Download Dependencies
Create requirements.txt (versions can be adjusted as needed):
```
torch>=2.1
transformers>=4.41.0
accelerate>=0.31.0
peft>=0.11.0
bitsandbytes>=0.43.0
datasets>=2.19.0
sentencepiece
```

Download dependencies to local directory offline_packages:
```bash
pip download -r requirements.txt -d ./offline_packages
```

Notes:
- If you need torch wheels that strictly match CUDA versions, you can manually download corresponding cu118/cu121 versions from official index URLs.
- If there are private environment constraints, you can include common tools (numpy, pandas, scikit-learn, etc.) in requirements.txt.

### 2) Download Model Snapshots (DeepSeek Series Example)
Use Python in a networked environment to execute once, downloading the complete model to local folder:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    local_dir="./models/DeepSeek-R1-Distill-Qwen-14B",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-70B",
    local_dir="./models/DeepSeek-R1-Distill-Qwen-70B",
    local_dir_use_symlinks=False
)
```

Recommendations:
- If you need other scales like 7B/32B, repeat the above steps.
- Record a "model registry" JSON containing local model paths, source repositories, commit or date, and usage descriptions for easy traceability and version management.

---

## II. Offline Model Loading

### 1) Offline Dependency Installation
Execute in GPU offline environment:
```bash
pip install --no-index --find-links ./offline_packages -r requirements.txt
```

### 2) Load Models from Local Directory (No Network Required)
Subsequent examples use local_dir pointing to local directories, for example:
```
./models/DeepSeek-R1-Distill-Qwen-14B
./models/DeepSeek-R1-Distill-Qwen-70B
```

---

## III. Inference: Full Precision and Reduced Precision

### Principle Overview
- Full Precision (FP16/FP32)
  - Weights represented as floating-point numbers (16/32 bits), high precision, large memory usage
  - A100 is FP16-friendly, 14B/30B run stably on 80GB memory
- Quantization (8bit/4bit)
  - Quantize weights to 8/4-bit integers for storage, approximately restored during inference
  - Significantly reduced memory usage, throughput usually comparable to FP16
  - Trade-off is slight quality loss in output; acceptable impact for most text analysis tasks

Experience:
- A100 80GB: 14B/30B recommended FP16; 70B suggested 4bit (or multi-GPU/distributed)
- For long context and multi-concurrency, quantization can significantly alleviate memory pressure

### Case A: Full Precision Inference (14B, FP16)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

local_dir = "./models/DeepSeek-R1-Distill-Qwen-14B"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate(prompt, max_new_tokens=256, temperature=0.7):
    ins = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ins,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0], skip_special_tokens=True)

print(generate("Please explain the role of topic models in text analysis in three sentences."))
```

### Case B: Reduced Precision Inference (70B, 4bit Quantization)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

local_dir = "./models/DeepSeek-R1-Distill-Qwen-70B"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

quant = BitsAndBytesConfig(
    load_in_4bit=True,                     # 4-bit quantization: significantly reduced memory usage
    bnb_4bit_quant_type="nf4",             # nf4 performs well for LLMs
    bnb_4bit_compute_dtype=torch.float16   # Still use FP16 for computation
)

model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    quantization_config=quant
)

print("4bit quantized model loaded successfully")
```

Notes:
- Long text OOM: Reduce max_new_tokens, use 4bit, or process text in segments
- Significant quality degradation after quantization: Try 8bit or switch to smaller model's FP16 version

---

## IV. Fine-tuning Case 1 (Unsupervised): QLoRA Adaptation with Only Large-Scale Corpora

### Scenario and Principle
- Only large-scale Chinese/English text, no paired "input-output"
- Goal: Make the model more aligned with your domain's language distribution and expression style
- Self-supervised language modeling: Predict next token based on previous context, no annotation needed
- LoRA: Only train small adaptation parameters, avoid modifying base weights
- QLoRA: LoRA fine-tuning on 4bit quantized base, more memory efficient

### Code (DeepSeek 14B, Self-supervised QLoRA, field text)
```python
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorForLanguageModeling, Trainer)
from peft import LoraConfig, get_peft_model, TaskType

local_dir = "./models/DeepSeek-R1-Distill-Qwen-14B"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

# Load base model with 4bit quantization
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    quantization_config=quant
)

# LoRA configuration (only train small parameters for key modules)
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(base, lora)
model.print_trainable_parameters()

# Local JSONL, each line: {"text":"……"}
ds = load_dataset("json", data_files="data_unsup.jsonl")["train"]

block_size = 1024
def tok_fn(examples):
    return tok(examples["text"], truncation=True, max_length=block_size)

tok_ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

# Self-supervised LM: labels=inputs
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir="./qlora_unsup_out",
    per_device_train_batch_size=2,     # Can be increased appropriately for A100 80G
    gradient_accumulation_steps=8,
    num_train_epochs=1,                # Start small to observe loss, then increase
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=collator)
trainer.train()

# Only save LoRA adapters (small files, easy for versioning and distribution)
trainer.save_model()
tok.save_pretrained("./qlora_unsup_out")
```

### Usage Effects and Recommendations
- Output style more aligned with your corpus, terminology and expressions more natural
- Won't force specific task outputs; if you need structured outputs for specific tasks, consider further instruction fine-tuning
- Parameter tuning order: batch size → seq_len (block_size) → learning rate; prioritize observing loss curves and memory usage

---

## V. Fine-tuning Case 2 (Supervised): Opinion Analysis (Output -2 to 2)

### Scenario and Principle
- Input a text, output discrete sentiment intensity scores: -2, -1, 0, 1, 2
- Instruction fine-tuning: Provide "task description (instruction) + input + standard answer (output)"
- Emphasize "output only numbers" in both training and inference to reduce off-topic and redundant text
- Also use QLoRA to reduce memory usage and minimize overfitting risk

### Data Format (JSONL)
Each line is one training sample:
```json
{"instruction":"Please read the text and output sentiment intensity, range -2 to 2, negative strongest is -2, positive strongest is 2, output only numbers.","input":"Text content...","output":"1"}
```

### Training Code (DeepSeek 14B, QLoRA)
```python
import torch, re
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType

local_dir = "./models/DeepSeek-R1-Distill-Qwen-14B"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    quantization_config=quant
)

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(base, lora)
model.print_trainable_parameters()

# Training set: data_opinion.jsonl
ds = load_dataset("json", data_files="data_opinion.jsonl")["train"]

def format_ex(ex):
    instr = ex.get("instruction","Please output as required")
    inp = ex.get("input","")
    out = str(ex.get("output","0")).strip()
    prompt = f"{instr}\nText: {inp}\nAnswer (numbers only): "
    X = tok(prompt, truncation=True, max_length=1024)
    with tok.as_target_tokenizer():
        y = tok(out, truncation=True, max_length=8)
    X["labels"] = y["input_ids"]
    return X

tok_ds = ds.map(format_ex, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="./qlora_opinion_out",
    per_device_train_batch_size=4,   # Can be increased appropriately for A100 80G
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds)
trainer.train()
trainer.save_model()
tok.save_pretrained("./qlora_opinion_out")
```

### Inference (Output Numbers Only)
```python
import re, torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_dir = "./models/DeepSeek-R1-Distill-Qwen-14B"
lora_dir = "./qlora_opinion_out"

tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
quant = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base = AutoModelForCausalLM.from_pretrained(
    base_dir, device_map="auto", quantization_config=quant
)
model = PeftModel.from_pretrained(base, lora_dir)

def predict_score(text):
    prompt = (
        "Please read the text and output sentiment intensity, range -2 to 2, negative strongest is -2, positive strongest is 2, output only numbers.\n"
        f"Text: {text}\nAnswer (numbers only): "
    )
    ins = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ins, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
        )
    s = tok.decode(out[0], skip_special_tokens=True)
    # Extract first valid number, limit range -2..2
    m = re.search(r"(?<!\d)-?[0-2](?!\d)", s)
    return int(m.group(0)) if m else None

print(predict_score("This initiative greatly improved the service experience."))
```

---

## VI. Resources and Practical Points (A100 80GB)

- Inference
  - FP16: 14B/30B stable; 32B also feasible; pay attention to KV cache usage for long context
  - 4bit: Can support 70B and higher concurrency, suitable for batch processing and long text
- Fine-tuning
  - QLoRA: 7B–30B most convenient; 14B balanced for common data scales
  - Parameter tuning order: batch size → sequence length (block_size) → learning rate
  - Monitor nvidia-smi and training loss, reduce seq_len or increase accumulation steps if necessary
- Offline Steady State
  - transformers only loads models and tokenizers from local directories
  - All dependencies from offline_packages, pip uses --no-index and --find-links
  - For version updates, repeat "networked phase" to download new packages and models, then switch back to offline
- Data and Compliance
  - Training data should be desensitized; avoid logging original text to disk
  - Only distribute LoRA adapters, maintain base models uniformly to reduce redundancy

---

## VII. Common Issues and Troubleshooting

- CUDA OOM
  - Use 4bit quantization; shorten max_new_tokens/sequence length; reduce batch/concurrency
- Unstable output or redundant text
  - Clearly specify "output only numbers/only JSON" in prompts, add few-shot examples if necessary
- Quality degradation after quantization
  - Try 8bit or switch to smaller model's FP16 version; or do additional supervised fine-tuning for specific tasks
- Offline loading failure
  - Confirm from_pretrained points to local directory; requirements installed completely through offline packages; disable environment variables/configurations that automatically pull from network

---

## VIII. Appendix: Complete Script Example
```python
# Usage:
# python batch_sum.py --model_dir ./models/DeepSeek-R1-Distill-Qwen-14B \
#   --input data.csv --text_col text --out out.csv --max_new_tokens 128

import argparse, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out", default="out.csv")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", torch_dtype=torch.float16
    )

    def infer(t):
        prompt = f"Please summarize the key points of the following text in one sentence:\n{t}\nSummary: "
        ins = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ins, max_new_tokens=args.max_new_tokens,
                temperature=0.2, do_sample=True,
                pad_token_id=tok.eos_token_id
            )
        s = tok.decode(out[0], skip_special_tokens=True)
        return s.split("Summary: ")[-1].strip()

    df = pd.read_csv(args.input)
    df["result"] = df[args.text_col].astype(str).apply(infer)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("saved to", args.out)

if __name__ == "__main__":
    main()
```

## IX. Detailed Tutorial Links
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch): How to train, pre-train, and fine-tune large models from scratch
- [Self LLM](https://github.com/datawhalechina/self-llm/tree/master): Open source large model deployment and fine-tuning cases
- [Happy LLM](https://github.com/datawhalechina/happy-ll): Deep understanding of large language model principles and training processes

# 在本地服务器上运行开源大模型（DeepSeek + A100 80GB，纯离线场景）

## 目录

- [在本地服务器上运行开源大模型（DeepSeek + A100 80GB，纯离线场景）](#在本地服务器上运行开源大模型deepseek--a100-80gb纯离线场景)
  - [目录](#目录)
  - [一、环境准备+模型权重缓存](#一环境准备模型权重缓存)
    - [1) 生成并下载依赖](#1-生成并下载依赖)
    - [2) 下载模型快照（DeepSeek 系列示例）](#2-下载模型快照deepseek-系列示例)
  - [二、离线加载模型](#二离线加载模型)
    - [1) 离线安装依赖](#1-离线安装依赖)
    - [2) 从本地目录加载模型（无需联网）](#2-从本地目录加载模型无需联网)
  - [三、推理：全精度与降精度](#三推理全精度与降精度)
    - [原理简述](#原理简述)
    - [案例A：全精度推理（14B，FP16）](#案例a全精度推理14bfp16)
    - [案例B：降精度推理（70B，4bit 量化）](#案例b降精度推理70b4bit-量化)
  - [四、微调案例1（无监督）：仅有大型语料库的 QLoRA 适配](#四微调案例1无监督仅有大型语料库的-qlora-适配)
    - [场景与原理](#场景与原理)
    - [代码（DeepSeek 14B，自监督 QLoRA，字段 text）](#代码deepseek-14b自监督-qlora字段-text)
    - [使用效果与建议](#使用效果与建议)
  - [五、微调案例2（有监督）：Opinion Analysis（输出 -2～2）](#五微调案例2有监督opinion-analysis输出--22)
    - [场景与原理](#场景与原理-1)
    - [数据格式（JSONL）](#数据格式jsonl)
    - [训练代码（DeepSeek 14B，QLoRA）](#训练代码deepseek-14bqlora)
    - [推理（仅输出数字）](#推理仅输出数字)
  - [六、资源与实践要点（A100 80GB）](#六资源与实践要点a100-80gb)
  - [七、常见问题与排障](#七常见问题与排障)
  - [八、附：脚本完整示例](#八附脚本完整示例)
  - [九、详细教程链接](#九详细教程链接)

---

硬件前提：NVIDIA A100 80GB（本文以此卡为例）  
网络前提：GPU 环境完全离线，但可在同一台机器的"有网环境"先下载依赖与模型，再切换到"GPU 离线环境"使用

目标：在离线环境中完成模型缓存与安装，并给出三个具有代表性的案例：
- 推理：全精度与降精度（含原理说明）
- 微调：仅有大型语料库的无监督 QLoRA 适配
- 微调：Opinion Analysis（输入文本，输出 -2～2 的数字）

---

## 一、环境准备+模型权重缓存

### 1) 生成并下载依赖
创建 requirements.txt（可按需增减版本）：
```
torch>=2.1
transformers>=4.41.0
accelerate>=0.31.0
peft>=0.11.0
bitsandbytes>=0.43.0
datasets>=2.19.0
sentencepiece
```

下载依赖到本地目录 offline_packages：
```bash
pip download -r requirements.txt -d ./offline_packages
```

提示：
- 如需与 CUDA 版本严格匹配的 torch 轮子，可改用官方索引地址手动下载对应 cu118/cu121 版本。
- 若有私有环境约束，可将常用工具（numpy、pandas、scikit-learn等）一并写入 requirements.txt。

### 2) 下载模型快照（DeepSeek 系列示例）
使用 Python 在联网环境执行一次，将模型完整落地到本地文件夹：
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

建议：
- 如需 7B/32B 等其他规模，重复执行上面步骤。
- 记录一个“模型注册表”JSON，包含模型本地路径、来源仓库、commit 或日期、用途说明，便于后续溯源与版本管理。

---

## 二、离线加载模型

### 1) 离线安装依赖
在 GPU 离线环境执行：
```bash
pip install --no-index --find-links ./offline_packages -r requirements.txt
```

### 2) 从本地目录加载模型（无需联网）
后续示例中的 local_dir 统一指向本地目录，例如：
```
./models/DeepSeek-R1-Distill-Qwen-14B
./models/DeepSeek-R1-Distill-Qwen-70B
```

---

## 三、推理：全精度与降精度

### 原理简述
- 全精度（FP16/FP32）
  - 权重用浮点数表示（16/32 位），精度高、显存占用大
  - A100 对 FP16 友好，14B/30B 在 80GB 显存上运行稳定
- 量化（8bit/4bit）
  - 将权重量化为 8/4 位整数存储，推理时近似还原
  - 显存占用显著降低，吞吐通常与 FP16 接近
  - 代价是输出质量可能有轻微损失；对多数文本分析任务影响可接受

经验：
- A100 80GB：14B/30B 推荐 FP16；70B 建议 4bit（或多卡/分布式）
- 长上下文与多并发时，量化能显著缓解显存压力

### 案例A：全精度推理（14B，FP16）
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

print(generate("请用三句话解释主题模型在文本分析中的作用。"))
```

### 案例B：降精度推理（70B，4bit 量化）
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

local_dir = "./models/DeepSeek-R1-Distill-Qwen-70B"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

quant = BitsAndBytesConfig(
    load_in_4bit=True,                     # 4位量化：显存占用显著下降
    bnb_4bit_quant_type="nf4",             # nf4 对 LLM 性能较好
    bnb_4bit_compute_dtype=torch.float16   # 计算仍用 FP16
)

model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    quantization_config=quant
)

print("4bit 量化模型加载完成")
```

提示：
- 长文本 OOM：减少 max_new_tokens、使用 4bit、或对文本分段处理
- 量化后质量明显下降：尝试 8bit 或改用较小模型的 FP16 版本

---

## 四、微调案例1（无监督）：仅有大型语料库的 QLoRA 适配

### 场景与原理
- 只有大规模中文/英文文本，没有成对“输入-输出”
- 目标：让模型更贴合你的领域语言分布与表达风格
- 自监督语言建模：根据前文预测下一个 token，无需标注
- LoRA：仅训练小量适配参数，避免修改基础权重
- QLoRA：在 4bit 量化的基础上进行 LoRA 微调，显存更省

### 代码（DeepSeek 14B，自监督 QLoRA，字段 text）
```python
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorForLanguageModeling, Trainer)
from peft import LoraConfig, get_peft_model, TaskType

local_dir = "./models/DeepSeek-R1-Distill-Qwen-14B"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

# 4bit 量化加载基础模型
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

# LoRA 配置（仅训练关键模块的小量参数）
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(base, lora)
model.print_trainable_parameters()

# 本地 JSONL，每行：{"text":"……"}
ds = load_dataset("json", data_files="data_unsup.jsonl")["train"]

block_size = 1024
def tok_fn(examples):
    return tok(examples["text"], truncation=True, max_length=block_size)

tok_ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

# 自监督 LM：labels=inputs
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir="./qlora_unsup_out",
    per_device_train_batch_size=2,     # A100 80G 可适当增大
    gradient_accumulation_steps=8,
    num_train_epochs=1,                # 先小跑观察 loss，再增大
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=collator)
trainer.train()

# 仅保存 LoRA 适配器（小文件，便于版本化与分发）
trainer.save_model()
tok.save_pretrained("./qlora_unsup_out")
```

### 使用效果与建议
- 输出风格更贴近你的语料，术语与表达更自然
- 不会强制形成特定任务输出；若需面向任务的结构化输出，建议进一步做指令式微调
- 调参顺序建议：batch size → seq_len（block_size）→ 学习率；优先观察 loss 曲线与显存占用

---

## 五、微调案例2（有监督）：Opinion Analysis（输出 -2～2）

### 场景与原理
- 输入一段文本，输出离散情绪强度分值：-2、-1、0、1、2
- 指令微调：提供“任务说明（instruction）+ 输入（input）+ 标准答案（output）”
- 训练与推理中都强调“仅输出数字”，减少跑题与多余文本
- 同样使用 QLoRA，降低显存占用并减小过拟合风险

### 数据格式（JSONL）
每行一条训练样本：
```json
{"instruction":"请阅读文本并输出情感强度，范围-2到2，负最强为-2，正最强为2，仅输出数字。","input":"文本内容……","output":"1"}
```

### 训练代码（DeepSeek 14B，QLoRA）
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

# 训练集：data_opinion.jsonl
ds = load_dataset("json", data_files="data_opinion.jsonl")["train"]

def format_ex(ex):
    instr = ex.get("instruction","请按要求输出")
    inp = ex.get("input","")
    out = str(ex.get("output","0")).strip()
    prompt = f"{instr}\n文本：{inp}\n答案（仅数字）："
    X = tok(prompt, truncation=True, max_length=1024)
    with tok.as_target_tokenizer():
        y = tok(out, truncation=True, max_length=8)
    X["labels"] = y["input_ids"]
    return X

tok_ds = ds.map(format_ex, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="./qlora_opinion_out",
    per_device_train_batch_size=4,   # A100 80G 可适当增大
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

### 推理（仅输出数字）
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
        "请阅读文本并输出情感强度，范围-2到2，负最强为-2，正最强为2，仅输出数字。\n"
        f"文本：{text}\n答案（仅数字）："
    )
    ins = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ins, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
        )
    s = tok.decode(out[0], skip_special_tokens=True)
    # 抽取首个有效数字，限定范围 -2..2
    m = re.search(r"(?<!\d)-?[0-2](?!\d)", s)
    return int(m.group(0)) if m else None

print(predict_score("这项举措极大改善了服务体验。"))
```

---

## 六、资源与实践要点（A100 80GB）

- 推理
  - FP16：14B/30B 稳定；32B 也可行；长上下文注意 KV 缓存占用
  - 4bit：可支持 70B 与更高并发，适合批量处理与长文本
- 微调
  - QLoRA：7B–30B 最顺手；14B 常见数据规模更平衡
  - 调参顺序：batch size → 序列长度（block_size）→ 学习率
  - 监控 nvidia-smi 与训练 loss，必要时降低 seq_len 或增大累积步数
- 离线稳态
  - transformers 仅从本地目录加载模型与分词器
  - 所有依赖来自 offline_packages，pip 使用 --no-index 与 --find-links
  - 更新版本时，重复“联网阶段”下载新包与模型，再切回离线
- 数据与合规
  - 训练数据尽量脱敏；日志避免落盘原文
  - 仅分发 LoRA 适配器，基础模型统一维护以减少冗余

---

## 七、常见问题与排障

- CUDA OOM
  - 使用 4bit 量化；缩短 max_new_tokens/序列长度；减小 batch/并发
- 输出不稳定或多余文本
  - 提示词中明确“仅输出数字/仅输出 JSON”，必要时加入 few-shot 示例
- 量化后质量下降
  - 尝试 8bit 或改用较小模型的 FP16 版本；或针对目标任务再做少量监督微调
- 离线加载失败
  - 确认 from_pretrained 指向本地目录；requirements 已通过离线包完整安装；禁用自动从网拉取的环境变量/配置

---

## 八、附：脚本完整示例
```python
# 用法：
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
        prompt = f"请用一句话总结以下文本要点：\n{t}\n总结："
        ins = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ins, max_new_tokens=args.max_new_tokens,
                temperature=0.2, do_sample=True,
                pad_token_id=tok.eos_token_id
            )
        s = tok.decode(out[0], skip_special_tokens=True)
        return s.split("总结：")[-1].strip()

    df = pd.read_csv(args.input)
    df["result"] = df[args.text_col].astype(str).apply(infer)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("saved to", args.out)

if __name__ == "__main__":
    main()
```

## 九、详细教程链接
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch): 如何从零开始训练、预训练、微调大模型
- [Self LLM](https://github.com/datawhalechina/self-llm/tree/master): 开源大模型部署、微调案例
- [Happy LLM](https://github.com/datawhalechina/happy-ll): 深入了解大语言模型的原理和训练过程
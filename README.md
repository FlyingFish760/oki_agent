# oki — 本地个人 AI 助手

基于 LLM 微调的本地个人助手：有个人人格、记忆，可在个人电脑上对话、接受并完成命令、处理事务。
数据本地存储，**个人记忆由用户自己掌控**。

- **Baseline**：Qwen3.6-35B-A3B（MoE，3B 激活）
- **推理引擎**：Ollama（GGUF Q4_K_M，约 20GB，可跑在 32GB 显存上）
- **demo 主线**：人格 QLoRA 微调（LoRA 管语气，记忆管事实）

## 分层架构

```
交互层 (interface/CLI)
  └─ 编排/Agent 层 (agent: orchestrator + planner)
       ├─ 工具层 (agent/tools: 文件/日历 + 安全 gate)
       ├─ 记忆层 (memory: SQLite 事实 + Chroma 情景)
       └─ 人格系统 (persona: system prompt + 旋钮 + 自我画像)
            └─ 推理引擎层 (inference: Ollama)
                 └─ 模型层 (base + 人格 LoRA adapter，见 finetune/)
```

## 目录结构

```
oki/                     运行时包
  inference/             Ollama 客户端（流式、A/B 模型切换）
  persona/               人格装配（persona card + 可调旋钮 -> system prompt）
  memory/                记忆：models/store/extractor/retriever/consolidation
  agent/                 orchestrator（主循环）+ planner + tools/（含安全 gate）
  interface/             CLI（流式输出 + 危险操作确认）
configs/                 model.yaml / persona.yaml / memory.yaml
finetune/                微调流水线（demo 主线）
  persona_card.md        人格卡（造数据种子 + 评估 rubric，唯一真源）
  data/                  generate.py（蒸馏造数据）+ capability_keep.py（能力保持）
  configs/qlora.yaml     QLoRA 配置（MoE：只挂 q/k/v/o，不碰 router/experts）
  train_qlora.py         训练
  eval/                  judge.py（人格一致性）/ capability.py / ab_compare.py
  export/                merge_lora.py + Modelfile（GGUF / ADAPTER 两条部署路线）
data/                    运行时本地数据（gitignore，用户掌控）
```

## 快速开始

```powershell
# 1. 运行时依赖
uv sync

# 2. 起 Ollama 并准备 base 模型
ollama serve
ollama pull qwen3-35b-a3b        # 或用 Modelfile 从本地 GGUF 创建

# 3. 跑 CLI
uv run oki
```

## 运行推理

推理默认走本地 Ollama 服务（HTTP `http://127.0.0.1:11434`）。封装见
[oki/inference/ollama_client.py](oki/inference/ollama_client.py)，支持流式输出与
base / base+adapter 的模型切换。

### 0）准备 Ollama 与模型

```powershell
# 起服务（若未随系统自启）
ollama serve

# 看本机已有模型的确切名字
ollama list
```

模型来源二选一：

```powershell
# A) 直接从库里拉一个 Qwen3 tag（名字以 `ollama list`/官方库为准）
ollama pull qwen3:latest

# B) 已有本地 GGUF：用 Modelfile 从文件创建（推荐给 35B-A3B Q4_K_M）
#    Modelfile 里写 `FROM ./your-model-Q4_K_M.gguf`
ollama create qwen3-35b-a3b -f finetune/export/Modelfile
```

拿到确切模型名后，把它填进 [configs/model.yaml](configs/model.yaml) 的
`chat_model` 与 `ab.base_model`（微调后模型名填 `ab.persona_model`）。

### 1）快速自检推理通路

```powershell
# 直接问 Ollama HTTP API（不依赖本项目，先确认服务通）
curl http://127.0.0.1:11434/api/chat -d '{\"model\":\"qwen3:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":false}'
```

### 2）交互式对话（流式）

```powershell
uv run oki            # 读取 configs/*.yaml，走人格 + 记忆 + 工具主循环
```

### 3）从 Python 直接调用

```python
from oki.inference.ollama_client import OllamaClient

client = OllamaClient(model="qwen3:latest")
for delta in client.chat([{"role": "user", "content": "用一句话介绍你自己"}]):
    print(delta, end="", flush=True)
```

### 4）记录"微调前"基线（Day1）

```powershell
# 对 finetune/eval/prompts_eval.txt 跑 base 模型，存下基线快照，留给 Day4 对照
uv run python finetune/eval/baseline.py --model qwen3:latest
```

### 5）base vs base+adapter 并排对比（Day4）

```powershell
# 读取 configs/model.yaml 的 ab.base_model / ab.persona_model 两个模型并排出结果
uv run python finetune/eval/ab_compare.py
```

> 说明：本地首 token 有延迟，交互层默认**流式输出**改善体感；`thinking`
> 展示策略（show/fold/hide）在 [configs/model.yaml](configs/model.yaml) 配置。

## 微调流水线（demo 主线）

```powershell
uv sync --extra finetune

# Day2 造数据：英文日常对话数据（从英文开源数据集抽 user intent，再用 teacher 生成 oki 回复）
# 数据源：facebook/empathetic_dialogues + OpenAssistant/oasst1 + HuggingFaceH4/ultrachat_200k
python finetune/data/generate.py --mode daily-en --dry-run-sources --n-train 10 --n-eval 5
python finetune/data/generate.py --mode daily-en --persona-card finetune/persona_card-daily_en.md --n-train 300 --n-eval 50 --teacher openai-compatible --model <teacher-model-name>
python finetune/data/capability_keep.py

# Day3 训练（QLoRA，单卡）
python finetune/train_qlora.py --config finetune/configs/qlora.yaml

# Day4 评估 + 部署 + A/B
python finetune/eval/ab_compare.py
python finetune/export/merge_lora.py --base Qwen/Qwen3.6-35B-A3B --adapter finetune/output/oki-persona-lora --out finetune/output/oki-persona-merged
# 转 GGUF Q4_K_M 后：ollama create oki-persona -f finetune/export/Modelfile
```

## MoE + 微调关键约束

- LoRA **只挂 attention 投影 q/k/v/o**，不碰 router / 路由专家，避免破坏专家路由稳定性。
- 训练数据的 **thinking trace** 要统一处理（strip 或 keep），别混，否则污染推理链。
- 掺入 **能力保持样本**（中性语气）对抗灾难性遗忘。

## 现状

骨架已搭好，各模块留有 `TODO(DayX)` 标注实现点。运行时逻辑（记忆检索接向量库、
function-calling、巩固/遗忘）按一周 plan 逐步填充。

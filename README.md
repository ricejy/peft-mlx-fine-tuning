# PEFT MLX Fine-Tuning — Banking Q&A LoRA Experiment

An experiment in on-device, parameter-efficient fine-tuning using Apple's [MLX](https://github.com/ml-explore/mlx) framework and [llama.cpp](https://github.com/ggerganov/llama.cpp) for model conversion. The goal is to adapt a small quantised LLM into a structured-output banking assistant using Low-Rank Adaptation (LoRA), entirely on Apple Silicon — no cloud GPUs required.

---

## What This Experiment Explores

| Topic | Detail |
|---|---|
| Base model | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Fine-tuning method | LoRA (rank 8, scale 20.0, top 16 layers) |
| Training framework | `mlx-lm` on Apple Silicon |
| Post-training | Adapter fusion → GGUF conversion via `llama.cpp` |
| Task | Structured JSON output for a banking chatbot |
| Dataset size | 2,416 train / 809 validation examples |

The experiment asks: can a 1.5B parameter model, fine-tuned locally on consumer hardware, reliably output valid structured JSON for a domain-specific assistant task?

---

## Dataset

The training data is a collection of banking Q&A pairs modelled after an OCBC mobile app assistant. Every example instructs the model to respond **only** with a JSON object — no markdown, no prose — using three fixed keys:

```json
{
  "response": "Human-readable answer to the customer query.",
  "in_app":   true,
  "product":  "product name"
}
```

Topics include precious metals investing, fund transfers, account management, and other retail banking features. The data is formatted using the OpenAI chat template (system / user / assistant turns) stored as JSONL:

```jsonl
{"messages": [
  {"role": "system",    "content": "You are an OCBC customer assistant..."},
  {"role": "user",      "content": "I want to buy gold"},
  {"role": "assistant", "content": "{\"response\":\"...\",\"in_app\":true,\"product\":\"precious metals\"}"}
]}
```

---

## Project Structure

```
.
├── data/
│   ├── train.jsonl          # 2,416 training examples
│   └── valid.jsonl          # 809 validation examples
├── adapters/
│   └── adapter_iter_one/
│       ├── adapter_config.json          # Full training config
│       ├── 0000100_adapters.safetensors # Checkpoint every 100 iters
│       ├── ...
│       └── adapters.safetensors         # Final adapter (iter 1500)
└── mlx_test.py              # Inference script (streamed output)
```

---

## Workflow

### 1. Fine-tune with MLX LoRA

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data data \
  --adapter-path adapters/adapter_iter_one \
  --iters 1500 \
  --batch-size 2 \
  --num-layers 16 \
  --learning-rate 2e-4 \
  --lora-rank 8 \
  --lora-scale 20.0 \
  --save-every 100 \
  --steps-per-eval 200
```

LoRA parameters are injected into the top 16 transformer layers. Checkpoints are saved every 100 iterations to `adapters/adapter_iter_one/`.

### 2. Run inference with the trained adapter

```python
# mlx_test.py
from mlx_lm import load, stream_generate

model, tokenizer = load(
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    adapter_path="adapters/adapter_iter_one"
)

messages = [{"role": "user", "content": "How do I transfer funds?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

for response in stream_generate(model, tokenizer, prompt, max_tokens=256):
    print(response.text, end="", flush=True)
```

```bash
python mlx_test.py
```

### 3. Fuse adapter weights into the base model

Once satisfied with the adapter, merge it into the base model weights for a self-contained model:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters/adapter_iter_one \
  --save-path fused_model
```

### 4. Convert to GGUF with llama.cpp

The fused model can be converted to GGUF format for use with `llama.cpp`, enabling inference on a wider range of devices and runtimes (Ollama, LM Studio, etc.):

```bash
# Clone llama.cpp if you haven't already
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt

# Convert the fused MLX/HuggingFace model to GGUF
python convert_hf_to_gguf.py ../fused_model \
  --outfile ../fused_model.gguf \
  --outtype q4_k_m
```

After conversion, the GGUF model can be served locally:

```bash
llama.cpp/llama-cli -m fused_model.gguf -p "How do I transfer funds?"
```

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | 8 |
| Scale (alpha) | 20.0 |
| Dropout | 0.0 |
| Layers targeted | Top 16 |
| Optimizer | Adam (`lr=2e-4`) |
| Iterations | 1,500 |
| Batch size | 2 |
| Max sequence length | 2,048 |

---

## Planned Experiments

This repository will expand to cover LoRA fine-tuning across other open-source models available on HuggingFace. The same banking Q&A dataset and structured JSON output task will be used across all runs, making it straightforward to compare how different model families and sizes respond to the same fine-tuning recipe.

Models under consideration include (but are not limited to):

- **Llama 3 / 3.1 / 3.2** (Meta) — various sizes from 1B to 8B
- **Gemma 2** (Google) — 2B and 9B variants
- **Phi-3 / Phi-3.5** (Microsoft) — small but capable instruction-tuned models
- **Mistral 7B / Mistral Nemo** (Mistral AI)
- **SmolLM2** (HuggingFace) — sub-1B models for extremely constrained environments

Each experiment will be saved in its own adapter directory (e.g. `adapters/llama3_1b_iter_one/`) and documented with its training config, so results are reproducible and easy to compare.

---

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10+
- [`mlx-lm`](https://github.com/ml-explore/mlx-lm): `pip install mlx-lm`
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (for GGUF conversion only)

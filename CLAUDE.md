# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project fine-tunes `mlx-community/Qwen2.5-1.5B-Instruct-4bit` using LoRA (via `mlx-lm`) on an OCBC banking chatbot dataset. The model is trained to return structured JSON responses with keys `response`, `in_app`, and `product`.

## Commands

### Fine-tuning
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

### Inference (with adapter)
```bash
python mlx_test.py
```

### Generate without adapter
```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --prompt "your prompt here" \
  --max-tokens 256
```

### Fuse adapter into model weights
```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters/adapter_iter_one \
  --save-path fused_model
```

## Architecture

- **`mlx_test.py`** — loads the base model with a LoRA adapter and streams a response using `stream_generate`. Uses the chat template to format input messages.
- **`data/train.jsonl` / `data/valid.jsonl`** — JSONL files where each line is `{"messages": [...]}` in OpenAI chat format (system + user + assistant turns). The assistant always outputs a JSON string.
- **`adapters/adapter_iter_one/`** — LoRA checkpoint directory. Contains `.safetensors` files saved every 100 iterations (up to iter 1500), plus `adapter_config.json` storing the full training configuration.

## Data Format

Each training example must follow this structure:
```json
{"messages": [
  {"role": "system", "content": "You are an OCBC customer assistant..."},
  {"role": "user", "content": "user question"},
  {"role": "assistant", "content": "{\"response\":\"...\",\"in_app\":true,\"product\":\"...\"}"}
]}
```

The assistant output is always a JSON string (not a JSON object) — it is the raw text the model should generate.

## Notes

- This project targets **Apple Silicon (Mac)** via the MLX framework. It will not run on Windows/Linux without porting to a different training stack.
- LoRA is applied to the top 16 transformer layers (`num_layers: 16`) with rank 8 and scale 20.0.
- Checkpoint at iter 1500 (`adapters.safetensors`) is the final trained adapter.

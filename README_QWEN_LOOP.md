# `llama.cpp-qwen-loop` — Recurrent-Loop Inference (`1..16` Loops) for Qwen3.5 / Qwen3.8

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CympUser/llama.cpp-qwen-loop/blob/qwen35-loop/try_in_colab.ipynb)
[![Download Windows CUDA Release](https://img.shields.io/badge/Windows_CUDA-1--Click_Release_v1.0.0-blue?logo=windows)](https://github.com/CympUser/llama.cpp-qwen-loop/releases/tag/v1.0.0)

This fork extends [`brianchmielNV/llama.cpp`](https://github.com/brianchmielNV/llama.cpp) (Dory support) with native **recurrent-loop execution** for the hybrid **Qwen3.5 / Qwen3.8** architecture (Gated DeltaNet + Full Attention `3:1`), enabling test-time compute scaling from `1` to `6+` loops using the same quantized weights in VRAM.

## 1-Click Quickstart (Windows CUDA — No Build Required)

1. Download **`llama-cpp-qwen-loop-v1.0.0-win-cuda-x64.zip`** from [Releases](https://github.com/CympUser/llama.cpp-qwen-loop/releases/tag/v1.0.0) and unzip it.
2. Double-click **`run.bat`** (automatically downloads the base GGUF and Loop-6 adapter, starts `llama-server.exe`, and opens `http://localhost:8080`).
3. Switch between **1-Loop, 2-Loop, 4-Loop, and 6-Loop** live inside the Web UI dropdown!

## What Changed in This Engine

1. **Virtual Layer Partitioning (`src/models/qwen35.cpp`)**:
   - Splits `32` physical trunk blocks into **Prelude (`0..3`)**, **Recurrent Core (`4..27` repeated `K` times)**, and **Coda (`28..31`)** — yielding `152` virtual layers at `K=6` without duplicating model weights.
   - Controlled via GGUF KV overrides: `qwen35.num_loops=int:K` (`1..16`) and `qwen35.recurrent_kv_cache_mode=str:per_loop|last_loop`.
2. **Identity-Preserving Loop Bridges & Per-Iteration LoopLoRA (`src/llama-adapter.{h,cpp}`, `src/models/qwen35.cpp`)**:
   - Supports GGUF adapter tensors `loop.<k>.in_scale.weight`, `loop.<k>.embd.weight`, `loop.<k>.out_gate.weight` and per-iteration LoRA pairs `loop.<k>.blk.<p>.<stem>.weight.lora_{a,b}` (`k=0` stays frozen so `K=1` preserves exact base behavior).
3. **`llama-server` Memory Hardening (`tools/server/server-task.cpp`, `tools/server/server-context.cpp`)**:
   - Guards recurrent prompt-cache checkpoint cloning/creation against `std::bad_alloc` (preventing crashes when `K=6` creates `~234 MiB` DeltaNet checkpoints).
4. **C++/CUDA Verification (`tests/test-qwen35-loop.cpp`)**:
   - Verifies zero-init identity (`NMSE ~ 2.4e-10`), CPU vs CUDA parity across `K=1..6` (`NMSE <= 1.46e-08`), and prefill vs step-by-step decode (`NMSE = 3.44e-09`).

## Build (Windows / CUDA)

```powershell
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-cli llama-server test-qwen35-loop -j
.\build\bin\Release\test-qwen35-loop.exe
```

## Quick Run (`K=1..6` Loops)

Download the base GGUF and the LoopLoRA adapter from the companion Hugging Face repository, then run:

```powershell
.\build\bin\Release\llama-server.exe `
  -m .\models\Qwen3.8-9B-Distill-uncensored-heretic.Q4_K_S.gguf `
  --lora .\models\qwen3.8-9b-heretic-loop6-lora.gguf `
  --override-kv qwen35.num_loops=int:6,qwen35.recurrent_kv_cache_mode=str:per_loop `
  -c 8192 -np 1 --cache-ram 1024 --ctx-checkpoints 2 `
  -ngl 999 --fit off --cache-type-k q8_0 --cache-type-v q8_0 --repeat-penalty 1.15 --port 8080
```

- Change `int:6` to any value in `int:1`..`int:6` at startup.
- If using a vision projector (`--mmproj`), add `--no-mmproj-offload` to keep the vision encoder in system RAM and save VRAM for context.
- See [`models.example.ini`](models.example.ini) for ready-to-use multi-loop presets (`--models-preset models.example.ini`).

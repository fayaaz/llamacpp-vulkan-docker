# Running Qwen3.8 over llama.cpp RPC between two nodes

## Hardware

| Node | CPU | RAM | GPU | VRAM |
|---|---|---|---|---|
| primary | AMD Ryzen 9 9950X3D | 64 GB | AMD Radeon RX 9700 XT | 16 GB |
| worker | AMD Ryzen 5 3600 | 16 GB | AMD Radeon RX 6600 XT | 8 GB |

Combined VRAM: ~24 GB. Network: LAN between the two nodes. RPC server runs on the 6600 XT box (worker), llama-server runs on the 9700 XT box (primary).

## llama.cpp RPC overview

llama.cpp's `--rpc` backend splits the model computation across machines. The master node loads the GGUF and streams hidden states; the RPC nodes execute tensor ops on those states and return results. RPC traffic is small (a few kB of hidden state per layer), so inference on RPC isn't bandwidth-limited — it's latency-limited.

Important operational details:

- `split-mode = layer` is required for cross-device splitting. `split-mode = none` pins everything to `main-gpu` and the RPC worker is only probed, never used.
- `--fit` computes an offload plan across local GPUs and the RPC device. If KV cache or a draft model does not fit in the combined VRAM, they spill to system RAM first; the model still runs, but slowly.
- The RPC server always exports all backends it was built with (Vulkan/CPU here). Our custom image `ghcr.io/fayaaz/llama-cpp-vulkan-rpc` is built with `GGML_VULKAN=ON` and `GGML_RPC=ON` from upstream `.devops/vulkan.Dockerfile`, and includes `ggml-rpc-server`.

## Client setup (primary / 9070 XT)

`config.ini` preset example (`qwen-3.8-27b-uncensored-100k`):

```ini
model = /models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf
ctx-size = 102400
parallel = 1
split-mode = layer
fit = on
fit-target = 128
fit-ctx = 102400
flash-attn = on
jinja = on
cache-type-k = q8_0
cache-type-v = q4_0
batch-size = 2048
ubatch-size = 256
threads = 32
reasoning = on
reasoning-effort = xhigh

spec-type = draft-mtp
spec-draft-n-max = 2
```

`docker-compose.yml` supplies the RPC endpoint via env:

```yaml
environment:
  LLAMA_ARG_RPC: "${LLAMA_ARG_RPC}"
```

`.env` (gitignored, create locally):

```
LLAMA_ARG_RPC=<worker-ip>:30552
```

The server is started with:

```
docker compose up -d server
```

The 6600 XT side runs the RPC worker listening on port 50052

## Verified RPC usage

Before RPC was actually put to use, the 6600 XT sat mostly idle during inference. Once `split-mode = layer` replaced `none` in the preset, `amdgpu_top` on the weak node showed:

- `Total VRAM Usage: 6188 / 8176 MiB`
- `gpu_activity GFX: 54%`, ~58 W total board power

I.e. the 6600 XT is holding ~6 GiB of weights/KV and doing real compute during generation.

## Model + config matrix and measured tok/s

Tokens/s numbers are from the last load-bearing requests on primary (`llama-server` router mode, single parallel slot). `prompt eval` is tokens/s of prompt processing, `eval` is generated tokens/s. Where a model has `spec-type = draft-mtp`, decode goes through the MTP draft head.

| Preset | Main GGUF | ctx | split | draft | prompt eval tok/s | decode tok/s |
|---|---|---|---|---|---|---|
| `qwen-3.8-27b-100k` | Qwen3.8-27B-IQ4_XS (no MTP) | 102k | layer | none | 22.9 | 4.9 |
| `qwen-3.8-27b-uncensored-100k` | Qwen3.8-27B-Uncensored-IQ4_XS | 102k | layer | fused MTP (`n_max=2`) | 82.2 | 24.2 |
| `qwen-3.8-27b-q4km` | Qwen3.8-27B-UD-Q4_K_M | 65k | layer | separate MTP Q4_0 | 116.3 | 40.2 |
| `qwen-3.8-27b-udq6k` | Qwen3.8-27B-UD-Q6_K | 131k | layer | separate MTP Q4_0 | 102.9 | 14.8 |
| `qwen-3.8-27b-q4km-vision-100k` | Qwen3.8-27B-UD-Q4_K_M | 102k | layer | separate MTP Q4_0 | 85.7 | 29.9 |
| `ornith-1.0-35b` | ornith-1.0-35b-Q4_K_M | 65k | layer | fused MTP | 295.8 | 26.8 |
| `ornith-1.0-35b-vision` | Ornith-1.0-35B-MTP-APEX-I-Compact | 65k | layer | fused MTP | 230.4 | 26.1 |
| `ornith-1.0-9b` | Ornith-1.0-9B-Q8_0 | 65k | layer | none | 558.3 | 32.7 |
| `ornith-1.5-35b-abliterated` | Huihui-Ornith-1.5-35B-A3B-abliterated | 65k | layer | none | 233.1 | 26.2 |

Notes on the measurements:

- The MTP draft, when active, raises decode throughput roughly 1.6× over the same base without it. All uncensored presets have the MTP head fused in the main file; for the base model we attach a stand-alone MTP draft via `spec-draft-model`.
- `qwen-3.8-27b-udq6k` fits in ~25 GiB total (weights ~21 GiB + mmproj ~0.9 GiB + 128k KV ~3.2 GiB compressed). That is marginally over the 24 GiB node pair, so some layers may sit in system RAM. It still runs without error at ~19 tok/s decode.
- Prompt processing on a long 9.7k prompt ran at ~386–399 tok/s on the Q6 preset and ~256 tok/s on the IQ4 uncensored one, both with the draft enabled; v-cache hits made the IQ4 prompt path faster (up to ~154 t/s on long prompts in prior loads).

## Practical KV cache sizing

Weight sizes are ~15.3 GiB (IQ4_XS), ~16 GiB (Q4_K_M), ~19 GiB (Q5_K_M) and ~21 GiB (Q6_K). The only part that scales with context length on Qwen3.8 is the KV cache of the 16 full-attention layers; the 48 Gated DeltaNet layers hold a fixed-size recurrent state. Compact KV math from the model card: at `cache-type-k = q8_0`, `cache-type-v = q4_0` the 16 attention layers cost ~26.6 KB/token ≈ 2.5 GiB at 102400 ctx and ~6.5 GiB at native 262144 ctx.

So on the 24 GiB pair, practical model + ctx budgets are:

- IQ4_XS + 102k ctx + mmproj: ~15.3 + 2.5 + 0.9 ≈ 18.7 GiB — comfortable.
- Q4_K_M + 131k ctx + mmproj: ~16 + 3.2 + 0.9 ≈ 20 GiB — comfortable.
- Q5_K_M + 131k ctx + mmproj: ~19.8 + 3.2 + 0.9 ≈ 24 GiB — right at the edge.
- Q6_K + 131k ctx + mmproj: ~21 + 3.2 + 0.9 ≈ 25 GiB — offloads some state to RAM, still runs at ~19 t/s.

Dropping `cache-type-k` to `q4_0` roughly halves the Q8 half of KV and buys a couple of GiB headroom, at a measurable quality cost under heavy reasoning (upstream ggml-org/llama.cpp#23470). We chose q8_0 keys everywhere in the presets.

## What does not transfer over RPC

- llama.cpp RPC has no authentication and should not be exposed beyond the LAN (the `ggml-rpc-server` binary prints that warning itself).
- The RPC worker does not need model files. Only the GGUF + mmproj on the master node matter. `--fit` converges to a layer split without consulting the RPC node's disk.
- Draft-KV is incremental. If you tie a draft model that doesn't fit, the router loads it before the target weights and you can OOM that slot; keep it small (MTP Q4_0 is only ~1.9 GiB).

## Improving RPC performance (from the SharedLLM post and upstream PR state)

Observations and tuning recommendations targeted at your pair:

1. **Same worker+server build**: RPC and server must be the same GGML ABI version. We build both from the same image (`ghcr.io/fayaaz/llama-cpp-vulkan-rpc:latest`). If Arches ever uses a different tag, hang or sudden decode stalls are the first symptom. Pin by digest, not `latest`, before production use.


3. **layer split is the model, not the VRAM**: with `--split-mode layer`, llama.cpp assigns whole layers (and KV blocks) to each device proportionally to how `--fit` computes the budget. There is no tensor chunking within a matmul — the RPC device gets separate fused operations, not half a GEMM. So inter-machine RPC does not trade the same way as an MTP draft would.

4. **Draft offload**: MTP drafts are strictly profitable because the main LLM is the bottleneck for tokens. On primary, make sure the draft stays local: for the base Q4_K_M profile we use the separate `mtp-Qwen3.8-27B-Q4_0.gguf` (≈1.9 GiB); the uncensored GGUF fuses the draft into the main file. Both go through RPC when `split-mode=layer` — you could pin it to slot 0 by putting it first in `tensor-split` if it benchmarks better.

5. **Context size**: Q6_K at 131k is tight (~25 GiB); measured on combined 9700 XT + 6600 XT it spills to RAM or glued layers and decodes at ~19 t/s. For Q6, stay at 65k. For Q5, Q4, IQ4, 131k is achievable.

6. **Network is 900–950 MB/s**: RPC hides behind the GPU streams; raw LAN bandwidth isn't the decode-performance limiter. The decode path is latency-bound per layer, so moving the worker closer or enabling 2.5Gbps on both ends helps prefill and graph splice more than steady decode.

7. **No `--n-cpu-moe` mixing**: with RPC and a single layer split, MoE weights must live on one device's allocation; we did not blend `--n-cpu-moe` into the same profile as RPC (keep separate sections).

8. **Relevant upstream PRs**:
   - ggml-org/llama.cpp#8032 — RPC buffer hibernation (not yet merged)
   - ggml-org/llama.cpp#24524 — CUDA-only MoE expert cache prototype (not merged, CUDA-only)

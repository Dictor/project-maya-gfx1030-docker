<p align="center"><img src="assets/maya-logo.png" width="112" alt="Project Maya logo"></p>

<h1 align="center">Project Maya — AMD RDNA2 (<code>gfx1030</code>) Fork</h1>

<p align="center">
  <b>GLM-5.3-Flash (Maya-S &amp; Maya-M) optimized for AMD RDNA2 (<code>gfx1030</code>) GPUs &amp; AVX2-only Unraid / Linux hosts</b><br>
  Based on upstream <a href="https://github.com/mw00/project-maya">Project Maya</a> <code>v1.1.69</code> · Branch: <code>feat/gfx1030-rdna2-opt</code><br>
  Docker image: <code>kimdictor/project-maya-gfx1030:v1.1.69.1</code> (also tagged <code>:v1.1.69</code>, <code>:latest</code>; fork revision tracked in <code>docker/VERSION.gfx1030</code>)
</p>

---

This fork builds on upstream **[Project Maya](https://github.com/mw00/project-maya)** (created by Peasant Smith [@mw00](https://github.com/mw00) and co-maintained with [@needmorevram](https://github.com/needmorevram)) to target **AMD RDNA2 (`gfx1030`)** cards — such as **Dual Radeon PRO W6800 32 GB**, RX 6800/6900, and PRO V620 — running **GLM-5.3-Flash (`Maya-S` and `Maya-M`)** on headless Linux / Unraid servers with AVX2-only CPUs (e.g., Haswell Xeon without AVX-512), THP `[always]`, and HDD-stored GGUF shards.

For full upstream documentation covering all models, NVIDIA/Metal/Windows setup, and web UI features, see [docs/README.md](docs/README.md). A Korean Unraid deployment guide is also available in [docs/DOCKER_GFX1030.md](docs/DOCKER_GFX1030.md).

---

## What Differs from Upstream `main` (`v1.1.69` → `feat/gfx1030-rdna2-opt`)

- **RDNA2 (`gfx1030`) HIP decode optimizations** (`src/kernels/cuda/glm_fast.cu`, `src/kernels/cuda/iq_kquant.cuh`, `include/strata/kernels/fresh_load.hpp`, `src/ds4/ds4_pcie.cuh`):
  - Enables the tuned wave32 expert kernels on `gfx103x` for **Maya-S** (`IQ2_XXS`/`IQ2_S`) and **Maya-M** (`IQ2_S`/`IQ3_XXS`/`IQ3_S`), adds tuned `IQ3_S` down + multi-token window-dedup support, and defaults `MAYA_GLM_GU22_TUNED` ON for `gfx103x`.
  - Enables the optimized dense `Q6_K`/BF16 GEMV kernel (`MAYA_GLM_MV_RDNA`) on `gfx103x`, using native `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) with hoisted per-activation `q6_u_bias`.
  - Replaces LDS `ds_bpermute` warp reductions with VALU `v_permlanex16_b32` + DPP `row_xmask`, replaces emulated `__vsubss4`/`__vcmpne4`/`__vsub4` with bit-parallel 2-op rewrites (`q6_m32`, `q3_sub4`, `hip_fast_signs`), adds `s_sleep` (`__nanosleep`) backoff to HIP spin-waits, and enables 100 MHz `wall_clock64()` timers on `gfx103x`.
- **RDNA2 (`gfx1030`) prefill acceleration** (`src/kernels/cuda/glm_batch.cu`, `src/core/glm_prefill.cu`, `src/core/glm_prefill_dsa.cu`, `src/prefill/gemm.cu`):
  - **`mla_attn_fdot2`** (`MAYA_GLM_MLA_FDOT2=1`): RDNA2 prompt MLA attention using native `v_dot2_f32_f16` (`__builtin_amdgcn_fdot2`) with per-head power-of-two query scaling and FP32 softmax/accumulation.
  - **FP16-output prompt GEMMs** (`MAYA_GLM_PROMPT_F16=1`): runs `rocBLAS` FP16-output prompt GEMMs with in-place row widening (`widen_rows_f16`) and optional overflow diagnostics (`MAYA_GLM_PROMPT_F16_CHECK=1`).
- **RDNA2 stability, vision & tooling fixes** (`src/core/glm_dflash.cu`, `src/core/glm_fast_path.cu`, `src/core/glm_model.cu`, `serve/server.py`, `maya.py`):
  - Keeps auxiliary RDNA2 streams (DFlash2 drafter/inject, prefetch CE, head-split) at default stream priority to avoid cross-stream stalls (`#621`).
  - Fixes dual-GPU HIP vision pinning in `serve/server.py` (`HIP_VISIBLE_DEVICES` pinned to GPU 0 to match `VLEND`).
  - Unlocks `--calibrate` / `tools/calibrate_glm.py` for the HIP backend and adds `MAYA_PORTABLE=1` (`-DSTRATA_PORTABLE=ON`) for native Linux AVX2-baseline builds.
- **Minimal 2-stage Unraid / Linux Docker image** (`docker/Dockerfile.gfx1030`, `docker/entrypoint-gfx1030.sh`):
  - Built on `ubuntu:24.04` with AMD TheRock ROCm 7 (`rocm[libraries,devel]==7.13.0a20260515`, `gfx103X-all`) and stripped to minimal runtime `/opt/rocm/lib` + `rocBLAS` Tensile tables.
  - Ships pre-compiled and pre-stamped `/opt/maya/build-hip/maya` (GLM-5.3-Flash only), `/opt/maya/build-vision/bin/maya-vision` (HIP `gfx1030` + portable AVX2), `/opt/maya/build-vision-cpu/bin/maya-vision` (CPU-only portable AVX2), and 9 on-GPU HIP test/benchmark binaries under `/opt/maya/tests/`.
  - Enforces build-time checks baked into `/opt/maya/image/`: zero AVX-512 `zmm` instructions in host non-dispatched/ggml/vision binaries (`host-isa-check.txt`), verified `gfx1030` ISA with `spill_vgpr_count=0` (`gfx1030-isa-check.txt`), zero unresolved `ldd` libraries, and a 20-minute `HEALTHCHECK` (`--start-period=1200s`) for one-time HDD cold loading.

| GLM-5.3-Flash Quant | GGUF Size | Total Memory Needed | Dual W6800 (64 GB VRAM) + ~100 GiB RAM |
| --- | ---: | ---: | --- |
| **Maya-S** (`Maya-S-v2-IQ2_XXS`) | 96.5 GB | ~104 GB | 100% VRAM + pinned RAM resident (`MAYA_GLM_RAM_RESIDENT=1`) |
| **Maya-M** (`Maya-M-IQ2_S`) | 116.0 GB | ~124 GB | 100% VRAM + pinned RAM resident (`MAYA_GLM_RAM_RESIDENT=1`) |

---

## Quick Start (Docker Compose)

```yaml
services:
  maya:
    image: kimdictor/project-maya-gfx1030:v1.1.69.1 # or :latest
    container_name: maya-gfx1030
    restart: unless-stopped
    devices:
      - /dev/dri:/dev/dri
      - /dev/kfd:/dev/kfd
    group_add:
      - video
      - render
    ipc: host
    cap_add:
      - IPC_LOCK
    ulimits:
      memlock:
        soft: -1
        hard: -1
    ports:
      - "8080:8080"
    volumes:
      # Unraid HDD tip: mount the direct disk path (/mnt/diskX/...) instead of
      # /mnt/user/... FUSE to speed up the one-time cold load into VRAM + RAM.
      - /mnt/disk1/gguf:/models:ro
      # Keep /data (~2 GB pack index/dense weights + configs + logs) on NVMe/SSD cache.
      - /mnt/cache/appdata/maya:/data:rw
    environment:
      # Auto-packs on first start into /data/pack/<model> and writes /data/maya-<model>-hip.json.
      # Switch to /models/Maya-M to run Maya-M, or set MAYA_CONFIG=/data/maya-<model>-hip.json.
      MAYA_GGUF_DIR: /models/Maya-S-v2-IQ2_XXS
      MAYA_GPUS: "0,1"
      MAYA_CONTEXT: "32768"
      # Default is 1: loads all off-VRAM experts into pinned RAM once at startup -> zero HDD reads during decode.
      MAYA_GLM_RAM_RESIDENT: "1"
      # Optional:
      # MAYA_VISION: gpu          # gpu (GPU 0 on-demand via VLEND) or cpu
      # MAYA_API_KEY: "your-key"
```

```bash
docker compose up -d
```

- **HDD storage + zero decode disk I/O:** With `MAYA_GLM_RAM_RESIDENT=1` and default startup warm-up (`MAYA_GLM_WARM` on), the container reads GGUF shards from HDD **once** at startup into VRAM + pinned RAM, then serves decode with zero disk reads. Mounting `/mnt/diskX/...` directly instead of Unraid's `/mnt/user/...` FUSE mount eliminates FUSE overhead during that initial 96.5–116 GB sequential load.
- **Entrypoint modes (`docker/entrypoint-gfx1030.sh`):**
  1. **No args (default above):** launches via `MAYA_CONFIG`, `MAYA_PACK`, `MAYA_GGUF`, or `MAYA_GGUF_DIR` (or auto-discovers a single `/data/maya-*.json`).
  2. **`serve` / flags:** `docker compose run --rm maya --config /data/maya-<model>-hip.json --port 8080` forwards flags to `/opt/maya/serve/server.py`.
  3. **Command passthrough:** any non-flag command executes directly (e.g., `/opt/maya/tests/*`, `bash`, `cat /opt/maya/image/gfx1030-isa-check.txt`).
- **Config JSON precedence:** Auto-generated configs leave `"env": {}` so container environment variables always take effect. In a hand-edited `MAYA_CONFIG` JSON, keys inside its `"env"` object override container `-e` variables.

---

## Environment Variable Reference

### Container & Model Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MAYA_GGUF_DIR` | *(unset)* | Directory with model `.gguf` shard(s); auto-packs via `tools/iq_pack.py --compat-bf16` on first start |
| `MAYA_GGUF` | *(unset)* | Explicit path to the first `.gguf` shard (alternative to `MAYA_GGUF_DIR`) |
| `MAYA_PACK` | *(unset)* | Pre-built pack directory (e.g., `/data/pack/maya-s`) |
| `MAYA_CONFIG` | *(unset)* | Existing server config JSON path (e.g., `/data/maya-<model>-hip.json`) |
| `MAYA_DATA` | `/data` | Persistent directory for packs, generated configs, and logs |
| `MAYA_GPUS` | `0,1` | Comma-separated GPU indices passed to `--gpu` |
| `MAYA_CONTEXT` | `32768` | Maximum context length (`--max-context`) |
| `MAYA_KV_STREAMING` | `on` | `--kv-streaming` setting |
| `MAYA_KV` | *(unset)* | Optional `--kv` cache format override |
| `MAYA_LAYER_SPLIT` | `auto` | Multi-GPU layer split strategy |
| `MAYA_HOST` / `MAYA_PORT` | `0.0.0.0` / `8080` | Server bind address and port (also honors `HOST` / `PORT`) |
| `MAYA_API_KEY` | *(unset)* | Optional API key (`--api-key`) |
| `MAYA_VISION` | `off` | Enable vision encoder: `gpu` (GPU 0 via `VLEND`) or `cpu` (needs `mmproj-GLM-5.3-Flash-F16.gguf` + `GLM-5.3-Flash-vocab.gguf`) |
| `MAYA_VISION_DIR` | *(auto-searched)* | Override search directory for vision GGUF files (also checks `<gguf>/vision`, `/models/vision`, `/data/vision`) |

### Runtime Defaults & `gfx1030` Optimization / Rollback Toggles

All defaults in `docker/entrypoint-gfx1030.sh` can be overridden via Compose `environment:` or `docker run -e`.

| Variable | Default | Purpose / Rollback |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | RDNA2 `gfx1030` ROCm runtime override |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Prevents ROCm scratch reallocation stalls |
| `GPU_PINNED_MIN_XFER_SIZE` | `1048576` | HIP pinned DMA threshold (bytes) |
| `MAYA_UPDATE_CHECK` | `0` | Disables in-container git self-update |
| `MAYA_GLM_RESERVE_MB` | `700` | Headless VRAM reserve per GPU (reclaims ~4.6 GiB across 2× W6800 vs. 3072 MiB desktop default) |
| `MAYA_GLM_PIN_PIECE_MB` | `1024` | Piecewise `hipHostMalloc` sizing to avoid Direct Compaction freezes under Unraid THP `[always]` |
| `MAYA_GLM_RAM_RESIDENT` | `1` | Keeps all off-VRAM experts in RAM after startup warm-up (`0` restores disk-backed tiering) |
| `MAYA_GLM_PREFILL_MIN_CHUNKS` | `2,512` | Pipelines prompts across dual GPUs instead of idling GPU 1 |
| `MAYA_GLM_PREFILL_SUB` | `1024` | Prompt sub-batch size |
| `MAYA_GLM_PREFILL_MB` | `4096` | Prompt scratch ceiling borrowed from expert pool tail |
| `MAYA_GLM_PF_CE` | `0` | Disables prefetch copy-engine high-priority stream (`#621`) |
| `MAYA_GLM_PROMPT_F16` | `1` | `gfx1030` FP16-output prompt GEMMs + widen (`0` rolls back to FP32-out; set `MAYA_GLM_PROMPT_F16_CHECK=1` to log non-finite counts) |
| `MAYA_GLM_MLA_FDOT2` | `1` | `gfx1030` `v_dot2_f32_f16` prompt MLA attention (`0` rolls back to F32) |
| `MAYA_GLM_MV_RDNA` | `1` | `gfx1030` optimized dense `Q6_K`/BF16 GEMV (`0` rolls back to original kernels) |
| `MAYA_GLM_GU22_TUNED` | *(ON on `gfx103x`)* | Tuned `IQ2_S` gate/up expert path for Maya-M (`0` rolls back to original) |
| `MAYA_HIP_EXPERTS_LEGACY` | *(unset)* | Set `1` to roll back all tuned `gfx1030` wave32 expert kernels to legacy path |
| `MAYA_GLM_STREAM_PRIO` / `MAYA_GLM_DFLASH_PRIO` | *(0 on `gfx103x`)* | Set `1` to opt back into high-priority auxiliary HIP streams |
| `MAYA_PORTABLE` | *(unset)* | Native Linux builds only: `MAYA_PORTABLE=1 ./maya.sh --backend hip --setup` enables `-DSTRATA_PORTABLE=ON` (AVX2 baseline) |

---

## Tools & Verification

### 1. GGUF Packing (Auto or Manual)

Setting `MAYA_GGUF_DIR` (or `MAYA_GGUF`) automatically runs `tools/iq_pack.py --compat-bf16` once into `/data/pack/<model>`. To pack manually:

```bash
docker compose run --rm maya \
  /opt/maya/.venv/bin/python /opt/maya/tools/iq_pack.py \
    --gguf /models/Maya-S-v2-IQ2_XXS/<first-shard>.gguf \
    --out /data/pack/maya-s \
    --compat-bf16
```

### 2. HIP CPU-Lane / PCIe-Share Calibration (`--calibrate`)

Measures optimal `MAYA_GLM_PCIE_SHARE`, `MAYA_GLM_CPU_LANE` (and `MAYA_GLM_READ_CHUNKS` if disk-backed) on your CPU + PCIe topology. Stop the running server first so GPUs are idle:

```bash
docker compose stop maya
docker compose run --rm maya \
  /opt/maya/.venv/bin/python /opt/maya/tools/calibrate_glm.py \
    /data/maya-<model>-hip.json
```

On native Linux installs, run `./maya.sh --calibrate` to measure and persist settings into the config automatically.

### 3. On-GPU HIP Parity & Benchmark Suite (`/opt/maya/tests/`)

The image includes 9 compiled HIP tests and synthetic fixtures (`/opt/maya/data/glm5-synth*`). Run all parity checks on GPU 0:

```bash
docker compose run --rm -e HIP_VISIBLE_DEVICES=0 maya bash -c '
  set -euo pipefail
  cd /opt/maya
  /opt/maya/tests/hip_intrinsics
  /opt/maya/tests/hip_glm_handoff
  /opt/maya/tests/hip_prefill_mmq_parity
  /opt/maya/tests/hip_glm_prefill_attention
  /opt/maya/tests/glm_mv_bench --parity-only
  /opt/maya/tests/hip_expert_kernel_bench --parity-only
  /opt/maya/tests/glm_attn_bench --parity-only
  /opt/maya/tests/glm_layer_parity --selftest
  /opt/maya/tests/glm_model_test --selftest
'
```

Run microbenchmarks / kernel sweeps on GPU 0:

```bash
docker compose run --rm -e HIP_VISIBLE_DEVICES=0 maya /opt/maya/tests/hip_expert_kernel_bench --sweep
docker compose run --rm -e HIP_VISIBLE_DEVICES=0 maya /opt/maya/tests/glm_mv_bench
docker compose run --rm -e HIP_VISIBLE_DEVICES=0 maya /opt/maya/tests/glm_attn_bench
```

*(Other flags: `hip_expert_kernel_bench` also supports `--shared` and `--window`; `glm_mv_bench` and `glm_attn_bench` also support `--sweep`.)*

### 4. Baked Build, Version & ISA Verification Reports (`/opt/maya/image/`)

Inspect the baked image version (`/opt/maya/image/version.txt`), compile-time AMDGPU `gfx1030` ISA report, and Haswell AVX2 host portability report baked into the image (no GPU required):

```bash
docker run --rm kimdictor/project-maya-gfx1030:v1.1.69.1 cat /opt/maya/image/version.txt
docker run --rm kimdictor/project-maya-gfx1030:v1.1.69.1 cat /opt/maya/image/gfx1030-isa-check.txt
docker run --rm kimdictor/project-maya-gfx1030:v1.1.69.1 cat /opt/maya/image/host-isa-check.txt
docker run --rm kimdictor/project-maya-gfx1030:v1.1.69.1 cat /opt/maya/image/cpu-flags.txt
```

### 5. Local Docker Build

Build locally with Docker BuildKit (Windows Docker Desktop or Linux). Image version combines upstream `VERSION` + `docker/VERSION.gfx1030` (`v<UPSTREAM>.<REV>`, e.g., `v1.1.69.1`):

```bash
docker build \
  -f docker/Dockerfile.gfx1030 \
  --build-arg IMAGE_VERSION=v1.1.69.1 \
  -t kimdictor/project-maya-gfx1030:v1.1.69.1 \
  -t kimdictor/project-maya-gfx1030:1.1.69.1 \
  -t kimdictor/project-maya-gfx1030:v1.1.69 \
  -t kimdictor/project-maya-gfx1030:1.1.69 \
  -t kimdictor/project-maya-gfx1030:latest \
  .
```

Optional `--build-arg` overrides: `HIP_ARCHITECTURES` (`gfx1030`), `LLAMA_CPP_COMMIT` (`3cf03257f219afbe7334045ff7c6a06ac68c627d`), `THEROCK_INDEX` (`https://rocm.nightlies.amd.com/v2/gfx103X-all/`), `THEROCK_SPEC` (`rocm[libraries,devel]==7.13.0a20260515`), `JOBS` (`nproc / 2`).

---

## Further Documentation & Credits

- **Docs:** [docs/DOCKER_GFX1030.md](docs/DOCKER_GFX1030.md) (Unraid guide, KR) · [docs/SETTINGS.md](docs/SETTINGS.md) · [docs/AMD_MAYA.md](docs/AMD_MAYA.md) · [docs/MEMORY.md](docs/MEMORY.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/README.md](docs/README.md)
- **Upstream:** [Project Maya](https://github.com/mw00/project-maya) is created and led by **Peasant Smith** ([@mw00](https://github.com/mw00)), co-maintained with [@needmorevram](https://github.com/needmorevram), built on [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp).
- **License:** MIT License ([LICENSE](LICENSE)); third-party notices in `third_party/` apply.

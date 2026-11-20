#!/bin/sh
# Entrypoint for Project Maya AMD RDNA2 (gfx1030 / Dual Radeon PRO W6800) container.
#
# Launch modes:
#   1. Explicit command passthrough (e.g. direct serve/server.py, /opt/maya/tests/*, bash):
#        docker run ... <image> /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya --config ...
#        docker run ... <image> /opt/maya/tests/hip_intrinsics
#   2. Direct serve/server.py flags (or leading "serve"):
#        docker run ... <image> --config /data/maya-s-hip.json --port 8080
#   3. Env-driven Unraid launch (no args):
#        -e MAYA_CONFIG=/data/maya-s-hip.json
#        -e MAYA_PACK=/data/pack/maya-s
#        -e MAYA_GGUF_DIR=/models/Maya-S-v2-IQ2_XXS   (auto-packs via tools/iq_pack.py + writes config)
#
# All defaults below are overridable via container `-e KEY=VALUE`.
set -eu

cd /opt/maya || exit 1

# ------------------------------------------------------------------------------
# 1. ROCm / HIP runtime & Unraid Dual-W6800 (gfx1030) GLM-5.3-Flash defaults
# ------------------------------------------------------------------------------
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAYA_GGUF_PY="${MAYA_GGUF_PY:-/opt/maya/third_party/llama.cpp/gguf-py}"

# RDNA2 (gfx1030) ROCm/HIP stability
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export GPU_PINNED_MIN_XFER_SIZE="${GPU_PINNED_MIN_XFER_SIZE:-1048576}"

# Container hygiene (self-update is a git fast-forward; disabled inside container)
export MAYA_UPDATE_CHECK="${MAYA_UPDATE_CHECK:-0}"

# Unraid Dual-W6800 (2x 32GB VRAM, 110 GiB RAM, THP [always], HDD GGUFs) GLM defaults:
# - RESERVE_MB=700: headless Unraid needs no 3 GiB desktop GUI margin (+4.6 GiB VRAM across 2 cards)
# - PIN_PIECE_MB=1024: avoids 15+ min Direct Compaction freezes under Unraid host THP [always]
# - RAM_RESIDENT=1: keeps 100% of off-VRAM experts resident in RAM (zero HDD reads during decode;
#   MAYA_GLM_WARM stays at its default ON so warm-up fills VRAM + RAM once at startup)
# - PREFILL_MIN_CHUNKS=2,512: pipelines prompts across both W6800 cards instead of idling GPU 1
# - PREFILL_SUB=1024 / PREFILL_MB=4096: larger prompt sub-batches inside borrowed pool tail
# - PF_CE=0: avoids RDNA2 high-priority cross-stream deadlock (#621)
export MAYA_GLM_RESERVE_MB="${MAYA_GLM_RESERVE_MB:-700}"
export MAYA_GLM_PIN_PIECE_MB="${MAYA_GLM_PIN_PIECE_MB:-1024}"
export MAYA_GLM_RAM_RESIDENT="${MAYA_GLM_RAM_RESIDENT:-1}"
export MAYA_GLM_PREFILL_MIN_CHUNKS="${MAYA_GLM_PREFILL_MIN_CHUNKS:-2,512}"
export MAYA_GLM_PREFILL_SUB="${MAYA_GLM_PREFILL_SUB:-1024}"
export MAYA_GLM_PREFILL_MB="${MAYA_GLM_PREFILL_MB:-4096}"
export MAYA_GLM_PF_CE="${MAYA_GLM_PF_CE:-0}"

# RDNA2 (gfx1030) kernel optimization switches (all cleanly toggleable to 0)
export MAYA_GLM_PROMPT_F16="${MAYA_GLM_PROMPT_F16:-1}"
export MAYA_GLM_MLA_FDOT2="${MAYA_GLM_MLA_FDOT2:-1}"
export MAYA_GLM_MV_RDNA="${MAYA_GLM_MV_RDNA:-1}"

PYBIN="/opt/maya/.venv/bin/python"
SERVER="/opt/maya/serve/server.py"
HOST="${MAYA_HOST:-${HOST:-0.0.0.0}}"
PORT="${MAYA_PORT:-${PORT:-8080}}"
DATA_DIR="${MAYA_DATA:-/data}"

# Warn if host CPU lacks AVX2 (Haswell Xeon E5-2666 v3 has AVX2; older CPUs do not)
if [ -r /proc/cpuinfo ] && grep -q '^flags' /proc/cpuinfo && ! grep -m1 '^flags' /proc/cpuinfo | grep -qw avx2; then
  echo "[maya-gfx1030] WARNING: host CPU lacks AVX2; portable AVX2 binaries may raise SIGILL." >&2
fi

# ------------------------------------------------------------------------------
# 2. Mode 1: Explicit command passthrough
# ------------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
  case "$1" in
    serve)
      shift
      ;;
    -*)
      ;;
    *)
      exec "$@"
      ;;
  esac

  # Mode 2: Direct serve/server.py flags (fill missing --engine / --host / --port / --config)
  has_engine=0
  has_host=0
  has_port=0
  has_config=0
  for arg in "$@"; do
    case "$arg" in
      --engine|--engine=*) has_engine=1 ;;
      --host|--host=*)     has_host=1 ;;
      --port|--port=*)     has_port=1 ;;
      --config|--config=*) has_config=1 ;;
    esac
  done
  if [ "$has_config" -eq 0 ] && [ -n "${MAYA_CONFIG:-}" ]; then
    set -- --config "$MAYA_CONFIG" "$@"
  fi
  if [ "$has_port" -eq 0 ]; then
    set -- --port "$PORT" "$@"
  fi
  if [ "$has_host" -eq 0 ]; then
    set -- --host "$HOST" "$@"
  fi
  if [ "$has_engine" -eq 0 ]; then
    set -- --engine maya "$@"
  fi
  exec "$PYBIN" "$SERVER" "$@"
fi

# ------------------------------------------------------------------------------
# 3. Mode 3: Env-driven launch (MAYA_CONFIG / MAYA_PACK / MAYA_GGUF / MAYA_GGUF_DIR)
# ------------------------------------------------------------------------------
CFG=""

if [ -n "${MAYA_CONFIG:-}" ]; then
  if [ ! -f "$MAYA_CONFIG" ]; then
    echo "[maya-gfx1030] ERROR: MAYA_CONFIG='$MAYA_CONFIG' does not exist." >&2
    exit 1
  fi
  CFG="$MAYA_CONFIG"
elif [ -n "${MAYA_PACK:-}" ] || [ -n "${MAYA_GGUF:-}" ] || [ -n "${MAYA_GGUF_DIR:-}" ]; then
  mkdir -p "$DATA_DIR"
  CFG="$("$PYBIN" - <<'PYEOF'
import json, os, pathlib, re, subprocess, sys

sys.path.insert(0, "/opt/maya/tools")
import maya_quants as Q

data_dir = pathlib.Path(os.environ.get("MAYA_DATA") or "/data")
pack_env = (os.environ.get("MAYA_PACK") or "").strip()
gguf_env = (os.environ.get("MAYA_GGUF") or "").strip()
gguf_dir_env = (os.environ.get("MAYA_GGUF_DIR") or "").strip()

def find_first_gguf() -> pathlib.Path | None:
    if gguf_env:
        p = pathlib.Path(gguf_env)
        if not p.is_file():
            sys.exit(f"[maya-gfx1030] ERROR: MAYA_GGUF='{p}' does not exist.")
        return p
    if gguf_dir_env:
        d = pathlib.Path(gguf_dir_env)
        if not d.is_dir():
            sys.exit(f"[maya-gfx1030] ERROR: MAYA_GGUF_DIR='{d}' is not a directory.")
        cands = sorted(
            f for f in d.glob("*.gguf")
            if f.is_file() and not f.name.startswith("mmproj-") and not f.name.endswith("-vocab.gguf")
        )
        first = [f for f in cands if "-00001-of-" in f.name]
        if first:
            return first[0]
        if cands:
            return cands[0]
        sys.exit(f"[maya-gfx1030] ERROR: no model .gguf files found in MAYA_GGUF_DIR='{d}'.")
    return None

def slug_for(path: pathlib.Path) -> str:
    stem = re.sub(r"-00001-of-\d+$", "", path.stem, flags=re.I)
    return re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-") or "glm5-flash"

gguf_first = find_first_gguf()
if pack_env:
    pack_dir = pathlib.Path(pack_env)
    tag = slug_for(pack_dir)
elif gguf_first is not None:
    tag = slug_for(gguf_first)
    pack_dir = data_dir / "pack" / tag
else:
    sys.exit("[maya-gfx1030] ERROR: set MAYA_CONFIG, MAYA_PACK, MAYA_GGUF, or MAYA_GGUF_DIR.")

need_pack = [pack_dir / "index.txt", pack_dir / "dense.bin", pack_dir / "tokenizer" / "vocab.json"]
if not all(p.is_file() for p in need_pack):
    if gguf_first is None:
        sys.exit(f"[maya-gfx1030] ERROR: incomplete pack at '{pack_dir}' and no MAYA_GGUF/MAYA_GGUF_DIR provided.")
    pack_dir.mkdir(parents=True, exist_ok=True)
    print(f"[maya-gfx1030] Indexing GGUF pack into {pack_dir} from {gguf_first} ...", file=sys.stderr, flush=True)
    subprocess.run(
        [
            "/opt/maya/.venv/bin/python",
            "/opt/maya/tools/iq_pack.py",
            "--gguf", str(gguf_first),
            "--out", str(pack_dir),
            "--compat-bf16",
        ],
        check=True,
        stdout=sys.stderr,
    )

gpus_raw = (os.environ.get("MAYA_GPUS") or "0,1").strip()
gpus = [int(x.strip()) for x in gpus_raw.split(",") if x.strip()]
ctx = int(os.environ.get("MAYA_CONTEXT") or "32768")
kv_stream = (os.environ.get("MAYA_KV_STREAMING") or "on").strip()
kv_fmt = (os.environ.get("MAYA_KV") or "").strip()
layer_split = (os.environ.get("MAYA_LAYER_SPLIT") or "auto").strip()
port = int(os.environ.get("MAYA_PORT") or os.environ.get("PORT") or "8080")
host = (os.environ.get("MAYA_HOST") or os.environ.get("HOST") or "0.0.0.0").strip()

args = [
    "--glm-pack", str(pack_dir),
    "--max-context", str(ctx),
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--kv-streaming", kv_stream,
]
if kv_fmt:
    args += ["--kv", kv_fmt]
profile = pathlib.Path("/opt/maya/data/expert-profile-glm.bin")
if profile.is_file():
    args += ["--expert-profile", str(profile)]

sampling, effort = Q.MODEL_DEFAULTS["glm"]
cfg = {
    "exe": "/opt/maya/build-hip/maya",
    "backend": "hip",
    "args": args,
    "cwd": "/opt/maya",
    "tokenizer": str(pack_dir / "tokenizer"),
    "model_name": "glm-5.3-flash",
    "gpu": gpus,
    "layer_split": layer_split,
    "sampling": dict(sampling),
    "reasoning_effort": effort,
    "lib_dirs": ["/opt/rocm/lib"],
    "host": host,
    "port": port,
    "log": str(data_dir / f"maya-{tag}-hip.log"),
    # Keep env empty so container -e MAYA_GLM_* overrides remain active without JSON shadowing
    "env": {},
}

# Optional vision encoder (MAYA_VISION=gpu | cpu)
vision_mode = (os.environ.get("MAYA_VISION") or "off").strip().lower()
if vision_mode in ("gpu", "cpu", "1", "on", "true"):
    vdir_cands = []
    if os.environ.get("MAYA_VISION_DIR"):
        vdir_cands.append(pathlib.Path(os.environ["MAYA_VISION_DIR"]))
    if gguf_first is not None:
        vdir_cands.extend([gguf_first.parent / "vision", gguf_first.parent.parent / "vision"])
    vdir_cands.extend([pathlib.Path("/models/vision"), data_dir / "vision"])
    mmproj = next((d / "mmproj-GLM-5.3-Flash-F16.gguf" for d in vdir_cands if (d / "mmproj-GLM-5.3-Flash-F16.gguf").is_file()), None)
    vocab = next((d / "GLM-5.3-Flash-vocab.gguf" for d in vdir_cands if (d / "GLM-5.3-Flash-vocab.gguf").is_file()), None)
    if mmproj and vocab:
        use_gpu = vision_mode != "cpu"
        vexe = "/opt/maya/build-vision/bin/maya-vision" if use_gpu else "/opt/maya/build-vision-cpu/bin/maya-vision"
        cfg["vision"] = {"exe": vexe, "mmproj": str(mmproj), "model": str(vocab), "gpu": use_gpu}
    else:
        print("[maya-gfx1030] WARNING: MAYA_VISION enabled but mmproj/vocab files not found; starting text-only.", file=sys.stderr, flush=True)

cfg_path = data_dir / f"maya-{tag}-hip.json"
cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
print(str(cfg_path))
PYEOF
)"
else
  # Auto-discover a single existing config under /data
  found_cfg=""
  found_n=0
  for p in "$DATA_DIR"/maya-*.json; do
    [ -f "$p" ] || continue
    case "$p" in
      *.shared-settings.json) continue ;;
    esac
    found_cfg="$p"
    found_n=$((found_n + 1))
  done
  if [ "$found_n" -eq 1 ]; then
    CFG="$found_cfg"
  fi
fi

if [ -z "$CFG" ]; then
  cat >&2 <<'EOF'
[maya-gfx1030] No model config or GGUF source specified.
  Examples:
    - Existing config: -e MAYA_CONFIG=/data/maya-maya-s-hip.json
    - Auto-pack GGUF:  -e MAYA_GGUF_DIR=/models/Maya-S-v2-IQ2_XXS -v /mnt/diskX/models:/models:ro -v /mnt/user/appdata/maya:/data
    - Direct server:   docker run ... <image> /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya --config /data/cfg.json --host 0.0.0.0 --port 8080
    - Run HIP tests:   docker run ... <image> /opt/maya/tests/hip_intrinsics
EOF
  exec "$PYBIN" "$SERVER" --help
fi

echo "[maya-gfx1030] Starting Project Maya (GLM-5.3-Flash, gfx1030) with config: $CFG"
echo "[maya-gfx1030] Defaults: RESERVE_MB=$MAYA_GLM_RESERVE_MB PIN_PIECE_MB=$MAYA_GLM_PIN_PIECE_MB RAM_RESIDENT=$MAYA_GLM_RAM_RESIDENT PREFILL_MIN_CHUNKS=$MAYA_GLM_PREFILL_MIN_CHUNKS PREFILL_SUB=$MAYA_GLM_PREFILL_SUB PREFILL_MB=$MAYA_GLM_PREFILL_MB PF_CE=$MAYA_GLM_PF_CE PROMPT_F16=$MAYA_GLM_PROMPT_F16 MLA_FDOT2=$MAYA_GLM_MLA_FDOT2 MV_RDNA=$MAYA_GLM_MV_RDNA"

set -- --engine maya --config "$CFG" --host "$HOST" --port "$PORT"
if [ -n "${MAYA_GPUS:-}" ]; then
  set -- "$@" --gpu "$MAYA_GPUS"
fi
if [ -n "${MAYA_API_KEY:-}" ]; then
  set -- "$@" --api-key "$MAYA_API_KEY"
fi
exec "$PYBIN" "$SERVER" "$@"

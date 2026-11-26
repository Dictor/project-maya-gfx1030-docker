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
export MAYA_GLM_EMB_RAM="${MAYA_GLM_EMB_RAM:-1}"
export MAYA_GLM_DENSE_MMQ="${MAYA_GLM_DENSE_MMQ:-1}"
export MAYA_GLM_BF16_F16="${MAYA_GLM_BF16_F16:-1}"

# Hy4-Flash (maya-hy4) RDNA2 (gfx1030) defaults (overridable via -e)
export HY4_PROMPT_F16="${HY4_PROMPT_F16:-1}"
export HY4_EG_RAM="${HY4_EG_RAM:-0}"
export HY4_LANE_RAM="${HY4_LANE_RAM:-1}"
export HY4_LOAD_SEQ="${HY4_LOAD_SEQ:-1}"
export HY4_EG_POPULATE="${HY4_EG_POPULATE:-0}"
export HY4_EXPERT_MMQ="${HY4_EXPERT_MMQ:-1}"
export HY4_PROMPT_PIPE="${HY4_PROMPT_PIPE:-1}"
export HY4_DENSE_MMQ="${HY4_DENSE_MMQ:-1}"
export HY4_ATTN_FDOT2="${HY4_ATTN_FDOT2:-1}"
export HY4_PIPE_BALANCE="${HY4_PIPE_BALANCE:-1}"
export HY4_STREAM_ASYNC="${HY4_STREAM_ASYNC:-1}"
export HY4_ATTN_64H="${HY4_ATTN_64H:-1}"

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
import importlib.util, json, os, pathlib, re, shutil, subprocess, sys

sys.path.insert(0, "/opt/maya/tools")
import maya_quants as Q

data_dir = pathlib.Path(os.environ.get("MAYA_DATA") or "/data")
pack_env = (os.environ.get("MAYA_PACK") or "").strip()
gguf_env = (os.environ.get("MAYA_GGUF") or "").strip()
gguf_dir_env = (os.environ.get("MAYA_GGUF_DIR") or "").strip()
engine_env = (os.environ.get("MAYA_ENGINE") or "").strip().lower()

_hy4_pack_mod = None
def _hy4_pack():
    global _hy4_pack_mod
    if _hy4_pack_mod is None:
        spec = importlib.util.spec_from_file_location("hy4_pack_gguf", "/opt/maya/tools/hy4/pack_gguf.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _hy4_pack_mod = mod
    return _hy4_pack_mod

def inspect_gguf(path: pathlib.Path):
    try:
        g = _hy4_pack().Gguf(path)
        return g.kv.get("general.architecture"), g.kv.get("general.type"), g.kv.get("split.index")
    except (Exception, SystemExit):
        return None, None, None

def slug_for(path: pathlib.Path) -> str:
    stem = re.sub(r"-00001-of-\d+$", "", path.stem if path.suffix else path.name, flags=re.I)
    return re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-") or "maya-model"

def detect_family() -> str:
    if engine_env in ("hy4", "glm"):
        return engine_env
    if pack_env:
        p = pathlib.Path(pack_env)
        if (p / "hy4.txt").is_file() or (p / "maya-hy4" / "hy4.txt").is_file():
            return "hy4"
        if (p / "index.txt").is_file():
            return "glm"
    if gguf_env:
        p = pathlib.Path(gguf_env)
        if p.is_file():
            arch, _, _ = inspect_gguf(p)
            if arch == "hy_v4_flash":
                return "hy4"
    if gguf_dir_env:
        d = pathlib.Path(gguf_dir_env)
        if d.is_dir():
            for f in sorted(d.glob("*.gguf")):
                if f.is_file():
                    arch, _, _ = inspect_gguf(f)
                    if arch == "hy_v4_flash":
                        return "hy4"
    return "glm"

def link_into(src: pathlib.Path, dst_dir: pathlib.Path, required_desc: str = "file"):
    link_path = dst_dir / src.name
    if link_path.is_symlink():
        if os.readlink(link_path) != str(src):
            try:
                link_path.unlink()
                os.symlink(str(src), str(link_path))
            except OSError as exc:
                sys.exit(f"[maya-gfx1030] ERROR: cannot update {required_desc} symlink {link_path} -> {src}: {exc}")
    elif not link_path.exists():
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(str(src), str(link_path))
        except OSError as exc:
            if not (pack_env and (dst_dir / src.name).is_file()):
                sys.exit(f"[maya-gfx1030] ERROR: cannot place {required_desc} {link_path} -> {src}: {exc}")

gpus_raw = (os.environ.get("MAYA_GPUS") or "0,1").strip()
gpus = [int(x.strip()) for x in gpus_raw.split(",") if x.strip()]
ctx = int(os.environ.get("MAYA_CONTEXT") or "32768")
kv_fmt = (os.environ.get("MAYA_KV") or "").strip()
port = int(os.environ.get("MAYA_PORT") or os.environ.get("PORT") or "8080")
host = (os.environ.get("MAYA_HOST") or os.environ.get("HOST") or "0.0.0.0").strip()

family = detect_family()

if family == "hy4":
    src_dir = None
    if gguf_dir_env:
        d = pathlib.Path(gguf_dir_env)
        if not d.is_dir():
            sys.exit(f"[maya-gfx1030] ERROR: MAYA_GGUF_DIR='{d}' is not a directory.")
        src_dir = d
    elif gguf_env:
        p = pathlib.Path(gguf_env)
        if not p.is_file():
            sys.exit(f"[maya-gfx1030] ERROR: MAYA_GGUF='{p}' does not exist.")
        src_dir = p.parent

    if pack_env:
        p = pathlib.Path(pack_env)
        if (p / "maya-hy4" / "hy4.txt").is_file() or (not (p / "hy4.txt").is_file() and src_dir is not None and p.name != "maya-hy4"):
            model_dir = p
            pack_dir = p / "maya-hy4"
        else:
            pack_dir = p
            model_dir = p.parent
        tag = slug_for(model_dir if pack_dir.name == "maya-hy4" and model_dir.name else pack_dir)
    elif src_dir is not None:
        trunk = None
        for f in sorted(src_dir.glob("*.gguf")):
            if not f.is_file():
                continue
            arch, kind, split_idx = inspect_gguf(f)
            if arch == "hy_v4_flash" and kind in (None, "model") and "engram" not in f.name.lower():
                if split_idx == 0 or "-00001-of-" in f.name or trunk is None:
                    trunk = f
                    if split_idx == 0 or "-00001-of-" in f.name:
                        break
        tag = slug_for(trunk if trunk is not None else src_dir)
        model_dir = data_dir / "pack" / tag
        pack_dir = model_dir / "maya-hy4"
    else:
        sys.exit("[maya-gfx1030] ERROR: set MAYA_CONFIG, MAYA_PACK, MAYA_GGUF, or MAYA_GGUF_DIR.")

    ancestor = model_dir
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    can_write_model_dir = os.access(ancestor, os.W_OK)
    if can_write_model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
    if src_dir is not None and (can_write_model_dir or not pack_env):
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and (f.suffix in (".gguf", ".dfly2") or "scales" in f.name.lower()):
                link_into(f, model_dir, "hy4 file")
        raw_gguf_files = sorted(src_dir.glob("*.gguf_file"))
        if raw_gguf_files:
            names = ", ".join(f.name for f in raw_gguf_files)
            print(f"[maya-gfx1030] NOTE: found '.gguf_file' file(s) ({names}); rename extension to '.gguf' for Hy4 indexing.",
                  file=sys.stderr, flush=True)
    for extra_var in ("MAYA_HY4_ENGRAM", "MAYA_HY4_MTP", "MAYA_HY4_MMPROJ", "MAYA_HY4_DRAFTER"):
        val = (os.environ.get(extra_var) or "").strip()
        if val:
            ep = pathlib.Path(val)
            if not ep.is_file():
                sys.exit(f"[maya-gfx1030] ERROR: {extra_var}='{ep}' does not exist.")
            if can_write_model_dir or not pack_env:
                link_into(ep, model_dir, extra_var)

    need_hy4 = [
        pack_dir / "hy4.txt",
        pack_dir / "tensors.txt",
        pack_dir / "tokenizer" / "vocab.json",
    ]
    need_reindex = not all(p.is_file() for p in need_hy4)
    reindex_reasons = []
    if not need_reindex and model_dir.is_dir():
        indexed_names = set()
        for line in (pack_dir / "tensors.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) == 3 and parts[0] == "file":
                indexed_names.add(pathlib.Path(parts[2]).name)
        for name in sorted(indexed_names):
            if not (model_dir / name).is_file():
                reindex_reasons.append(f"removed {name}")
        for f in sorted(model_dir.iterdir()):
            if f.suffix not in (".gguf", ".dfly2") or not f.is_file():
                continue
            if f.name not in indexed_names:
                arch, kind, _ = inspect_gguf(f)
                if arch in ("hy_v4_flash", "clip", "hy_v4_flash_dfly") or kind in ("mtp", "mmproj", "drafter"):
                    reindex_reasons.append(f"added {f.name}")
        scales_in_dir = sorted(p.name for p in model_dir.glob("*.engram-scales.bf16") if p.is_file())
        indexed_scales = ""
        for line in (pack_dir / "hy4.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("engram_scales "):
                indexed_scales = line.split(None, 1)[1].strip()
                break
        if len(scales_in_dir) == 1 and scales_in_dir[0] != indexed_scales:
            reindex_reasons.append(f"engram_scales {scales_in_dir[0]}")
        if reindex_reasons and os.access(pack_dir, os.W_OK):
            need_reindex = True
    if need_reindex:
        if src_dir is None and not any(model_dir.glob("*.gguf")):
            sys.exit(f"[maya-gfx1030] ERROR: incomplete hy4 pack at '{pack_dir}' and no MAYA_GGUF/MAYA_GGUF_DIR provided.")
        pack_dir.mkdir(parents=True, exist_ok=True)
        reason_str = f" ({', '.join(reindex_reasons)})" if reindex_reasons else ""
        print(f"[maya-gfx1030] Indexing Hy4-Flash pack into {pack_dir} from {model_dir}{reason_str} ...", file=sys.stderr, flush=True)
        subprocess.run(
            [
                "/opt/maya/.venv/bin/python",
                "/opt/maya/tools/hy4/pack_gguf.py",
                "--model", str(model_dir),
                "--out", str(pack_dir),
            ],
            check=True,
            stdout=sys.stderr,
        )

    hy4_meta = {}
    for line in (pack_dir / "hy4.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.strip().partition(" ")
        if k:
            hy4_meta[k] = v.strip()

    has_mtp = int(float(hy4_meta.get("n_mtp") or "0")) >= 1
    has_dfly2 = int(float(hy4_meta.get("dfly_layers") or "0")) >= 1
    has_vision = "vision.dim" in hy4_meta

    hy4_tp = (os.environ.get("MAYA_HY4_TP") or "auto").strip()
    hy4_reserve = (os.environ.get("MAYA_HY4_RESERVE_GB") or "0.7").strip()
    hy4_kv_stream = (os.environ.get("MAYA_HY4_KV_STREAMING") or "auto").strip().lower()
    if hy4_kv_stream not in ("auto", "on", "off"):
        sys.exit(f"[maya-gfx1030] ERROR: MAYA_HY4_KV_STREAMING='{hy4_kv_stream}' must be auto, on, or off.")
    hy4_draft = (os.environ.get("MAYA_HY4_DRAFT") or "auto").strip().lower()
    hy4_vision = (os.environ.get("MAYA_HY4_VISION") or "auto").strip().lower()

    args = [
        "--pack", str(pack_dir),
        "--max-context", str(ctx),
        "--tp", hy4_tp,
        "--reserve-gb", hy4_reserve,
        "--kv-streaming", hy4_kv_stream,
    ]
    if kv_fmt:
        if kv_fmt.lower() in ("auto", "f16", "q8"):
            args += ["--kv", kv_fmt.lower()]
        else:
            print(f"[maya-gfx1030] WARNING: ignoring GLM-only MAYA_KV='{kv_fmt}' for Hy4-Flash (supported: auto, f16, q8).",
                  file=sys.stderr, flush=True)

    if hy4_draft in ("mtp",) or (hy4_draft == "auto" and has_mtp):
        args.append("--mtp")
    elif hy4_draft in ("dfly2",) or (hy4_draft == "auto" and not has_mtp and has_dfly2 and len(gpus) <= 1):
        args.append("--dfly2")
    elif hy4_draft not in ("auto", "off", "none", "0", "false"):
        sys.exit(f"[maya-gfx1030] ERROR: MAYA_HY4_DRAFT='{hy4_draft}' must be auto, mtp, dfly2, or off.")

    if has_vision and hy4_vision in ("off", "0", "false", "no"):
        args += ["--vision", "off"]

    for env_k, flag in (
        ("MAYA_HY4_CHUNK", "--chunk"),
        ("MAYA_HY4_CPU_THREADS", "--cpu-threads"),
        ("MAYA_HY4_VRAM_EXPERTS", "--vram-experts"),
        ("MAYA_HY4_DRAFT_N", "--draft"),
        ("MAYA_HY4_DRAFT_FLOOR", "--draft-floor"),
        ("MAYA_HY4_DRAFT_SURE", "--draft-sure"),
        ("MAYA_HY4_DRAFT_STREAK", "--draft-streak"),
    ):
        val = (os.environ.get(env_k) or "").strip()
        if val:
            args += [flag, val]

    if "--mtp" in args:
        dm = os.environ.get("MAYA_HY4_DRAFT_MORE")
        dm_val = "2" if dm is None else dm.strip()
        if dm_val:
            args += ["--draft-more", dm_val]

    hy4_eg_ram_cfg = (os.environ.get("MAYA_HY4_ENGRAM_RAM") or "").strip().lower()
    if hy4_eg_ram_cfg:
        if hy4_eg_ram_cfg in ("0", "off", "false", "no"):
            args += ["--engram-ram", "off"]
        elif hy4_eg_ram_cfg in ("1", "on", "true", "yes", "ram"):
            args += ["--engram-ram", "on"]
        elif hy4_eg_ram_cfg in ("2", "lock", "mlock"):
            args += ["--engram-ram", "lock"]
        else:
            sys.exit(f"[maya-gfx1030] ERROR: MAYA_HY4_ENGRAM_RAM='{hy4_eg_ram_cfg}' must be off, on, or lock.")

    hy4_lane_ram_cfg = (os.environ.get("MAYA_HY4_LANE_RAM") or "").strip().lower()
    if hy4_lane_ram_cfg:
        if hy4_lane_ram_cfg in ("0", "off", "false", "no"):
            args += ["--lane-ram", "off"]
        elif hy4_lane_ram_cfg in ("1", "on", "true", "yes", "ram"):
            args += ["--lane-ram", "on"]
        elif hy4_lane_ram_cfg in ("2", "lock", "mlock"):
            args += ["--lane-ram", "lock"]
        else:
            sys.exit(f"[maya-gfx1030] ERROR: MAYA_HY4_LANE_RAM='{hy4_lane_ram_cfg}' must be off, on, or lock.")

    learned_profile = pack_dir / "expert-profile.bin"
    if learned_profile.is_file():
        args += ["--profile", str(learned_profile)]
    if os.access(pack_dir, os.W_OK):
        args += ["--profile-out", str(learned_profile)]

    sampling, effort = Q.MODEL_DEFAULTS["hy4"]
    cfg = {
        "exe": "/opt/maya/build-hip/maya-hy4",
        "backend": "hip",
        "args": args,
        "cwd": "/opt/maya",
        "tokenizer": str(pack_dir / "tokenizer"),
        "model_name": "hy4-flash",
        "gpu": gpus,
        "sampling": dict(sampling),
        "reasoning_effort": effort,
        "lib_dirs": ["/opt/rocm/lib"],
        "host": host,
        "port": port,
        "log": str(data_dir / f"maya-{tag}-hip.log"),
        "installer": {"family": "hy4"},
        # Keep env empty so container -e HY4_* overrides remain active without JSON shadowing
        "env": {},
    }
    cfg_path = data_dir / f"maya-{tag}-hip.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(str(cfg_path))
    sys.exit(0)

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

def resolve_shards(first: pathlib.Path) -> list[pathlib.Path]:
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if m:
        total = int(m.group(2))
        shards = [
            first.with_name(first.name[:m.start()] + f"-{i:05d}-of-{total:05d}.gguf")
            for i in range(1, total + 1)
        ]
    else:
        shards = [first]
    missing = [str(s) for s in shards if not s.is_file()]
    if missing:
        sys.exit(f"[maya-gfx1030] ERROR: missing model shard(s): {', '.join(missing)}")
    return shards

gguf_first = find_first_gguf()
if pack_env:
    pack_dir = pathlib.Path(pack_env)
    model_dir = pack_dir.parent
    tag = slug_for(pack_dir.parent if pack_dir.name == "pack" and pack_dir.parent.name else pack_dir)
elif gguf_first is not None:
    tag = slug_for(gguf_first)
    model_dir = data_dir / "pack" / tag
    pack_dir = model_dir / "pack"
else:
    sys.exit("[maya-gfx1030] ERROR: set MAYA_CONFIG, MAYA_PACK, MAYA_GGUF, or MAYA_GGUF_DIR.")

if gguf_first is not None:
    shards = resolve_shards(gguf_first)
    can_write_model_dir = os.access(model_dir if model_dir.exists() else model_dir.parent, os.W_OK)
    if can_write_model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        link_path = model_dir / shard.name
        if link_path.is_symlink():
            if os.readlink(link_path) != str(shard):
                try:
                    link_path.unlink()
                    os.symlink(str(shard), str(link_path))
                except OSError as exc:
                    sys.exit(f"[maya-gfx1030] ERROR: cannot update shard symlink {link_path} -> {shard}: {exc}")
        elif not link_path.exists():
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                os.symlink(str(shard), str(link_path))
            except OSError as exc:
                if not (pack_env and (model_dir / shard.name).is_file()):
                    sys.exit(f"[maya-gfx1030] ERROR: cannot place shard {link_path} -> {shard}: {exc}")
    for shard in shards:
        if not (model_dir / shard.name).is_file():
            sys.exit(f"[maya-gfx1030] ERROR: shard '{model_dir / shard.name}' missing beside pack '{pack_dir}'.")

need_pack = [
    pack_dir / "index.txt",
    pack_dir / "native_experts.txt",
    pack_dir / "dense.bin",
    pack_dir / "tokenizer" / "vocab.json",
]
if not all(p.is_file() for p in need_pack):
    if gguf_first is None:
        sys.exit(f"[maya-gfx1030] ERROR: incomplete pack at '{pack_dir}' and no MAYA_GGUF/MAYA_GGUF_DIR provided.")
    pack_dir.mkdir(parents=True, exist_ok=True)
    gguf_for_pack = model_dir / gguf_first.name
    gguf_arg = gguf_for_pack if gguf_for_pack.is_file() else gguf_first
    print(f"[maya-gfx1030] Indexing GGUF pack into {pack_dir} from {gguf_arg} ...", file=sys.stderr, flush=True)
    subprocess.run(
        [
            "/opt/maya/.venv/bin/python",
            "/opt/maya/tools/iq_pack.py",
            "--gguf", str(gguf_arg),
            "--out", str(pack_dir),
            "--compat-bf16",
        ],
        check=True,
        stdout=sys.stderr,
    )

tmpl_dst = pack_dir / "tokenizer" / "chat_template.jinja"
tmpl_src = pathlib.Path("/opt/maya/serve/glm5_chat_template.jinja")
if not tmpl_dst.is_file() and tmpl_src.is_file() and os.access(tmpl_dst.parent, os.W_OK):
    shutil.copyfile(tmpl_src, tmpl_dst)

kv_stream = (os.environ.get("MAYA_KV_STREAMING") or "on").strip()
layer_split = (os.environ.get("MAYA_LAYER_SPLIT") or "auto").strip()

args = [
    "--glm-pack", str(pack_dir),
    "--max-context", str(ctx),
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--kv-streaming", kv_stream,
]
if kv_fmt:
    args += ["--kv", kv_fmt]
learned_profile = pack_dir / "expert-profile.bin"
shipped_profile = pathlib.Path("/opt/maya/data/expert-profile-glm.bin")
if learned_profile.is_file():
    args += ["--expert-profile", str(learned_profile)]
elif shipped_profile.is_file():
    args += ["--expert-profile", str(shipped_profile)]
if os.access(pack_dir, os.W_OK):
    args += ["--expert-profile-save", str(learned_profile)]

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
    - Existing config:       -e MAYA_CONFIG=/data/maya-maya-s-hip.json
    - Auto-pack GLM GGUF:    -e MAYA_GGUF_DIR=/models/Maya-S-v2-IQ2_XXS -v /mnt/diskX/models:/models:ro -v /mnt/user/appdata/maya:/data
    - Auto-pack Hy4 GGUF:    -e MAYA_GGUF_DIR=/models/Hy4-Flash-Maya-S -v /mnt/diskX/models:/models:ro -v /mnt/user/appdata/maya:/data
    - Direct server:         docker run ... <image> /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya --config /data/cfg.json --host 0.0.0.0 --port 8080
    - Run HIP tests:         docker run ... <image> /opt/maya/tests/hip_intrinsics
EOF
  exec "$PYBIN" "$SERVER" --help
fi

if grep -q 'maya-hy4' "$CFG"; then
  echo "[maya-gfx1030] Starting Project Maya (Hy4-Flash, gfx1030) with config: $CFG"
  echo "[maya-gfx1030] Hy4 Defaults: HY4_PROMPT_F16=$HY4_PROMPT_F16 HY4_EG_RAM=$HY4_EG_RAM HY4_LANE_RAM=$HY4_LANE_RAM HY4_LOAD_SEQ=$HY4_LOAD_SEQ HY4_EG_POPULATE=$HY4_EG_POPULATE HY4_EXPERT_MMQ=$HY4_EXPERT_MMQ HY4_PROMPT_PIPE=$HY4_PROMPT_PIPE HY4_DENSE_MMQ=$HY4_DENSE_MMQ HY4_ATTN_FDOT2=$HY4_ATTN_FDOT2 HY4_PIPE_BALANCE=$HY4_PIPE_BALANCE HY4_STREAM_ASYNC=$HY4_STREAM_ASYNC HY4_ATTN_64H=$HY4_ATTN_64H"
else
  echo "[maya-gfx1030] Starting Project Maya (GLM-5.3-Flash, gfx1030) with config: $CFG"
  echo "[maya-gfx1030] Defaults: RESERVE_MB=$MAYA_GLM_RESERVE_MB PIN_PIECE_MB=$MAYA_GLM_PIN_PIECE_MB RAM_RESIDENT=$MAYA_GLM_RAM_RESIDENT EMB_RAM=$MAYA_GLM_EMB_RAM PREFILL_MIN_CHUNKS=$MAYA_GLM_PREFILL_MIN_CHUNKS PREFILL_SUB=$MAYA_GLM_PREFILL_SUB PREFILL_MB=$MAYA_GLM_PREFILL_MB PF_CE=$MAYA_GLM_PF_CE PROMPT_F16=$MAYA_GLM_PROMPT_F16 MLA_FDOT2=$MAYA_GLM_MLA_FDOT2 MV_RDNA=$MAYA_GLM_MV_RDNA DENSE_MMQ=$MAYA_GLM_DENSE_MMQ BF16_F16=$MAYA_GLM_BF16_F16"
fi

set -- --engine maya --config "$CFG" --host "$HOST" --port "$PORT"
if [ -n "${MAYA_GPUS:-}" ]; then
  set -- "$@" --gpu "$MAYA_GPUS"
fi
if [ -n "${MAYA_API_KEY:-}" ]; then
  set -- "$@" --api-key "$MAYA_API_KEY"
fi
if [ "${MAYA_DRY_RUN:-0}" = "1" ]; then
  echo "[maya-gfx1030] DRY_RUN config=$CFG"
  exit 0
fi
exec "$PYBIN" "$SERVER" "$@"

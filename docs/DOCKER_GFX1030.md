# Project Maya — AMD RDNA2 (`gfx1030`) Dual-W6800 Unraid Docker Guide

AMD RDNA2 (`gfx1030`: **Dual Radeon PRO W6800 32 GB** / RX 6800 / 6900 / PRO V620) 및 **AVX2-only 호스트 CPU**(예: Unraid Intel Xeon E5-2666 v3 Haswell, AVX-512 미지원, 110 GiB RAM, THP `[always]`)에서 **GLM-5.3-Flash (`Maya-S` 및 `Maya-M`)**를 실행하기 위한 경량 2-Stage Docker 빌드 및 Unraid 운영 가이드입니다.

---

## 1. 이미지 구성 (What Is Inside)

[`docker/Dockerfile.gfx1030`](../docker/Dockerfile.gfx1030)은 AMD TheRock ROCm 7 휠(`rocm[libraries,devel]==7.13.0a20260515`, `gfx103X-all`)을 사용해 Builder 스테이지에서 컴파일하고, Runner(`ubuntu:24.04`) 스테이지에는 런타임 `.so` 및 `rocBLAS` Tensile 테이블(`/opt/rocm/lib`)만 남깁니다.

- **GLM-5.3-Flash 엔진 (`build-hip/maya`)**: `gfx1030` 전용, `-DSTRATA_PORTABLE=ON`(AVX2 baseline), `-DSTRATA_PREFILL_MMQ=ON`, `-DSTRATA_NATIVE_EXPERTS=ON`. (빌드 시간과 이미지 크기 절감을 위해 `maya-ds4`, `maya-mm3`, `maya-hy4`는 제외)
- **비전 인코더 (`maya-vision`)**:
  - `/opt/maya/build-vision/bin/maya-vision`: HIP `gfx1030` + CPU portable AVX2 (듀얼 GPU 환경에서 `serve/server.py`가 `HIP_VISIBLE_DEVICES`를 GPU 0으로 고정해 `VLEND`와 일치)
  - `/opt/maya/build-vision-cpu/bin/maya-vision`: CPU-only portable AVX2 (ROCm 의존성 없음)
- **HIP 정합성/벤치마크 바이너리 (`/opt/maya/tests/`)**:
  `hip_intrinsics`, `hip_glm_handoff`, `hip_prefill_mmq_parity`, `hip_glm_prefill_attention`, `glm_mv_bench`, `hip_expert_kernel_bench`, `glm_attn_bench`, `glm_layer_parity`, `glm_model_test` (+ `data/glm5-synth*` 픽스처 포함)
- **빌드 타임 자동 검증**:
  - `CMakeCache.txt` 검증 (`STRATA_PORTABLE=ON`, `GGML_NATIVE=OFF`, `GGML_AVX2=ON`, AVX-512/VNNI/AMX 비활성화)
  - `objdump -d` 검증: `maya-vision`, `libggml-cpu.a`, `strata_core/engine/kernels/prefill/mmq` 및 비-CPUID-디스패치 호스트 오브젝트 전체 `zmm == 0` (Haswell `SIGILL` 원천 차단)
  - `clang-offload-bundler` + `llvm-objdump` 검증: `gfx1030` 디바이스 코드 오브젝트 및 `v_dot4_i32_i8` 명령어 생성 확인
  - `ldd` 검증: 모든 바이너리에서 `"not found"` 0건 확인
  - `build-hip/MAYA-BUILD.json` 및 `build-vision/MAYA-BUILD.json` 사전 스탬핑 (컨테이너 시작 시 재컴파일 방지)

---

## 2. Dual W6800 (64 GB VRAM) + 110 GiB RAM 메모리 계산 및 HDD 저장소 팁

| 모델 (GLM-5.3-Flash) | GGUF 다운로드 크기 | 전체 메모리 요구량 | Dual W6800 (64 GB) + 100 GB 가용 RAM |
| --- | ---: | ---: | --- |
| **Maya-S** (`Maya-S-v2-IQ2_XXS`) | 96.5 GB | ~104 GB | **100% VRAM + RAM 상주** (`MAYA_GLM_RAM_RESIDENT=1`) |
| **Maya-M** (`Maya-M-IQ2_S`) | 116.0 GB | ~124 GB | **100% VRAM + RAM 상주** (`MAYA_GLM_RAM_RESIDENT=1`) |

- **HDD에 GGUF 샤드를 보관해도 디코드 중 디스크 I/O 0**: `MAYA_GLM_RAM_RESIDENT=1`과 기본 워밍업(`MAYA_GLM_WARM` 기본 ON)이 시작 시 단 1회 모든 Expert를 VRAM(~53 GiB) + pinned RAM(Maya-S ~27 GiB, Maya-M ~55 GiB)에 적재합니다.
- **Unraid HDD 마운트 경로 팁 (중요)**: Unraid FUSE 레이어(`/mnt/user/...`) 대신 **실제 디스크 경로(`/mnt/diskX/...`)를 직접 마운트**하면 시작 시 1회성 HDD 콜드 로드(96.5~116 GB 순차 읽기)에서 FUSE 오버헤드를 제거해 로딩 시간을 크게 줄일 수 있습니다.
- **인덱스/팩(`/data`)은 NVMe 캐시 권장**: `/mnt/cache/appdata/maya:/data`에 마운트하면 `tools/iq_pack.py`가 생성하는 소형 인덱스/dense 팩(~2 GB)과 로그가 빠르게 동작합니다.
- **긴 콜드 로드 헬스체크**: 이미지의 `HEALTHCHECK`는 `--start-period=1200s`(20분)로 설정되어 있어 Maya-S/Maya-M HDD 워밍업 도중 조기 `unhealthy` 판정이 발생하지 않습니다. (`serve/server.py`도 `READY`까지 별도 시작 타임아웃 없이 대기합니다.)

---

## 3. Docker 이미지 빌드 (Build)

로컬 PC(Windows Docker Desktop / Linux)에서 BuildKit으로 빌드합니다.

```bash
docker build \
  -f docker/Dockerfile.gfx1030 \
  -t docker.io/kimdictor/project-maya-gfx1030:v1.1.69 \
  -t docker.io/kimdictor/project-maya-gfx1030:1.1.69 \
  -t docker.io/kimdictor/project-maya-gfx1030:latest \
  .
```

주요 `--build-arg` 옵션:

| Build Arg | 기본값 | 설명 |
| --- | --- | --- |
| `HIP_ARCHITECTURES` | `gfx1030` | 대상 AMDGPU 아키텍처 |
| `LLAMA_CPP_COMMIT` | `3cf03257f219afbe7334045ff7c6a06ac68c627d` | `setup.py`에 고정된 `llama.cpp` 커밋 |
| `THEROCK_INDEX` | `https://rocm.nightlies.amd.com/v2/gfx103X-all/` | AMD TheRock ROCm 7 인덱스 |
| `THEROCK_SPEC` | `rocm[libraries,devel]==7.13.0a20260515` | TheRock ROCm 패키지 스펙 |
| `JOBS` | `(nproc / 2)` | 병렬 컴파일 작업 수 |

---

## 4. Unraid 실행 방법 (Run on Unraid)

### 방법 A: HDD GGUF 폴더에서 자동 팩 생성 + 즉시 서빙 (권장)

최초 1회 실행 시 `tools/iq_pack.py --compat-bf16`로 `/data/pack/<model>`을 자동 생성하고, 이후 재시작에서는 기존 팩을 바로 사용합니다. `Maya-S`와 `Maya-M`은 별도 팩/설정 이름으로 저장되므로 `MAYA_GGUF_DIR`만 바꿔 손쉽게 전환할 수 있습니다.

```bash
docker run -d --name maya-gfx1030 --restart unless-stopped \
  --device=/dev/dri \
  --device=/dev/kfd \
  --group-add video \
  --group-add render \
  --ulimit memlock=-1:-1 \
  --ipc=host \
  --cap-add=IPC_LOCK \
  -p 8080:8080 \
  -v /mnt/disk1/gguf:/models:ro \
  -v /mnt/cache/appdata/maya:/data:rw \
  -e MAYA_GGUF_DIR=/models/Maya-S-v2-IQ2_XXS \
  -e MAYA_GPUS=0,1 \
  -e MAYA_CONTEXT=32768 \
  kimdictor/project-maya-gfx1030:v1.1.69
```

> **Maya-M으로 전환할 때**: `-e MAYA_GGUF_DIR=/models/Maya-M`으로 변경하고 컨테이너를 재시작하면 됩니다.
> **비전 활성화(선택)**: `mmproj-GLM-5.3-Flash-F16.gguf`와 `GLM-5.3-Flash-vocab.gguf`가 `/models/vision` 또는 `<MAYA_GGUF_DIR>/vision`에 있을 때 `-e MAYA_VISION=gpu` (GPU 0 on-demand lending) 또는 `-e MAYA_VISION=cpu`를 추가합니다.

### 방법 B: 기존 설정 JSON 직접 실행 (`docs/DOCKER.md` 호환)

```bash
# Env 방식 (JSON을 수정하지 않고 그대로 사용)
docker run --rm \
  --device=/dev/dri --device=/dev/kfd \
  --group-add video --group-add render \
  --ulimit memlock=-1:-1 --ipc=host --cap-add=IPC_LOCK \
  -p 8080:8080 \
  -v /mnt/disk1/gguf:/models:ro \
  -v /mnt/cache/appdata/maya:/data:rw \
  -e MAYA_CONFIG=/data/maya-maya-s-hip.json \
  kimdictor/project-maya-gfx1030:v1.1.69

# 또는 serve/server.py 직접 호출 (passthrough)
docker run --rm \
  --device=/dev/dri --device=/dev/kfd \
  --group-add video --group-add render \
  --ulimit memlock=-1:-1 --ipc=host --cap-add=IPC_LOCK \
  -p 8080:8080 \
  -v /mnt/disk1/gguf:/models:ro \
  -v /mnt/cache/appdata/maya:/data:rw \
  kimdictor/project-maya-gfx1030:v1.1.69 \
  /opt/maya/.venv/bin/python /opt/maya/serve/server.py \
    --engine maya --config /data/maya-maya-s-hip.json --host 0.0.0.0 --port 8080
```

> 참고: 직접 작성한 `MAYA_CONFIG` JSON의 `"env"` 블록에 명시된 키는 컨테이너 환경변수보다 우선합니다 (`serve/server.py:child_env`).

---

## 5. 기본 환경변수 및 토글 (Overridable Defaults)

`docker/entrypoint-gfx1030.sh`는 아래 Unraid Dual-W6800 기본값을 설정하며, 모든 항목은 `-e KEY=VALUE`로 덮어쓸 수 있습니다.

| 환경변수 | 기본값 | 용도 / 비고 |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | RDNA2 `gfx1030` ROCm 호환성 |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | ROCm 스크래치 재할당 스톨 방지 |
| `GPU_PINNED_MIN_XFER_SIZE` | `1048576` | HIP pinned DMA 임계값 |
| `MAYA_UPDATE_CHECK` | `0` | 컨테이너 내 git 자가 업데이트 비활성화 |
| `MAYA_GLM_RESERVE_MB` | `700` | 헤드리스 Unraid VRAM 여유분 (기본 3072 -> 700으로 카드 2장 합계 +4.6 GiB 확보) |
| `MAYA_GLM_PIN_PIECE_MB` | `1024` | Unraid THP `[always]` 환경에서 대형 단일 `hipHostMalloc` Direct Compaction 프리징 방지 |
| `MAYA_GLM_RAM_RESIDENT` | `1` | 오프-VRAM Expert 전체 RAM 상주 (HDD 디코드 읽기 제거) |
| `MAYA_GLM_PREFILL_MIN_CHUNKS` | `2,512` | 단일 청크 프롬프트를 2장 W6800에 파이프라이닝 |
| `MAYA_GLM_PREFILL_SUB` | `1024` | 프롬프트 서브배치 크기 |
| `MAYA_GLM_PREFILL_MB` | `4096` | 프롬프트 임시 버퍼 상한 (풀 tail 대여) |
| `MAYA_GLM_PF_CE` | `0` | RDNA2 고우선순위 크로스-스트림 데드락(`#621`) 방지 |
| `MAYA_GLM_PROMPT_F16` | `1` | `gfx1030` FP16-output 프롬프트 GEMM 가속 (`0`으로 즉시 롤백; `MAYA_GLM_PROMPT_F16_CHECK=1`로 overflow 진단) |
| `MAYA_GLM_MLA_FDOT2` | `1` | `gfx1030` `v_dot2_f32_f16` MLA 프롬프트 어텐션 가속 (`0`으로 즉시 롤백) |
| `MAYA_GLM_MV_RDNA` | `1` | `gfx1030` 최적화된 dense `Q6_K`/BF16 GEMV (`0`으로 즉시 롤백) |
| `MAYA_HIP_EXPERTS_LEGACY` | (미설정) | `1` 설정 시 최적화된 `gfx1030` wave32 expert 커널을 레거시 경로로 롤백 |

---

## 6. Unraid에서 HIP 정합성/성능 테스트 및 CPU Lane 캘리브레이션 실행

### HIP 패리티 및 벤치마크 테스트 (`/opt/maya/tests/`)

```bash
# 핵심 HIP 정합성 테스트 (단일 GPU 0 기준)
docker run --rm --device=/dev/dri --device=/dev/kfd \
  --group-add video --group-add render --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=0 \
  kimdictor/project-maya-gfx1030:v1.1.69 bash -c '
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

# W6800 (30 WGP / 128 MB Infinity Cache) 커널 스윕 벤치마크
docker run --rm --device=/dev/dri --device=/dev/kfd \
  --group-add video --group-add render -e HIP_VISIBLE_DEVICES=0 \
  kimdictor/project-maya-gfx1030:v1.1.69 \
  /opt/maya/tests/hip_expert_kernel_bench --parity-only --sweep
```

### HIP CPU Lane / PCIe Share 캘리브레이션 (`--calibrate`)

`tools/calibrate_glm.py`는 이제 HIP 백엔드에서도 사용 가능합니다. 서버가 중지된 상태에서 실행하여 Haswell Xeon + PCIe 3.0 듀얼 W6800의 최적 `MAYA_GLM_PCIE_SHARE` / `MAYA_GLM_CPU_LANE`을 측정할 수 있습니다.

```bash
docker run --rm --device=/dev/dri --device=/dev/kfd \
  --group-add video --group-add render --ulimit memlock=-1:-1 --ipc=host --cap-add=IPC_LOCK \
  -v /mnt/disk1/gguf:/models:ro -v /mnt/cache/appdata/maya:/data:rw \
  kimdictor/project-maya-gfx1030:v1.1.69 \
  /opt/maya/.venv/bin/python /opt/maya/tools/calibrate_glm.py /data/maya-glm-5.3-flash-maya-s-v2-iq2_xxs-hip.json
```

---

## 7. 로컬 리눅스 네이티브 HIP 빌드에서 Portable AVX2 활성화

Docker 외부(네이티브 Linux)에서 `./maya.sh --backend hip`을 빌드할 때도 AVX-512 없는 CPU(Haswell 등)로 바이너리를 옮겨야 한다면 아래 환경변수로 `-DSTRATA_PORTABLE=ON`을 켤 수 있습니다(`maya.py` 엔진 및 `maya-vision` 모두 지원):

```bash
MAYA_PORTABLE=1 ./maya.sh --backend hip --setup ...
```

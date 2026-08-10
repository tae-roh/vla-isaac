"""vla-train venv 스모크 테스트 — Phase 3 (SFT) 환경 검증.

확인 항목:
  1. torch + CUDA + bf16 지원
  2. transformers / peft / flash-attn / dlimp import
  3. openvla-oft 의 prismatic 패키지가 import 되고, LIBERO 상수셋이 우리 스펙과 일치
  4. (--full) RLDS 데이터셋이 열리고 배치 하나를 뽑을 수 있는지
  5. (--full) OpenVLA 7B 로드 + 1 스텝 forward

실행:
    python env/smoke/check_vla_train.py
    python env/smoke/check_vla_train.py --full --data-root datasets/rlds --dataset-name vla_pick

3번이 이 스크립트의 핵심이다. openvla-oft 는 실행 커맨드에 "libero" 가 들어
있는지로 상수셋을 고르는데, 그 자동 감지가 빗나가면 ACTION_DIM 이 14(ALOHA)로
잡히면서 SFT 가 조용히 잘못된 차원으로 학습된다. 여기서 잡지 못하면
Phase 4 에서 원인불명 성능 붕괴로 나타난다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import check, finish, load_vla_spec, warn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="모델 로드 + forward 까지 검증")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/rlds"))
    parser.add_argument("--dataset-name", default="vla_pick")
    parser.add_argument("--vla-path", default="openvla/openvla-7b")
    args = parser.parse_args()

    print("=== vla-train 환경 스모크 테스트 ===\n")

    spec = None

    def _spec():
        nonlocal spec
        spec = load_vla_spec()
        spec.assert_consistent()
        return "configs/vla_spec.py 정합성 OK"

    check("공유 스펙 로드", _spec)

    def _torch():
        import torch

        if not torch.cuda.is_available():
            raise AssertionError("CUDA 사용 불가.")
        if not torch.cuda.is_bf16_supported():
            raise AssertionError("bf16 미지원 GPU — SFT 레시피가 bf16 전제다.")
        gpu = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"torch {torch.__version__} / {gpu} / {total:.0f}GB"

    check("torch + CUDA + bf16", _torch)

    def _libs():
        import peft
        import transformers

        import flash_attn  # noqa: F401

        return f"transformers {transformers.__version__}, peft {peft.__version__}"

    check("transformers / peft / flash-attn", _libs)

    def _rlds_libs():
        import dlimp  # noqa: F401
        import tensorflow as tf
        import tensorflow_datasets  # noqa: F401

        # TF 가 GPU 를 잡으면 학습 배치가 줄어든다. CPU 판을 썼는지 확인.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            warn(
                f"TensorFlow 가 GPU {len(gpus)}개를 잡고 있다. tensorflow-cpu 를 설치했는지 "
                "확인할 것 — VRAM 을 선점해 SFT 배치 크기를 깎는다."
            )
        return f"tensorflow {tf.__version__} (GPU {len(gpus)}개)"

    check("RLDS 스택 (tensorflow / dlimp)", _rlds_libs)

    # -- 핵심: 상수셋 일치 검증 ----------------------------------------------
    def _constants():
        from prismatic.vla.constants import (
            ACTION_DIM,
            NUM_ACTIONS_CHUNK,
            PROPRIO_DIM,
        )

        mismatches = []
        if ACTION_DIM != spec.ACTION_DIM:
            mismatches.append(f"ACTION_DIM {ACTION_DIM} != {spec.ACTION_DIM}")
        if PROPRIO_DIM != spec.PROPRIO_DIM:
            mismatches.append(f"PROPRIO_DIM {PROPRIO_DIM} != {spec.PROPRIO_DIM}")
        if NUM_ACTIONS_CHUNK != spec.NUM_ACTIONS_CHUNK:
            mismatches.append(
                f"NUM_ACTIONS_CHUNK {NUM_ACTIONS_CHUNK} != {spec.NUM_ACTIONS_CHUNK}"
            )
        if mismatches:
            raise AssertionError(
                "openvla-oft 상수셋이 우리 스펙과 어긋났다: "
                + "; ".join(mismatches)
                + "\n  원인: prismatic/vla/constants.py 의 detect_robot_platform() 은 "
                "실행 커맨드에 'libero' 가 들어 있어야 LIBERO 상수셋(7/8/8)을 고른다.\n"
                "  해결: 학습 스크립트 파일명이나 인자에 'libero' 를 포함시키거나, "
                "constants.py 를 직접 편집해 LIBERO 셋을 강제할 것."
            )
        return f"ACTION_DIM={ACTION_DIM}, PROPRIO_DIM={PROPRIO_DIM}, CHUNK={NUM_ACTIONS_CHUNK}"

    check("openvla-oft 상수셋 ↔ 스펙 일치", _constants)

    if not args.full:
        print("\n--full 없이 실행됨 → 데이터셋/모델 로드 검증은 건너뛴다.")
        return finish("vla-train")

    # -- 데이터셋 ------------------------------------------------------------
    def _dataset():
        import tensorflow_datasets as tfds

        builder = tfds.builder(args.dataset_name, data_dir=str(args.data_root))
        info = builder.info
        n = info.splits["train"].num_examples
        if n == 0:
            raise AssertionError("train split 이 비어 있다.")
        return f"{args.dataset_name}: train {n} 에피소드"

    check("RLDS 데이터셋 열기", _dataset)

    def _norm_stats():
        import json

        path = args.data_root / spec.NORM_STATS_FILENAME
        if not path.exists():
            raise AssertionError(
                f"{path} 없음. scripts/convert_hdf5_to_rlds.py 가 생성해야 한다. "
                "이 파일이 없으면 RFT 에서 액션 디토크나이즈 범위가 어긋난다."
            )
        stats = json.loads(path.read_text())
        q01 = stats["action"]["q01"]
        if len(q01) != spec.ACTION_DIM:
            raise AssertionError(
                f"norm_stats 의 액션 차원 {len(q01)} != 스펙 {spec.ACTION_DIM}"
            )
        return f"action q01/q99 {spec.ACTION_DIM}차원 확인"

    check("액션 정규화 통계", _norm_stats)

    # -- 모델 ----------------------------------------------------------------
    def _model():
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            args.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda:0")
        model.eval()

        from PIL import Image
        import numpy as np

        dummy = Image.fromarray(
            np.zeros((spec.IMAGE_HEIGHT, spec.IMAGE_WIDTH, 3), dtype=np.uint8)
        )
        inputs = processor(spec.build_prompt(), dummy).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs)
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        return f"{n_params:.1f}B params, logits {tuple(out.logits.shape)}"

    check("OpenVLA 로드 + 1 forward", _model)

    return finish("vla-train")


if __name__ == "__main__":
    sys.exit(main())

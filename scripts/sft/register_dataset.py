"""openvla-oft 를 우리 태스크에 맞게 설정한다 (데이터셋 등록 + 상수셋 고정).

openvla-oft 는 임의의 RLDS 데이터셋을 그냥 읽지 못한다. 두 곳에 항목이 있어야 한다:
  1. OXE_DATASET_CONFIGS          — 관측 키 매핑 (어느 카메라가 primary 인지 등)
  2. OXE_STANDARDIZATION_TRANSFORMS — 에피소드를 표준 형식으로 바꾸는 함수

이 스크립트는 두 항목을 **멱등하게** 삽입한다. 이미 있으면 아무것도 하지 않는다.
직접 손으로 고쳐도 되지만, Day 2 에 H100 인스턴스를 새로 띄운 상태에서
파일 두 개를 찾아 편집하는 것보다 한 번 실행하는 편이 빠르고 덜 틀린다.

사용:
    python scripts/sft/register_dataset.py --oft-dir ~/openvla-oft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_NAME = "vla_pick"
_PIN_MARKER = "# --- vla-isaac: 상수셋 고정 ---"

_PIN_BLOCK = f'''

{_PIN_MARKER}
# scripts/sft/register_dataset.py 가 추가한 블록. 손으로 지우지 말 것.
#
# 왜 필요한가:
#   openvla-oft 의 detect_robot_platform() 은 sys.argv 를 전부 이어붙인 문자열에
#   "libero"/"aloha"/"bridge" 가 있는지로 상수셋을 고른다. 즉 **실행 커맨드에
#   우연히 섞인 문자열이 학습 차원을 바꾼다.** 예를 들어 저장소를 ~/bridge/ 아래
#   두기만 해도 BRIDGE 상수셋이 잡힌다.
#
#   기본 폴백이 LIBERO 라 대부분은 우연히 맞지만, 그건 "안전"이 아니라 "운"이다.
#   틀렸을 때의 증상은 조용하다 — SFT 가 잘못된 액션 차원으로 하룻밤을 돌고,
#   문제는 Phase 4 에서 원인불명 성능 붕괴로만 드러난다.
#
#   모듈 끝에서 다시 대입해 감지 결과와 무관하게 LIBERO 값으로 확정한다.
#   우리 Isaac Lab 태스크가 이 스펙에 맞춰 설계되어 있다 (configs/vla_spec.py).
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
ROBOT_PLATFORM = "LIBERO"
print(
    f"[vla-isaac] 상수셋 고정: NUM_ACTIONS_CHUNK={{NUM_ACTIONS_CHUNK}}, "
    f"ACTION_DIM={{ACTION_DIM}}, PROPRIO_DIM={{PROPRIO_DIM}}"
)
'''

_TRANSFORM_FN = f'''

def {DATASET_NAME}_dataset_transform(trajectory: dict) -> dict:
    """vla_pick (Isaac Lab Mimic 증강) — 이미 표준 레이아웃이라 손댈 게 거의 없다.

    액션은 [dx,dy,dz,drx,dry,drz,gripper] 7차원으로 LIBERO 와 동일하고,
    proprio 는 8차원(eef 위치 3 + axis-angle 3 + 그리퍼 2)으로 역시 동일하다.
    이 일치는 우연이 아니라 configs/vla_spec.py 에서 의도적으로 맞춘 것이다.
    """
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory
'''

_CONFIG_ENTRY = f'''    "{DATASET_NAME}": {{
        "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": None}},
        "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }},
'''


def patch_transforms(oft_dir: Path) -> bool:
    path = oft_dir / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "transforms.py"
    if not path.exists():
        raise FileNotFoundError(f"transforms.py 를 찾을 수 없다: {path}")

    text = path.read_text(encoding="utf-8")
    if f"{DATASET_NAME}_dataset_transform" in text:
        print(f"  transforms.py — 이미 등록됨, 건너뜀")
        return False

    marker = "OXE_STANDARDIZATION_TRANSFORMS = {"
    if marker not in text:
        raise RuntimeError(
            f"{path} 에서 OXE_STANDARDIZATION_TRANSFORMS 를 찾지 못했다. "
            "openvla-oft 구조가 바뀌었을 수 있다 — 수동으로 등록할 것."
        )

    # 변환 함수는 레지스트리 dict **바로 앞에** 넣는다. 파일 끝에 붙이면
    # dict 리터럴이 정의되는 시점에 함수가 아직 없어서 import 하는 순간
    # NameError 로 죽는다 (dict 가 함수 객체를 값으로 담기 때문).
    entry = f'    "{DATASET_NAME}": {DATASET_NAME}_dataset_transform,\n'
    text = text.replace(
        marker,
        _TRANSFORM_FN.strip("\n") + "\n\n\n" + marker + "\n" + entry,
        1,
    )

    path.write_text(text, encoding="utf-8")
    print(f"  transforms.py — 등록 완료")
    return True


def patch_configs(oft_dir: Path) -> bool:
    path = oft_dir / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "configs.py"
    if not path.exists():
        raise FileNotFoundError(f"configs.py 를 찾을 수 없다: {path}")

    text = path.read_text(encoding="utf-8")
    if f'"{DATASET_NAME}"' in text:
        print(f"  configs.py — 이미 등록됨, 건너뜀")
        return False

    marker = "OXE_DATASET_CONFIGS = {"
    if marker not in text:
        raise RuntimeError(
            f"{path} 에서 OXE_DATASET_CONFIGS 를 찾지 못했다. 수동으로 등록할 것."
        )
    text = text.replace(marker, marker + "\n" + _CONFIG_ENTRY, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  configs.py — 등록 완료")
    return True


_ACTION_HEAD_MARKER = "# --- vla-isaac: 이산 토큰 경로 가드 ---"

_ACTION_HEAD_ANCHOR = """    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:"""

_ACTION_HEAD_FIX = f'''{_ACTION_HEAD_MARKER}
    # 상류 finetune.py 는 use_l1_regression / use_diffusion 중 하나는 켜져 있다고
    # 가정하고 그 분기에서만 action_head 를 바인딩하는데, 학습 루프(1040줄 근처)는
    # 플래그와 무관하게 action_head=action_head 로 넘긴다. 우리는 GRPO 호환을 위해
    # 둘 다 끄므로(README §2 이산 액션 토큰 + CE) 첫 배치에서
    #   UnboundLocalError: local variable 'action_head' referenced before assignment
    # 로 죽는다. None 으로 미리 묶어 두면 끝이다 — run_forward_pass 는 두 플래그가
    # False 면 action_head 를 아예 건드리지 않고, 체크포인트 저장부도
    # `action_head is not None` 을 명시적으로 확인한다.
    action_head = None

{_ACTION_HEAD_ANCHOR}'''


def patch_finetune(oft_dir: Path) -> bool:
    """이산 토큰 경로에서 finetune.py 가 UnboundLocalError 로 죽는 것을 막는다."""
    path = oft_dir / "vla-scripts" / "finetune.py"
    if not path.exists():
        raise FileNotFoundError(f"finetune.py 를 찾을 수 없다: {path}")

    text = path.read_text(encoding="utf-8")
    if _ACTION_HEAD_MARKER in text:
        print("  finetune.py — 이미 패치됨, 건너뜀")
        return False

    if _ACTION_HEAD_ANCHOR not in text:
        raise RuntimeError(
            f"{path} 에서 action_head 초기화 지점을 찾지 못했다. 상류가 바뀌었을 수 "
            "있다 — cfg.use_l1_regression 분기 앞에 `action_head = None` 을 직접 넣을 것."
        )

    path.write_text(text.replace(_ACTION_HEAD_ANCHOR, _ACTION_HEAD_FIX, 1), encoding="utf-8")
    print("  finetune.py — 이산 토큰 경로 가드(action_head = None) 삽입 완료")
    return True


def pin_constants(oft_dir: Path) -> bool:
    """상수셋을 LIBERO 로 확정한다 (argv 문자열 감지에 의존하지 않도록).

    모듈 끝에 재대입 블록을 덧붙이는 방식이라, detect_robot_platform() 이
    무엇을 골랐든 최종 값은 우리가 정한 것이 된다. 상류 코드 구조가 바뀌어도
    깨지지 않는 방식이다 (내부를 고치지 않고 끝에 붙이기만 하므로).
    """
    path = oft_dir / "prismatic" / "vla" / "constants.py"
    if not path.exists():
        raise FileNotFoundError(f"constants.py 를 찾을 수 없다: {path}")

    text = path.read_text(encoding="utf-8")
    if _PIN_MARKER in text:
        print("  constants.py — 이미 고정됨, 건너뜀")
        return False

    if "NormalizationType" not in text:
        raise RuntimeError(
            f"{path} 에서 NormalizationType 을 찾지 못했다. openvla-oft 구조가 "
            "바뀌었을 수 있다 — 상수셋을 수동으로 확인할 것."
        )

    path.write_text(text + _PIN_BLOCK, encoding="utf-8")
    print("  constants.py — LIBERO 상수셋(7/8/8)으로 고정 완료")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oft-dir", type=Path, default=Path.home() / "openvla-oft")
    parser.add_argument(
        "--skip-pin",
        action="store_true",
        help="상수셋 고정을 건너뛴다 (권장하지 않음 — argv 문자열 감지에 맡기게 된다)",
    )
    args = parser.parse_args()

    if not args.oft_dir.exists():
        print(f"openvla-oft 를 찾을 수 없다: {args.oft_dir}", file=sys.stderr)
        return 2

    print(f"openvla-oft 설정: {args.oft_dir}")
    changed = (
        patch_configs(args.oft_dir)
        | patch_transforms(args.oft_dir)
        | patch_finetune(args.oft_dir)
    )
    if not args.skip_pin:
        changed |= pin_constants(args.oft_dir)

    print()
    print("완료." if changed else "이미 설정되어 있었다 (변경 없음).")
    print("\n확인:")
    print(f"  python -c \"from prismatic.vla.datasets.rlds.oxe.configs import "
          f"OXE_DATASET_CONFIGS; print('{DATASET_NAME}' in OXE_DATASET_CONFIGS)\"")
    print("  python -c \"from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM, "
          "NUM_ACTIONS_CHUNK; print(ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK)\"")
    print("    → 7 8 8 이어야 정상")
    return 0


if __name__ == "__main__":
    sys.exit(main())

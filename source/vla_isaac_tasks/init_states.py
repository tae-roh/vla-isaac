# =============================================================================
# vla_isaac_tasks/init_states.py
#
# 초기 상태 뱅크 — 개정 §3. **RL 코드보다 먼저 있어야 하는 인프라다.**
#
# 왜 리셋 때 샘플링하지 않는가:
#   GRPO 는 동일한 s₀ 에서 G개 궤적을 뽑는 것을 전제한다. 리스폰마다 배치가
#   달라지면
#       Â_i = (R_i − mean(R)) / std(R)
#   가 "정책이 잘했는가"가 아니라 "이번 리스폰이 쉬웠는가"를 재게 되어 보상이
#   노이즈가 된다. 그래서 초기 상태를 **미리 만들어 파일로 굳혀 두고** 인덱스로
#   꺼내 쓴다 (LIBERO 의 set_init_state 와 같은 발상).
#
#   나중에 붙이면 그 이전 실험 결과가 전부 재현 불가가 된다.
#
# 이 모듈은 **numpy 만** 쓴다. Isaac Sim 없이도 뱅크를 만들고 검사할 수 있어야
# 하기 때문이다 (scripts/make_init_states.py 가 어느 venv 에서든 돈다).
#
# 뱅크 한 줄:  [x, y, yaw] × NUM_BLOCKS + [target_block_idx]
# =============================================================================

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:
    from .spec import SPEC
except ImportError:
    # 패키지 밖에서 파일로 직접 로드된 경우 (scripts/make_init_states.py).
    # 패키지 __init__ 은 gymnasium 을 import 하는데, 뱅크를 만드는 데는 필요 없다.
    import importlib.util as _ilu

    _s = _ilu.spec_from_file_location(
        "vla_spec", Path(__file__).resolve().parents[2] / "configs" / "vla_spec.py"
    )
    SPEC = _ilu.module_from_spec(_s)
    sys.modules.setdefault("vla_spec", SPEC)
    _s.loader.exec_module(SPEC)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def bank_path(name: str) -> Path:
    """뱅크 이름 → 파일 경로. 이름은 split 이름을 쓴다 ('train', 'eval_base', ...)."""
    return _REPO_ROOT / SPEC.INIT_STATE_DIR / f"{name}.npz"


# -----------------------------------------------------------------------------
# 샘플링
# -----------------------------------------------------------------------------
def sample_bank(size: int, seed: int) -> np.ndarray:
    """초기 상태 뱅크를 만든다. Shape (size, SPEC.INIT_STATE_DIM).

    제약 (개정 §2-6):
      - 블록이 서로 겹치지 않는다. 최소 간격을 블록 대각선보다 크게 잡아
        yaw 와 무관하게 성립시킨다 — 회전까지 고려한 정확한 겹침 판정을
        구현할 이유가 없다.
      - 전부 박스 안쪽, 벽에서 여유를 두고 스폰한다. 벽에 낀 채로 시작하면
        실패 원인이 정책인지 초기 배치인지 구분되지 않는다.
      - 타깃 블록은 반드시 보이도록 → 겹침이 없으므로 자동으로 만족된다.
        "묻힌 타깃"은 별도 hard split 으로 만든다 (지금은 만들지 않는다).
    """
    rng = np.random.default_rng(seed)
    bx, by = SPEC.BOX_CENTER
    half_x = 0.5 * SPEC.BOX_INNER_SIZE[0] - SPEC.BLOCK_SPAWN_WALL_MARGIN
    half_y = 0.5 * SPEC.BOX_INNER_SIZE[1] - SPEC.BLOCK_SPAWN_WALL_MARGIN
    n = SPEC.NUM_BLOCKS

    # 타깃 블록은 **균등 배분**한다. 그냥 뽑으면 64개짜리 홀드아웃에서 어떤
    # 색은 12번, 어떤 색은 25번 나온다 — 성공률이 뽑기 운에 좌우되고,
    # Language split 의 숫자가 그만큼 흔들린다.
    # (트레이 단일화로 조합이 9종 → 3종이 되어 배분이 더 촘촘해졌다.)
    targets = np.arange(size) % n
    rng.shuffle(targets)

    rows = np.zeros((size, SPEC.INIT_STATE_DIM), dtype=np.float64)
    for k in range(size):
        xy = _sample_non_overlapping(rng, n, bx, by, half_x, half_y)
        yaw = rng.uniform(-math.pi, math.pi, size=n)
        rows[k, : 3 * n] = np.stack([xy[:, 0], xy[:, 1], yaw], axis=1).reshape(-1)
        rows[k, 3 * n] = targets[k]
    return rows


def _sample_non_overlapping(rng, n, cx, cy, half_x, half_y) -> np.ndarray:
    """최소 간격을 지키는 xy n개. 실패하면 통째로 다시 뽑는다."""
    min_sep = SPEC.BLOCK_MIN_SEPARATION
    for _ in range(200):
        pts = np.stack(
            [
                rng.uniform(cx - half_x, cx + half_x, size=n),
                rng.uniform(cy - half_y, cy + half_y, size=n),
            ],
            axis=1,
        )
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        d[np.diag_indices(n)] = np.inf
        if d.min() >= min_sep:
            return pts
    raise RuntimeError(
        f"블록 {n}개를 최소 간격 {min_sep}m 로 배치하지 못했다. "
        "BOX_INNER_SIZE 를 키우거나 BLOCK_MIN_SEPARATION / "
        "BLOCK_SPAWN_WALL_MARGIN 을 줄일 것 (configs/vla_spec.py)."
    )


# -----------------------------------------------------------------------------
# 저장 / 로드
# -----------------------------------------------------------------------------
def save_bank(name: str, bank: np.ndarray, seed: int) -> Path:
    """뱅크를 저장한다. 생성 시드도 함께 굳혀 재생성 경로를 남긴다."""
    path = bank_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        states=bank.astype(np.float64),
        seed=np.int64(seed),
        num_blocks=np.int64(SPEC.NUM_BLOCKS),
        # 지오메트리가 바뀌면 예전 뱅크는 의미가 달라진다. 로드할 때 대조한다.
        box_center=np.asarray(SPEC.BOX_CENTER, dtype=np.float64),
        box_inner=np.asarray(SPEC.BOX_INNER_SIZE, dtype=np.float64),
    )
    return path


def load_bank(name: str) -> np.ndarray:
    """뱅크를 읽는다. 지오메트리가 바뀌었으면 조용히 넘어가지 않고 막는다."""
    path = bank_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"초기 상태 뱅크가 없다: {path}\n"
            f"  python scripts/make_init_states.py --name {name}"
        )
    data = np.load(path)
    if int(data["num_blocks"]) != SPEC.NUM_BLOCKS:
        raise ValueError(
            f"{path.name}: 블록 수 {int(data['num_blocks'])} != 현재 스펙 "
            f"{SPEC.NUM_BLOCKS}. 뱅크를 다시 만들 것 — 안 그러면 이전 실험과 "
            "초기 상태가 달라져 비교가 성립하지 않는다."
        )
    if not np.allclose(data["box_inner"], SPEC.BOX_INNER_SIZE):
        raise ValueError(
            f"{path.name}: 박스 안치수가 뱅크 생성 시점과 다르다 "
            f"({data['box_inner']} → {SPEC.BOX_INNER_SIZE}). 뱅크를 다시 만들 것."
        )
    states = np.asarray(data["states"], dtype=np.float64)
    # ★ 차원 검사. 트레이 단일화로 target_slot_idx 가 빠져 한 줄이 짧아졌다
    #   (3N+2 → 3N+1). 이걸 안 막으면 예전 뱅크의 슬롯 인덱스를 타깃 블록으로
    #   읽어 지시문과 실제 타깃이 조용히 어긋난다 — 에러 없이 데이터만 오염된다.
    if states.shape[1] != SPEC.INIT_STATE_DIM:
        raise ValueError(
            f"{path.name}: 상태 차원 {states.shape[1]} != 현재 스펙 "
            f"{SPEC.INIT_STATE_DIM}. 태스크 지오메트리가 바뀌었다 — 뱅크를 다시 "
            f"만들 것:\n  python scripts/make_init_states.py --name {name} --force"
        )
    return states


def ensure_bank(name: str, size: int, seed: int) -> np.ndarray:
    """없으면 만들어서라도 돌려준다 (학습용 뱅크의 편의 경로).

    평가용 홀드아웃은 이걸 쓰지 말 것 — 반드시 스크립트로 명시 생성해 커밋한다.
    """
    try:
        return load_bank(name)
    except FileNotFoundError:
        bank = sample_bank(size, seed)
        save_bank(name, bank, seed)
        return bank


# -----------------------------------------------------------------------------
# 자체 검사
# -----------------------------------------------------------------------------
def _demo() -> None:
    """같은 시드가 같은 뱅크를 주는지 + 제약이 실제로 지켜지는지."""
    a = sample_bank(16, seed=7)
    b = sample_bank(16, seed=7)
    assert np.array_equal(a, b), "같은 시드가 다른 뱅크를 냈다 — 재현성이 깨졌다."
    assert not np.array_equal(a, sample_bank(16, seed=8)), "시드가 무시되고 있다."
    assert a.shape == (16, SPEC.INIT_STATE_DIM)

    n = SPEC.NUM_BLOCKS
    xy = a[:, : 3 * n].reshape(16, n, 3)[..., :2]
    d = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
    d[:, np.arange(n), np.arange(n)] = np.inf
    assert d.min() >= SPEC.BLOCK_MIN_SEPARATION - 1e-9, "블록이 겹쳤다."

    half = 0.5 * np.asarray(SPEC.BOX_INNER_SIZE) - SPEC.BLOCK_SPAWN_WALL_MARGIN
    off = np.abs(xy - np.asarray(SPEC.BOX_CENTER))
    assert (off <= half + 1e-9).all(), "블록이 박스 밖에 스폰됐다."

    tgt = a[:, 3 * n].astype(int)
    assert tgt.min() >= 0 and tgt.max() < n
    print(f"초기 상태 뱅크 검사 통과 (상태 차원 {SPEC.INIT_STATE_DIM}).")


if __name__ == "__main__":
    _demo()

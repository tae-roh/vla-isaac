# =============================================================================
# vla_isaac_tasks/mdp/events.py
#
# 리셋 시 씬 초기화. **랜덤 샘플링을 하지 않는다** (개정 §3).
#
# 초기 배치·타깃 블록·타깃 슬롯은 전부 미리 만들어 둔 초기 상태 뱅크에서
# 인덱스로 꺼낸다. 이유는 init_states.py 머리말 참조 — 한 줄로 요약하면
# "GRPO 는 같은 s₀ 에서 G개를 굴려야 하고, 그러려면 s₀ 를 지정할 수 있어야 한다".
#
# 인덱스를 정하는 두 경로:
#   1) 순회(기본)   — 뱅크를 순서대로 돈다. 데이터 생성·SFT 학습용.
#   2) 강제 지정    — set_forced_indices() 로 못 박는다.
#                     GRPO 그룹(전 env 동일 인덱스)과 평가 홀드아웃이 이걸 쓴다.
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .. import init_states
from ..scene_assets import block_name
from ..spec import SPEC

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# 런타임 상태. 프로세스 하나에 환경 하나라는 전제다 (Isaac Sim 이 그렇다).
_BANK: dict = {"name": None, "states": None, "cursor": 0, "forced": None}


# -----------------------------------------------------------------------------
# 뱅크 선택 / 인덱스 강제
# -----------------------------------------------------------------------------
def use_bank(name: str) -> np.ndarray:
    """활성 뱅크를 바꾼다. 평가 split 을 갈아끼울 때 쓴다."""
    _BANK["states"] = init_states.load_bank(name)
    _BANK["name"] = name
    _BANK["cursor"] = 0
    _BANK["forced"] = None
    return _BANK["states"]


def set_forced_indices(indices) -> None:
    """다음 리셋부터 쓸 뱅크 인덱스를 못 박는다.

    - int 하나  → 모든 env 가 같은 초기 상태. **GRPO 그룹이 이 경로다.**
    - 리스트    → env 별 인덱스. 평가 홀드아웃을 순서대로 돌 때 쓴다.
    - None      → 순회 모드로 되돌린다.
    """
    if indices is None:
        _BANK["forced"] = None
    elif isinstance(indices, (int, np.integer)):
        _BANK["forced"] = int(indices)
    else:
        _BANK["forced"] = [int(i) for i in indices]


def active_bank_name() -> str | None:
    return _BANK["name"]


def _rows_for(
    env_ids: torch.Tensor, bank_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """이번 리셋에 쓸 뱅크 행들과 그 인덱스를 고른다."""
    if _BANK["states"] is None or _BANK["name"] != bank_name:
        # 학습용 뱅크는 없으면 만들어 준다 — 첫 실행이 파일 없음으로 죽지 않게.
        # 평가 홀드아웃은 scripts/make_init_states.py 로 명시 생성해 커밋한다.
        _BANK["states"] = init_states.ensure_bank(
            bank_name, SPEC.TRAIN_BANK_SIZE, seed=0
        )
        _BANK["name"] = bank_name
        _BANK["cursor"] = 0

    states = _BANK["states"]
    k = len(states)
    n = len(env_ids)
    forced = _BANK["forced"]

    if isinstance(forced, int):
        idx = np.full(n, forced % k, dtype=np.int64)
    elif forced is not None:
        env_np = env_ids.detach().cpu().numpy()
        idx = np.asarray([forced[int(e) % len(forced)] for e in env_np]) % k
    else:
        idx = (_BANK["cursor"] + np.arange(n)) % k
        _BANK["cursor"] = int((_BANK["cursor"] + n) % k)

    return states[idx], idx


# -----------------------------------------------------------------------------
# 리셋 이벤트
# -----------------------------------------------------------------------------
def reset_scene_from_bank(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    bank_name: str = "train",
) -> None:
    """블록 배치 + 타깃(블록·슬롯)을 초기 상태 뱅크에서 복원한다.

    ★ 이 함수가 초기 상태의 **유일한** 출처다. 여기 말고 다른 곳에서 블록을
      흔들면 (예: 별도의 pose 랜덤화 이벤트) 인덱스로 s₀ 를 지정한다는 전제가
      깨지고, GRPO 의 advantage 가 다시 노이즈가 된다.
    """
    rows, idx = _rows_for(env_ids, bank_name)
    device = env.device
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device=device)

    for b in range(SPEC.NUM_BLOCKS):
        asset: RigidObject = env.scene[block_name(b)]
        root = asset.data.default_root_state[env_ids].clone()

        root[:, 0] = rows_t[:, 3 * b + 0]
        root[:, 1] = rows_t[:, 3 * b + 1]
        # 블록 반높이만큼 띄워 정확히 테이블 위에 놓는다. 모두 같은 크기라
        # 예전처럼 "조금 띄워 놓고 떨어뜨리는" 보정이 필요 없다 —
        # 낙하 시간이 없으니 리셋 직후 관측이 곧바로 안정 상태다.
        root[:, 2] = SPEC.TABLE_HEIGHT + 0.5 * SPEC.BLOCK_SIZE[2]

        yaw = rows_t[:, 3 * b + 2]
        half = 0.5 * yaw
        root[:, 3] = torch.cos(half)   # w
        root[:, 4] = 0.0
        root[:, 5] = 0.0
        root[:, 6] = torch.sin(half)   # z
        root[:, 7:] = 0.0

        root[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

    # 타깃 지정 — 관측·성공판정·지시문이 모두 이 두 버퍼를 읽는다.
    tgt_block = rows_t[:, 3 * SPEC.NUM_BLOCKS].long()
    tgt_slot = rows_t[:, 3 * SPEC.NUM_BLOCKS + 1].long()
    _buffer(env, "_vla_target_block", torch.long)[env_ids] = tgt_block
    _buffer(env, "_vla_target_slot", torch.long)[env_ids] = tgt_slot
    _buffer(env, "_vla_init_index", torch.long)[env_ids] = torch.as_tensor(
        idx, dtype=torch.long, device=device
    )


def reset_episode_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor) -> None:
    """에피소드 래치·카운터를 지운다.

    - `_vla_grasp_lift_latch` : 리프트 중 파지했는가 (pushcut 방지 항)
    - `_vla_success_hold_counter` : 성공 유지 스텝 수

    래치는 조건이 성립할 때만 켜지므로 이전 에피소드 값이 남으면 **밀어서
    넣은 궤적이 성공으로 잡힌다.** 성공 판정의 핵심 항이라 반드시 지운다.
    """
    _buffer(env, "_vla_grasp_lift_latch", torch.bool)[env_ids] = False
    _buffer(env, "_vla_success_hold_counter", torch.long)[env_ids] = 0


def _buffer(env, attr: str, dtype: torch.dtype) -> torch.Tensor:
    """env 에 붙은 (num_envs,) 버퍼를 가져오거나 만든다."""
    buf = getattr(env, attr, None)
    if buf is None or buf.shape[0] != env.num_envs or buf.dtype != dtype:
        buf = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
        setattr(env, attr, buf)
    return buf

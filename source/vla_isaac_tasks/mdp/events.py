# =============================================================================
# vla_isaac_tasks/mdp/events.py
#
# 리셋 시 씬 초기화. **랜덤 샘플링을 하지 않는다** (개정 §3).
#
# 초기 배치와 타깃 블록은 전부 미리 만들어 둔 초기 상태 뱅크에서 인덱스로
# 꺼낸다. 이유는 init_states.py 머리말 참조 — 한 줄로 요약하면
# "GRPO 는 같은 s₀ 에서 G개를 굴려야 하고, 그러려면 s₀ 를 지정할 수 있어야 한다".
#
# 인덱스를 정하는 두 경로:
#   1) 순회(기본)   — 뱅크를 순서대로 돈다. 데이터 생성·SFT 학습용.
#   2) 강제 지정    — set_forced_indices() 로 못 박는다.
#                     GRPO 그룹(전 env 동일 인덱스)과 평가 홀드아웃이 이걸 쓴다.
# =============================================================================

from __future__ import annotations

import os
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


# 뱅크 행 역조회 허용 오차 [m]. 블록 N개의 xy 를 이어 붙인 벡터의 L2 거리다.
# 실측상 정확히 일치하는 행은 거리 0.00mm 이고, 서로 다른 행은 수 cm 떨어져
# 있으므로 이 값은 매우 넉넉하다. 넘으면 "뱅크에서 온 배치가 아니다" 로 본다.
_BANK_MATCH_TOL = 0.005


def recover_target_from_scene(env, env_ids=None) -> int:
    """현재 블록 배치로 뱅크 행을 역조회해 타깃 블록을 복원한다.

    ★ 왜 필요한가 — replay / annotate 경로가 타깃을 잃어버린다.
      ManagerBasedEnv.reset_to() 는 이 순서로 돈다:
          _reset_idx()            → reset 이벤트 실행 (reset_scene_from_bank)
          scene.reset_to(state)   → 블록 위치를 **파일 값으로 덮어씀**
      블록 위치는 복원되지만 `_vla_target_block` 은 파이썬 버퍼라 물리 상태에
      들어 있지 않다. 그래서 이벤트가 뱅크 커서에서 뽑은 값이 그대로 남고,
      데모가 실제로 지시했던 블록과 어긋난다 (실측 10개 중 7개 불일치).
      성공 판정이 엉뚱한 블록을 보게 되어 replay 성공률이 0 이 된다.

      초기 배치는 연속 난수라 뱅크 행을 유일하게 식별한다. 그래서 덮어쓰기가
      끝난 뒤 배치로 행을 되찾으면 타깃이 정확히 복원된다.
      정상 롤아웃에서는 방금 그 행을 다시 찾으므로 아무것도 바뀌지 않는다.

    Returns:
        복원에 성공한 env 수.
    """
    states = _BANK["states"]
    if states is None:
        return 0

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device)
    if env_ids.numel() == 0:
        return 0

    n = SPEC.NUM_BLOCKS
    cur = []
    for b in range(n):
        asset: RigidObject = env.scene[block_name(b)]
        pos = asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
        cur.append(pos[:, :2])
    cur_np = torch.cat(cur, dim=-1).detach().cpu().numpy()          # (N, 2n)

    bank_xy = np.concatenate(
        [states[:, [3 * b, 3 * b + 1]] for b in range(n)], axis=1
    )                                                               # (K, 2n)
    dist = np.linalg.norm(bank_xy[None, :, :] - cur_np[:, None, :], axis=-1)
    j = dist.argmin(axis=1)
    dmin = dist[np.arange(len(j)), j]
    tgt = states[j, 3 * n].astype(np.int64)

    buf = _buffer(env, "_vla_target_block", torch.long)
    idx_buf = _buffer(env, "_vla_init_index", torch.long)
    recovered = 0
    for i, e in enumerate(env_ids.tolist()):
        if dmin[i] <= _BANK_MATCH_TOL:
            buf[e] = int(tgt[i])
            idx_buf[e] = int(j[i])
            recovered += 1
    return recovered


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
    """블록 배치 + 타깃 블록을 초기 상태 뱅크에서 복원한다.

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

    # 타깃 지정 — 관측·성공판정·지시문이 모두 이 버퍼를 읽는다.
    # 트레이는 하나뿐이라 "어느 블록인가" 만 남았다.
    tgt_block = rows_t[:, 3 * SPEC.NUM_BLOCKS].long()
    _buffer(env, "_vla_target_block", torch.long)[env_ids] = tgt_block
    _buffer(env, "_vla_init_index", torch.long)[env_ids] = torch.as_tensor(
        idx, dtype=torch.long, device=device
    )

    _announce_targets(env_ids, tgt_block, idx)


# -----------------------------------------------------------------------------
# 텔레옵용 타깃 안내
# -----------------------------------------------------------------------------
# ★ 왜 필요한가
#   블록 3개는 색만 다르고 형상·크기가 같다. 지시문을 모르면 사람은 어느 것을
#   집어야 하는지 알 방법이 없는데, record_demos.py 는 관측을 화면에 찍어 주지
#   않는다. 타깃을 틀리면 성공 판정이 안 뜨고 → EXPORT_SUCCEEDED_ONLY 라
#   에피소드가 통째로 버려진다. 에러도 안 나서 "왜 저장이 안 되지" 로만 보인다.
#
#   뱅크를 미리 출력해 두고 세는 방법도 있지만, 폐기(R) 도 리셋이라 커서를
#   소비하므로 한 번 어긋나면 계속 어긋난다. 리셋 때마다 환경이 직접 말하게 하는
#   것이 유일하게 어긋나지 않는 방법이다.
#
#   기본은 꺼 둔다 — 야간 배치 생성(num_envs=30)에서 매 리셋마다 찍으면 로그가
#   쓸모없어진다. 텔레옵 수집 때만 VLA_ANNOUNCE_TARGET=1 로 켠다.
def _announce_targets(env_ids, tgt_block, idx) -> None:
    """리셋된 env 의 지시문을 stdout 에 찍는다 (VLA_ANNOUNCE_TARGET=1 일 때만)."""
    if os.environ.get("VLA_ANNOUNCE_TARGET", "") not in ("1", "true", "True"):
        return
    ids = env_ids.detach().cpu().numpy().tolist()
    blocks = tgt_block.detach().cpu().numpy().tolist()
    for e, b, i in zip(ids, blocks, np.atleast_1d(idx).tolist()):
        instr = SPEC.instruction_for(int(b))
        print(
            f"\n{'=' * 60}\n"
            f"  [env {e} / 뱅크 #{i}]  ▶  {instr.upper()}\n"
            f"{'=' * 60}",
            flush=True,
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

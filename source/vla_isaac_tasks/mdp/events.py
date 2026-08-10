# =============================================================================
# vla_isaac_tasks/mdp/events.py
#
# 리셋 시 랜덤화. "리스폰마다 형태가 조금씩 달라지는 자재" 를 강체로 근사하는
# 세 축이 여기서 만들어진다:
#
#   축 1. 형상 — MultiAssetSpawnerCfg 가 env 별로 6종 중 하나를 배정 (materials.py)
#   축 2. 크기 — prestartup 스케일 랜덤화 (아래)
#   축 3. 자세 — reset 마다 위치/yaw 랜덤화 (아래)
#
# 축 1·2 는 스폰 시점에 고정된다 (PhysX 가 런타임 지오메트리 변경을 지원하지 않음).
# 축 3 만 매 리셋 바뀐다. 따라서 데이터 다양성은 num_envs 를 키워서 얻는다 —
# Phase 2 에서 --num_envs 20~40 을 쓰라는 지침이 속도뿐 아니라 다양성 문제인 이유다.
# =============================================================================

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from ..spec import SPEC

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_material_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    x_range: tuple[float, float] = (0.38, 0.60),
    y_range: tuple[float, float] = (-0.22, 0.05),
    z_offset: float = 0.03,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """자재를 테이블 위 무작위 위치·yaw 로 재배치한다.

    ★ y_range 상한이 0.05 인 것은 의도된 값이다.
      목표 영역 중심이 y=+0.35 (SPEC.GOAL_REGION_CENTER) 이므로, 스폰 범위를
      목표에서 충분히 떨어뜨려 **자재가 이미 목표 안에 놓인 채로 시작하는 일이
      없도록** 한다. 그런 에피소드는 아무것도 안 해도 성공으로 잡혀서
      RFT 보상을 오염시키고, 성공 판정 유지 카운터의 자연 초기화도 깨뜨린다.
      SPEC.GOAL_REGION_CENTER 를 바꾸면 이 범위도 함께 검토할 것.

    z 는 자재 크기가 6종마다 달라 정확한 안착 높이를 미리 알 수 없으므로,
    조금 띄워 놓고 물리로 떨어뜨린다 (z_offset).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    def _uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=device) * (hi - lo) + lo

    root_states = asset.data.default_root_state[env_ids].clone()

    root_states[:, 0] = _uniform(*x_range)
    root_states[:, 1] = _uniform(*y_range)
    root_states[:, 2] = SPEC.TABLE_HEIGHT + z_offset

    # yaw 만 랜덤화한다. roll/pitch 까지 흔들면 원뿔·캡슐이 굴러가 버려
    # 텔레옵 데모를 만들기가 지나치게 어려워진다.
    yaw = _uniform(*yaw_range)
    half = yaw * 0.5
    root_states[:, 3] = torch.cos(half)   # w
    root_states[:, 4] = 0.0               # x
    root_states[:, 5] = 0.0               # y
    root_states[:, 6] = torch.sin(half)   # z

    # 속도는 0 으로. 남은 속도가 있으면 리셋 직후 물체가 튀어 나간다.
    root_states[:, 7:] = 0.0

    # 월드 좌표로 올린다.
    root_states[:, :3] += env.scene.env_origins[env_ids]

    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids=env_ids)


def reset_success_hold_counter(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
) -> None:
    """성공 유지 카운터를 리셋한다 (terminations.task_success 가 쓰는 버퍼).

    카운터는 조건이 깨지면 스스로 0 이 되지만, 리셋 직후 첫 판정 전에 이전
    에피소드의 값이 남아 있으면 한 스텝짜리 가짜 성공이 뜰 수 있다.
    명시적으로 지워 두는 편이 안전하다.
    """
    counter = getattr(env, "_vla_success_hold_counter", None)
    if counter is not None:
        counter[env_ids] = 0

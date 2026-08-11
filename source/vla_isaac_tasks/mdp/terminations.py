# =============================================================================
# vla_isaac_tasks/mdp/terminations.py
#
# 에피소드 종료 조건. 성공 판정은 세 곳이 공유한다:
#   - 텔레옵 녹화 (record_demos.py 가 성공 시 저장하고 다음 에피소드로 넘어감)
#   - Mimic 증강 (생성된 궤적이 성공했는지 판정)
#   - RFT 의 0/1 보상
# 셋이 같은 함수를 부르는 것이 핵심이다.
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ..spec import SPEC
from .observations import placed_signal, target_block_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HOLD_COUNTER_ATTR = "_vla_success_hold_counter"


def task_success(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hold_steps: int = SPEC.SUCCESS_HOLD_STEPS,
) -> torch.Tensor:
    """성공 조건이 hold_steps 연속으로 유지되면 True. Shape (num_envs,).

    ★ 왜 즉시 종료하지 않고 유지 스텝을 두는가

      계획서 §Phase1-3 의 경고 지점이다. 성공하는 순간 녹화를 끊으면,
      나중에 replay_demos.py 로 재생할 때 마지막 프레임에서 성공 조건이
      아슬아슬하게 재트리거되지 않아 데모가 통째로 버려진다. Isaac Lab 물리는
      env.reset 경유 재생에서 결정론적으로 재현되지 않기 때문이다.
      몇 스텝 여유를 두면 재생 시에도 확실히 조건을 통과한다.

      부수 효과로 "물체를 놓자마자 굴러 나가는" 가짜 성공도 걸러진다.

    카운터는 조건이 깨지면 0 으로 돌아간다. 에피소드 리셋 시 블록은 전부 박스
    안에 스폰되고 grasped_during_lift 래치도 꺼져 있으므로, 첫 스텝에서 조건이
    False 가 되어 카운터가 자연히 초기화된다.
    """
    cond = placed_signal(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)

    counter = getattr(env, _HOLD_COUNTER_ATTR, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, _HOLD_COUNTER_ATTR, counter)

    counter = torch.where(cond, counter + 1, torch.zeros_like(counter))
    setattr(env, _HOLD_COUNTER_ATTR, counter)

    return counter >= hold_steps


def object_dropped(
    env: "ManagerBasedRLEnv",
    minimum_height: float = -0.05,
) -> torch.Tensor:
    """타깃 블록이 테이블 아래로 떨어졌으면 True — 에피소드를 조기 종료한다.

    떨어진 뒤에도 계속 굴리면 롤아웃 시간만 낭비된다. RFT 에서는 이 조기 종료가
    스텝당 비용을 눈에 띄게 줄여 준다.

    타깃이 아닌 블록이 떨어지는 것은 종료 사유가 아니다 — 성공 술어가 타깃만
    보므로 그 에피소드는 그대로 실패로 끝난다. 굳이 다른 종료 경로를 만들면
    "왜 끝났는지" 해석만 늘어난다.
    """
    return target_block_pose(env)[:, 2] < (SPEC.TABLE_HEIGHT + minimum_height)

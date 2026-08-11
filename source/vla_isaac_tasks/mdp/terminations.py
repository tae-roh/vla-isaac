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


def update_success_hold(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """성공 조건 유지 카운터를 **한 스텝분** 전진시킨다. Shape (num_envs,).

    ★ 반드시 스텝당 정확히 한 번만 불러야 한다. 그래서 호출부는 환경의
      step() 하나로 못 박아 두었다 (vla_env.VlaEnvMixin). 왜 그래야 하는지는
      task_success() 의 주석 참조.
    """
    cond = placed_signal(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)

    counter = getattr(env, _HOLD_COUNTER_ATTR, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    counter = torch.where(cond, counter + 1, torch.zeros_like(counter))
    setattr(env, _HOLD_COUNTER_ATTR, counter)
    return counter


def task_success(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hold_steps: int = SPEC.SUCCESS_HOLD_STEPS,
) -> torch.Tensor:
    """성공 조건이 hold_steps 연속으로 유지되었으면 True. Shape (num_envs,).

    ★ 왜 즉시 종료하지 않고 유지 스텝을 두는가

      계획서 §Phase1-3 의 경고 지점이다. 성공하는 순간 녹화를 끊으면,
      나중에 replay_demos.py 로 재생할 때 마지막 프레임에서 성공 조건이
      아슬아슬하게 재트리거되지 않아 데모가 통째로 버려진다. Isaac Lab 물리는
      env.reset 경유 재생에서 결정론적으로 재현되지 않기 때문이다.
      몇 스텝 여유를 두면 재생 시에도 확실히 조건을 통과한다.

      부수 효과로 "물체를 놓자마자 굴러 나가는" 가짜 성공도 걸러진다.

    ★ 이 함수는 **카운터를 읽기만 한다.** 예전에는 호출될 때마다 카운터를
      올렸는데, 그러면 "몇 번 불렸는가" 를 세게 되어 호출 빈도가 다른 소비자
      사이에서 의미가 달라진다. 실제로 이것 때문에 replay 가 항상 실패했다:

          replay_demos.py 는 env_cfg.terminations = {} 로 종료 조건을 전부 끈 뒤
          success_term.func() 를 **에피소드 끝에 한 번만** 부른다.
          → 카운터가 1 까지밖에 못 올라가 1 >= 48 이 영원히 False.
          → 데모가 아무리 멀쩡해도 성공률이 0/10 으로 나온다.

      이제 전진은 update_success_hold() 가 환경 step() 에서 스텝당 한 번씩
      담당하고, 이 함수는 그 결과만 본다. 호출 횟수와 무관해졌으므로
      replay / Mimic 생성 / RFT 어느 경로에서 불러도 같은 뜻이 된다.

    카운터는 조건이 깨지면 0 으로 돌아가고, 리셋에서는
    events.reset_episode_buffers 가 지운다.
    """
    counter = getattr(env, _HOLD_COUNTER_ATTR, None)
    if counter is None or counter.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
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

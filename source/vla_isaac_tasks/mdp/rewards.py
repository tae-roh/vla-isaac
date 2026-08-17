# =============================================================================
# vla_isaac_tasks/mdp/rewards.py
#
# RFT 용 보상. 계획서 §2-2 의 결정에 따라 **이진 0/1 만** 쓴다.
#
# 왜 shaping 을 넣지 않는가:
#   GRPO 는 그룹 내에서 advantage 를 정규화하는 critic-free 알고리즘이라
#   sparse 0/1 보상에 잘 맞는다. SimpleVLA-RL 의 검증된 성과(LIBERO-Long 97.6,
#   데모 1개 SFT 에서 17.3 → 91.7)가 전부 이 설정이다.
#   여기에 거리 기반 shaping 을 섞으면 그 레시피에서 벗어나면서, 보상 해킹
#   (목표 근처를 맴돌며 shaping 만 챙기는 행동)의 여지가 생긴다.
#   3~4일 안에 그걸 진단할 시간이 없다.
#
# 디버깅용 보조 항은 아래에 두되 **가중치 0** 으로 둔다 — 로그로는 보이지만
# 학습 신호에는 들어가지 않는다. 학습 커브가 평평할 때 "정책이 접근은 하는데
# 파지를 못 하는 것인지" 를 구분하는 데 쓴다.
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import placed_signal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ★ 진단 항목을 "가중치 0 인 보상 항" 으로 두지 않는다.
#   Isaac Lab 의 RewardManager 는 로그에 남기는 값도 weight 를 곱한 뒤라
#   (episode_sums += func() * weight * dt), weight=0 이면 **로그도 0 이다.**
#   진단이 되는 것처럼 보이지만 아무것도 알려주지 않는다.
#   → 진단은 관측(yaw_err, subtask_terms/grasp_lift)으로 내보내고, 롤아웃
#     워커가 그 평균을 보고한다 (rft/isaaclab_rollout_worker.py 의 diag).


def success_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """태스크 성공 시 1.0, 아니면 0.0. RFT 의 유일한 학습 신호다.

    ★ 2026-08-16 수정 — **유지 조건까지 반영한다.**

      예전에는 `placed_signal()` 의 **순간값**을 돌려주었다. 그런데 종료 조건
      `task_success()` 는 같은 술어가 SUCCESS_HOLD_STEPS(16스텝 = 2초) 연속
      유지될 때만 참이다. 두 정의가 갈라져 있어서, 블록이 트레이 영역을 느리게
      **스치기만 해도** 보상이 1 로 뜨고 성공으로 집계됐다
      (settled 임계가 0.03 m/s 라 통과 중에도 만족된다).

      실측 증거: SFT 체크포인트 12 에피소드에서 `diag["success"]` 는 2건을
      잡았지만, 12개 **전부** 300스텝을 완주했다 — 즉 task_success 종료가
      한 번도 발동하지 않았고 실제 안착은 0건이었다. 영상에도 트레이에 들어가는
      장면이 없다. 그 상태로 RFT 를 돌리면 "안착" 이 아니라 "스침" 을 강화한다.

      이제 종료 조건과 **같은 카운터**를 읽으므로 보상·종료·평가가 한 정의다
      (placed_signal 의 주석이 원래 의도한 바이기도 하다).

    ⚠ 카운터 전진은 VlaEnvMixin.step() 의 update_success_hold() 가 담당한다.
      이 함수는 읽기만 한다 — 여기서 전진시키면 호출 횟수만큼 중복 계수된다.
    """
    from .terminations import task_success

    return task_success(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg).float()



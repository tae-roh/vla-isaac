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

from .observations import grasp_lift_signal, placed_signal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """태스크 성공 시 1.0, 아니면 0.0. RFT 의 유일한 학습 신호다."""
    return placed_signal(env, robot_cfg=robot_cfg, object_cfg=object_cfg).float()


def grasp_lift_progress(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """진단용 — 파지 후 들어올림 성공 여부. 가중치 0 으로 등록해 로그로만 본다."""
    return grasp_lift_signal(
        env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg
    ).float()

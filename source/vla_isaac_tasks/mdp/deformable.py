# =============================================================================
# vla_isaac_tasks/mdp/deformable.py
#
# 변형체(FEM) 자재용 관측·판정. Phase 4b 스트레치 전용.
#
# 왜 강체 버전을 재사용할 수 없는가:
#   DeformableObject 에는 단일 강체 pose 가 존재하지 않는다. 상태는 노드 집합
#   (nodal_pos_w) 으로 표현되고, "위치" 는 노드 평균으로 정의할 수밖에 없다.
#   이것이 계획서 §2-3 에서 "SkillGen 증강을 변형체에 직접 적용 불가" 라고
#   판단한 것과 같은 이유다 — 증강은 오브젝트 SE(3) pose 를 전제로 하기 때문.
#
#   RL 은 보상만 있으면 되므로 pose 가정이 필요 없고, 그래서 변형체는
#   RFT 단계에만 들어간다.
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

from ..spec import SPEC
from .observations import gripper_pos

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def deformable_center(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """변형체의 노드 평균 위치 (env 로컬). Shape (num_envs, 3)."""
    obj: DeformableObject = env.scene[object_cfg.name]
    # nodal_pos_w: (num_envs, num_nodes, 3)
    center_w = obj.data.nodal_pos_w.mean(dim=1)
    return center_w - env.scene.env_origins


def deformable_spread(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """노드가 중심에서 얼마나 퍼져 있는지 (변형 정도의 대리 지표). Shape (num_envs, 1).

    학습 신호로는 쓰지 않는다. FEM 이 발산하면(파지 중 물체가 폭발하는 전형적
    실패) 이 값이 급증하므로, 로그로 보면서 solver 파라미터를 잡는 데 쓴다.
    계획서 §6 "변형체 파지 물리 불안정" 리스크의 조기 감지 장치다.
    """
    obj: DeformableObject = env.scene[object_cfg.name]
    nodal = obj.data.nodal_pos_w
    center = nodal.mean(dim=1, keepdim=True)
    return torch.linalg.norm(nodal - center, dim=-1).mean(dim=1, keepdim=True)


def deformable_placed_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    node_fraction: float = 0.8,
) -> torch.Tensor:
    """변형체가 목표 영역에 놓였는가. Shape (num_envs,) bool.

    계획서 §Phase4b-5 의 정의를 따르되 한 가지를 강화했다:
    노드 평균만 보면 자재가 목표 경계에 길게 걸쳐 있어도 성공으로 잡힌다.
    그래서 **노드의 node_fraction 이상이 영역 안에 있을 것** 을 함께 요구한다.

    Args:
        node_fraction: 목표 영역 안에 있어야 하는 노드 비율.
    """
    obj: DeformableObject = env.scene[object_cfg.name]
    nodal_local = obj.data.nodal_pos_w - env.scene.env_origins.unsqueeze(1)

    gx, gy = SPEC.GOAL_REGION_CENTER
    goal_xy = torch.tensor([gx, gy], device=env.device, dtype=nodal_local.dtype)

    # (1) 노드 다수가 목표 반경 안
    dxy = torch.linalg.norm(nodal_local[..., :2] - goal_xy, dim=-1)
    inside_ratio = (dxy < SPEC.GOAL_REGION_RADIUS).float().mean(dim=1)
    in_region = inside_ratio >= node_fraction

    # (2) 테이블 위에 내려놓았는가 (노드 평균 높이)
    height = nodal_local[..., 2].mean(dim=1) - SPEC.TABLE_HEIGHT
    on_table = height < SPEC.GOAL_HEIGHT_TOLERANCE

    # (3) 정지 — 노드 속도 평균
    settled = (
        torch.linalg.norm(obj.data.nodal_vel_w, dim=-1).mean(dim=1)
        < SPEC.SUCCESS_LIN_VEL_THRESHOLD
    )

    # (4) 그리퍼 개방
    released = gripper_pos(env, robot_cfg).sum(dim=-1) > SPEC.GRIPPER_OPEN_QPOS_SUM

    return in_region & on_table & settled & released


def deformable_success_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """변형체 RFT 의 0/1 보상."""
    return deformable_placed_signal(
        env, robot_cfg=robot_cfg, object_cfg=object_cfg
    ).float()


def deformable_dropped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float = -0.05,
) -> torch.Tensor:
    """변형체가 테이블 아래로 떨어졌는가."""
    center = deformable_center(env, object_cfg)
    return center[:, 2] < (SPEC.TABLE_HEIGHT + minimum_height)

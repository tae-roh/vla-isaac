# =============================================================================
# vla_isaac_tasks/mdp/observations.py
#
# 관측 항목. 세 갈래로 쓰인다:
#   1) policy 그룹  — VLA 입력 (이미지) + Mimic 이 요구하는 eef pose
#   2) subtask 그룹 — Mimic 자동 어노테이션용 휴리스틱 시그널 (--auto)
#   3) proprio      — LIBERO 스펙(8차원)과 동일 레이아웃. USE_PROPRIO 일 때만 사용
#
# Isaac Lab 의 Franka 스태킹 태스크가 이미 갖고 있는 함수는 재구현하지 않고
# 그대로 가져다 쓴다 (ee_frame_pos / ee_frame_quat / gripper_pos / object_grasped).
# 검증된 코드를 다시 쓰는 쪽이 블라인드 코딩 리스크가 낮다.
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

# Isaac Lab 스태킹 태스크의 검증된 구현을 재사용한다.
#   ee_frame_pos / ee_frame_quat : Mimic 의 get_robot_eef_pose 가 읽는 값
#   gripper_pos                  : 평행 그리퍼/석션을 모두 처리해 준다
#   object_grasped               : eef-물체 거리 + 그리퍼 닫힘 여부 판정
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import (  # noqa: F401
    ee_frame_pos,
    ee_frame_quat,
    gripper_pos,
    object_grasped,
)

from ..spec import SPEC

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# 자재(오브젝트) 상태
# -----------------------------------------------------------------------------
def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """자재 위치를 로봇 베이스 좌표계로. Shape (num_envs, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        obj.data.root_pos_w[:, :3],
    )
    return pos_b


def object_pose_in_env_frame(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """자재의 위치+쿼터니언을 env 로컬 좌표계로. Shape (num_envs, 7).

    Mimic 의 get_object_poses 가 이 값을 4x4 행렬로 바꿔 쓴다.
    env 원점을 빼는 것이 중요하다 — 월드 좌표를 그대로 쓰면 env 격자 간격이
    궤적 변환에 섞여 들어가 증강 결과가 전부 어긋난다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    return torch.cat([pos, obj.data.root_quat_w], dim=-1)


def object_lin_vel(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """자재 선속도. Shape (num_envs, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w


def object_height_above_table(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """자재가 테이블 상면에서 얼마나 떠 있는지. Shape (num_envs, 1)."""
    obj: RigidObject = env.scene[object_cfg.name]
    z_local = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z_local - SPEC.TABLE_HEIGHT).unsqueeze(-1)


def goal_position_in_env_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """목표 영역 중심 (env 로컬). Shape (num_envs, 3).

    스펙 상수에서 만들어 낸 고정값이다. 목표 위치를 랜덤화하려면 여기가 아니라
    이벤트로 바꾸고, 그 순간 SFT 데이터도 다시 만들어야 한다.
    """
    gx, gy = SPEC.GOAL_REGION_CENTER
    goal = torch.tensor(
        [gx, gy, SPEC.TABLE_HEIGHT], device=env.device, dtype=torch.float32
    )
    return goal.unsqueeze(0).repeat(env.num_envs, 1)


# -----------------------------------------------------------------------------
# Proprioception — LIBERO 8차원 레이아웃과 정확히 일치
# -----------------------------------------------------------------------------
def proprio_state(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """LIBERO 와 동일한 8차원 proprio. Shape (num_envs, 8).

    레이아웃: eef 위치(3) + eef 방향 axis-angle(3) + 그리퍼 관절(2)

    SimpleVLA-RL 의 rob_rollout._obs_to_input() 이 LIBERO 에서 만드는 벡터와
    같은 순서·같은 의미여야 한다. 여기가 어긋나면 SFT 와 RFT 가 서로 다른 것을
    proprio 라고 부르게 되고, 증상은 "RFT 를 켜자 성능이 무너진다" 로 나타난다.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos = ee_frame.data.target_pos_w[..., 0, :] - env.scene.env_origins
    quat = ee_frame.data.target_quat_w[..., 0, :]

    # LIBERO 는 quat2axisangle 로 3차원 지수좌표를 쓴다. 동일하게 맞춘다.
    axis_angle = math_utils.axis_angle_from_quat(quat)

    grip = gripper_pos(env, robot_cfg)          # (num_envs, 2)
    out = torch.cat([pos, axis_angle, grip], dim=-1)

    if out.shape[-1] != SPEC.PROPRIO_DIM:
        raise RuntimeError(
            f"proprio 차원 {out.shape[-1]} != 스펙 {SPEC.PROPRIO_DIM}. "
            "configs/vla_spec.py 의 PROPRIO_LAYOUT 과 이 함수를 함께 맞출 것."
        )
    return out


# -----------------------------------------------------------------------------
# 서브태스크 종료 시그널 (Mimic --auto 어노테이션용)
# -----------------------------------------------------------------------------
# 계획서 §Phase1-5: 서브태스크는 최소 개수로. 여기서는 2개로 압축했다.
#   서브태스크 1 종료 = "자재를 잡고 충분히 들어올림"  → grasp_lift
#   서브태스크 2 종료 = 태스크 끝 (Mimic 규약상 마지막은 signal=None)
#
# 경계를 "파지 직후"가 아니라 "파지 후 들어올린 뒤"에 둔 것이 의도한 바다.
# 계획서 §Phase1-5 지침대로, 경계에서 로봇이 다른 물체(여기서는 테이블)와
# 충돌하지 않는 지점이어야 스티칭된 궤적의 생성 성공률이 높다.
def grasp_lift_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lift_height: float = SPEC.LIFT_HEIGHT_THRESHOLD,
) -> torch.Tensor:
    """자재를 파지한 채로 lift_height 이상 들어올렸는가. Shape (num_envs,) bool."""
    grasped = object_grasped(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=SPEC.GRASP_DISTANCE_THRESHOLD,
    )
    height = object_height_above_table(env, object_cfg).squeeze(-1)
    return torch.logical_and(grasped, height > lift_height)


# -----------------------------------------------------------------------------
# 서브태스크 "시작" 시그널 (SkillGen 전용)
# -----------------------------------------------------------------------------
# SkillGen 은 시작 경계가 없으면 데이터 생성이 실패한다 ("Start boundary signals are
# mandatory for SkillGen"). MimicGen 만 쓸 때는 이 항목들이 그냥 무시되므로,
# 두 방식을 분기 없이 같은 환경으로 지원할 수 있다.
#
# 의미: 시작~종료 사이 = 사람 데모를 그대로 재생할 접촉 구간(skill)
#       서브태스크 사이   = cuRobo 가 충돌 회피 경로를 새로 까는 자유공간 구간
def grasp_start_signal(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    start_distance: float = SPEC.APPROACH_START_DISTANCE,
) -> torch.Tensor:
    """eef 가 자재에 충분히 접근했는가 = 파지 접촉 구간의 시작. Shape (num_envs,) bool.

    이 지점 이전(자유공간 접근)은 cuRobo 가 계획하고, 이후(정렬·하강·파지)는
    사람 데모를 재생한다. 그래서 임계값이 실제 접촉 거리보다 넉넉해야 한다 —
    너무 좁게 잡으면 파지 직전의 정렬 동작까지 플래너에 맡기게 되어 파지가 실패한다.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    eef_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.linalg.norm(obj.data.root_pos_w - eef_pos_w, dim=-1)
    return dist < start_distance


def place_start_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    start_radius: float = SPEC.PLACE_START_RADIUS,
) -> torch.Tensor:
    """자재를 든 채로 목표 영역 위에 도달했는가 = 배치 접촉 구간의 시작.

    Shape (num_envs,) bool.

    "든 채로" 조건이 중요하다. 이게 없으면 자재가 우연히 목표 근처에 스폰된
    에피소드에서 시작 신호가 첫 프레임부터 켜져, 운반 구간이 통째로 사라진다.
    """
    grasped = object_grasped(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=SPEC.GRASP_DISTANCE_THRESHOLD,
    )
    obj: RigidObject = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w - env.scene.env_origins
    gx, gy = SPEC.GOAL_REGION_CENTER
    dxy = torch.linalg.norm(
        pos_local[:, :2]
        - torch.tensor([gx, gy], device=env.device, dtype=pos_local.dtype),
        dim=-1,
    )
    return torch.logical_and(grasped, dxy < start_radius)


def placed_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """자재가 목표 영역 안에 놓였는가 (성공 조건 원본). Shape (num_envs,) bool.

    RFT 의 0/1 보상과 성공 판정 termination 이 모두 이 함수를 부른다 —
    "보상과 평가 기준이 같은 코드에서 나온다"는 것이 중요하다. 따로 구현하면
    RFT 가 최적화하는 것과 우리가 측정하는 것이 미묘하게 달라진다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w - env.scene.env_origins
    gx, gy = SPEC.GOAL_REGION_CENTER

    # (1) 목표 영역 안 (xy 반경)
    dxy = torch.linalg.norm(
        pos_local[:, :2]
        - torch.tensor([gx, gy], device=env.device, dtype=pos_local.dtype),
        dim=-1,
    )
    in_region = dxy < SPEC.GOAL_REGION_RADIUS

    # (2) 테이블 위에 내려놓았는가 (공중에 든 채로 성공 처리되면 안 된다)
    height = pos_local[:, 2] - SPEC.TABLE_HEIGHT
    on_table = height < SPEC.GOAL_HEIGHT_TOLERANCE

    # (3) 정지 — 굴러가는 중에 성공이 뜨는 것을 막는다
    settled = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1) < SPEC.SUCCESS_LIN_VEL_THRESHOLD

    # (4) 그리퍼 개방 — "놓았다"는 것까지 확인해야 픽앤플레이스가 완결된다
    released = gripper_pos(env, robot_cfg).sum(dim=-1) > SPEC.GRIPPER_OPEN_QPOS_SUM

    return in_region & on_table & settled & released

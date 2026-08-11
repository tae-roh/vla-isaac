# =============================================================================
# vla_isaac_tasks/mdp/observations.py
#
# 관측 항목. 네 갈래로 쓰인다:
#   1) policy 그룹  — VLA 입력 (이미지) + Mimic 이 요구하는 eef/오브젝트 pose
#   2) subtask 그룹 — Mimic/SkillGen 어노테이션용 휴리스틱 시그널
#   3) proprio      — LIBERO 스펙(8차원)과 동일 레이아웃. USE_PROPRIO 일 때만 사용
#   4) 타깃 식별자  — 지시문을 만들 때 쓴다 (블록·슬롯 인덱스)
#
# ★ 이 태스크의 오브젝트는 "블록 N개 중 지시문이 지정한 하나" 다.
#   그래서 대부분의 함수가 **타깃 블록을 env 별로 골라내는** 것으로 시작한다.
#   Isaac Lab 의 SceneEntityCfg 는 env 별로 다른 엔티티를 가리킬 수 없으므로
#   (씬 엔티티 이름은 정적이다) 여기서는 N개를 전부 읽어 index_select 한다.
#   블록이 3개뿐이라 이 비용은 무시할 수 있다.
# =============================================================================

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

# Isaac Lab 스태킹 태스크의 검증된 구현을 재사용한다.
#   ee_frame_pos / ee_frame_quat : Mimic 의 get_robot_eef_pose 가 읽는 값
#   gripper_pos                  : 평행 그리퍼/석션을 모두 처리해 준다
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import (  # noqa: F401
    ee_frame_pos,
    ee_frame_quat,
    gripper_pos,
)

from ..scene_assets import block_name
from ..spec import SPEC

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# 타깃 선택 — 나머지 전부가 이 두 함수 위에 올라간다
# -----------------------------------------------------------------------------
def _index_buffer(env: "ManagerBasedRLEnv", attr: str) -> torch.Tensor:
    """리셋 이벤트가 채워 두는 (num_envs,) long 버퍼. 없으면 0 으로 만든다.

    ★ 0 으로 때우는 것이 맞다 — 예외를 던지면 **환경 생성 자체가 실패한다.**
      ObservationManager 는 __init__ 에서 관측 차원을 알아내려고 항목을 한 번
      호출하는데, 그 시점은 어떤 리셋 이벤트도 돌기 전이다.

    "이벤트를 빼먹으면 전부 0번 블록이 타깃" 이라는 위험은 남는다. 그건 여기서
    막지 않고 스모크 테스트(check_isaaclab.py 의 target_ids 확인)와 데이터 점검
    (convert_hdf5_to_rlds.py 의 지시문 분포)에서 잡는다 — 둘 다 "지시문이 한
    종류뿐" 이라는 형태로 곧바로 드러난다.
    """
    buf = getattr(env, attr, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, attr, buf)
    return buf


def target_block_index(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """각 env 의 타깃 블록 인덱스. Shape (num_envs,) long."""
    return _index_buffer(env, "_vla_target_block")


def target_slot_index(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """각 env 의 타깃 슬롯 인덱스. Shape (num_envs,) long."""
    return _index_buffer(env, "_vla_target_slot")


def all_block_poses(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """블록 전체의 pos+quat (env 로컬). Shape (num_envs, NUM_BLOCKS, 7)."""
    poses = []
    for b in range(SPEC.NUM_BLOCKS):
        obj: RigidObject = env.scene[block_name(b)]
        pos = obj.data.root_pos_w - env.scene.env_origins
        poses.append(torch.cat([pos, obj.data.root_quat_w], dim=-1))
    return torch.stack(poses, dim=1)


def _gather_target(env: "ManagerBasedRLEnv", per_block: torch.Tensor) -> torch.Tensor:
    """(num_envs, NUM_BLOCKS, D) 에서 env 별 타깃 블록의 행만 뽑는다."""
    idx = target_block_index(env).view(-1, 1, 1).expand(-1, 1, per_block.shape[-1])
    return per_block.gather(1, idx).squeeze(1)


def target_block_pose(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """타깃 블록의 pos+quat (env 로컬). Shape (num_envs, 7).

    Mimic 의 get_object_poses 가 이 값을 4x4 행렬로 바꿔 쓴다. env 원점을 빼는
    것이 중요하다 — 월드 좌표를 그대로 쓰면 env 격자 간격이 궤적 변환에 섞여
    들어가 증강 결과가 전부 어긋난다.
    """
    return _gather_target(env, all_block_poses(env))


def target_block_lin_vel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """타깃 블록 선속도. Shape (num_envs, 3)."""
    vels = torch.stack(
        [env.scene[block_name(b)].data.root_lin_vel_w for b in range(SPEC.NUM_BLOCKS)],
        dim=1,
    )
    return _gather_target(env, vels)


def target_pocket_pose(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """타깃 포켓의 pos+quat (env 로컬). Shape (num_envs, 7).

    포켓은 정지 구조물이라 씬에서 읽을 필요 없이 스펙에서 만들어 낸다.
    쿼터니언은 항등 — 포켓 장축이 x축과 나란하다는 것이 yaw 정렬의 기준이다.

    Mimic 의 두 번째 서브태스크(배치)가 이 pose 를 기준 오브젝트로 쓴다.
    타깃 슬롯이 env 마다 다르므로, 이게 없으면 증강된 궤적이 전부 같은 포켓으로
    간다 — 언어 조건이 데이터에서 사라진다.
    """
    centers = torch.tensor(
        [SPEC.pocket_center(i) for i in range(len(SPEC.SLOT_NAMES))],
        device=env.device,
        dtype=torch.float32,
    )                                                     # (num_slots, 2)
    xy = centers[target_slot_index(env)]                  # (num_envs, 2)
    z = torch.full_like(xy[:, :1], SPEC.TABLE_HEIGHT + 0.5 * SPEC.BLOCK_SIZE[2])
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    return torch.cat([xy, z, quat], dim=-1)


def target_ids(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[타깃 블록 idx, 타깃 슬롯 idx]. Shape (num_envs, 2) float.

    지시문 문자열을 만드는 재료다. 관측으로 내보내는 이유: 롤아웃 워커와
    HDF5→RLDS 변환이 **같은 소스**에서 지시문을 만들어야 하기 때문이다.
    두 곳에서 따로 만들면 SFT 데이터의 지시문과 RFT 롤아웃의 지시문이 어긋난다.
    """
    return torch.stack(
        [target_block_index(env).float(), target_slot_index(env).float()], dim=-1
    )


# -----------------------------------------------------------------------------
# 로봇 기준 상태
# -----------------------------------------------------------------------------
def target_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """타깃 블록 위치를 로봇 베이스 좌표계로. Shape (num_envs, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    pos_w = target_block_pose(env)[:, :3] + env.scene.env_origins
    pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], pos_w
    )
    return pos_b


def target_height_above_table(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """타깃 블록이 테이블 상면에서 얼마나 떠 있는지. Shape (num_envs, 1)."""
    z = target_block_pose(env)[:, 2]
    return (z - SPEC.TABLE_HEIGHT).unsqueeze(-1)


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
# 파지 판정
# -----------------------------------------------------------------------------
def _eef_pos(env: "ManagerBasedRLEnv", ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :] - env.scene.env_origins


def target_grasped(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """타깃 블록을 실제로 쥐고 있는가. Shape (num_envs,) bool.

    Isaac Lab 스태킹의 object_grasped 를 쓰지 않는 이유: 그쪽은 SceneEntityCfg 로
    오브젝트를 고정 지정하는데, 우리는 타깃이 env 마다 다르다.
    판정 내용 자체는 같다 — eef-블록 거리 + 그리퍼가 닫혀 있는가.

    ★ abs() 가 반드시 필요하다. Isaac Lab 의 gripper_pos 는 두 번째 손가락의
      부호를 뒤집어 [f1, -f2] 를 돌려준다 (LIBERO/robosuite 의 Panda qpos 규약과
      맞추기 위한 것이고, proprio 는 그 규약이 맞다). 그런데 Franka 의 두 손가락은
      둘 다 0~+0.04 로 대칭 이동하므로 그냥 sum() 하면 f1 - f2 ≈ 0 이 되어
      **항상 "닫힘"** 으로 판정된다 → 빈손으로 블록 옆을 지나가도 파지로 잡힌다.
      (반대 방향 버그를 개정 전 placed_signal 에서 이미 한 번 겪었다)
    """
    dist = torch.linalg.norm(
        target_block_pose(env)[:, :3] - _eef_pos(env, ee_frame_cfg), dim=-1
    )
    closed = gripper_pos(env, robot_cfg).abs().sum(dim=-1) < SPEC.GRIPPER_OPEN_QPOS_SUM
    return (dist < SPEC.GRASP_DISTANCE_THRESHOLD) & closed


# -----------------------------------------------------------------------------
# 서브태스크 종료 시그널 (Mimic --auto 어노테이션용)
# -----------------------------------------------------------------------------
# 서브태스크는 2개다 (개정 §4-2). 가능한 한 적게 나눈다 — 구간을 이어붙일 때마다
# 스티칭이 늘고 동작이 덜 부드러워지며 실패 데모가 늘어난다.
#   1. 박스에서 블록 파지 + 벽 위로 리프트   → grasp_lift
#   2. 포켓에 정렬해 안착                    → 마지막이므로 종료 시그널 없음
def grasp_lift_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    lift_height: float = SPEC.LIFT_HEIGHT_THRESHOLD,
) -> torch.Tensor:
    """타깃 블록을 쥔 채 박스 벽 위로 들어올렸는가. Shape (num_envs,) bool.

    ★ 경계를 파지 직후가 아니라 **벽 위로 뺀 뒤**에 두는 것이 핵심이다
      (개정 §4-2). 플래너가 이어붙일 자유공간 구간에 박스 벽이 걸리면
      SkillGen 생성이 그 지점에서 실패한다.

    같은 조건이 성공 술어의 grasped_during_lift 항(래치)에도 쓰인다 —
    "들어올려서 옮겼다"와 "벽 모서리로 끌어 넘겼다"를 가르는 항이다.
    """
    grasped = target_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)
    height = target_height_above_table(env).squeeze(-1)
    return grasped & (height > lift_height)


# -----------------------------------------------------------------------------
# 서브태스크 "시작" 시그널 (SkillGen 전용)
# -----------------------------------------------------------------------------
# SkillGen 은 시작 경계가 없으면 생성이 실패한다. MimicGen 만 쓸 때는 무시되므로
# 두 방식을 분기 없이 같은 환경으로 지원할 수 있다.
def grasp_start_signal(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    start_distance: float = SPEC.APPROACH_START_DISTANCE,
) -> torch.Tensor:
    """eef 가 타깃 블록에 충분히 접근했는가 = 파지 접촉 구간의 시작.

    이 지점 이전(자유공간 접근)은 cuRobo 가 계획하고, 이후(정렬·하강·파지)는
    사람 데모를 재생한다. 그래서 임계값이 실제 접촉 거리보다 넉넉해야 한다.
    """
    dist = torch.linalg.norm(
        target_block_pose(env)[:, :3] - _eef_pos(env, ee_frame_cfg), dim=-1
    )
    return dist < start_distance


def place_start_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    start_radius: float = SPEC.PLACE_START_RADIUS,
) -> torch.Tensor:
    """블록을 든 채로 타깃 포켓 위에 도달했는가 = 배치 접촉 구간의 시작.

    "든 채로" 조건이 중요하다. 이게 없으면 블록이 우연히 포켓 근처에 있는
    에피소드에서 시작 신호가 첫 프레임부터 켜져 운반 구간이 통째로 사라진다.
    """
    grasped = target_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)
    dxy = torch.linalg.norm(
        target_block_pose(env)[:, :2] - target_pocket_pose(env)[:, :2], dim=-1
    )
    return grasped & (dxy < start_radius)


# -----------------------------------------------------------------------------
# 성공 술어 (sparse binary) — 개정 §2-5
# -----------------------------------------------------------------------------
#   success = in_pocket AND yaw_aligned AND settled AND grasped_during_lift
#
# dense shaping 은 넣지 않는다. 목표까지의 거리 항은 "밀기/끌기"를 적극적으로
# 보상해 파지 없는 정책으로 수렴시킨다.
def yaw_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """타깃 블록의 yaw 오차를 대칭 차수로 접은 값 [rad]. Shape (num_envs,).

    2:1 직육면체는 180도 돌려도 같은 자세다. 접지 않으면 물리적으로 완벽히
    안착한 절반의 에피소드가 실패로 잡힌다.
    """
    quat = target_block_pose(env)[:, 3:7]
    _, _, yaw = math_utils.euler_xyz_from_quat(quat)
    period = 2.0 * math.pi / SPEC.BLOCK_SYMMETRY_ORDER
    folded = torch.remainder(yaw + 0.5 * period, period) - 0.5 * period
    return folded.abs()


def yaw_error_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """yaw 오차를 관측으로. Shape (num_envs, 1).

    ★ 진단 채널이다. 클리어런스를 조였을 때 곡선이 주저앉는 원인이
      "포켓에 못 넣는 것"인지 "각도를 못 맞추는 것"인지 가른다.
      가중치 0 짜리 보상 항으로 두면 로그에도 0 만 남는다 (rewards.py 머리말).
    """
    return yaw_error(env).unsqueeze(-1)


def grasped_during_lift(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """에피소드 중 한 번이라도 "쥔 채로 벽 위까지 들어올렸는가". 래치.

    박스 벽이 있어 밀어내기는 어렵지만, 벽 모서리로 끌어 넘기는 해는 여전히
    가능하다. 이 항이 그 pushcut 을 막는다. 래치라서 놓은 뒤에도 유지된다 —
    성공 시점에는 이미 그리퍼가 열려 있기 때문이다.

    ★ 래치는 판정할 때마다 갱신된다. 성공 판정이 매 스텝 호출되므로 별도의
      업데이트 훅이 필요 없다. 리셋에서 지우는 것은 events.reset_episode_buffers.
    """
    latch = getattr(env, "_vla_grasp_lift_latch", None)
    if latch is None or latch.shape[0] != env.num_envs:
        latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._vla_grasp_lift_latch = latch
    latch |= grasp_lift_signal(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)
    return latch


def placed_signal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """성공 조건 원본. Shape (num_envs,) bool.

    RFT 의 0/1 보상과 성공 판정 termination 이 모두 이 함수를 부른다 —
    "보상과 평가 기준이 같은 코드에서 나온다"는 것이 중요하다. 따로 구현하면
    RFT 가 최적화하는 것과 우리가 측정하는 것이 미묘하게 달라진다.
    """
    clearance = getattr(env.cfg, "pocket_clearance", SPEC.POCKET_CLEARANCE)
    block = target_block_pose(env)
    pocket = target_pocket_pose(env)

    # (1) 포켓 안 — xy 공차는 클리어런스에서 파생된다 (split 마다 함께 조여진다).
    dxy = torch.linalg.norm(block[:, :2] - pocket[:, :2], dim=-1)
    in_pocket = dxy < SPEC.pocket_xy_tolerance(clearance)
    # 레일 위에 걸쳐 있는 상태와 구분한다. xy 만 보면 얹혀 있어도 성공이 된다.
    in_pocket &= block[:, 2] < SPEC.pocket_seat_z_max()

    # (2) yaw 정렬 — 대칭 차수를 고려한다.
    yaw_aligned = yaw_error(env) < SPEC.SUCCESS_YAW_TOLERANCE

    # (3) 정지 — 굴러가거나 튀는 중에 성공이 뜨는 것을 막는다.
    settled = (
        torch.linalg.norm(target_block_lin_vel(env), dim=-1)
        < SPEC.SUCCESS_LIN_VEL_THRESHOLD
    )

    # (4) 리프트 구간 동안 파지 — 밀기·끌기 배제.
    lifted = grasped_during_lift(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)

    return in_pocket & yaw_aligned & settled & lifted

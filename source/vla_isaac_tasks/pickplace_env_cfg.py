# =============================================================================
# vla_isaac_tasks/pickplace_env_cfg.py
#
# 기본 태스크 환경 <VlaPick-v0> — 텔레옵 수집 · 재생 · 최종 정책 평가에 쓴다.
# (Mimic 증강용 환경 2종은 pickplace_mimic_env_cfg.py 에서 이 클래스를 상속한다)
#
# 태스크: 형상이 랜덤한 자재를 집어 목표 영역으로 옮긴다.
#
# ★ 이 파일의 모든 수치는 configs/vla_spec.py 에서 온다.
#   직접 상수를 박지 말 것 — SFT 데이터를 만든 설정이 곧 RFT 설정이어야 하고,
#   두 곳에 같은 숫자를 적어 두면 반드시 한쪽만 바뀐다.
# =============================================================================

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg, Se3GamepadCfg, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from . import mdp
from .materials import make_material_spawner
from .spec import SPEC

##
# 로봇 프리셋
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# =============================================================================
# 씬
# =============================================================================
@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Franka + 형상 랜덤 자재 + 목표 마커 + 3인칭 카메라."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.4, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.62, 0.58, 0.52), roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.9
            ),
        ),
        # 상면이 z=TABLE_HEIGHT 에 오도록 두께의 절반만큼 내린다.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, SPEC.TABLE_HEIGHT - 0.025)
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1800.0),
    )

    # 자재 — env 별로 6종 중 하나가 배정된다 (materials.py 참조).
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=make_material_spawner(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.1, SPEC.TABLE_HEIGHT + 0.04), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # 목표 영역 — 시각 표시 전용. 충돌체가 없어야 자재가 걸리지 않는다.
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.CylinderCfg(
            radius=SPEC.GOAL_REGION_RADIUS,
            height=0.002,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.85, 0.35), emissive_color=(0.05, 0.30, 0.10)
            ),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(*SPEC.GOAL_REGION_CENTER, SPEC.TABLE_HEIGHT + 0.002)
        ),
    )

    # 엔드이펙터 프레임 — Mimic 의 get_robot_eef_pose 가 이 값을 쓴다.
    # 링크 이름과 오프셋은 전부 configs/vla_spec.py 의 embodiment 섹션에서 온다.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SPEC.BASE_LINK_NAME}",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SPEC.EEF_BODY_NAME}",
                name="end_effector",
                offset=OffsetCfg(pos=SPEC.EEF_FRAME_OFFSET),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SPEC.FINGER_LINK_NAMES[0]}",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=SPEC.FINGER_FRAME_OFFSET),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SPEC.FINGER_LINK_NAMES[1]}",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=SPEC.FINGER_FRAME_OFFSET),
            ),
        ],
    )

    # 3인칭 단일 카메라. 손목캠은 넣지 않는다 (계획서 §2-1: 단일 뷰 스펙).
    # TiledCamera 를 쓰는 이유: 병렬 env 렌더가 CameraCfg 보다 크게 빠르다.
    # Phase 2 의 야간 생성과 Phase 4 롤아웃이 모두 렌더 바운드라 이 차이가 일정에 직결된다.
    table_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=SPEC.IMAGE_HEIGHT,
        width=SPEC.IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=SPEC.CAMERA_FOCAL_LENGTH,
            focus_distance=SPEC.CAMERA_FOCUS_DISTANCE,
            horizontal_aperture=SPEC.CAMERA_HORIZONTAL_APERTURE,
            clipping_range=SPEC.CAMERA_CLIPPING_RANGE,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=SPEC.CAMERA_POS,
            rot=SPEC.CAMERA_ROT,
            convention=SPEC.CAMERA_CONVENTION,
        ),
    )


# =============================================================================
# MDP
# =============================================================================
@configclass
class ActionsCfg:
    """7차원 액션 = IK 상대 델타 포즈(6) + 이진 그리퍼(1).

    LIBERO/OpenVLA 의 액션 레이아웃과 정확히 같다 (configs/vla_spec.py 참조).
    이 일치 덕분에 openvla-oft 와 SimpleVLA-RL 의 모델 코드를 고칠 필요가 없다.
    """

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[SPEC.ARM_JOINT_REGEX],
        body_name=SPEC.EEF_BODY_NAME,
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=SPEC.IK_ACTION_SCALE,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=SPEC.EEF_BODY_OFFSET
        ),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[SPEC.GRIPPER_JOINT_REGEX],
        open_command_expr={SPEC.GRIPPER_COMMAND_REGEX: SPEC.GRIPPER_OPEN_QPOS},
        close_command_expr={SPEC.GRIPPER_COMMAND_REGEX: SPEC.GRIPPER_CLOSE_QPOS},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """VLA 입력 + Mimic 이 요구하는 상태.

        concatenate_terms=False 가 중요하다. Mimic 의 get_robot_eef_pose 가
        obs_buf["policy"]["eef_pos"] 처럼 **키로** 접근하기 때문에, 이어 붙이면
        Mimic 파이프라인 전체가 KeyError 로 죽는다.
        """

        # --- VLA 가 실제로 보는 것 ---
        table_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg(SPEC.CAMERA_SENSOR_NAME),
                "data_type": "rgb",
                # normalize=False → uint8 [0,255] 로 받는다.
                # OpenVLA 의 프로세서가 자체 정규화를 하므로 여기서 하면 이중 정규화가 된다.
                "normalize": False,
            },
        )

        # --- Mimic 필수 (get_robot_eef_pose / action_to_target_eef_pose) ---
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # --- proprio (스펙에서 켤 때만 VLA 에 전달된다) ---
        proprio = ObsTerm(func=mdp.proprio_state)

        # --- 상태 기반 디버깅/평가용 ---
        object_pose = ObsTerm(func=mdp.object_pose_in_env_frame)
        object_pos_b = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        goal_pos = ObsTerm(func=mdp.goal_position_in_env_frame)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Mimic/SkillGen 자동 어노테이션(--auto)용 휴리스틱 시그널.

        계획서 §Phase1-6. 이게 없으면 annotate_demos.py 를 수동(B/N/S 키)으로
        돌려야 하는데, 데모를 다시 수집할 때마다 반복 비용이 든다.

        서브태스크는 2개로 압축했다 (§Phase1-5): 스티칭 구간이 적을수록
        생성 성공률이 높다.

        ★ 종료 시그널 1개 + 시작 시그널 2개를 모두 등록한다.
          MimicGen 은 종료 경계만 쓰지만, SkillGen 은 **시작 경계가 필수**다
          (없으면 데이터 생성이 실패한다). 시작 시그널은 SkillGen 을 쓰지 않을 때
          그냥 무시되므로, 둘을 다 두면 분기 코드 없이 양쪽을 지원할 수 있다.
          Day 1 에 cuRobo 가 막혀 MimicGen 으로 후퇴하더라도 이 환경을 그대로 쓴다.
        """

        # --- 종료 경계 (MimicGen + SkillGen 공통) ---
        grasp_lift = ObsTerm(
            func=mdp.grasp_lift_signal,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # --- 시작 경계 (SkillGen 전용) ---
        # 이 지점 이전의 자유공간은 cuRobo 가 계획하고, 이후는 사람 데모를 재생한다.
        grasp_start = ObsTerm(
            func=mdp.grasp_start_signal,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        place_start = ObsTerm(
            func=mdp.place_start_signal,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    # 자재 크기 랜덤화 — PhysX 제약상 prestartup 에서만 가능하다.
    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": {"x": (0.85, 1.25), "y": (0.85, 1.25), "z": (0.85, 1.25)},
        },
    )

    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_material = EventTerm(
        func=mdp.randomize_material_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")},
    )

    reset_hold_counter = EventTerm(
        func=mdp.reset_success_hold_counter,
        mode="reset",
        params={},
    )


@configclass
class RewardsCfg:
    """RFT 의 0/1 보상. 계획서 §2-2 대로 shaping 없이 이진만 쓴다."""

    success = RewTerm(
        func=mdp.success_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    # 진단 전용 — weight=0.0 이므로 학습 신호에 들어가지 않고 로그에만 남는다.
    # 커브가 평평할 때 "접근·파지까지는 되는데 배치를 못 하는 것"인지 구분한다.
    grasp_lift_diag = RewTerm(
        func=mdp.grasp_lift_progress,
        weight=0.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.object_dropped, params={"object_cfg": SceneEntityCfg("object")}
    )
    success = DoneTerm(
        func=mdp.task_success,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """<VlaPick-v0> — 텔레옵 수집 / 재생 / 정책 평가용."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        # ★ 반드시 False.
        #   MultiAssetSpawnerCfg 는 env 마다 다른 지오메트리를 만들기 때문에
        #   PhysX 복제 최적화를 쓸 수 없다. True 로 두면 모든 env 가 같은 형상이
        #   되는데 **에러가 나지 않아서**, 형상 랜덤화가 되고 있다고 착각한 채로
        #   데이터를 전부 만들어 버리게 된다.
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = SPEC.DECIMATION
        self.episode_length_s = SPEC.EPISODE_LENGTH_S
        self.sim.dt = SPEC.SIM_DT
        self.sim.render_interval = SPEC.DECIMATION

        # 파지 안정성에 직결되는 솔버 설정. 자재가 미끄러지면 Mimic 생성 성공률이
        # 떨어지고, 그러면 Day 1 야간 배치가 헛돈다.
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Isaac Lab 의 그리퍼 상태 판정 헬퍼가 환경 cfg 에서 찾는 값들.
        # ★ 세 개가 한 세트다. 하나만 넣으면 다음 것에서 죽는다:
        #     gripper_joint_names 없음 → NotImplementedError: Cannot find gripper_joint_names
        #     gripper_open_val 없음    → AttributeError: no attribute 'gripper_open_val'
        #     gripper_threshold 없음   → AttributeError: no attribute 'gripper_threshold'
        #   (이 리비전 isaaclab 0.54.4 의 stack.mdp.observations.object_grasped 가
        #    hasattr 로 gripper_joint_names 만 확인하고 나머지 둘은 그냥 참조한다.)
        # 액션 cfg 의 joint_names 는 정규식이라 그걸로는 대체되지 않는다.
        self.gripper_joint_names = list(SPEC.GRIPPER_JOINT_NAMES)
        self.gripper_open_val = SPEC.GRIPPER_OPEN_QPOS
        self.gripper_threshold = SPEC.GRIPPER_STATUS_THRESHOLD

        # ---------------------------------------------------------------
        # 텔레옵 장치 (계획서 §2-4: SpaceMouse 없이 키보드/게임패드)
        # ---------------------------------------------------------------
        # ★ 여기를 None 으로 두면 record_demos.py 가 죽는다.
        #   그 스크립트의 setup_teleop_device() 는 이렇게 판정한다:
        #       if hasattr(env_cfg, "teleop_devices") and \
        #          args_cli.teleop_device in env_cfg.teleop_devices.devices:
        #   teleop_devices=None 이면 hasattr 는 True 를 돌려주고, 곧바로
        #   None.devices 에 접근해 AttributeError 가 난다.
        #   "설정 안 함" 이 아니라 "속성은 있는데 None" 이라는 상태가 문제다.
        #
        # ★ 게임패드는 여기에 등록하지 않으면 아예 못 쓴다.
        #   위 판정이 실패했을 때의 폴백 경로는 keyboard 와 spacemouse 만 만들고,
        #   그 외 값이면 에러를 찍고 종료한다. 계획서 §2-4 가 게임패드를
        #   "데모 품질이 크게 다르다" 며 강력 권장하므로 반드시 등록해 둔다.
        self.teleop_devices = DevicesCfg(
            devices={
                # 키보드: 감도를 낮게 잡는다. 크면 한 번 누를 때마다 팔이 튀어
                # 궤적에 계단이 생기고, 그건 Mimic 생성 성공률을 직접 깎는다.
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                # 게임패드: 아날로그 스틱이라 연속 조작이 되므로 감도를 키워도
                # 궤적이 매끄럽다. dead_zone 은 스틱 중립에서의 드리프트를 막는다 —
                # 이게 없으면 손을 떼도 팔이 천천히 흐르면서 "일시정지 없는 데모"
                # 체크리스트를 만족시키기 어려워진다.
                "gamepad": Se3GamepadCfg(
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.6,
                    dead_zone=0.01,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class PickPlaceVisuomotorEnvCfg(PickPlaceEnvCfg):
    """카메라 관측을 반드시 켜야 하는 변형. `--enable_cameras` 와 함께 쓴다.

    현재는 기본 cfg 도 카메라를 포함하고 있어 내용이 같지만, 클래스를 분리해 둔다:
    나중에 상태 기반 생성(계획서 §3-3 선택지 B)으로 전환할 때 기본 cfg 에서
    카메라를 떼어내는 것이 이 구조라면 한 줄로 끝난다.
    """

    pass

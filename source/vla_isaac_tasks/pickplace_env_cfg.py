# =============================================================================
# vla_isaac_tasks/pickplace_env_cfg.py
#
# 기본 태스크 환경 <VlaPlace-v0> — 텔레옵 수집 · 재생 · 최종 정책 평가에 쓴다.
# (Mimic 증강용 환경 2종은 pickplace_mimic_env_cfg.py 에서 이 클래스를 상속한다)
#
# 태스크 (개정 §2): 얕은 박스 안의 블록 3개 중 **지시문이 지정한 하나**를 꺼내
#                   트레이의 **지시문이 지정한 포켓**에 안착시킨다.
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
from .scene_assets import box_wall, make_block_cfg, tray_rail
from .spec import SPEC

##
# 로봇 프리셋
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# 아래 씬은 블록 3개를 필드로 명시 선언한다 (InteractiveSceneCfg 는 선언된
# 필드만 씬에 올린다). 개수를 바꾸려면 필드도 함께 늘려야 한다.
assert SPEC.NUM_BLOCKS == 3, (
    f"SPEC.NUM_BLOCKS={SPEC.NUM_BLOCKS} 인데 씬에는 블록 필드가 3개뿐이다. "
    "PickPlaceSceneCfg 의 block_* 필드를 함께 늘리고, BLOCK_COLORS / "
    "BLOCK_ATTRS 도 같이 맞출 것."
)


# =============================================================================
# 씬
# =============================================================================
@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Franka + 소스 박스(블록 3개) + 맞춤 포켓 트레이 + 3인칭 카메라."""

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

    # --- 블록 3개 (색만 다르고 형상·크기는 동일) ---
    block_0: RigidObjectCfg = make_block_cfg(0)
    block_1: RigidObjectCfg = make_block_cfg(1)
    block_2: RigidObjectCfg = make_block_cfg(2)

    # --- 소스 박스: 벽 4장. 벽이 있어야 "밀어내기" 해가 막힌다 ---
    box_x_pos = box_wall("box_x_pos")
    box_x_neg = box_wall("box_x_neg")
    box_y_pos = box_wall("box_y_pos")
    box_y_neg = box_wall("box_y_neg")

    # --- 타깃 트레이: 정사각 하나, 레일 4장 ---
    tray_x_pos = tray_rail("tray_x_pos")
    tray_x_neg = tray_rail("tray_x_neg")
    tray_y_pos = tray_rail("tray_y_pos")
    tray_y_neg = tray_rail("tray_y_neg")

    # 엔드이펙터 프레임 — Mimic 의 get_robot_eef_pose 가 이 값을 쓴다.
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

    # 3인칭 단일 카메라. 손목캠은 넣지 않는다 (단일 뷰 스펙).
    # TiledCamera 를 쓰는 이유: 병렬 env 렌더가 CameraCfg 보다 크게 빠르다.
    # ★ 박스와 타깃 트레이가 **둘 다** 화각에 들어와야 한다.
    #   dump_obs_reference.py 의 PNG 로 확인할 것 (스펙의
    #   assert_workspace_visible() 이 렌더 없이 먼저 걸러 준다).
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

        # --- 타깃 (Mimic 의 get_object_poses + 지시문 생성) ---
        # ★ target_ids 가 지시문의 유일한 출처다. 롤아웃 워커와 RLDS 변환이
        #   같은 값에서 문장을 만들어야 SFT 와 RFT 의 지시문이 어긋나지 않는다.
        target_pose = ObsTerm(func=mdp.target_block_pose)
        tray_pose = ObsTerm(func=mdp.target_tray_pose)
        target_ids = ObsTerm(func=mdp.target_ids)

        # --- 진단 채널 (정책은 안 본다. 롤아웃 워커가 평균만 보고한다) ---
        yaw_err = ObsTerm(func=mdp.yaw_error_obs)

        # --- 상태 기반 디버깅/평가용 ---
        target_pos_b = ObsTerm(func=mdp.target_position_in_robot_root_frame)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Mimic/SkillGen 자동 어노테이션(--auto)용 휴리스틱 시그널.

        서브태스크는 2개다 (개정 §4-2). 가능한 한 적게 나눈다 — 구간을 선형
        보간/플래닝으로 이어붙이므로 서브태스크가 많을수록 스티칭이 늘고
        동작이 덜 부드러워지며 실패 데모가 늘어난다.

        ★ 종료 시그널 1개 + 시작 시그널 2개를 모두 등록한다.
          MimicGen 은 종료 경계만 쓰지만, SkillGen 은 **시작 경계가 필수**다.
          시작 시그널은 MimicGen 에서 그냥 무시되므로, 둘을 다 두면 분기 코드
          없이 양쪽을 지원할 수 있다.
        """

        # --- 종료 경계 (MimicGen + SkillGen 공통) ---
        # 파지 직후가 아니라 "블록을 들어 박스 벽 위로 뺀 뒤"가 경계다.
        grasp_lift = ObsTerm(
            func=mdp.grasp_lift_signal,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # --- 시작 경계 (SkillGen 전용) ---
        grasp_start = ObsTerm(
            func=mdp.grasp_start_signal,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        place_start = ObsTerm(
            func=mdp.place_start_signal,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """리셋. **랜덤 샘플링이 없다** (개정 §3).

    블록 배치·타깃 블록·타깃 슬롯은 전부 초기 상태 뱅크에서 인덱스로 꺼낸다.
    여기에 pose 랜덤화 항을 추가하면 "인덱스로 s₀ 를 지정한다"는 전제가 깨지고
    GRPO advantage 가 다시 노이즈가 된다.
    """

    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # 팔 초기 자세까지 뱅크에 넣지는 않았다. 관절 노이즈는 씬 난이도가
            # 아니라 시작 자세의 미세 변동이라 그룹 내 분산에 거의 기여하지
            # 않는다. 완전 결정론이 필요하면 이 범위를 0 으로 둘 것.
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_scene = EventTerm(
        func=mdp.reset_scene_from_bank,
        mode="reset",
        params={"bank_name": "train"},
    )
    reset_buffers = EventTerm(func=mdp.reset_episode_buffers, mode="reset", params={})


@configclass
class RewardsCfg:
    """RFT 의 0/1 보상. shaping 없이 이진만 쓴다 (개정 §2-5)."""

    success = RewTerm(
        func=mdp.success_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    # 진단은 여기 두지 않는다. RewardManager 는 로그에 남기는 값에도 weight 를
    # 곱하므로 weight=0 인 항은 **로그도 0** 이다 — 진단이 되는 것처럼 보이지만
    # 아무것도 알려주지 않는다. 진단 채널은 관측(yaw_err, subtask_terms/grasp_lift)
    # 이고, 롤아웃 워커가 그 평균을 step 응답에 실어 보낸다.


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(func=mdp.object_dropped, params={})
    success = DoneTerm(
        func=mdp.task_success,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """<VlaPlace-v0> — 텔레옵 수집 / 재생 / 정책 평가용."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        # 모든 env 의 지오메트리가 같으므로 PhysX 복제 최적화를 쓸 수 있다.
        # (형상 랜덤 자재를 쓰던 시절에는 False 여야 했다 — 그 제약이 사라졌다)
        replicate_physics=True,
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

        # 파지 안정성에 직결되는 솔버 설정. 블록이 미끄러지면 Mimic 생성
        # 성공률이 떨어지고, 그러면 야간 배치가 헛돈다.
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
        # 텔레옵 장치
        # ---------------------------------------------------------------
        # ★ 여기를 None 으로 두면 record_demos.py 가 죽는다.
        #   그 스크립트의 setup_teleop_device() 는 이렇게 판정한다:
        #       if hasattr(env_cfg, "teleop_devices") and \
        #          args_cli.teleop_device in env_cfg.teleop_devices.devices:
        #   teleop_devices=None 이면 hasattr 는 True 를 돌려주고, 곧바로
        #   None.devices 에 접근해 AttributeError 가 난다.
        #
        # ★ 게임패드는 여기에 등록하지 않으면 아예 못 쓴다 (폴백 경로는
        #   keyboard/spacemouse 만 만든다).
        self.teleop_devices = DevicesCfg(
            devices={
                # 키보드: 감도를 낮게 잡는다. 크면 한 번 누를 때마다 팔이 튀어
                # 궤적에 계단이 생긴다.
                # ★ SkillGen 전제에서는 계단이 transit 구간에선 무의미하다
                #   (플래너가 새로 깐다). 품질이 중요한 건 접촉 구간 두 곳뿐이다.
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                # 게임패드: 아날로그 스틱이라 연속 조작이 되므로 감도를 키워도
                # 궤적이 매끄럽다. dead_zone 은 스틱 중립에서의 드리프트를 막는다.
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
    나중에 상태 기반 생성으로 전환할 때 기본 cfg 에서 카메라를 떼어내는 것이
    이 구조라면 한 줄로 끝난다.
    """

    pass

# =============================================================================
# ⚠ 보류 중 — gym 에 등록되어 있지 않다. import 하면 깨진다.
#
#   태스크 개정(§8)으로 변형체는 **후속 확장**으로 밀렸고, 시작점도 바뀌었다:
#   체적 FEM 큐브가 아니라 **박막(2D)** 부터 한다. 사전학습에 천/케이블 데이터가
#   미량 존재하고, 체적 FEM보다 훨씬 싸기 때문이다.
#
#   또 이 파일은 강체 씬(PickPlaceSceneCfg)을 상속하는데, 그 씬은 이제 블록 3개 +
#   박스 + 포켓 트레이다. 여기 있는 단일 `object` 전제와 mdp/deformable.py 의
#   목표영역(GOAL_REGION_*) 전제가 모두 스펙에서 사라졌다.
#
#   되살릴 때 할 일:
#     - 성공 술어를 새 술어(in_tray/settled/grasped_during_lift)로 포팅
#     - 박막 지오메트리로 교체 (얇은 메시는 convex hull cooking 이 GPU 에서
#       실패해 CPU 폴백이 나고, 그러면 변형체 충돌이 아예 동작하지 않는다.
#       그 경고 로그를 반드시 확인할 것)
#     - vla_isaac_tasks/__init__.py 에 등록 추가
#
# 아래는 개정 전 코드를 그대로 남긴 것이다. 참고용.
# =============================================================================
# vla_isaac_tasks/deformable_env_cfg.py
#
# <VlaPick-Deformable-v0> — 변형체(PhysX FEM) 자재 환경.
#
# 위치: Phase 4b 스트레치. 강체 RFT 개선 커브를 확보한 뒤에만 손댄다.
#       Day 1 에는 scripts/test_deformable_grasp.py 로 "그리퍼가 이걸 잡을 수
#       있는가" 만 확인하고 파라미터를 확보한 뒤 넘어간다 (계획서 §Phase1-7).
#
# ★ 강체 환경과 관측·액션 스펙이 완전히 동일해야 한다.
#   그래야 강체로 SFT 한 정책을 그대로 이 환경에 태울 수 있고, 그 도메인 갭을
#   RL 이 메우는 것이 이 프로젝트의 스토리다 (계획서 §2-3).
#   달라지는 것은 오브젝트의 물리 표현과 그에 딸린 판정 함수뿐이다.
# =============================================================================

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .mdp import deformable as deform_mdp
from .pickplace_env_cfg import PickPlaceEnvCfg, PickPlaceSceneCfg
from .spec import SPEC


@configclass
class DeformableSceneCfg(PickPlaceSceneCfg):
    """강체 씬에서 자재만 FEM 변형체로 교체한다."""

    object: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.07, 0.05, 0.05),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                # ---- Day 1 예비 테스트에서 확보할 파라미터들 ----
                # 아래는 "잡히기는 하는" 출발점이다. 실제 값은
                # scripts/test_deformable_grasp.py 로 스윕해 확정하고 여기에 반영할 것.
                rest_offset=0.0,
                contact_offset=0.001,
                # 해상도가 높을수록 사실적이지만 시뮬이 급격히 느려진다.
                # RFT 롤아웃 비용에 직결되므로 낮게 시작한다.
                simulation_hexahedral_resolution=8,
                solver_position_iteration_count=20,
                # 자기 충돌은 비싸다. 자루형 자재를 접히게 하려면 필요하지만
                # 3~4일 일정에서는 끄고 간다.
                self_collision=False,
                vertex_velocity_damping=0.005,
                sleep_damping=1.0,
                sleep_threshold=0.05,
                settling_threshold=0.1,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                # 낮을수록 물렁하다. 너무 낮으면 파지 중 관통·발산이 일어나므로
                # 계획서 §6 의 fallback ("변형 강도 축소(준강체) 후 점진 증가")대로
                # 단단한 쪽에서 시작해 내려온다.
                youngs_modulus=5.0e5,
                poissons_ratio=0.4,
                dynamic_friction=1.0,
                elasticity_damping=0.001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.45, 0.25), roughness=0.8
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.5, -0.1, SPEC.TABLE_HEIGHT + 0.04)
        ),
        debug_vis=False,
    )


@configclass
class DeformableEventCfg:
    """변형체는 강체와 리셋 방식이 다르다 (root pose 가 아니라 nodal state)."""

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
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # 강체 환경의 randomize_material_pose 와 같은 범위를 쓴다.
            # 목표 영역(y=+0.35)과 겹치지 않게 유지하는 것이 중요하다.
            "position_range": {"x": (-0.06, 0.10), "y": (-0.12, 0.08), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    reset_hold_counter = EventTerm(
        func=mdp.reset_success_hold_counter, mode="reset", params={}
    )


@configclass
class DeformableRewardsCfg:
    success = RewTerm(
        func=deform_mdp.deformable_success_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class DeformableTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=deform_mdp.deformable_dropped,
        params={"object_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(
        func=deform_mdp.deformable_placed_signal,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceDeformableEnvCfg(PickPlaceEnvCfg):
    """<VlaPick-Deformable-v0>"""

    scene: DeformableSceneCfg = DeformableSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        # 변형체는 env 별 메시가 독립이라 물리 복제를 쓸 수 없다.
        replicate_physics=False,
    )
    events: DeformableEventCfg = DeformableEventCfg()
    rewards: DeformableRewardsCfg = DeformableRewardsCfg()
    terminations: DeformableTerminationsCfg = DeformableTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 관측에서 강체 전용 항목을 제거한다.
        # (configclass 특성상 __post_init__ 에서 정리하는 편이 안전하다)
        self.observations.policy.object_pose = None
        self.observations.policy.object_pos_b = None
        self.observations.subtask_terms = None

        # FEM 은 무겁다. 계획서 §Phase4b-5 의 지침대로 num_envs 를 8~16 으로
        # 유지하고, GPU 소프트바디 버퍼를 넉넉히 잡는다.
        self.sim.physx.gpu_max_soft_body_contacts = 1024 * 1024
        # 변형체는 더 잘게 적분해야 안정적이다.
        self.sim.dt = SPEC.SIM_DT / 2.0
        self.decimation = SPEC.DECIMATION * 2      # 정책 주기는 강체와 동일하게 유지
        self.sim.render_interval = self.decimation

# =============================================================================
# vla_isaac_tasks/materials.py
#
# "비정형 자재" 의 강체 근사 — 형상이 다른 자재 6종.
#
# 계획서 §2-3 의 결정에 따른 것이다:
#   SkillGen/MimicGen 증강은 "오브젝트의 SE(3) pose 기준으로 궤적을 변환 재배치"
#   하는 원리라, 단일 강체 pose 가 정의되지 않는 FEM 변형체에는 적용할 수 없다.
#   → SFT 데이터는 강체 + 형상 랜덤화로 만들고, 변형체는 RFT 에서 투입한다.
#
# 외부 USD 에셋을 받지 않고 절차적 프리미티브로 구성한 이유:
#   Day 1 에 에셋 수급이 막히면 그 자리에서 일정이 무너진다. 프리미티브는
#   Nucleus 접속 없이 즉시 생성되고, 형상 다양성이라는 목적에는 충분하다.
#   (색·크기·형태가 모두 달라 카메라 관측에서 확실히 구분된다)
#
# ★ 형상 다양성이 실제로 생기는 방식 — 오해하기 쉬운 지점
#   MultiAssetSpawnerCfg 는 **스폰 시점에 env 별로** 에셋을 하나씩 무작위 배정한다.
#   리셋마다 형상이 바뀌지는 않는다 (PhysX 는 런타임에 지오메트리를 못 바꾼다).
#   따라서 다양성은 "num_envs 를 키워 여러 형상을 동시에 굴리는 것"에서 나온다.
#   → Phase 2 데이터 생성에서 --num_envs 를 20~40 으로 올리라는 지침이
#     생성 속도뿐 아니라 데이터 다양성 측면에서도 중요한 이유다.
#   스케일 랜덤화도 같은 제약을 받아 mode="prestartup" 으로만 걸 수 있다.
# =============================================================================

from __future__ import annotations

import isaaclab.sim as sim_utils

from .spec import SPEC

# 자재 기본 물성 — 6종이 공유한다.
# 마찰을 다소 높게 잡았다: 파지 실패는 Mimic 생성 성공률을 직접 깎아먹고,
# 생성 성공률이 낮으면 Day 1 야간 배치가 통째로 헛돈다.
_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=0.08)
_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2,
    dynamic_friction=1.0,
    restitution=0.0,
)


def _common(color: tuple[float, float, float]) -> dict:
    return dict(
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS_PROPS,
        collision_props=_COLLISION_PROPS,
        physics_material=_PHYSICS_MATERIAL,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
    )


# 자재 6종. 모두 Franka 그리퍼(최대 개방 약 0.08m)로 잡을 수 있는 크기여야 한다 —
# 이 제약을 어기면 텔레옵 데모 자체가 안 만들어진다.
MATERIAL_VARIANTS: list = [
    # 1) 각진 블록 — 파지가 가장 쉬운 기준 형상
    sim_utils.MeshCuboidCfg(size=(0.055, 0.055, 0.055), **_common((0.85, 0.25, 0.20))),
    # 2) 납작하고 긴 블록 — 파지 방향이 제한된다
    sim_utils.MeshCuboidCfg(size=(0.090, 0.040, 0.035), **_common((0.20, 0.55, 0.85))),
    # 3) 원기둥 (누운 자세로 스폰되면 굴러간다 → 정지 판정이 의미를 갖는다)
    sim_utils.MeshCylinderCfg(radius=0.028, height=0.075, **_common((0.30, 0.75, 0.35))),
    # 4) 캡슐 — 자루형 자재 근사. 표면이 곡면이라 미끄러지기 쉽다
    sim_utils.MeshCapsuleCfg(radius=0.026, height=0.055, **_common((0.85, 0.70, 0.20))),
    # 5) 원뿔 — 무게중심이 낮고 파지점이 좁다 (가장 어려운 축)
    #    반지름 0.036 에서 시작했다가 0.028 로 줄였다: 밑면 지름이 스케일 상한
    #    1.25배에서 9.0cm 가 되어 그리퍼 최대 개방폭 8cm 를 넘었기 때문이다.
    #    (assert_graspable 이 잡아냈다. 안 잡았으면 텔레옵 중에야 알았을 문제다)
    sim_utils.MeshConeCfg(radius=0.028, height=0.075, **_common((0.65, 0.35, 0.80))),
    # 6) 작은 정육면체 — 크기 축 다양성
    sim_utils.MeshCuboidCfg(size=(0.042, 0.042, 0.042), **_common((0.55, 0.55, 0.58))),
]

NUM_MATERIAL_VARIANTS = len(MATERIAL_VARIANTS)


def _graspable_widths(cfg) -> tuple[float, ...]:
    """자재를 잡을 때 그리퍼가 벌려야 하는 폭 후보들을 돌려준다.

    직육면체는 세 축 중 아무 축으로나 접근할 수 있으므로 후보가 3개,
    회전체(원기둥/캡슐/원뿔)는 지름 하나뿐이다.
    """
    if isinstance(cfg, sim_utils.MeshCuboidCfg):
        return tuple(cfg.size)
    radius = getattr(cfg, "radius", None)
    if radius is not None:
        return (2.0 * radius,)
    return ()


def assert_graspable(max_scale: float = 1.25) -> None:
    """모든 자재가 Franka 그리퍼로 잡을 수 있는 크기인지 확인한다.

    왜 필요한가: 자재 크기와 그리퍼 최대 개방폭(8cm)은 서로 다른 파일에 있어서
    한쪽만 바꾸기 쉽다. 너무 두꺼운 자재는 **에러 없이** 그냥 안 잡히고,
    그 사실은 Day 1 오후에 텔레옵을 30분 해 본 뒤에야 드러난다.
    그때는 이미 그날 일정이 밀린 뒤다.

    스케일 랜덤화(EventCfg 의 scale_range 상한)까지 감안해 검사한다 —
    1.25배로 커진 상태에서도 잡혀야 하기 때문이다.
    """
    limit = SPEC.GRIPPER_MAX_WIDTH
    problems = []
    for i, cfg in enumerate(MATERIAL_VARIANTS):
        widths = _graspable_widths(cfg)
        if not widths:
            continue
        # 가장 얇은 축으로 접근하면 되므로 최소값만 통과하면 된다.
        thinnest = min(widths) * max_scale
        if thinnest >= limit:
            problems.append(
                f"  자재 {i} ({type(cfg).__name__}): 최소 파지폭 "
                f"{thinnest * 100:.1f}cm ≥ 그리퍼 한계 {limit * 100:.1f}cm"
            )

    if problems:
        raise AssertionError(
            "그리퍼로 잡을 수 없는 자재가 있다 (스케일 상한 "
            f"{max_scale}배 반영):\n" + "\n".join(problems) +
            "\n자재 크기를 줄이거나 EventCfg 의 scale_range 상한을 낮출 것."
        )


# import 시점에 검사한다. 자재를 고치는 순간 알 수 있어야 의미가 있다.
assert_graspable()


def make_material_spawner() -> sim_utils.MultiAssetSpawnerCfg:
    """자재 6종 중 하나를 env 별로 무작위 배정하는 스포너를 만든다.

    ⚠ 이 스포너를 쓰면 씬 설정에서 반드시 ``replicate_physics=False`` 여야 한다.
      env 마다 물리 지오메트리가 다르므로 PhysX 복제 최적화를 쓸 수 없다.
      켜 둔 채로 두면 모든 env 가 같은 형상이 되어 버리는데, 에러가 나지 않아
      "형상 랜덤화가 되고 있다고 착각한 채" 데이터를 다 만들게 된다.
      pickplace_env_cfg.py 의 씬 설정에서 명시적으로 꺼 두었다.
    """
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=MATERIAL_VARIANTS,
        random_choice=True,
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS_PROPS,
        collision_props=_COLLISION_PROPS,
    )

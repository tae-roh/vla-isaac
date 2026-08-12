# =============================================================================
# vla_isaac_tasks/scene_assets.py
#
# 씬 구성물 — 블록 N개 / 타깃 트레이.
#
# ★ 2026-08-12: 얕은 소스 박스(벽 4장)를 제거했다. 블록은 테이블 위
#   스폰 영역(SPEC.SPAWN_CENTER / SPAWN_AREA_SIZE)에 그냥 놓인다 —
#   그 영역에 대응하는 프림은 없다. 이유는 vla_spec.py 의 스폰 영역 절 참조:
#   벽이 사람 데모를 관통시켜 증강 파지 실패의 70% 를 만들고 있었다.
#
# ★ 왜 전부 프리미티브 직육면체인가
#   - 콜라이더가 박스 프리미티브라 시뮬 비용이 최저다. RFT 롤아웃 비용에 직결된다.
#   - 얇은 메시의 convex hull cooking 실패(GPU→CPU 폴백 → 충돌이 아예 안 먹는
#     증상)를 원천 회피한다. 포켓을 "파내지" 않고 레일로 둘러싸는 이유다.
#   - Nucleus 에셋 수급이 막혀도 그 자리에서 일정이 무너지지 않는다.
#
# ★ 블록 3개는 색만 다르고 형상·크기가 같다 (개정 §2-4).
#   "하나만 튀는" 구성이면 모델이 지시문을 무시하고 "튀는 색으로 가라"만 배워도
#   만점이 나온다. 그러면 언어 채널이 죽고, Language split 이 의미를 잃는다.
#
# ★ 형상 랜덤화를 없앤 부수 효과: env 마다 지오메트리가 같아졌으므로
#   replicate_physics=True 를 다시 쓸 수 있다 (PhysX 복제 최적화 = 생성/롤아웃
#   속도 이득). MultiAssetSpawnerCfg 와 그에 딸린 함정도 함께 사라졌다.
# =============================================================================

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from .spec import SPEC

# 블록 물성. 마찰을 높게 잡는다 — 파지 실패는 Mimic 생성 성공률을 직접 깎는다.
_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_BLOCK_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2, dynamic_friction=1.0, restitution=0.0
)
# 정지된 구조물(박스·트레이)은 마찰만 있으면 된다. 반발은 0 — 블록이 벽이나
# 레일에 튕겨 포켓 밖으로 나가면 성공률만 떨어지고 배우는 것은 없다.
_STATIC_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.9, dynamic_friction=0.8, restitution=0.0
)

# 속성어(SPEC.BLOCK_ATTRS) 순서와 1:1 대응해야 한다.
BLOCK_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.15, 0.12),   # red
    (0.15, 0.35, 0.85),   # blue
    (0.20, 0.70, 0.30),   # green
)


def assert_consistent() -> None:
    """색 팔레트와 스펙의 속성어 개수가 맞는지. import 시점에 확인한다."""
    assert len(BLOCK_COLORS) >= SPEC.NUM_BLOCKS, (
        f"블록 {SPEC.NUM_BLOCKS}개인데 색은 {len(BLOCK_COLORS)}개뿐이다."
    )
    # 그리퍼로 잡을 수 있는가 — 짧은 변으로 접근한다.
    assert min(SPEC.BLOCK_SIZE[:2]) < SPEC.GRIPPER_MAX_WIDTH, (
        f"블록 파지폭 {min(SPEC.BLOCK_SIZE[:2]) * 100:.1f}cm ≥ 그리퍼 한계 "
        f"{SPEC.GRIPPER_MAX_WIDTH * 100:.1f}cm — 텔레옵 데모 자체가 안 만들어진다."
    )


assert_consistent()


# -----------------------------------------------------------------------------
# 블록
# -----------------------------------------------------------------------------
def block_name(idx: int) -> str:
    """씬 엔티티 이름. 관측·이벤트·Mimic 이 모두 이 이름을 쓴다."""
    return f"block_{idx}"


def make_block_cfg(idx: int) -> RigidObjectCfg:
    """블록 idx 의 씬 설정.

    초기 위치는 박스 안에 일렬로 벌려 둔다. 실제 배치는 리셋 이벤트가
    초기 상태 뱅크에서 꺼내 덮어쓰므로 여기 값은 "스폰 시점에 서로 겹치지만
    않으면 되는" 자리다.
    """
    bx, by = SPEC.SPAWN_CENTER
    spread = SPEC.BLOCK_MIN_SEPARATION
    y0 = by + (idx - (SPEC.NUM_BLOCKS - 1) / 2.0) * spread
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Block_{idx}",
        spawn=sim_utils.CuboidCfg(
            size=SPEC.BLOCK_SIZE,
            rigid_props=_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=SPEC.BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=_BLOCK_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BLOCK_COLORS[idx], roughness=0.7
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(bx, y0, SPEC.TABLE_HEIGHT + 0.5 * SPEC.BLOCK_SIZE[2]),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# 정지 구조물 — 공통 생성기
# -----------------------------------------------------------------------------
def _static_box(prim: str, size, pos, color) -> AssetBaseCfg:
    """충돌체를 가진 정지 직육면체 하나.

    ★ prim 은 **평평한 이름**이어야 한다 ("Box/box_x_pos" 처럼 중첩 금지).
      isaaclab.sim.utils.prims 의 @clone 데코레이터는 prim_path 를 마지막 '/'
      기준으로 쪼갠 뒤 부모 경로를 find_matching_prim_paths() 로 찾는데,
      중간 그룹 Xform 은 아무도 만들어 주지 않는다 (Isaac Lab 은 자동 생성하지
      않는다). 그래서 중첩하면
          RuntimeError: Unable to find source prim path: '/World/envs/env_.*/Box'
      로 씬 생성이 죽는다. 부모를 먼저 선언하는 방법도 있지만 필드 순서에
      의존하게 되어, 나중에 순서만 바뀌어도 같은 곳에서 다시 깨진다.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=_STATIC_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.8
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


# -----------------------------------------------------------------------------
# 타깃 트레이 (하나, 정사각) — 레일 4장
# -----------------------------------------------------------------------------
# 트레이 바닥은 테이블 상면 그대로다. 바닥판을 깔지 않으면 프림이 하나 줄고,
# 블록 바닥 접촉 판정도 단순해진다 (트레이 바닥 = 테이블 상면 = z 0).
#
# ★ 2026-08-11 개편: 좁은 포켓 3개(레일 2 + 칸막이 6 = 8프림) → 정사각 트레이
#   1개(레일 4장). 한 변이 블록 긴 축의 1.2배(72mm)라, 블록 대각선(67mm)보다
#   커서 yaw 를 맞추지 않고 비스듬히 넣어도 들어간다. 그게 성공 판정에서 yaw 를
#   뺄 수 있는 근거다 (SPEC.SUCCESS_REQUIRE_YAW).
TRAY_RAIL_NAMES = ("tray_x_pos", "tray_x_neg", "tray_y_pos", "tray_y_neg")
_TRAY_COLOR = (0.30, 0.30, 0.34)


def tray_rail(name: str) -> AssetBaseCfg:
    """트레이 레일 하나. name 은 TRAY_RAIL_NAMES 중 하나.

    x 레일이 모서리까지 덮고(길이 s+2r), y 레일은 그 사이(길이 s)만 채운다 —
    소스 박스 벽과 같은 규약이라 두 구조물의 코너 처리가 일관된다.
    """
    s = SPEC.TRAY_INNER_SIZE
    r, d = SPEC.TRAY_RAIL_THICKNESS, SPEC.TRAY_DEPTH
    tx, ty = SPEC.TRAY_CENTER
    z = SPEC.TABLE_HEIGHT + 0.5 * d

    layout = {
        "tray_x_pos": ((r, s + 2 * r, d), (tx + 0.5 * s + 0.5 * r, ty, z)),
        "tray_x_neg": ((r, s + 2 * r, d), (tx - 0.5 * s - 0.5 * r, ty, z)),
        "tray_y_pos": ((s, r, d), (tx, ty + 0.5 * s + 0.5 * r, z)),
        "tray_y_neg": ((s, r, d), (tx, ty - 0.5 * s - 0.5 * r, z)),
    }
    size, pos = layout[name]
    return _static_box(name, size, pos, _TRAY_COLOR)

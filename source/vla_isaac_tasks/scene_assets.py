# =============================================================================
# vla_isaac_tasks/scene_assets.py
#
# 씬 구성물 — 블록 N개 / 얕은 소스 박스 / 맞춤 포켓 트레이.
# (개정 §2. 이전의 "형상 랜덤 자재 6종"(materials.py)을 대체한다)
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
    bx, by = SPEC.BOX_CENTER
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
    """충돌체를 가진 정지 직육면체 하나."""
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
# 소스 박스 (얕은 상자) — 벽 4장
# -----------------------------------------------------------------------------
BOX_WALL_NAMES = ("box_x_pos", "box_x_neg", "box_y_pos", "box_y_neg")
_BOX_COLOR = (0.45, 0.42, 0.40)


def box_wall(name: str) -> AssetBaseCfg:
    """박스 벽 하나. name 은 BOX_WALL_NAMES 중 하나."""
    bx, by = SPEC.BOX_CENTER
    ix, iy = SPEC.BOX_INNER_SIZE
    t, h = SPEC.BOX_WALL_THICKNESS, SPEC.BOX_WALL_HEIGHT
    z = SPEC.TABLE_HEIGHT + 0.5 * h

    # x 벽이 모서리까지 덮고(길이 iy+2t), y 벽은 그 사이(길이 ix)만 채운다.
    layout = {
        "box_x_pos": ((t, iy + 2 * t, h), (bx + 0.5 * ix + 0.5 * t, by, z)),
        "box_x_neg": ((t, iy + 2 * t, h), (bx - 0.5 * ix - 0.5 * t, by, z)),
        "box_y_pos": ((ix, t, h), (bx, by + 0.5 * iy + 0.5 * t, z)),
        "box_y_neg": ((ix, t, h), (bx, by - 0.5 * iy - 0.5 * t, z)),
    }
    size, pos = layout[name]
    return _static_box(f"Box/{name}", size, pos, _BOX_COLOR)


# -----------------------------------------------------------------------------
# 트레이 (맞춤 포켓 N개) — 레일 2 + 칸막이 2N
# -----------------------------------------------------------------------------
# 포켓 바닥은 테이블 상면 그대로다. 바닥판을 깔지 않으면 프림이 하나 줄고,
# 블록 안착 높이 계산도 단순해진다 (성공 판정이 SPEC.pocket_seat_z_max() 하나).
#
# 클리어런스가 바뀌면 이 레일들의 위치·크기가 전부 바뀐다. split 별 환경은
# apply_clearance() 로 한 번에 갈아끼운다 — 씬 cfg 를 split 마다 복제하지 않는다.
TRAY_RAIL_NAMES = ("tray_x_pos", "tray_x_neg") + tuple(
    f"pocket{i}_{s}" for i in range(SPEC.NUM_BLOCKS) for s in ("y_pos", "y_neg")
)
_TRAY_COLOR = (0.30, 0.30, 0.34)


def tray_rail(name: str, clearance: float | None = None) -> AssetBaseCfg:
    """트레이 레일 하나. name 은 TRAY_RAIL_NAMES 중 하나."""
    px, py = SPEC.pocket_inner_size(clearance)
    r, d = SPEC.POCKET_RAIL_THICKNESS, SPEC.POCKET_DEPTH
    tx, ty = SPEC.TRAY_CENTER
    z = SPEC.TABLE_HEIGHT + 0.5 * d

    if name.startswith("tray_x"):
        # 긴 레일 2장은 포켓 열 전체를 따라 이어진다.
        span = (SPEC.NUM_BLOCKS - 1) * SPEC.POCKET_PITCH + py + 2 * r
        sign = 1.0 if name.endswith("pos") else -1.0
        return _static_box(
            f"Tray/{name}",
            (r, span, d),
            (tx + sign * (0.5 * px + 0.5 * r), ty, z),
            _TRAY_COLOR,
        )

    # pocket{i}_y_{pos,neg} — 포켓 하나의 앞뒤 칸막이.
    idx = int(name[len("pocket")])
    _, cy = SPEC.pocket_center(idx)
    sign = 1.0 if name.endswith("pos") else -1.0
    return _static_box(
        f"Tray/{name}",
        (px + 2 * r, r, d),
        (tx, cy + sign * (0.5 * py + 0.5 * r), z),
        _TRAY_COLOR,
    )


def apply_clearance(scene_cfg, clearance: float) -> None:
    """씬의 포켓 레일을 주어진 클리어런스로 다시 만든다.

    클리어런스 split 환경(`VlaPlace-c1mm-v0` 등)이 __post_init__ 에서 부른다.
    성공 판정 공차도 같은 값에서 파생되어야 하므로, 호출부는 반드시
    `env_cfg.pocket_clearance` 도 함께 갱신할 것 — 둘은 한 쌍이다.
    """
    for name in TRAY_RAIL_NAMES:
        setattr(scene_cfg, name, tray_rail(name, clearance))

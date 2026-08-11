# =============================================================================
# vla_isaac_tasks
#
# 태스크 등록. Mimic 파이프라인은 단계마다 다른 태스크 이름을 쓴다:
#
#   VlaPlace-v0                    텔레옵 수집 · 재생 · 정책 평가
#                                  → record_demos.py, replay_demos.py, eval_rollout.py
#   VlaPlace-Mimic-v0              상태 기반 어노테이션/증강 (예비)
#   VlaPlace-Visuomotor-Mimic-v0   카메라 포함 증강 ★ 본 프로젝트 사용
#                                  → annotate_demos.py, generate_dataset.py (+ --enable_cameras)
#
# 여기에 **클리어런스 split** 이 더해진다 (개정 §2-3b). 위 3종은 기본 split
# (SPEC.POCKET_CLEARANCE) 이고, 사다리의 각 값마다 같은 3종이 꼬리표를 달고
# 다시 등록된다:
#
#   VlaPlace-c5mm-v0 / -c2mm- / -c1mm- / -c0p5mm-   (+ 각각의 Mimic 변형)
#
# SR_SFT(c) vs SR_SFT+RL(c) 곡선이 이 프로젝트의 핵심 결과물이므로, 곡선의 점
# 하나가 태스크 이름 하나에 대응해야 실험이 헷갈리지 않는다.
#
# import 만으로 등록이 일어나므로, 스크립트에서 `import vla_isaac_tasks` 한 줄이면 된다.
#
# 이전 이름(VlaPick-*)은 태스크가 바뀌면서 폐기했다. 옛 이름으로 실행하면
# gym 이 NameNotFound 로 즉시 죽는다 — 조용히 다른 태스크가 도는 것보다 낫다.
# =============================================================================

from functools import partial

import gymnasium as gym

from .spec import SPEC  # noqa: F401  — import 시점에 스펙 정합성이 검사된다

_ENTRY_MIMIC = f"{__name__}.pickplace_mimic_env:PickPlaceMimicEnv"


def _make_cfg(clearance: float | None, kind: str):
    """환경 cfg 인스턴스를 만든다 (gym 의 env_cfg_entry_point 콜러블).

    클리어런스 split 마다 configclass 를 8개씩 복제하는 대신, 인스턴스를 만들고
    set_clearance() 로 포켓 레일과 판정 공차를 함께 갈아끼운다. 둘이 따로 놀면
    포켓만 좁아지고 성공 공차는 그대로여서 곡선이 조용히 틀어진다.
    """
    from .pickplace_env_cfg import PickPlaceEnvCfg
    from .pickplace_mimic_env_cfg import (
        PickPlaceMimicEnvCfg,
        PickPlaceVisuomotorMimicEnvCfg,
        datagen_name,
    )

    cfg = {
        "base": PickPlaceEnvCfg,
        "mimic": PickPlaceMimicEnvCfg,
        "visuomotor_mimic": PickPlaceVisuomotorMimicEnvCfg,
    }[kind]()

    if clearance is not None:
        cfg.set_clearance(clearance)
        if hasattr(cfg, "datagen_config"):
            # __post_init__ 은 기본 클리어런스로 이름을 지었다. split 이름으로 고친다.
            cfg.datagen_config.name = datagen_name(clearance)
    return cfg


def _register(suffix: str, clearance: float | None) -> None:
    gym.register(
        id=f"VlaPlace-{suffix}v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": partial(_make_cfg, clearance, "base")},
        disable_env_checker=True,
    )
    gym.register(
        id=f"VlaPlace-{suffix}Mimic-v0",
        entry_point=_ENTRY_MIMIC,
        kwargs={"env_cfg_entry_point": partial(_make_cfg, clearance, "mimic")},
        disable_env_checker=True,
    )
    gym.register(
        id=f"VlaPlace-{suffix}Visuomotor-Mimic-v0",
        entry_point=_ENTRY_MIMIC,
        kwargs={
            "env_cfg_entry_point": partial(_make_cfg, clearance, "visuomotor_mimic")
        },
        disable_env_checker=True,
    )


# 기본 split (스펙의 POCKET_CLEARANCE) — 데모 수집과 첫 SFT 는 여기서 한다.
_register("", None)

# 클리어런스 사다리. 기본값과 같은 값도 별칭으로 함께 등록해 둔다 —
# 실험 로그에 "어느 공차였나" 가 태스크 이름으로 남는 편이 안전하다.
for _c in SPEC.CLEARANCE_LADDER:
    _register(f"{SPEC.clearance_tag(_c)}-", _c)

__all__ = ["SPEC"]

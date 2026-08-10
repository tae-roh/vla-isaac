# =============================================================================
# vla_isaac_tasks
#
# 태스크 3종을 gym 에 등록한다.
#
# 계획서 §Phase1 의 경고 그대로, Mimic 파이프라인은 환경을 3개 등록해야 하고
# 단계마다 다른 태스크 이름을 쓴다:
#
#   VlaPick-v0                    텔레옵 수집 · 재생 · 정책 평가
#                                 → record_demos.py, replay_demos.py, eval_rollout.py
#   VlaPick-Mimic-v0              상태 기반 어노테이션/증강 (예비)
#   VlaPick-Visuomotor-Mimic-v0   카메라 포함 증강 ★ 본 프로젝트 사용
#                                 → annotate_demos.py, generate_dataset.py (+ --enable_cameras)
#
# import 만으로 등록이 일어나므로, 스크립트에서 `import vla_isaac_tasks` 한 줄이면 된다.
# =============================================================================

import gymnasium as gym

from .spec import SPEC  # noqa: F401  — import 시점에 스펙 정합성이 검사된다

##
# 1) 기본 환경 — 텔레옵 / 재생 / 평가
##
gym.register(
    id="VlaPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:PickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

##
# 2) Mimic (상태 기반)
##
gym.register(
    id="VlaPick-Mimic-v0",
    entry_point=f"{__name__}.pickplace_mimic_env:PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pickplace_mimic_env_cfg:PickPlaceMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

##
# 3) Mimic (visuomotor) — 본 프로젝트가 실제로 쓰는 환경
##
gym.register(
    id="VlaPick-Visuomotor-Mimic-v0",
    entry_point=f"{__name__}.pickplace_mimic_env:PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pickplace_mimic_env_cfg:PickPlaceVisuomotorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

##
# 4) 변형체 (Phase 4b 스트레치) — 강체 RFT 커브를 확보한 뒤에만 쓴다
##
gym.register(
    id="VlaPick-Deformable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.deformable_env_cfg:PickPlaceDeformableEnvCfg"
        ),
    },
    disable_env_checker=True,
)

__all__ = ["SPEC"]

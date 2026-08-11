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
# ★ 클리어런스 split 은 폐지했다 (2026-08-11 개편).
#   예전에는 포켓 클리어런스 사다리(5/2/1/0.5mm)마다 같은 3종을 꼬리표 붙여
#   다시 등록해 총 15종이었고, SR_SFT(c) vs SR_SFT+RL(c) 곡선이 결과물이었다.
#   그 태스크는 **사람이 텔레옵으로 원본 데모를 만들 수 없어서** (xy 공차 6.5mm
#   vs 키보드 1스텝 25mm) 파이프라인 첫 단계부터 막혔다. 트레이를 하나로 넓히면서
#   난이도 축이 사라졌고, 등록도 3종으로 돌아왔다.
#
# import 만으로 등록이 일어나므로, 스크립트에서 `import vla_isaac_tasks` 한 줄이면 된다.
#
# 이전 이름(VlaPick-*, VlaPlace-c*mm-*)은 폐기했다. 옛 이름으로 실행하면
# gym 이 NameNotFound 로 즉시 죽는다 — 조용히 다른 태스크가 도는 것보다 낫다.
# =============================================================================

import gymnasium as gym

from .spec import SPEC  # noqa: F401  — import 시점에 스펙 정합성이 검사된다

_ENTRY_MIMIC = f"{__name__}.pickplace_mimic_env:PickPlaceMimicEnv"


def _make_cfg(kind: str):
    """환경 cfg 인스턴스를 만든다 (gym 의 env_cfg_entry_point 콜러블)."""
    from .pickplace_env_cfg import PickPlaceEnvCfg
    from .pickplace_mimic_env_cfg import (
        PickPlaceMimicEnvCfg,
        PickPlaceVisuomotorMimicEnvCfg,
    )

    return {
        "base": PickPlaceEnvCfg,
        "mimic": PickPlaceMimicEnvCfg,
        "visuomotor_mimic": PickPlaceVisuomotorMimicEnvCfg,
    }[kind]()


def _cfg_entry(kind: str):
    """env_cfg_entry_point 로 넘길 콜러블을 만든다.

    ★ functools.partial 을 쓰면 안 된다. 이 리비전의
      isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry 는 콜러블
      엔트리포인트에 대해 호출보다 **먼저** inspect.getfile() 을 부르는데,
      partial 은 그 인자로 허용되지 않아 TypeError 로 죽는다. 게다가 그 예외는
      스크립트의 finally: simulation_app.close() 에 삼켜져 트레이스백도 없이
      exit 0 으로 끝난다 — PNG 가 한 장도 안 나오는데 에러도 없는 상태가 된다.
      평범한 함수는 __code__ 가 있어 inspect.getfile() 을 통과한다.
    """

    def make_cfg():
        return _make_cfg(kind)

    return make_cfg


gym.register(
    id="VlaPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": _cfg_entry("base")},
    disable_env_checker=True,
)
gym.register(
    id="VlaPlace-Mimic-v0",
    entry_point=_ENTRY_MIMIC,
    kwargs={"env_cfg_entry_point": _cfg_entry("mimic")},
    disable_env_checker=True,
)
gym.register(
    id="VlaPlace-Visuomotor-Mimic-v0",
    entry_point=_ENTRY_MIMIC,
    kwargs={"env_cfg_entry_point": _cfg_entry("visuomotor_mimic")},
    disable_env_checker=True,
)

__all__ = ["SPEC"]

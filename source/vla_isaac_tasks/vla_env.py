# =============================================================================
# vla_isaac_tasks/vla_env.py
#
# reset_to() 이후 타깃 블록을 되찾는 환경 믹스인.
#
# ★ 이게 왜 필요한가 (실측으로 확인한 버그)
#
#   이 태스크에서 "어느 블록이 타깃인가" 는 물리 상태가 아니라 파이썬 버퍼
#   (`_vla_target_block`) 에 들어 있다. 그런데 Isaac Lab 의 재생 경로는
#
#       reset_to(state)
#         → _reset_idx()           reset 이벤트 실행 → 뱅크 커서에서 타깃 뽑음
#         → scene.reset_to(state)  블록 위치만 파일 값으로 덮어씀
#         → observation_manager.compute()
#
#   순서로 돈다. 위치는 복원되지만 타깃은 **뱅크 커서 값이 그대로 남는다.**
#   데모가 실제로 지시했던 블록과 어긋나고(실측 10개 중 7개), 성공 판정이
#   엉뚱한 블록을 보게 되어 replay 성공률이 0/10 으로 나왔다.
#
#   같은 경로를 annotate_demos.py 도 쓰므로, 고치지 않으면 Mimic 서브태스크
#   어노테이션이 전부 틀린 블록 기준으로 찍힌다.
#
# ★ 왜 이벤트 안에서 못 고치는가
#   이벤트는 scene.reset_to() **전에** 돈다. 그 시점에는 파일의 배치가 아직
#   씬에 들어오지 않아서 무엇을 복원해야 하는지 알 수 없다. 덮어쓰기가 끝난
#   뒤에 손대야 하고, 그 지점이 reset_to() 의 바깥이다.
#
# ★ 관측을 다시 계산하는 이유
#   super().reset_to() 는 마지막에 관측을 계산해 돌려준다. 타깃을 그 뒤에
#   고치면 방금 돌려준 관측의 target_ids 는 옛 값이다. 재생 첫 프레임부터
#   지시문이 어긋나므로 다시 계산해서 반환한다.
# =============================================================================

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.events import recover_target_from_scene
from .mdp.terminations import update_success_hold


class VlaEnvMixin:
    """이 태스크의 환경이 공통으로 갖춰야 하는 두 가지.

    1) reset_to() 뒤 타깃 블록 복원 (아래 reset_to)
    2) 성공 유지 카운터를 스텝당 정확히 한 번 전진 (아래 step)

    정상 롤아웃(env.reset())에는 1)의 영향이 없다 — 그 경로는 reset_to 를
    거치지 않고, 거치더라도 방금 배치한 그 행을 다시 찾으므로 값이 같다.
    """

    def step(self, action):
        """스텝 후 성공 유지 카운터를 전진시킨다.

        ★ 왜 termination 함수가 아니라 여기인가
          replay_demos.py 는 `env_cfg.terminations = {}` 로 종료 조건을 통째로
          끄고 성공 함수를 에피소드 끝에 한 번만 부른다. Mimic 생성도
          `env_cfg.terminations = None` 이다. 카운터 전진을 termination 안에
          두면 그 경로들에서 카운터가 자라지 않아, 멀쩡한 데모도 성공으로
          잡히지 않는다 (실제로 replay 가 0/10 이었다).
          step() 은 어떤 소비자도 끄지 않으므로 여기가 유일하게 안전한 자리다.

          termination 매니저는 이 갱신 **전에** 카운터를 읽으므로 성공이
          한 스텝 늦게 잡힌다. 24Hz 에서 1/24 초라 무시할 수 있다.
        """
        out = super().step(action)
        update_success_hold(self)
        return out

    def reset_to(
        self,
        state: dict,
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        recover_target_from_scene(self, env_ids)
        # 타깃이 바뀌었을 수 있으므로 관측을 다시 만든다.
        # update_history=False — super() 에서 이미 히스토리를 밀었다.
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, self.extras


class VlaPlaceEnv(VlaEnvMixin, ManagerBasedRLEnv):
    """VlaPlace-v0 (텔레옵 수집 · 재생 · 평가) 환경."""

    pass


__all__ = ["VlaEnvMixin", "VlaPlaceEnv"]

# =============================================================================
# vla_isaac_tasks/pickplace_mimic_env_cfg.py
#
# Mimic 환경 설정 2종:
#   <VlaPlace-Mimic-v0>             — 상태 기반 (빠름. 예비)
#   <VlaPlace-Visuomotor-Mimic-v0>  — 카메라 포함 (★ 본 프로젝트가 실제로 쓰는 것)
#
# state 기반과 visuomotor 중 하나를 골랐으면 모든 명령에서 일관되게 유지해야
# 한다. 우리는 Visuomotor 로 간다 — annotate_demos.py 와 generate_dataset.py
# 양쪽 모두 --task VlaPlace-Visuomotor-Mimic-v0 을 쓸 것.
# =============================================================================

from __future__ import annotations

import os

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .pickplace_env_cfg import PickPlaceEnvCfg, PickPlaceVisuomotorEnvCfg
from .pickplace_mimic_env import EEF_NAME
from .spec import SPEC


# 생성 데이터셋 이름. 클리어런스 split 이 없어져 하나로 고정됐다.
DATAGEN_NAME = "vla_place"

# 마지막 서브태스크의 식별 이름. SkillGen 이 시작 시그널 dict 의 키로 쓴다.
# pickplace_mimic_env.get_subtask_start_signals() 가 같은 이름을 써야 한다 —
# 어긋나면 생성이 KeyError 로 죽는다. 그래서 여기서 한 번만 정의하고 공유한다.
LAST_SUBTASK_NAME = "place_done"


def _build_subtask_configs() -> list[SubTaskConfig]:
    """서브태스크 2개. MimicGen 과 SkillGen 양쪽에서 그대로 쓴다.

    개정 §4-2 를 그대로 따랐다:
      - 개수를 최소로. 구간을 이어붙일 때마다 스티칭이 늘고 동작이 덜 부드러워지며
        실패 데모가 늘어난다.
      - 경계는 파지 직후가 아니라 **블록을 들어 박스 벽 위로 뺀 뒤**에 둔다.
        플래너가 이어붙일 구간에 박스 벽이 걸리면 그 지점에서 생성이 실패한다.
      - num_interpolation_steps 는 5 전후에서 시작해 튜닝 (0=급격, 20=지연).

    ★ object_ref 가 서브태스크마다 다르다.
      1) 파지 → "target"  : 지시문이 지정한 블록의 pose 기준으로 궤적을 변환
      2) 배치 → "tray"    : 타깃 트레이의 pose 기준으로 궤적을 변환
      "target" 은 env 마다 값이 다르다 (get_object_poses 참조). 여기를 고정
      엔티티 이름으로 두면 증강 데이터가 전부 같은 블록으로 가고, 언어 조건이
      데이터에서 사라진다. "tray" 는 트레이가 하나뿐이라 모든 env 가 같지만,
      배치 구간의 기준 프레임으로는 여전히 필요하다.

    ★ 시작 경계(SkillGen 용)는 여기에 적지 않는다.
      SubTaskConfig 에는 subtask_start_signal 같은 필드가 **없다**.
      Isaac Lab 은 시작 신호를 환경의 별도 메서드로 받는다:
          get_subtask_term_signals()   → 종료 경계 (MimicGen + SkillGen 공통)
          get_subtask_start_signals()  → 시작 경계 (SkillGen 전용)
      annotate_demos.py 가 --annotate_subtask_start_signals 와 함께 후자를 호출한다.

    마지막 서브태스크의 subtask_term_signal 은 MimicGen 에서는 None 이지만
    SkillGen 에서는 이름이 있어야 한다 (아래 서브태스크 2 주석 참조).
    """
    common = dict(
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        # ★ 생성 시 액션에 주입하는 가우시안 노이즈. 데이터 다양성을 위한 것인데,
        #   우리 데모의 액션 크기에 비해 과했다. 실측:
        #       사람 데모 평균 액션 크기   0.0037
        #       생성 데이터 스텝간 변화    0.0355 (17.7mm) ← 신호와 같은 크기
        #       생성 데이터 방향 반전      38.6%  (사람 데모는 1.8%)
        #   0.03 은 Isaac Lab Franka stack 레퍼런스 값인데, 그 태스크는 사람
        #   데모의 액션이 우리보다 훨씬 크다. 우리 데모는 텔레옵 감도를 낮춰
        #   곱게 만든 것이라 같은 노이즈가 신호를 덮는다 (SNR 약 1:8).
        #   참고로 AgiBot 레퍼런스는 0.01, G1 은 0.003 을 쓴다.
        action_noise=float(os.environ.get("VLA_ACTION_NOISE", 0.003)),
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )

    return [
        # 서브태스크 1: 박스에서 타깃 블록을 파지하고 벽 위로 들어올린다.
        SubTaskConfig(
            object_ref="target",
            subtask_term_signal="grasp_lift",
            # 시그널이 뜬 시점에서 10~20 스텝 뒤를 경계로 삼는다. 여유를 두면
            # 파지가 확실히 안정된 뒤에 구간이 끊겨 스티칭이 매끄럽다.
            subtask_term_offset_range=(10, 20),
            description="Pick the target block out of the box",
            next_subtask_description="Place the block into the tray",
            **common,
        ),
        # 서브태스크 2: 타깃 트레이에 넣는다. (마지막)
        #
        # ★ 마지막인데도 subtask_term_signal 에 이름이 있다 — SkillGen 규약이다.
        #   MimicGen 만 쓸 때는 None 이 맞다. 하지만 SkillGen 은 이 이름을
        #   **시작 시그널 dict 의 키**로 쓴다:
        #       datagen_info_pool.py:143
        #       subtask_start_signals[eef_subtask_signal_name]
        #   annotate_demos.py 도 수동 모드 검증에서 같은 말을 한다 —
        #   "each subtask (including the last) must specify 'subtask_term_signal'.
        #    The last subtask's term signal name is used as the final start signal name."
        #
        #   ⚠ 그 검증은 --auto 모드에는 없다. None 으로 두면 어노테이션은 조용히
        #     통과하고, 생성 단계에서 KeyError 로 터진다 (실제로 그렇게 당했다).
        #
        #   이 이름으로 **종료 시그널을 만들 필요는 없다.** 마지막 서브태스크의
        #   종료 인덱스는 이름이 아니라 에피소드 길이로 정해진다
        #   (datagen_info_pool.py:152-154). get_subtask_term_signals() 는 그대로
        #   grasp_lift 하나만 돌려주면 된다.
        SubTaskConfig(
            object_ref="tray",
            subtask_term_signal=LAST_SUBTASK_NAME,
            subtask_term_offset_range=(0, 0),
            description="Place the block into the tray",
            **common,
        ),
    ]


def _apply_datagen_config(cfg) -> None:
    """두 Mimic cfg 가 공유하는 datagen 설정.

    ★ use_skillgen 은 여기서 건드리지 않는다 — generate_dataset.py 의
      `--use_skillgen` 플래그가 지배한다. 의도된 설계다: SkillGen 을 기본으로
      쓰되, cuRobo 가 막히면 **플래그 하나만 빼서** MimicGen 으로 후퇴할 수 있다.
    """
    cfg.datagen_config.name = DATAGEN_NAME
    # 생성 성공을 보장할 때까지 재시도한다.
    #
    # ★ True 면 generation_num_trials 가 **성공 개수**이고 채울 때까지 무한
    #   재시도한다 (generation.py:132-137 의 check_val 분기). 성공률이 낮을 때
    #   이게 몇 백 회씩 돌아 GPU 를 태운다 — SkillGen 0% 일 때 296회까지 갔다.
    #   VLA_TRIALS_ARE_ATTEMPTS=1 로 끄면 generation_num_trials 가 **시도 횟수**
    #   상한이 되어, 성공률 탐색용 짧은 실행을 정확히 N회에서 끊을 수 있다.
    cfg.datagen_config.generation_guarantee = os.environ.get(
        "VLA_TRIALS_ARE_ATTEMPTS", ""
    ) not in ("1", "true", "True")
    # 실패 궤적은 버린다. SFT 데이터에 실패가 섞이면 정책이 실패를 학습한다.
    #
    # ★ 진단할 때만 VLA_KEEP_FAILED=1 로 켠다. 그러면 실패분이
    #   <output>_failed.hdf5 로 따로 나와 replay 로 "어디서 무너지는지" 를 볼 수
    #   있다. generate_dataset.py 에는 이걸 켜는 CLI 플래그가 없고 cfg 가 유일한
    #   스위치라, 코드를 직접 고쳤다 되돌리는 대신 환경변수로 뺐다 —
    #   되돌리는 것을 잊으면 야간 배치가 실패분까지 쓰면서 디스크를 몇 배로 쓴다.
    cfg.datagen_config.generation_keep_failed = os.environ.get(
        "VLA_KEEP_FAILED", ""
    ) in ("1", "true", "True")
    cfg.datagen_config.max_num_failures = 50
    cfg.datagen_config.generation_num_trials = 10   # CLI 인자로 덮어쓴다
    cfg.datagen_config.generation_select_src_per_subtask = True
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.seed = 1

    cfg.subtask_configs[EEF_NAME] = _build_subtask_configs()


@configclass
class PickPlaceMimicEnvCfg(PickPlaceEnvCfg, MimicEnvCfg):
    """<VlaPlace-Mimic-v0> — 상태 기반 Mimic."""

    def __post_init__(self):
        super().__post_init__()
        _apply_datagen_config(self)


@configclass
class PickPlaceVisuomotorMimicEnvCfg(PickPlaceVisuomotorEnvCfg, MimicEnvCfg):
    """<VlaPlace-Visuomotor-Mimic-v0> — 카메라 관측 포함. 본 프로젝트가 쓰는 환경.

    실행 시 반드시 `--enable_cameras` 를 붙여야 한다. 빼먹으면 카메라 센서가
    초기화되지 않아 관측에 이미지가 비어 들어오고, 그대로 생성된 데이터셋은
    학습 중반에야 이상을 드러낸다.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_datagen_config(self)

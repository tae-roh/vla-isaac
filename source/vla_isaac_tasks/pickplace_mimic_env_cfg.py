# =============================================================================
# vla_isaac_tasks/pickplace_mimic_env_cfg.py
#
# Mimic 환경 설정 2종:
#   <VlaPick-Mimic-v0>             — 상태 기반 (빠름. 계획서 §3-3 선택지 B 대비용)
#   <VlaPick-Visuomotor-Mimic-v0>  — 카메라 포함 (★ 본 프로젝트가 실제로 쓰는 것)
#
# 계획서 §Phase1 의 경고: state 기반과 visuomotor 중 하나를 골랐으면 모든 명령에서
# 일관되게 유지해야 한다. 우리는 Visuomotor 로 간다 — annotate_demos.py 와
# generate_dataset.py 양쪽 모두 --task VlaPick-Visuomotor-Mimic-v0 을 쓸 것.
# =============================================================================

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .pickplace_env_cfg import PickPlaceEnvCfg, PickPlaceVisuomotorEnvCfg
from .pickplace_mimic_env import EEF_NAME


def _build_subtask_configs() -> list[SubTaskConfig]:
    """서브태스크 2개. MimicGen 과 SkillGen 양쪽에서 그대로 쓴다.

    계획서 §Phase1-5 를 그대로 따랐다:
      - 개수를 최소로 (approach→grasp→transport→place 를 2개로 압축).
        서브태스크가 많을수록 스티칭 구간이 늘어 생성 성공률이 떨어진다.
      - 경계를 "로봇이 다른 물체와 충돌하지 않을 지점"에 둔다.
        → "파지 직후"가 아니라 "파지 후 들어올린 뒤"(grasp_lift).
      - num_interpolation_steps 는 5 전후에서 시작해 튜닝 (0=급격, 20=지연).

    ★ 시작 경계(SkillGen 용)는 여기에 적지 않는다.
      SubTaskConfig 에는 subtask_start_signal 같은 필드가 **없다**.
      Isaac Lab 은 시작 신호를 환경의 별도 메서드로 받는다:
          get_subtask_term_signals()   → 종료 경계 (MimicGen + SkillGen 공통)
          get_subtask_start_signals()  → 시작 경계 (SkillGen 전용)
      annotate_demos.py 가 --annotate_subtask_start_signals 와 함께 후자를 호출해
      obs/datagen_info/subtask_start_signals/ 아래에 기록한다.
      → 구현은 pickplace_mimic_env.py 에 있다.

      SkillGen 은 그 경계로 궤적을 셋으로 나눈다:
          자유공간 접근 → [grasp_start] 파지 접촉구간 [grasp_lift] →
          자유공간 운반 → [place_start] 배치 접촉구간 → 끝
      접촉구간은 사람 데모를 재생하고, 자유공간은 cuRobo 가 충돌 회피 경로를 깐다.
      MimicGen 은 시작 신호를 무시하고 자유공간을 선형 보간으로 메운다.
      → 이 cfg 하나로 두 방식을 다 커버한다. cuRobo 가 막혀도 고칠 것이 없다.

    마지막 서브태스크는 규약상 subtask_term_signal=None 이다.
    """
    common = dict(
        object_ref="object",          # get_object_poses 의 키와 일치해야 한다
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.03,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )

    return [
        # 서브태스크 1: 자재를 파지하고 들어올린다.
        #   시작 = eef 가 자재에 접근 완료 (여기부터 사람 데모)
        #   종료 = 파지 후 들어올림
        SubTaskConfig(
            subtask_term_signal="grasp_lift",
            # 시그널이 뜬 시점에서 10~20 스텝 뒤를 경계로 삼는다. 여유를 두면
            # 파지가 확실히 안정된 뒤에 구간이 끊겨 스티칭이 매끄럽다.
            subtask_term_offset_range=(10, 20),
            description="Grasp the material and lift it",
            next_subtask_description="Move the material to the target zone and release",
            **common,
        ),
        # 서브태스크 2: 목표 영역에 배치하고 놓는다.
        #   시작 = 자재를 든 채로 목표 영역 위 도달 (여기부터 사람 데모)
        #   종료 = 없음 (마지막)
        SubTaskConfig(
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            description="Move the material to the target zone and release",
            **common,
        ),
    ]


def _apply_datagen_config(cfg) -> None:
    """두 Mimic cfg 가 공유하는 datagen 설정.

    ★ use_skillgen 은 여기서 건드리지 않는다 — generate_dataset.py 의
      `--use_skillgen` 플래그가 지배한다. 의도된 설계다:
      SkillGen 을 기본으로 쓰되, cuRobo 가 막히면 **플래그 하나만 빼서**
      MimicGen 으로 후퇴할 수 있어야 한다. 여기에 True 를 박아 두면
      후퇴할 때 코드를 고쳐야 하고, 그건 Day 1 밤에 하고 싶은 일이 아니다.
    """
    cfg.datagen_config.name = "vla_pick_material_D0"
    # 생성 성공을 보장할 때까지 재시도한다. 성공률이 낮은 태스크에서 켜 두면
    # 목표 개수를 실제로 채울 수 있다.
    cfg.datagen_config.generation_guarantee = True
    # 실패 궤적은 버린다. SFT 데이터에 실패가 섞이면 정책이 실패를 학습한다.
    cfg.datagen_config.generation_keep_failed = False
    cfg.datagen_config.max_num_failures = 50
    cfg.datagen_config.generation_num_trials = 10   # CLI 인자로 덮어쓴다
    cfg.datagen_config.generation_select_src_per_subtask = True
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.seed = 1

    cfg.subtask_configs[EEF_NAME] = _build_subtask_configs()


@configclass
class PickPlaceMimicEnvCfg(PickPlaceEnvCfg, MimicEnvCfg):
    """<VlaPick-Mimic-v0> — 상태 기반 Mimic."""

    def __post_init__(self):
        super().__post_init__()
        _apply_datagen_config(self)


@configclass
class PickPlaceVisuomotorMimicEnvCfg(PickPlaceVisuomotorEnvCfg, MimicEnvCfg):
    """<VlaPick-Visuomotor-Mimic-v0> — 카메라 관측 포함. 본 프로젝트가 쓰는 환경.

    실행 시 반드시 `--enable_cameras` 를 붙여야 한다. 빼먹으면 카메라 센서가
    초기화되지 않아 관측에 이미지가 비어 들어오고, 그대로 생성된 데이터셋은
    학습 중반에야 이상을 드러낸다.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_datagen_config(self)

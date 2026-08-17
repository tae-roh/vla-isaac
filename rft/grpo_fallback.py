"""자작 GRPO 루프 — veRL 어댑터가 막혔을 때의 fallback.

★ 언제 이걸로 갈아타는가
  Day 3 정오까지 SimpleVLA-RL 어댑터로 롤아웃 1회가 안 돌면 전환한다.
  계획서 §6 도 같은 판단을 적어 두었다: "num_envs 가 작아 veRL 스케일링 이점이
  제한적 → 단일 노드에선 자작이 오히려 단순할 수 있음".

  전환 비용이 작은 이유: 환경·보상·관측·브리지·평가 코드가 전부 그대로 쓰인다.
  갈아끼우는 것은 "정책을 어떻게 업데이트하는가" 한 조각뿐이다.

★ veRL 대비 포기하는 것 (알고 쓸 것)
  - vLLM 가속 생성 → HF generate 를 쓴다. 느리다. num_envs 가 작아 감당은 된다.
  - FSDP 멀티 GPU 샤딩 → 단일 GPU LoRA 만. 7B LoRA + bf16 이면 48GB 에 들어간다.
  - 정교한 KV 캐시/시퀀스 패킹 최적화

  즉 "느리지만 확실히 도는 것" 과 "빠르지만 이틀 안에 못 붙일 수도 있는 것"의
  교환이다. 3~4일 예산에서 유의미한 결과를 보장해야 하므로 이 선택지가 존재한다.

GRPO 요약:
  1. 같은 초기 상태에서 G개 궤적을 샘플링 (temperature > 1 로 탐색)
  2. 그룹 안에서 보상을 정규화 → advantage  (critic 불필요)
  3. PPO 스타일 클리핑으로 정책 업데이트

사용 예:
    python rft/grpo_fallback.py --config rft/configs/grpo_rigid.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# ★ 체크포인트의 원격 코드(modeling_prismatic.py)가 `import prismatic` 을 요구한다.
#   eval_rollout.py 와 **같은 셰임**을 쓴다 — 두 경로가 서로 다른 prismatic 을
#   보면 액션 마스크 규약이 갈라져도 알 방법이 없다 (--verify-checkpoint 는
#   같은 프로세스 안에서 두 경로를 비교하므로 이 어긋남을 잡지 못한다).
#   셰임에는 torch 2.7 호환 수정이 들어 있다 (vendor/README.md 참조).
sys.path.insert(0, str(REPO_ROOT / "vendor"))

_spec_mod = importlib.util.spec_from_file_location(
    "vla_spec", REPO_ROOT / "configs" / "vla_spec.py"
)
SPEC = importlib.util.module_from_spec(_spec_mod)
sys.modules["vla_spec"] = SPEC
_spec_mod.loader.exec_module(SPEC)

from rft.ipc_bridge import RolloutClient  # noqa: E402


# =============================================================================
@dataclass
class GRPOConfig:
    checkpoint: str = ""
    task: str = "VlaPlace-v0"
    isaaclab_python: str = str(Path.home() / "env_isaaclab" / "bin" / "python")

    # 롤아웃
    group_size: int = 8           # 같은 초기 상태에서 굴릴 궤적 수 (G)
    num_envs: int = 8             # 워커의 배치 크기. group_size 와 같게 두면 한 라운드 = 한 그룹
    groups_per_step: int = 2      # 한 업데이트에 쓸 **유효** 그룹 수
    # dynamic sampling (DAPO). 전멸/전승 그룹은 advantage 가 0 이라 버리고 다시
    # 뽑아 배치를 채운다. False 면 예전처럼 건너뛰기만 한다.
    dynamic_sampling: bool = True
    # 한 스텝에서 시도할 그룹 수의 상한. 0 = groups_per_step 의 3배.
    # 정책이 태스크를 못 풀면 유효 그룹이 영원히 안 나오므로 상한이 필요하다.
    max_group_attempts: int = 0
    max_steps_per_episode: int = SPEC.MAX_EPISODE_STEPS
    temperature: float = 1.6      # 개정 §5: SimpleVLA-RL 설정값
    # 정책 forward 를 몇 개씩 묶어 돌릴지. 예전에는 1개씩 순차로 돌아 96GB 중
    # 35GB 만 썼고, groups_per_step 을 늘려도 VRAM 이 아니라 **시간만** 늘었다.
    # ★ 상한은 _recompute_logp 다 — 거긴 그래디언트가 살아 있어 활성값이
    #   배치에 비례한다. OOM 이면 이 값을 4/2 로 낮춘다 (수집·평가는 no_grad
    #   라 여유가 크지만 같은 손잡이로 통일했다).
    policy_batch_size: int = 8
    # 초기 상태 뱅크 (개정 §3). 학습용 뱅크를 쓰고, 평가는 별도 홀드아웃을 쓴다.
    bank: str = "train"
    # 커리큘럼 어닐링 스케줄: [[시작스텝, 뱅크이름], ...] (시작스텝 오름차순).
    # 비어 있으면 `bank` 를 끝까지 쓴다.
    #
    # ★ 왜 어닐링인가 — 고정 혼합비는 커리큘럼이 아니다.
    #   reverse curriculum(Florensa 2017) 계열의 동력은 "쉬운 시작 상태로 신호를
    #   만든 뒤, 성공률이 오르면 원래 분포로 되돌리는" 것이다. 되돌리지 않으면
    #   커리큘럼 분포에서만 잘하고 목표 분포로 전이된다는 보장이 없다.
    #
    # ⚠ 뱅크는 **행 순서가 섞여 있어야** 한다. sample_bank 는 커리큘럼 행을
    #   앞쪽에 몰아 넣는데(k < n_curriculum), 학습은 커서 0 부터 순차로 순회하므로
    #   섞지 않으면 "30% 혼합" 뱅크가 실제로는 **100% 커리큘럼**으로 소비된다.
    #   (train_mix 가 그랬다 — 인덱스 0~1228 이 전부 커리큘럼)
    #   train_c30/c20/c10 은 생성 후 셔플해 두었다.
    bank_schedule: list = field(default_factory=list)

    # --- 성공률 기반 커리큘럼 어닐링 (bank_schedule 보다 우선한다) ---
    # bank_stages 가 비어 있지 않으면 스텝 기준(bank_schedule) 대신 이쪽을 쓴다.
    #
    # ★ 왜 스텝이 아니라 성공률인가
    #   reverse curriculum 의 정석은 "정책이 실제로 잘하게 됐을 때" 난이도를
    #   올리는 것이다. 스텝 기준은 정책 상태와 무관하게 시간표대로 밀어붙이므로,
    #   아직 못 푸는데 어려워지거나 이미 쉬운데 오래 머무르는 일이 생긴다.
    #
    # ★ 판정은 eval_base 홀드아웃으로만 한다 (평가 시점에만 갱신).
    #   학습 롤아웃 성공률은 스텝당 16궤적이라 노이즈가 크고, 무엇보다
    #   **커리큘럼이 섞인 분포**라 "목표 분포에서 잘하게 됐는가" 를 답하지 못한다.
    #   eval_success_rate 는 diag["success"] 기반이라 단계형 보상에 오염되지 않는다.
    #
    # ★ 승급 전용(monotonic). 강등은 넣지 않는다 — 평가 지점이 적어(10스텝마다)
    #   진동하면 회복할 기회가 없다.
    bank_stages: list = field(default_factory=list)
    # eval_base 성공률이 이 값 이상이면 다음 단계로 승급.
    promote_eval_success: float = 0.30
    # 한 단계에 이 스텝 수를 넘게 머무르면 성공률과 무관하게 승급한다.
    # ★ 없으면 임계에 영영 못 미칠 때 원래 분포(마지막 단계)에 도달하지 못해
    #   목표 분포 성능을 아예 못 재게 된다. 마감이 있는 실행에서는 필수다.
    #   0 이면 데드라인 없음(성공률로만 승급).
    promote_deadline_steps: int = 20

    # 액션 언노멀라이즈에 쓸 데이터셋 통계 키. 체크포인트의 norm_stats 안에 있어야
    # 한다 (SFT 가 dataset_statistics.json 으로 함께 저장한다).
    unnorm_key: str = "vla_pick"

    # --- LoRA ---
    # ★ 체크포인트 본체는 SFT LoRA 가 **병합된** 가중치다
    #   (scripts/sft/run_sft_libero_spec.sh 가 --merge_lora_during_training True,
    #    openvla-oft/vla-scripts/finetune.py:653-661 이 merge_and_unload 후 저장).
    #   그래서 from_pretrained 하면 7.54B 전 파라미터가 requires_grad=True 가 되고,
    #   그대로 두면 **LoRA 가 아니라 풀 파인튜닝**이 된다 (설계 의도와 다르고
    #   옵티마이저 상태만 30GB 다). 여기서 새 어댑터를 붙여 학습면을 좁힌다.
    #
    #   ⚠ ckpt/sft/lora_adapter/ 를 다시 얹으면 **이중 적용**이다. 그건 이미
    #     본체에 병합돼 있다. 반드시 새 어댑터를 0 에서 시작한다.
    #     (init_lora_weights="gaussian" 은 A 만 가우시안, B 는 0 → 붙인 직후
    #      정책은 SFT 와 수치적으로 동일하다. --verify-checkpoint 로 확인 가능)
    use_lora: bool = True
    # SFT 와 같은 형상. rank 를 바꾸면 SFT 가 쓰던 방향을 표현하지 못하거나
    # (낮출 때) 검증되지 않은 방향이 열린다(높일 때). 평가 분산이 ±10%p 라
    # rank 차이는 어차피 측정되지 않으므로 축을 늘리지 않는다.
    # (SimpleVLA-RL 은 LoRA 를 쓰지 않는다 — 8×A800 FSDP 풀 파인튜닝이라
    #  물려받을 rank 값 자체가 없다. 근거는 SFT 쪽에서 가져왔다.)
    lora_rank: int = 32
    lora_alpha: int = 16          # α/r = 0.5. SFT 와 동일한 보수적 스케일
    lora_dropout: float = 0.0
    # SFT 어댑터가 건드린 15개 모듈 그대로. LLM(q/k/v/o/gate/up/down)뿐 아니라
    # 비전 백본(qkv/proj/fc1/fc2 — DINOv2 + SigLIP 둘 다)과 lm_head 를 포함한다.
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv", "proj", "fc1", "fc2", "fc3", "q", "kv", "lm_head",
        ]
    )

    # 최적화
    learning_rate: float = 1e-6
    clip_eps: float = 0.2
    # clip-higher (DAPO). 상한을 넓혀 저확률 토큰의 상승 여지를 준다 —
    # 대칭 클리핑은 엔트로피를 빠르게 죽여 탐색이 멎는다. 개정 §5: [0.8, 1.28]
    clip_eps_high: float = 0.28
    kl_coef: float = 0.0          # 0 = KL 항 없음 (SimpleVLA-RL 기본)
    max_grad_norm: float = 1.0
    # 같은 롤아웃 배치로 밟는 그래디언트 스텝 수 (PPO inner epoch).
    #   1 = ratio≡1 이라 클리핑이 죽는다 → 실효 REINFORCE (+그룹 baseline).
    #   2+ = 두 번째 패스부터 ratio≠1 → 클리핑(트러스트 리전)이 실제로 작동한다.
    # SimpleVLA-RL 은 512궤적에 4회(mini_batch 128×4). 우리는 16궤적이라
    # 배치 대비 재사용 비율을 낮게(2) 잡는다 — 작은 배치에 4회는 과적합이다.
    ppo_epochs: int = 1
    total_steps: int = 300

    # 로깅/체크포인트
    log_dir: str = "logs/grpo"
    save_every: int = 25
    eval_every: int = 25          # 0 이면 끈다. 이제 실제로 쓰인다 (train() 참조)
    eval_episodes: int = 16       # eval_base 홀드아웃에서 뽑을 에피소드 수
    wandb_project: str = ""

    device: str = "cuda:0"
    seed: int = 0

    extra: dict = field(default_factory=dict)


def load_config(path: Path) -> GRPOConfig:
    """YAML 이 있으면 YAML, 없으면 JSON 으로 읽는다."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        raw = yaml.safe_load(text)
    except ImportError:
        raw = json.loads(text)
    known = {f for f in GRPOConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in raw.items() if k in known}
    kwargs["extra"] = {k: v for k, v in raw.items() if k not in known}
    return GRPOConfig(**kwargs)


# =============================================================================
def compute_group_advantages(rewards: np.ndarray) -> np.ndarray:
    """그룹 내 보상 정규화 → advantage.

    GRPO 의 핵심이자 critic 이 필요 없는 이유다. 같은 초기 상태에서 나온
    궤적들끼리만 비교하므로 상태 가치를 추정할 필요가 없다.

    Args:
        rewards: (num_groups, group_size) 0/1 보상

    Returns:
        같은 shape 의 advantage.

    그룹 전체가 성공(전부 1)이거나 전멸(전부 0)이면 표준편차가 0 이라
    학습 신호가 없다. 이 경우 advantage 를 0 으로 둔다 — 0 으로 나눠 NaN 이
    되면 그 스텝뿐 아니라 모델 가중치 전체가 오염된다.
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    adv = np.zeros_like(rewards)
    valid = std.squeeze(1) > 1e-6
    adv[valid] = (rewards[valid] - mean[valid]) / (std[valid] + 1e-8)
    return adv


def degenerate_fraction(rewards: np.ndarray) -> float:
    """학습 신호가 없는 그룹의 비율. 이게 계속 1.0 이면 커브가 오르지 않는다."""
    std = rewards.std(axis=1)
    return float((std <= 1e-6).mean())


def _diag_mean(diags: list, key: str) -> float:
    """워커 diag 들의 평균. 값이 하나도 없으면 nan (빈 리스트 경고 없이).

    ★ 반드시 **시도한 그룹 전체**의 diag 를 넘길 것. 유효 그룹만 넘기면
      전멸 그룹이 통계에서 빠져 숫자가 낙관적으로 부풀려진다.
    """
    vals = [d[key] for d in diags if isinstance(d, dict) and key in d]
    return float(np.mean(vals)) if vals else float("nan")


def is_degenerate(reward_row: np.ndarray) -> bool:
    """그룹 하나가 전멸(전부 0) 또는 전승(전부 1)인가 → advantage 가 0 이다."""
    return bool(np.std(reward_row) <= 1e-6)


def action_logits(
    model, processor, images, instructions, *, device, num_patches, n_action_tokens
):
    """관측 **배치** → 액션 구간 로짓 (B, n_action_tokens, vocab).

    ★ 2026-08-16 배치화. 이전에는 샘플 1개씩 순차로 돌아 96GB GPU 중 35GB 만
      쓰고 있었다. 모델 쪽은 원래 배치를 지원한다 —
      modeling_prismatic 의 forward() 멀티모달 경로가 `(B, seq_len, D)` 전제이고
      _prepare_input/labels_for_action_prediction 도 전부 `.shape[0]` 기반이다.
      ("Only batch size == 1 supported right now" 주석은 실제 코드와 맞지 않는
       stale 주석이다. B>1 을 막는 곳은 KV캐시 생성과 generate() 뿐인데 RFT 는
       둘 다 쓰지 않는다.)

    Args:
        images: (B, H, W, 3) uint8 또는 길이 B 시퀀스.
        instructions: 길이 B 문자열 시퀀스.

    ★ OpenVLA-OFT 는 **parallel decoding** 으로 학습된다. placeholder 액션 토큰
      56개(= 청크 8 × 액션 7)와 stop 토큰을 프롬프트 뒤에 붙여 **한 번의 forward**
      로 전체 청크를 예측하고, 액션 구간의 어텐션은 causal 이 아니라 양방향이다.
      그래서 `model.generate()` 로 한 토큰씩 뽑으면 **학습된 적 없는 방식**으로
      모델을 굴리게 된다 — 에러는 안 나고 정책 분포만 조용히 달라진다.
      (증상: 롤아웃 성공률이 eval_rollout 로 잰 SFT 성공률과 다르고, GRPO 의
       ratio 가 모델이 구현하지도 않는 분포를 비교한다)

      입력 구성은 모델 자신의 헬퍼(_prepare_input_for_action_prediction,
      _prepare_labels_for_action_prediction)에 맡긴다. 손으로 흉내 내면 상류가
      배치를 바꿀 때 조용히 어긋난다.

    ★ predict_action 이 forward 전에 하는 일이 **3가지**다. 하나라도 빠지면 안 된다:
        1) 특수 빈 토큰 29871 을 프롬프트 끝에 붙인다. 학습 때 "OUT:" 뒤에
           있던 토큰이라, 없으면 액션 구간의 위치가 한 칸 밀린다.
        2) placeholder 액션 토큰 56개 + stop 토큰을 붙인다.
        3) **labels 를 만든다.** 액션 마스크(_process_action_masks)가 이걸로
           "어디가 액션 구간인가"를 센다. labels 없이 forward 하면
           `None != IGNORE_INDEX` 가 파이썬 bool 이 되어
           `cumsum() received an invalid combination of arguments -
            got (bool, dim=int)` 로 죽는다 — torch 버전 문제로 오독하기 쉽다.

    로짓 슬라이스 위치가 predict_action 과 같은지는 --verify-checkpoint 로 확인한다.
    """
    import torch

    from prismatic.vla.constants import IGNORE_INDEX

    prompts = [SPEC.build_prompt(ins) for ins in instructions]
    pils = [SPEC.prepare_image_for_vla(img) for img in images]
    inputs = processor(prompts, pils).to(device, dtype=torch.bfloat16)

    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    batch = input_ids.shape[0]

    # ★ 프롬프트 길이가 전 행 동일한지 확인한다 — 배치화의 유일한 전제다.
    #   아래 num_prompt_tokens 는 **스칼라**라, 행마다 길이가 다르면 액션 구간
    #   슬라이스가 어긋난 채 **에러 없이** 엉뚱한 로짓을 정책으로 쓰게 된다.
    #   (이번 프로젝트에서 반복해서 당한 "조용히 실패" 유형이다)
    #   학습 경로는 SPEC.INSTRUCTION_TEMPLATE 하나로 고정이고 색 이름
    #   red/blue/green 이 모두 단일 토큰이라 22토큰으로 균일하다. 다만
    #   INSTRUCTION_TEMPLATES_EVAL 중 하나는 24토큰이라 섞이면 깨진다.
    _lens = attention_mask.sum(dim=-1)
    if batch > 1 and not bool((_lens == _lens[0]).all()):
        raise RuntimeError(
            f"배치 안에서 프롬프트 길이가 다르다: {_lens.tolist()}. "
            "액션 구간 슬라이스가 행마다 어긋나므로 배치 forward 를 쓸 수 없다. "
            "지시문 템플릿을 하나로 고정하거나, 좌측 패딩 + 행별 슬라이스로 "
            "action_logits 를 확장할 것."
        )

    # (1) 특수 빈 토큰. predict_action 과 같은 조건·같은 방식으로 붙인다.
    #     ★ 배치에서는 (B,1) 로 만들어야 한다. predict_action 은 (1,1) 로
    #       하드코딩돼 있어 B>1 이면 cat 이 dim0 불일치로 죽는다.
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (
                input_ids,
                torch.full(
                    (batch, 1), 29871, dtype=input_ids.dtype, device=input_ids.device
                ),
            ),
            dim=1,
        )

    # placeholder 를 붙이기 **전** 길이여야 한다 (predict_action 과 동일 규약).
    # ★ 29871 을 붙인 **뒤**에 세는 것도 규약의 일부다 — predict_action 이 그렇다.
    num_prompt_tokens = input_ids.shape[-1] - 1

    # (3) 액션 마스크용 labels. 아래 _prepare_input_for_action_prediction 이
    #     input_ids 를 늘리기 전 길이로 만들어야 한다 (predict_action 과 동일 순서).
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX

    # (2) placeholder 액션 토큰 + stop 토큰.
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs["pixel_values"],
        labels=labels,
    )
    start = num_patches + num_prompt_tokens
    # ★ 배치 차원을 보존한다. 예전에는 [0, ...] 로 첫 행만 집었다.
    logits = out.logits[:, start : start + n_action_tokens, :]
    if logits.shape[1] != n_action_tokens:
        raise RuntimeError(
            f"액션 구간 로짓이 {logits.shape[1]}개다 (기대 {n_action_tokens}). "
            f"시퀀스 길이 {out.logits.shape[1]}, 슬라이스 시작 {start}. "
            "num_patches 계산이나 프롬프트 길이 규약이 상류와 어긋났다."
        )
    return logits


def tokens_to_action(ids, vocab_size: int, bin_centers, action_stats) -> np.ndarray:
    """액션 토큰 → 연속 액션. openvla-oft 의 predict_action 과 같은 산술이다.

    모델 없이도 검사할 수 있도록 순수 함수로 뺐다 (--self-check).

    ★ 세 군데가 틀리기 쉽다. 셋 다 에러 없이 액션만 조용히 어긋난다:
      1. vocab_size 는 `config.vocab_size`(패딩 포함)가 아니라
         `text_config.vocab_size - pad_to_multiple_of` 다. 틀리면 pad 크기
         (보통 64)만큼 bin 이 통째로 밀린다.
      2. `-1` 이 필요하다. digitize 결과가 [1, n_bins] 인데 bin_centers 는
         n_bins-1 개뿐이다.
      3. 선형 환산이 아니라 **bin 의 중심값**이다 (반 bin 오프셋).
    """
    ids = np.asarray(ids)
    centers = np.asarray(bin_centers)
    idx = np.clip(vocab_size - ids - 1, 0, centers.shape[0] - 1)
    n = SPEC.NUM_ACTIONS_CHUNK * SPEC.ACTION_DIM
    normalized = centers[idx][:n].reshape(SPEC.NUM_ACTIONS_CHUNK, SPEC.ACTION_DIM)

    low = np.array(action_stats["q01"], dtype=np.float32)
    high = np.array(action_stats["q99"], dtype=np.float32)
    mask = np.array(
        action_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool
    )
    denorm = 0.5 * (normalized + 1.0) * (high - low) + low
    return np.where(mask, denorm, normalized).astype(np.float32)


def collect_groups(collect_fn, groups_per_step: int, max_attempts: int, cursor: int):
    """유효 그룹이 목표 개수를 채울 때까지 뽑는다 (DAPO 의 dynamic sampling).

    ★ 왜 건너뛰기만으로는 부족한가
      전멸/전승 그룹은 advantage 가 0 이라 그래디언트에 기여하지 않는다. 그냥
      건너뛰면 그 스텝의 유효 배치가 줄고, 성공률이 낮은 초기에는
      **업데이트가 거의 일어나지 않은 채 스텝만 흐른다.**
      SimpleVLA-RL/DAPO 는 버린 만큼 다시 뽑아 배치를 채운다.

    ★ 상한이 필요한 이유
      정책이 태스크를 아예 못 풀면 유효 그룹이 영원히 안 나온다. 상한 없이는
      한 스텝에서 무한히 롤아웃을 돌게 된다 — 롤아웃이 전체 시간의 대부분인
      구조라 그대로 밤을 날린다. 상한에 닿으면 모은 것으로 업데이트하고 넘어가고,
      그 사실은 로그의 group_attempts 로 드러난다.

    Args:
        collect_fn: `init_index -> (rewards(N,), payload)` 를 돌려주는 함수.
        cursor: 다음에 쓸 초기 상태 뱅크 인덱스. 시도마다 1씩 나아간다 —
            재시도에 같은 s₀ 를 다시 쓰면 같은 결과가 나와 재샘플링이 무의미하다.

    Returns:
        (used_rewards, used_payloads, all_rewards, cursor)
        all_rewards 는 버린 것까지 포함한다 (degenerate 비율 계산용).
    """
    used_rewards, used_payloads, all_rewards = [], [], []
    attempts = 0
    while len(used_rewards) < groups_per_step and attempts < max_attempts:
        rewards, payload = collect_fn(cursor)
        cursor += 1
        attempts += 1
        all_rewards.append(rewards)
        if not is_degenerate(rewards):
            used_rewards.append(rewards)
            used_payloads.append(payload)
    return used_rewards, used_payloads, all_rewards, cursor


# =============================================================================
class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        print(f"[grpo] 정책 로드: {cfg.checkpoint}")
        self.processor = AutoProcessor.from_pretrained(
            cfg.checkpoint, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(cfg.device)

        # LoRA 파라미터만 학습한다. 7B 전체를 열면 48GB 에 옵티마이저 상태가 안 들어간다.
        n_total = sum(p.numel() for p in self.model.parameters())
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            # ★ 병합된 체크포인트 위에 **새** 어댑터를 얹는다. 기존
            #   ckpt/sft/lora_adapter/ 를 얹으면 이중 적용이다 (이미 병합돼 있다).
            #   B=0 초기화라 붙인 직후 정책은 SFT 와 수치적으로 동일하다.
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=list(cfg.lora_target_modules),
                    init_lora_weights="gaussian",
                    bias="none",
                ),
            )

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "학습 가능한 파라미터가 없다. LoRA 어댑터가 붙은 체크포인트인지 확인할 것 "
                "(SFT 시 merge_lora_during_training=True 였다면 새로 LoRA 를 붙여야 한다)."
            )
        n_train = sum(p.numel() for p in trainable)
        # ★ 비율을 함께 찍는다. 100% 로 나오면 LoRA 가 안 붙어 풀 파인튜닝 중이라는
        #   뜻이다 — 에러 없이 메모리만 폭증하고 학습 성격이 달라지는 종류라
        #   로그 한 줄로 즉시 구분되게 둔다.
        print(f"[grpo] 학습 파라미터 {n_train / 1e6:.1f}M / 전체 {n_total / 1e9:.2f}B "
              f"({n_train / n_total * 100:.2f}%)"
              + (f" — LoRA r={cfg.lora_rank}, α={cfg.lora_alpha}"
                 if cfg.use_lora else " — ⚠ 풀 파인튜닝"))

        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)

        # 초기 상태 뱅크 커서. 그룹을 하나 뽑을 때마다 나아간다 (재샘플링 포함).
        self._init_cursor = 0
        # 한 스텝에서 시도한 그룹들의 diag (버린 그룹 포함). train() 이 스텝마다 비운다.
        self._step_diags: list = []
        # 현재 스텝에서 쓸 초기 상태 뱅크 (어닐링이 갱신한다).
        # bank_stages 를 쓰면 첫 단계에서, 아니면 cfg.bank 에서 시작한다.
        self._stage_idx: int = 0
        self._stage_entered_step: int = 0     # 현재 단계에 들어온 스텝 (데드라인 계산용)
        self._active_bank: str = (
            str(cfg.bank_stages[0]) if cfg.bank_stages else cfg.bank
        )

        # --- parallel decoding 에 필요한 상수 (체크포인트에서 읽는다) ---
        # 액션 토큰 수 = 청크 × 액션 차원. OFT 는 이만큼을 한 번에 예측한다.
        self._n_action_tokens = SPEC.NUM_ACTIONS_CHUNK * SPEC.ACTION_DIM
        # 로짓에서 액션 구간을 찾으려면 앞의 비전 패치 수를 알아야 한다.
        self._num_patches = (
            self.model.vision_backbone.get_num_patches()
            * self.model.vision_backbone.get_num_images_in_input()
        )
        if not hasattr(self.model, "_prepare_input_for_action_prediction"):
            raise RuntimeError(
                "체크포인트가 OpenVLA-OFT 의 parallel decoding 인터페이스를 노출하지 "
                "않는다 (_prepare_input_for_action_prediction 없음). "
                "trust_remote_code 로 로드된 modeling_prismatic.py 를 확인할 것."
            )
        if self.cfg.unnorm_key not in self.model.norm_stats:
            raise RuntimeError(
                f"unnorm_key '{self.cfg.unnorm_key}' 가 체크포인트의 norm_stats 에 "
                f"없다. 있는 키: {list(self.model.norm_stats)}. SFT 때의 "
                "dataset_statistics.json 이 체크포인트에 함께 저장됐는지 확인할 것."
            )
        print(f"[grpo] parallel decoding: 액션 토큰 {self._n_action_tokens}개, "
              f"비전 패치 {self._num_patches}개, unnorm_key '{cfg.unnorm_key}'")

        self.client = RolloutClient(
            isaaclab_python=cfg.isaaclab_python,
            worker_script=REPO_ROOT / "rft" / "isaaclab_rollout_worker.py",
            num_envs=cfg.num_envs,
            task=cfg.task,
            device="cuda:0",
        )
        self.client.start()

        self.writer = None
        if cfg.wandb_project:
            import wandb

            wandb.init(project=cfg.wandb_project, config=cfg.__dict__)
            self.writer = wandb

    # -------------------------------------------------------------------------
    def bank_for_step(self, step: int) -> str:
        """어닐링 스케줄에서 이 스텝의 뱅크를 고른다.

        스케줄이 비어 있으면 cfg.bank 를 그대로 쓴다. 스케줄이 있으면
        `시작스텝 <= step` 인 항목 중 마지막 것을 쓴다.
        """
        bank = self.cfg.bank
        for entry in self.cfg.bank_schedule or []:
            start, name = entry[0], entry[1]
            if step >= int(start):
                bank = str(name)
        return bank

    # -------------------------------------------------------------------------
    def maybe_promote(self, step: int, eval_success: float | None) -> str | None:
        """성공률 기반 커리큘럼 승급을 판정한다. 승급했으면 사유 문자열, 아니면 None.

        두 가지 경로로 승급한다:
          1. eval_base 성공률이 promote_eval_success 이상  ← 본래 의도
          2. 현재 단계 체류가 promote_deadline_steps 초과  ← 마감 대비 안전장치

        Args:
            step: 현재 스텝.
            eval_success: 방금 나온 eval_base 성공률. 평가가 없던 스텝이면 None.

        ★ 승급 전용이다. 성공률이 떨어져도 되돌아가지 않는다 — 평가가 드물어
          (10스텝마다) 진동하면 회복할 기회가 없기 때문이다.
        ★ 마지막 단계에 도달하면 더 올라갈 곳이 없으므로 항상 None 을 돌려준다.
        """
        stages = self.cfg.bank_stages or []
        if not stages or self._stage_idx >= len(stages) - 1:
            return None

        dwell = step - self._stage_entered_step
        reason = None
        if eval_success is not None and eval_success >= self.cfg.promote_eval_success:
            reason = (f"eval 성공률 {eval_success:.1%} ≥ "
                      f"{self.cfg.promote_eval_success:.0%}")
        elif self.cfg.promote_deadline_steps > 0 and dwell >= self.cfg.promote_deadline_steps:
            reason = f"데드라인 {dwell}스텝 ≥ {self.cfg.promote_deadline_steps}"

        if reason is None:
            return None

        self._stage_idx += 1
        self._stage_entered_step = step
        prev = self._active_bank
        self._active_bank = str(stages[self._stage_idx])
        # 뱅크가 바뀌면 커서를 0 으로 돌린다. 새 뱅크의 인덱스 공간이라
        # 이전 커서를 이어 쓰면 앞부분을 통째로 건너뛰게 된다.
        self._init_cursor = 0
        return f"{prev} → {self._active_bank} ({reason})"

    # -------------------------------------------------------------------------
    def _sample_actions(self, images: np.ndarray, instructions):
        """관측 배치 → (액션 청크, 샘플된 토큰, behavior log-prob).

        수집 단계는 그래디언트 없이 돈다. **업데이트 때 같은 (이미지, 토큰) 쌍으로
        log-prob 을 다시 계산해야** ratio 에 그래디언트가 흐른다 — 수집 때 만든
        log-prob 을 그대로 쓰면 ratio 가 항상 1 이 되어 학습이 전혀 일어나지 않는다.
        (조용히 도는 것처럼 보이는 종류의 버그라 명시해 둔다)

        계획서 §Phase4b-6 대로 토큰별 log-prob 을 처음부터 기록한다 — GRPO ratio 에
        필요할 뿐 아니라 KL 예산을 따질 때도 이 로그가 있어야 한다.

        ★ 액션 청크 하나가 **1회 forward** 로 나온다 (parallel decoding).
          예전 구현은 generate() 로 56스텝을 돌았다 — 학습 방식과 다를 뿐 아니라
          56배 느렸다.
        """
        import torch

        tokens_all, logps_all = [], []

        # ★ 배치 forward. 마이크로배치로 잘라 도는 이유는 메모리 상한 때문이다
        #   (수집은 no_grad 라 여유가 크지만, 같은 손잡이로 통일해 둔다).
        for lo in range(0, images.shape[0], max(self.cfg.policy_batch_size, 1)):
            hi = min(lo + max(self.cfg.policy_batch_size, 1), images.shape[0])
            with torch.no_grad():
                logits = self._action_logits(images[lo:hi], instructions[lo:hi])

            # 위치별 categorical 샘플링. 56개 위치가 서로 독립이다 —
            # parallel decoding 이라 앞 토큰이 뒤 토큰의 조건이 아니다.
            # (b, 56, V) → multinomial 은 2D 만 받으므로 (b*56, V) 로 펴서 뽑는다.
            logp_all = torch.log_softmax(logits.float() / self.cfg.temperature, dim=-1)
            b, n, v = logp_all.shape
            seq = torch.multinomial(
                logp_all.exp().reshape(b * n, v), num_samples=1
            ).reshape(b, n)

            # 궤적별 log-prob 은 56개 위치의 합 (dim=-1).
            tokens_all.append(seq)
            logps_all.append(
                logp_all.gather(-1, seq.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            )

        tokens = torch.cat(tokens_all, dim=0)
        logps = torch.cat(logps_all, dim=0)
        actions = np.stack([self._tokens_to_action(tokens[i]) for i in range(tokens.shape[0])])
        return actions, tokens, logps

    def _action_logits(self, images, instructions):
        """관측 배치 → 액션 구간 로짓 (B, 56, vocab). 샘플링·logp 재계산·평가가 공유한다."""
        return action_logits(
            self.model,
            self.processor,
            images,
            instructions,
            device=self.cfg.device,
            num_patches=self._num_patches,
            n_action_tokens=self._n_action_tokens,
        )

    def _recompute_logp(self, images: np.ndarray, tokens, instructions):
        """저장해 둔 (이미지, 토큰) 으로 현재 정책의 log-prob 을 **그래디언트와 함께** 계산.

        샘플링을 다시 하는 게 아니라 이미 샘플된 토큰의 확률을 현재 파라미터로
        다시 평가한다. 이게 PPO/GRPO ratio 의 분자다.

        parallel decoding 이라 위치가 고정이므로 causal 처럼 한 칸 밀어 정렬할
        필요가 없다 — 액션 구간 로짓에서 그대로 gather 한다.

        ★ temperature 는 샘플링과 여기에 **둘 다** 적용한다. 정책 π 의 정의가
          "temperature 로 스케일된 분포" 이기 때문이다. 한쪽만 적용하면 첫 스텝의
          ratio 가 1 이 아니게 되어, 업데이트 전부터 클리핑이 걸린다.
          (veRL 도 학습 시 로짓을 같은 온도로 나눈다)
        """
        import torch

        # ★ 여기는 그래디언트가 살아 있어 활성값이 배치 크기에 비례한다.
        #   policy_batch_size 로 잘라 도는 것이 OOM 방어의 유일한 손잡이다.
        logps = []
        bs = max(self.cfg.policy_batch_size, 1)
        for lo in range(0, images.shape[0], bs):
            hi = min(lo + bs, images.shape[0])
            logits = self._action_logits(images[lo:hi], instructions[lo:hi])
            logp_all = torch.log_softmax(logits.float() / self.cfg.temperature, dim=-1)
            picked = logp_all.gather(-1, tokens[lo:hi].unsqueeze(-1)).squeeze(-1)
            logps.append(picked.sum(dim=-1))

        return torch.cat(logps, dim=0)

    def _tokens_to_action(self, token_ids) -> np.ndarray:
        """액션 토큰 → 연속 액션.

        ★ 모델이 이미 들고 있는 값만 쓴다 (openvla-oft 의 predict_action 과 동일).
          손으로 다시 계산하면 아래 세 가지가 조용히 어긋난다:
            - `config.vocab_size` 는 **패딩된** 값이다. 규약이 요구하는 것은
              `text_config.vocab_size - pad_to_multiple_of` (= model.vocab_size).
              둘이 다르면 pad_to_multiple_of(보통 64)만큼 bin 이 통째로 밀린다.
            - `-1` 이 필요하다. np.digitize 가 [1, n_bins] 를 돌려주는데
              bin_centers 는 n_bins-1 개뿐이다.
            - 선형 환산이 아니라 **bin 의 중심값**이다 (반 bin 오프셋).
        """
        # 언노멀라이즈도 모델의 norm_stats 를 쓴다. 같은 숫자의 출처가 둘이면
        # 언젠가 갈라지고, 그때 증상은 "RFT 만 성능이 다르다" 로 나타난다.
        return tokens_to_action(
            token_ids.detach().cpu().numpy(),
            vocab_size=self.model.vocab_size,
            bin_centers=np.asarray(self.model.bin_centers),
            action_stats=self.model.norm_stats[self.cfg.unnorm_key]["action"],
        )

    # -------------------------------------------------------------------------
    def collect_group(self, init_index: int):
        """한 그룹(**같은 초기 상태**에서 num_envs 개 궤적)을 수집한다.

        ★ 초기 상태는 시드가 아니라 뱅크 인덱스로 지정한다 (개정 §3).
          시드만 맞추면 배치 안의 env 들이 서로 다른 배치를 받아, advantage 가
          "정책이 잘했는가"가 아니라 "이 env 가 쉬웠는가"를 재게 된다.
          정수 하나를 넘기면 전 env 가 동일한 s₀ 에서 출발한다 — GRPO 의 전제.

        업데이트에 필요한 (이미지, 토큰, 지시문) 을 함께 들고 나온다 — 그래야
        _recompute_logp 로 ratio 를 만들 수 있다. 지시문이 빠지면 프롬프트가
        수집 때와 달라져 ratio 가 무의미해진다.

        메모리 주의: 궤적당 스텝 수 × num_envs 개의 224² 이미지를 들고 있게 된다.
        max_steps_per_episode 를 키우면 여기가 먼저 터진다.
        """
        obs = self.client.reset(init_index=init_index, bank=self._active_bank)
        steps = int(np.ceil(self.cfg.max_steps_per_episode / SPEC.NUM_ACTIONS_CHUNK))

        transitions = []      # [(images, tokens, behavior_logp, instructions), ...]
        rewards = np.zeros(self.cfg.num_envs, dtype=np.float32)

        for _ in range(steps):
            images, instr = obs["image"], obs["instruction"]
            chunk, tokens, logp = self._sample_actions(images, instr)
            transitions.append((images, tokens, logp.detach(), instr))
            obs, rewards, done = self.client.step(chunk)
            if bool(done.all()):
                break

        return rewards, transitions, dict(self.client.last_diag)

    def _collect_for_sampling(self, init_index: int):
        """collect_groups 가 요구하는 형태로 감싼다 → (보상, 나머지 페이로드).

        ★ diag 는 여기서 따로 쌓는다. collect_groups 는 무신호 그룹의 페이로드를
          버리므로, 반환값만 보면 **버려진 그룹의 진단이 사라진다** — 그러면
          성공률·파지율이 "유효 그룹만" 기준이 되어 낙관적으로 부풀려진다.
          (transitions 는 이미지가 들어 있어 버리는 게 맞다. diag 만 남긴다)
        """
        rewards, transitions, diag = self.collect_group(init_index)
        self._step_diags.append(diag)
        return rewards, (transitions, diag)

    # -------------------------------------------------------------------------
    def evaluate(self, num_episodes: int) -> dict:
        """eval_base 홀드아웃에서 greedy 성공률을 잰다 (학습 스텝 사이에 낀다).

        ★ 왜 eval_rollout.py 를 서브프로세스로 부르지 않는가
          그러면 Isaac Sim 워커가 하나 더 뜬다 — 셰이더 캐시가 있어도 기동에
          수십 초~수 분이 들고, 단일 GPU에서 학습용 워커와 VRAM/렌더를
          나눠 써야 한다. 대신 학습에 이미 떠 있는 self.client 를 그대로 쓰고
          뱅크만 bank(train/train_mix) → eval_base 로 바꿔 reset 한다 —
          추가 프로세스 기동 비용이 0이다.

        ★ 왜 샘플링이 아니라 greedy 인가
          평가 프로토콜은 eval_rollout.py 와 동일해야 SFT/RFT 곡선이 비교
          가능하다 (개정 §6). RFT 롤아웃의 multinomial 샘플링을 그대로 쓰면
          eval_every 로 잰 "성공률"이 eval_rollout.py 로 잰 값과 어긋난다.

        비용: eval_episodes 를 작게 유지할 것 (기본 16). num_envs=8 이면
        라운드 2개 = 학습 그룹 2개를 도는 것과 같은 시간이라, eval_every=25~40
        스텝마다 한 번이면 전체 학습 시간에 몇 % 수준의 지연만 더한다.
        """
        import torch

        was_training = self.model.training
        self.model.eval()
        t0 = time.time()

        num_envs = self.cfg.num_envs
        num_rounds = int(np.ceil(num_episodes / num_envs))
        steps_per_round = int(np.ceil(self.cfg.max_steps_per_episode / SPEC.NUM_ACTIONS_CHUNK))

        successes: list[float] = []
        # ★ 단계 도달률을 함께 잰다. 원래 분포에서 성공률은 0% 라 그것만 보면
        #   커브가 평평한 직선이 되어 아무 정보가 없다. 인수인계 문서도 목표를
        #   "최종 성공률이 아니라 단계 도달률 개선" 으로 잡으라고 적었다.
        #   래치는 에피소드 안에서 단조 증가하므로 라운드 최댓값을 쓴다
        #   (eval_rollout.py 와 같은 방식).
        stages = {k: [] for k in ("grasped", "lifted", "in_tray")}
        try:
            for rnd in range(num_rounds):
                # eval_rollout.py 와 같은 인덱스 스킴 — 같은 뱅크·순서라 SFT 결과와
                # 직접 비교된다.
                indices = [
                    (rnd * num_envs + i) % SPEC.EVAL_HOLDOUT_SIZE for i in range(num_envs)
                ]
                obs = self.client.reset(init_indices=indices, bank="eval_base")
                done = np.zeros(num_envs, dtype=bool)
                round_stage = {k: 0.0 for k in stages}

                for _ in range(steps_per_round):
                    images, instr = obs["image"], obs["instruction"]
                    chunk = np.zeros(
                        (num_envs, SPEC.NUM_ACTIONS_CHUNK, SPEC.ACTION_DIM),
                        dtype=np.float32,
                    )
                    # 배치 forward + greedy. 학습 롤아웃과 같은 경로를 쓴다.
                    bs = max(self.cfg.policy_batch_size, 1)
                    with torch.no_grad():
                        for lo in range(0, num_envs, bs):
                            hi = min(lo + bs, num_envs)
                            logits = self._action_logits(images[lo:hi], instr[lo:hi])
                            ids = logits.argmax(dim=-1).cpu().numpy()   # (b, 56)
                            for j in range(hi - lo):
                                chunk[lo + j] = tokens_to_action(
                                    ids[j],
                                    vocab_size=self.model.vocab_size,
                                    bin_centers=np.asarray(self.model.bin_centers),
                                    action_stats=self.model.norm_stats[
                                        self.cfg.unnorm_key
                                    ]["action"],
                                )
                    obs, _, done = self.client.step(chunk)
                    for k in round_stage:
                        round_stage[k] = max(
                            round_stage[k], float(self.client.last_diag.get(k, 0.0))
                        )
                    if bool(done.all()):
                        break
                for k, v in round_stage.items():
                    stages[k].append(v)

                # diag 의 이동평균이 아니라 워커가 들고 있는 정확한 래치를 읽는다
                # (에피소드 단위 성공 플래그, N개 bool).
                successes.extend(np.asarray(self.client.get_success()).tolist())
        finally:
            if was_training:
                self.model.train()

        rate = float(np.mean(successes)) if successes else 0.0
        out = {
            "eval_success_rate": rate,
            "eval_episodes": len(successes),
            "eval_elapsed_s": time.time() - t0,
        }
        for k, v in stages.items():
            out[f"eval_{k}_frac"] = float(np.mean(v)) if v else float("nan")
        return out

    # -------------------------------------------------------------------------
    def train(self) -> int:
        cfg = self.cfg
        history = []
        t0 = time.time()

        torch = self.torch

        max_attempts = cfg.max_group_attempts or (3 * cfg.groups_per_step)
        if not cfg.dynamic_sampling:
            # 예전 동작: 정해진 횟수만 뽑고 무신호 그룹은 아래에서 건너뛴다.
            max_attempts = cfg.groups_per_step

        for step in range(cfg.total_steps):
            # 뱅크 인덱스로 s₀ 를 지정한다. 그룹 안에서는 전 env 가 동일한 s₀ 이고
            # (GRPO 전제), 커서는 시도마다 나아간다 — 재시도에 같은 s₀ 를 다시
            # 쓰면 같은 결과가 나와 재샘플링이 무의미해진다.
            # 커리큘럼 어닐링 — 이 스텝에서 쓸 뱅크를 정한다.
            # bank_stages 를 쓰면 승급은 **평가 직후**에만 일어나므로(아래 참조)
            # 여기서는 데드라인만 본다. bank_stages 가 없으면 예전 스텝 기준.
            if cfg.bank_stages:
                promoted = self.maybe_promote(step, eval_success=None)
                if promoted:
                    print(f"[grpo] 커리큘럼 승급: {promoted} (step {step})")
            else:
                prev_bank = self._active_bank
                self._active_bank = self.bank_for_step(step)
                if self._active_bank != prev_bank:
                    print(f"[grpo] 커리큘럼 어닐링: 뱅크 {prev_bank} → {self._active_bank} "
                          f"(step {step})")
                    # 뱅크가 바뀌면 커서를 0 으로 돌린다. 새 뱅크의 인덱스 공간이라
                    # 이전 커서를 이어 쓰면 앞부분을 통째로 건너뛰게 된다.
                    self._init_cursor = 0

            # 이 스텝에서 시도한 모든 그룹의 diag (버린 것 포함). 위 참조.
            self._step_diags = []
            used, payloads, attempted, self._init_cursor = collect_groups(
                collect_fn=self._collect_for_sampling,
                groups_per_step=cfg.groups_per_step,
                max_attempts=max_attempts,
                cursor=self._init_cursor,
            )
            group_transitions = [p[0] for p in payloads]

            # ★ 진단·성공률은 **버린 그룹까지 포함해** 센다.
            #   유효 그룹만 세면 전멸 그룹이 통계에서 사라져 숫자가 부풀려진다 —
            #   재샘플링을 켠 순간 로그가 조용히 낙관적으로 변하는 함정이다.
            #   (_collect_for_sampling 이 시도마다 self._step_diags 에 쌓는다)
            all_diags = list(self._step_diags)
            attempted_arr = np.stack(attempted)
            degen = degenerate_fraction(attempted_arr)

            # ★ reward 평균을 "성공률" 이라 부르면 안 된다.
            #   VLA_STAGED_REWARD=1 이면 reward 가 0.2/0.4/0.7/1.0 이라 평균이
            #   0.31 처럼 나오는데, 그것은 "31% 성공" 이 아니라 "파지·리프트까지
            #   간 에피소드가 섞인 평균 보상" 이다. 실제 outcome 성공률은 워커가
            #   diag["success"] 로 따로 보낸다 (eval_rollout.py 와 같은 규약).
            #   인수인계 문서가 함정으로 명시한 오독이라 둘을 분리해 기록한다.
            mean_reward = float(attempted_arr.mean())
            success_rate = _diag_mean(all_diags, "success")

            # 유효 그룹이 하나도 없으면 group_transitions 도 비어 있어 아래 업데이트
            # 루프가 그냥 돌지 않는다. 그 스텝은 로그만 남는다 —
            # SFT 베이스라인이 게이트(30%)를 못 넘겼다는 신호다.
            rewards = np.stack(used) if used else attempted_arr
            advantages = compute_group_advantages(rewards)

            # --- 정책 업데이트 (PPO inner epoch) ---
            # advantage 는 궤적 단위다. 궤적 안의 모든 액션 청크가 같은 advantage 를
            # 공유한다 (sparse 보상이라 credit assignment 를 더 쪼갤 근거가 없다).
            #
            # ★ epoch 1 에서는 파라미터가 수집 시점과 같아 ratio ≡ 1 이고, loss 는
            #   -mean(adv) = 0 으로 찍힌다 (그룹 정규화라 평균 0). 그래디언트는
            #   0 이 아니다 — 이 epoch 만 돌리면 실효 REINFORCE 다.
            # ★ epoch 2+ 는 첫 optimizer.step 이후라 ratio ≠ 1 이고, 그때부터
            #   클리핑(트러스트 리전)이 실제로 물린다. 우리 ratio 는 56토큰 logp 의
            #   **합**의 지수라 작은 드리프트도 증폭된다 — clip_frac 으로 감시한다.
            total_loss = 0.0
            num_terms = 0
            ratio_max = 1.0
            clip_hits = 0
            clip_total = 0

            for _ppo_epoch in range(max(cfg.ppo_epochs, 1)):
                self.optimizer.zero_grad()
                epoch_terms = 0

                for gi, transitions in enumerate(group_transitions):
                    adv_row = torch.tensor(
                        advantages[gi], device=cfg.device, dtype=torch.float32
                    )
                    # advantage 가 전부 0 인 그룹은 그래디언트가 0 이므로 건너뛴다
                    # (forward 비용만 나가고 얻는 게 없다).
                    if float(adv_row.abs().max()) < 1e-8:
                        continue

                    for images, tokens, behavior_logp, instructions in transitions:
                        new_logp = self._recompute_logp(images, tokens, instructions)
                        ratio = torch.exp(new_logp - behavior_logp.to(new_logp.dtype))
                        with torch.no_grad():
                            ratio_max = max(ratio_max, float(ratio.max()))
                            clip_hits += int(
                                (
                                    (ratio < 1 - cfg.clip_eps)
                                    | (ratio > 1 + cfg.clip_eps_high)
                                ).sum()
                            )
                            clip_total += int(ratio.numel())
                        unclipped = ratio * adv_row
                        clipped = (
                            torch.clamp(
                                ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps_high
                            )
                            * adv_row
                        )
                        loss = -torch.min(unclipped, clipped).mean()

                        if cfg.kl_coef > 0:
                            # 사전학습 정책 대비 KL 예산. SimpleVLA-RL 기본은 0 이지만,
                            # 정책이 발산하면 여기를 켜는 것이 첫 번째 대응이다.
                            approx_kl = (behavior_logp.to(new_logp.dtype) - new_logp).mean()
                            loss = loss + cfg.kl_coef * approx_kl

                        # 청크 수로 나눠 스케일을 맞춘 뒤 즉시 backward —
                        # 전체 그래프를 들고 있으면 VRAM 이 터진다.
                        (loss / max(len(transitions), 1)).backward()
                        total_loss += float(loss.item())
                        num_terms += 1
                        epoch_terms += 1

                if epoch_terms:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                    self.optimizer.step()

            mean_loss = total_loss / num_terms if num_terms else 0.0
            clip_frac = clip_hits / clip_total if clip_total else 0.0
            elapsed = time.time() - t0
            record = {
                "step": step,
                # outcome 성공률 (diag["success"]). 보고는 반드시 이 값으로 한다.
                "success_rate": success_rate,
                # 단계형 보상의 평균. 학습 신호의 세기를 보는 값이지 성공률이 아니다.
                "mean_reward": mean_reward,
                "loss": mean_loss,
                "degenerate_group_frac": degen,
                # 진단: 성공률이 낮을 때 "블록을 집어 들지 못하는 것"인지
                # "포켓에 못 넣는 것"인지 가른다 (보상에는 섞지 않는다).
                "grasped_frac": _diag_mean(all_diags, "grasped"),
                "lifted_frac": _diag_mean(all_diags, "lifted"),
                "in_tray_frac": _diag_mean(all_diags, "in_tray"),
                "yaw_err": _diag_mean(all_diags, "yaw_err"),
                # 트러스트 리전 감시. clip_frac 은 클리핑 경계 밖으로 나간 궤적
                # ratio 의 비율 (epoch 1 은 ratio≡1 이라 기여 0). 이게 0.5 를
                # 넘으면 한 스텝의 이동이 너무 크다 — lr 을 내릴 것.
                "ratio_max": ratio_max,
                "clip_frac": clip_frac,
                # 어느 뱅크로 학습했는지. 어닐링 구간을 나중에 되짚으려면 필요하다.
                "bank": self._active_bank,
                "stage_idx": self._stage_idx,
                "updated_terms": num_terms,
                # 재샘플링 비용. 이게 안 보이면 상한을 조정할 근거가 없다.
                "groups_used": len(used),
                "group_attempts": len(attempted),
                "elapsed_min": elapsed / 60,
            }
            history.append(record)
            # ★ 성공률과 평균보상을 **나란히** 찍는다. 하나만 찍으면 단계형 보상을
            #   켰을 때 어느 쪽인지 알 수 없다 (0.31 을 "31% 성공" 으로 읽는 사고).
            print(
                f"[grpo] step {step:4d} | 성공 {success_rate:.1%} | "
                f"보상 {mean_reward:.3f} | loss {mean_loss:+.4f} | "
                f"무신호그룹 {degen:.0%} | 그룹 {len(used)}/{len(attempted)} | "
                f"파지 {record['grasped_frac']:.0%} / 리프트 {record['lifted_frac']:.0%} "
                f"/ 진입 {record['in_tray_frac']:.0%} | "
                f"clip {clip_frac:.0%} r≤{ratio_max:.2f} | "
                f"{elapsed / 60:.1f}분"
            )
            if len(used) < cfg.groups_per_step:
                print(f"       ⚠ 유효 그룹 {len(used)}/{cfg.groups_per_step} — "
                      f"시도 {len(attempted)}회로 배치를 못 채웠다. 그룹이 전멸/전승으로 "
                      "쏠린다는 뜻이다. temperature 를 올리거나(탐색↑), "
                      "SFT 베이스라인이 30% 게이트를 넘는지 먼저 확인할 것.")

            # eval_base 홀드아웃 평가. save_every 와 같은 주기로 맞춰 두면
            # "이 체크포인트가 몇 % 였는가"를 바로 대응시킬 수 있다.
            if cfg.eval_every > 0 and cfg.eval_episodes > 0 and (step + 1) % cfg.eval_every == 0:
                eval_result = self.evaluate(cfg.eval_episodes)
                record.update(eval_result)
                print(
                    f"       [eval] eval_base 성공 {eval_result['eval_success_rate']:.1%} | "
                    f"파지 {eval_result['eval_grasped_frac']:.0%} / "
                    f"리프트 {eval_result['eval_lifted_frac']:.0%} / "
                    f"진입 {eval_result['eval_in_tray_frac']:.0%} "
                    f"({eval_result['eval_episodes']}에피소드, greedy) — "
                    f"{eval_result['eval_elapsed_s']:.0f}s"
                )
                # ★ 승급 판정은 여기서만 한다. 판정 기준이 eval_base 성공률이라
                #   평가가 갱신되는 시점 외에는 새 정보가 없다.
                #   다음 스텝부터 새 뱅크가 적용된다(이번 스텝의 롤아웃은 이미 끝).
                if cfg.bank_stages:
                    promoted = self.maybe_promote(
                        step + 1, eval_success=eval_result["eval_success_rate"]
                    )
                    if promoted:
                        print(f"       [grpo] 커리큘럼 승급: {promoted}")

            if self.writer:
                self.writer.log(record)
            (self.log_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            # save_every <= 0 이면 저장하지 않는다 (0 을 그대로 % 에 넣으면
            # ZeroDivisionError 로 학습이 통째로 죽는다 — 저장을 끄려다 잃는다).
            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                out = self.log_dir / f"checkpoint-{step + 1}"
                # ★ LoRA 를 쓰면 여기서 저장되는 것은 **어댑터뿐**이다(약 484MB).
                #   덕분에 저장이 빨라 학습 지연이 거의 없지만, 나중에 평가할 때는
                #   베이스(ckpt/sft) 위에 이 어댑터를 얹어야 한다 — 이 디렉터리만
                #   단독으로 eval_rollout.py 에 넘기면 모델을 못 찾는다.
                self.model.save_pretrained(out)
                self.processor.save_pretrained(out)
                print(f"       체크포인트 저장: {out}"
                      + ("  (LoRA 어댑터만 — 평가 시 베이스 위에 얹을 것)"
                         if cfg.use_lora else ""))

        self.client.close()
        return 0


# =============================================================================
def self_check() -> int:
    """dynamic sampling 수집 루프 점검. GPU·Isaac Sim 없이 돈다.

    여기서 깨지면 RFT 가 "도는 것처럼 보이는데 학습이 안 되는" 상태가 된다 —
    로그만으로는 구분이 어려운 종류라 실행 가능한 검사를 남긴다.
    """
    # (1) 무신호 그룹은 버리고 다시 뽑아 배치를 채운다.
    seen = []

    def scripted(idx):
        seen.append(idx)
        # 인덱스 0,1,2 는 전멸(무신호), 3부터 신호 있음.
        row = np.zeros(4, dtype=np.float32) if idx < 3 else np.array(
            [1, 0, 1, 0], dtype=np.float32
        )
        return row, f"payload-{idx}"

    used, payloads, attempted, cursor = collect_groups(scripted, 2, 10, cursor=0)
    assert len(used) == 2, f"유효 그룹 {len(used)} != 2 — 재샘플링이 안 됐다."
    assert payloads == ["payload-3", "payload-4"], payloads
    assert len(attempted) == 5, f"시도 {len(attempted)} != 5 (버린 3 + 쓴 2)"
    assert cursor == 5 and seen == [0, 1, 2, 3, 4], (
        f"커서가 안 나아갔다: {seen} → 재시도에 같은 s₀ 를 다시 쓰게 된다."
    )

    # (2) 전부 무신호여도 상한에서 멈춘다 (무한 루프 방지).
    calls = []

    def all_degenerate(idx):
        calls.append(idx)
        return np.ones(4, dtype=np.float32), None      # 전승 = 무신호

    used, _, attempted, cursor = collect_groups(all_degenerate, 2, 6, cursor=100)
    assert used == [] and len(attempted) == 6 and cursor == 106, (
        f"상한에서 멈추지 않았다: used={len(used)} attempts={len(attempted)}"
    )

    # (3) degenerate 비율은 **시도 전체** 기준이어야 한다 (버린 것 포함).
    frac = degenerate_fraction(np.stack([np.zeros(4), np.array([1, 0, 1, 0])]))
    assert abs(frac - 0.5) < 1e-9, frac

    # (4) 토큰 → 액션 왕복. OpenVLA 규약을 그대로 재현하는지 본다.
    n_bins = 256
    bins = np.linspace(-1, 1, n_bins)
    centers = (bins[:-1] + bins[1:]) / 2.0          # 255개
    vocab_size = 32000                              # text_config - pad_to_multiple_of
    n = SPEC.NUM_ACTIONS_CHUNK * SPEC.ACTION_DIM
    identity = {"q01": [-1.0] * SPEC.ACTION_DIM, "q99": [1.0] * SPEC.ACTION_DIM,
                "mask": [True] * SPEC.ACTION_DIM}

    # bin i 를 가리키는 토큰 id 는 vocab_size - (i + 1) 이다 (디코드의 역).
    for i in (0, 1, 127, centers.shape[0] - 1):
        ids = np.full(n, vocab_size - (i + 1))
        act = tokens_to_action(ids, vocab_size, centers, identity)
        assert act.shape == (SPEC.NUM_ACTIONS_CHUNK, SPEC.ACTION_DIM), act.shape
        assert np.allclose(act, centers[i], atol=1e-6), (
            f"bin {i} 왕복 실패: {act[0, 0]} != {centers[i]}"
        )

    # 선형 환산과 혼동하지 않았는지 — 반 bin 만큼 달라야 한다.
    linear = 0 / (n_bins - 1) * 2.0 - 1.0
    assert abs(centers[0] - linear) > 1e-4, (
        "bin 중심이 선형 환산과 같다 — centers 계산이 틀렸다."
    )

    # 범위 밖 토큰이 들어와도 죽지 않고 끝 bin 으로 잘려야 한다.
    edge = tokens_to_action(np.full(n, vocab_size + 10), vocab_size, centers, identity)
    assert np.allclose(edge, centers[0]), edge[0, 0]

    # 언노멀라이즈: [-1,1] → [q01,q99] 선형 사상.
    stats = {"q01": [0.0] * SPEC.ACTION_DIM, "q99": [2.0] * SPEC.ACTION_DIM,
             "mask": [True] * SPEC.ACTION_DIM}
    mid = tokens_to_action(np.full(n, vocab_size - 128), vocab_size, centers, stats)
    assert np.allclose(mid, 0.5 * (centers[127] + 1.0) * 2.0, atol=1e-6), mid[0, 0]

    # (5) 설정 필드가 학습 루프에서 **실제로 쓰이는지**.
    #     eval_every 는 한동안 GRPOConfig 에 있기만 하고 train() 어디에서도
    #     읽히지 않았다 — 설정에 값을 적어도 평가가 돌지 않는데 에러도 안 났다.
    #     필드 추가만으로는 재발을 막지 못하므로 소스를 직접 확인한다.
    import inspect

    # (5b) 커리큘럼 어닐링 스케줄이 실제로 뱅크를 바꾸는지.
    class _FakeCfg:
        bank = "train"
        bank_schedule = [[0, "train_c30"], [30, "train_c20"], [60, "train_c10"], [90, "train"]]

    _t = GRPOTrainer.__new__(GRPOTrainer)
    _t.cfg = _FakeCfg()
    for step, want in [(0, "train_c30"), (29, "train_c30"), (30, "train_c20"),
                       (89, "train_c10"), (90, "train"), (200, "train")]:
        got = _t.bank_for_step(step)
        assert got == want, f"어닐링 스케줄 오류: step {step} → {got} (기대 {want})"
    assert GRPOTrainer.__new__(GRPOTrainer).__class__ is GRPOTrainer
    # 스케줄이 비면 cfg.bank 를 그대로 써야 한다.
    _t.cfg.bank_schedule = []
    assert _t.bank_for_step(50) == "train"

    # (5c) 성공률 기반 승급 (maybe_promote).
    class _StageCfg:
        bank = "train"
        bank_schedule = []
        bank_stages = ["train_c30", "train_c20", "train_c10", "train"]
        promote_eval_success = 0.30
        promote_deadline_steps = 20

    def _fresh():
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.cfg = _StageCfg()
        t._stage_idx = 0
        t._stage_entered_step = 0
        t._active_bank = "train_c30"
        t._init_cursor = 123          # 승급 시 0 으로 리셋되는지 확인용
        return t

    # 임계 미달 + 데드라인 전 → 승급 없음
    t = _fresh()
    assert t.maybe_promote(5, eval_success=0.20) is None
    assert t._active_bank == "train_c30" and t._init_cursor == 123

    # 임계 도달 → 승급, 커서 리셋
    t = _fresh()
    assert t.maybe_promote(5, eval_success=0.30) is not None
    assert t._active_bank == "train_c20", t._active_bank
    assert t._init_cursor == 0, "승급 시 커서를 리셋하지 않았다"
    assert t._stage_entered_step == 5

    # 데드라인 초과 → 성공률 낮아도 승급
    t = _fresh()
    assert t.maybe_promote(20, eval_success=0.05) is not None
    assert t._active_bank == "train_c20"
    # 평가가 없는 스텝(None)에서도 데드라인은 작동해야 한다
    t = _fresh()
    assert t.maybe_promote(20, eval_success=None) is not None

    # 마지막 단계에서는 더 올라가지 않는다
    t = _fresh()
    t._stage_idx = 3
    t._active_bank = "train"
    assert t.maybe_promote(999, eval_success=1.0) is None
    assert t._active_bank == "train"

    # bank_stages 가 비면 승급 로직은 아무것도 하지 않는다 (스텝 기준으로 폴백)
    t = _fresh()
    t.cfg.bank_stages = []
    assert t.maybe_promote(999, eval_success=1.0) is None

    # 4단계를 데드라인만으로 끝까지 승급하면 마지막이 train 이어야 한다
    t = _fresh()
    for s in (20, 40, 60):
        t.maybe_promote(s, eval_success=0.0)
    assert t._active_bank == "train" and t._stage_idx == 3, t._active_bank

    train_src = inspect.getsource(GRPOTrainer.train)
    for field in ("save_every", "eval_every", "eval_episodes", "ppo_epochs",
                  "bank_stages"):
        assert f"cfg.{field}" in train_src, (
            f"GRPOConfig.{field} 가 train() 에서 쓰이지 않는다 — 설정에 값을 적어도 "
            "조용히 무시된다. 필드만 있고 배선이 없는 상태로 되돌아갔다."
        )
    # 평가는 반드시 홀드아웃 뱅크로. cfg.bank(train_mix)로 평가하면 커리큘럼
    # 성능이 섞여 원래 분포 성능이 부풀려진다.
    eval_src = inspect.getsource(GRPOTrainer.evaluate)
    assert '"eval_base"' in eval_src, (
        "evaluate() 가 eval_base 홀드아웃을 쓰지 않는다 — 학습 뱅크로 평가하면 "
        "커리큘럼이 섞여 숫자가 부풀려진다 (개정 §6)."
    )

    print("자체 검사 통과 (재샘플링 / 상한 / 커서 전진 / 토큰→액션 왕복 / "
          "설정 배선).")
    return 0


def verify_checkpoint(checkpoint: str, device: str, unnorm_key: str) -> int:
    """우리 정책 경로가 모델의 predict_action 과 같은 액션을 내는지 확인한다.

    ★ 이 검사가 이 파일에서 가장 중요하다.
      `_action_logits` 는 로짓에서 액션 구간을 **위치로** 잘라낸다
      (num_patches + num_prompt_tokens). 그 규약이 상류의 predict_action 과
      어긋나면 아무 에러 없이 엉뚱한 위치의 로짓을 정책으로 쓰게 된다.
      greedy 로 맞춰 두 경로를 직접 비교하는 것이 유일하게 확실한 확인이다.

    체크포인트가 필요하므로 GPU 인스턴스에서 돈다.
    """
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"[verify] 체크포인트 로드: {checkpoint}")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    n_tokens = SPEC.NUM_ACTIONS_CHUNK * SPEC.ACTION_DIM
    num_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    print(f"[verify] 액션 토큰 {n_tokens}, 비전 패치 {num_patches}, "
          f"vocab_size {model.vocab_size} (config {model.config.text_config.vocab_size} "
          f"- pad {model.config.pad_to_multiple_of}), bin_centers "
          f"{np.asarray(model.bin_centers).shape}")

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (SPEC.IMAGE_HEIGHT, SPEC.IMAGE_WIDTH, 3), dtype=np.uint8)
    instruction = SPEC.TASK_INSTRUCTION

    with torch.no_grad():
        # 우리 경로: 같은 로짓에서 greedy (샘플링만 빼면 롤아웃과 동일한 경로다)
        # ★ 배치 함수가 됐으므로 (1,H,W,3) / 길이 1 리스트로 감싼다.
        logits = action_logits(
            model, processor, image[None], [instruction],
            device=device, num_patches=num_patches, n_action_tokens=n_tokens,
        )[0]
        ours = tokens_to_action(
            logits.argmax(dim=-1).cpu().numpy(),
            vocab_size=model.vocab_size,
            bin_centers=np.asarray(model.bin_centers),
            action_stats=model.norm_stats[unnorm_key]["action"],
        )
        # 기준 경로: 모델 자신의 API
        inputs = processor(
            SPEC.build_prompt(instruction), SPEC.prepare_image_for_vla(image)
        ).to(device, dtype=torch.bfloat16)
        # ★ 반환 형태가 리비전마다 다르다. 이 체크포인트의 predict_action 은
        #   **튜플** (액션 ndarray, 히든스테이트 cuda Tensor) 를 돌려준다.
        #   그대로 np.asarray 하면 "can't convert cuda:0 device type tensor to
        #   numpy" 로 죽는다. eval_rollout.py 와 같은 방식으로 첫 원소만 쓴다 —
        #   두 곳의 해석이 갈라지면 비교 자체가 무의미해진다.
        ref = model.predict_action(**inputs, unnorm_key=unnorm_key)
        if isinstance(ref, tuple):
            ref = ref[0]
        if isinstance(ref, torch.Tensor):
            ref = ref.detach().float().cpu().numpy()
        ref = np.asarray(ref, dtype=np.float32).reshape(-1, SPEC.ACTION_DIM)[
            : SPEC.NUM_ACTIONS_CHUNK
        ]

    diff = float(np.abs(ours - ref).max())
    print(f"[verify] 최대 절대차 {diff:.3e}")
    print(f"  ours[0] = {np.round(ours[0], 4)}")
    print(f"  ref [0] = {np.round(ref[0], 4)}")
    if diff > 1e-3:
        print(
            "\n[verify] ✗ 두 경로가 다르다. 로짓 슬라이스 위치나 디토크나이즈가 "
            "상류와 어긋났다.\n"
            "  - 슬라이스: action_logits() 의 num_patches + num_prompt_tokens\n"
            "  - 디토크나이즈: tokens_to_action() 의 vocab_size / -1 / bin_centers\n"
            "  체크포인트의 modeling_prismatic.py 안 predict_action 과 대조할 것."
        )
        return 1
    print("\n[verify] ✓ 정책 경로가 predict_action 과 일치한다.")

    # -----------------------------------------------------------------------
    # 배치 ↔ 순차 동치 시험 — forward 배치화의 핵심 회귀 시험이다.
    #
    # 배치화는 "값은 그대로 두고 처리량만 올리는" 변경이어야 한다. 만약 슬라이스
    # 위치나 패딩 때문에 행마다 다른 로짓이 나오면 정책 분포가 조용히 달라지고,
    # 증상은 "커브가 안 오른다" 로만 보인다. 같은 관측을 배치로 한 번, 순차로
    # B 번 돌려 로짓이 일치하는지 직접 확인한다.
    # -----------------------------------------------------------------------
    B = 8
    imgs = rng.integers(
        0, 255, (B, SPEC.IMAGE_HEIGHT, SPEC.IMAGE_WIDTH, 3), dtype=np.uint8
    )
    # 지시문도 섞는다 — 색이 달라도 토큰 길이가 같아야 배치가 성립한다.
    instrs = [SPEC.instruction_for(i % SPEC.NUM_BLOCKS) for i in range(B)]

    with torch.no_grad():
        batched = action_logits(
            model, processor, imgs, instrs,
            device=device, num_patches=num_patches, n_action_tokens=n_tokens,
        )
        seq = torch.cat([
            action_logits(
                model, processor, imgs[i][None], [instrs[i]],
                device=device, num_patches=num_patches, n_action_tokens=n_tokens,
            )
            for i in range(B)
        ], dim=0)

    # ★ 판정은 **로짓 수준**에서 한다. 토큰(argmax) 완전일치를 요구하면 안 된다.
    #
    #   bf16 은 배치 크기에 따라 GEMM 타일링과 누적 순서가 달라져 로짓이 미세하게
    #   달라진다(실측 0.19). 그 결과 **근소한 동점** 위치에서 argmax 가 뒤집힌다.
    #   실측: 21/448 토큰이 뒤집혔고, 뒤집힌 위치의 top1-top2 격차는 중앙값 0.031
    #   최대 0.125 로 **전부 수치 오차보다 작았다**. 즉 모델이 사실상 무차별한
    #   지점들이다. 액션 공간 % 로 재면 그리퍼(사실상 이진, 범위 2.0)가 한 번만
    #   뒤집혀도 99% 로 보여 판정이 무의미해진다.
    #
    #   반면 **진짜 버그**(슬라이스 위치 오류, 패딩 혼입)는 서로 다른 위치의 로짓을
    #   비교하게 되므로 로짓 차이가 O(10) 이상으로 벌어지고 거의 모든 토큰이 어긋난다.
    #   따라서 로짓 최대차 하나로 두 경우가 깨끗하게 갈린다.
    ldiff = float((batched.float() - seq.float()).abs().max())
    n_flip = int((batched.argmax(-1) != seq.argmax(-1)).sum())
    gaps = []
    fi, fj = (batched.argmax(-1) != seq.argmax(-1)).nonzero(as_tuple=True)
    for i, j in zip(fi.tolist(), fj.tolist()):
        top2 = torch.topk(batched[i, j].float(), 2).values
        gaps.append(float(top2[0] - top2[1]))
    max_gap = max(gaps) if gaps else 0.0

    print(f"[verify] 배치({B}) ↔ 순차: 로짓 최대차 {ldiff:.4f}, "
          f"토큰 불일치 {n_flip}/{B * n_tokens}, "
          f"불일치 지점의 top1-top2 격차 최대 {max_gap:.4f}")

    TOL = 1.0   # bf16 노이즈(~0.2)는 통과, 슬라이스 오류(O(10))는 탈락
    if ldiff > TOL:
        print(
            f"\n[verify] ✗ 배치와 순차의 로짓이 {ldiff:.3f} 만큼 다르다 "
            f"(허용 {TOL}). 수치 오차가 아니라 로직 오류다.\n"
            "  - 프롬프트 길이 균일성 검사가 통과했는지\n"
            "  - out.logits[:, start:start+n, :] 슬라이스가 맞는지\n"
            "  - processor 가 패딩을 넣지 않았는지 (attention_mask 합 확인)"
        )
        return 1
    if gaps and max_gap > ldiff * 2:
        # 뒤집힘이 동점이 아닌 곳에서 났다면 수치 오차로 설명되지 않는다.
        print(f"\n[verify] ⚠ 불일치가 동점 지점이 아니다 "
              f"(격차 {max_gap:.3f} > 로짓오차 {ldiff:.3f}의 2배). 확인 필요.")
        return 1
    print("[verify] ✓ 배치 forward 가 순차와 일치한다 "
          "(불일치는 전부 동점 지점의 bf16 수치 오차).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="수집 루프·디토크나이즈만 점검하고 끝낸다 (GPU 불필요)")
    parser.add_argument("--verify-checkpoint", action="store_true",
                        help="정책 경로가 predict_action 과 일치하는지 확인하고 끝낸다 "
                             "(--checkpoint 필요, GPU 필요)")
    # 아래 둘은 --verify-checkpoint 전용이다. 학습은 config 의 값을 쓴다.
    parser.add_argument("--unnorm-key", default="vla_pick",
                        help="(--verify-checkpoint 전용)")
    parser.add_argument("--device", default="cuda:0",
                        help="(--verify-checkpoint 전용)")
    args = parser.parse_args()

    if args.self_check:
        return self_check()
    if args.verify_checkpoint:
        if not args.checkpoint:
            parser.error("--verify-checkpoint 에는 --checkpoint 가 필요하다.")
        return verify_checkpoint(args.checkpoint, args.device, args.unnorm_key)
    if args.config is None:
        parser.error("--config 가 필요하다 (또는 --self-check).")

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    if args.total_steps:
        cfg.total_steps = args.total_steps
    if not cfg.checkpoint:
        parser.error("체크포인트가 필요하다 (config 또는 --checkpoint).")

    print(SPEC.summary())
    return GRPOTrainer(cfg).train()


if __name__ == "__main__":
    sys.exit(main())

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
    # 초기 상태 뱅크 (개정 §3). 학습용 뱅크를 쓰고, 평가는 별도 홀드아웃을 쓴다.
    bank: str = "train"
    # 액션 언노멀라이즈에 쓸 데이터셋 통계 키. 체크포인트의 norm_stats 안에 있어야
    # 한다 (SFT 가 dataset_statistics.json 으로 함께 저장한다).
    unnorm_key: str = "vla_pick"

    # 최적화
    learning_rate: float = 1e-6
    clip_eps: float = 0.2
    # clip-higher (DAPO). 상한을 넓혀 저확률 토큰의 상승 여지를 준다 —
    # 대칭 클리핑은 엔트로피를 빠르게 죽여 탐색이 멎는다. 개정 §5: [0.8, 1.28]
    clip_eps_high: float = 0.28
    kl_coef: float = 0.0          # 0 = KL 항 없음 (SimpleVLA-RL 기본)
    max_grad_norm: float = 1.0
    total_steps: int = 300

    # 로깅/체크포인트
    log_dir: str = "logs/grpo"
    save_every: int = 25
    eval_every: int = 25
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


def is_degenerate(reward_row: np.ndarray) -> bool:
    """그룹 하나가 전멸(전부 0) 또는 전승(전부 1)인가 → advantage 가 0 이다."""
    return bool(np.std(reward_row) <= 1e-6)


def action_logits(
    model, processor, image, instruction, *, device, num_patches, n_action_tokens
):
    """관측 하나 → 액션 구간 로짓 (n_action_tokens, vocab).

    ★ OpenVLA-OFT 는 **parallel decoding** 으로 학습된다. placeholder 액션 토큰
      56개(= 청크 8 × 액션 7)와 stop 토큰을 프롬프트 뒤에 붙여 **한 번의 forward**
      로 전체 청크를 예측하고, 액션 구간의 어텐션은 causal 이 아니라 양방향이다.
      그래서 `model.generate()` 로 한 토큰씩 뽑으면 **학습된 적 없는 방식**으로
      모델을 굴리게 된다 — 에러는 안 나고 정책 분포만 조용히 달라진다.
      (증상: 롤아웃 성공률이 eval_rollout 로 잰 SFT 성공률과 다르고, GRPO 의
       ratio 가 모델이 구현하지도 않는 분포를 비교한다)

      입력 구성은 모델 자신의 헬퍼(_prepare_input_for_action_prediction)에 맡긴다.
      손으로 흉내 내면 상류가 배치를 바꿀 때 조용히 어긋난다.

    로짓 슬라이스 위치가 predict_action 과 같은지는 --verify-checkpoint 로 확인한다.
    """
    import torch

    inputs = processor(
        SPEC.build_prompt(instruction), SPEC.prepare_image_for_vla(image)
    ).to(device, dtype=torch.bfloat16)

    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    # placeholder 를 붙이기 **전** 길이여야 한다 (predict_action 과 동일 규약).
    num_prompt_tokens = input_ids.shape[-1] - 1
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs["pixel_values"],
    )
    start = num_patches + num_prompt_tokens
    logits = out.logits[0, start : start + n_action_tokens, :]
    if logits.shape[0] != n_action_tokens:
        raise RuntimeError(
            f"액션 구간 로짓이 {logits.shape[0]}개다 (기대 {n_action_tokens}). "
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
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "학습 가능한 파라미터가 없다. LoRA 어댑터가 붙은 체크포인트인지 확인할 것 "
                "(SFT 시 merge_lora_during_training=True 였다면 새로 LoRA 를 붙여야 한다)."
            )
        n_train = sum(p.numel() for p in trainable) / 1e6
        print(f"[grpo] 학습 파라미터 {n_train:.1f}M / "
              f"전체 {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B")

        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)

        # 초기 상태 뱅크 커서. 그룹을 하나 뽑을 때마다 나아간다 (재샘플링 포함).
        self._init_cursor = 0

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

        actions, tokens, logps = [], [], []

        for i in range(images.shape[0]):
            with torch.no_grad():
                logits = self._action_logits(images[i], instructions[i])

            # 위치별 categorical 샘플링. 56개 위치가 서로 독립이다 —
            # parallel decoding 이라 앞 토큰이 뒤 토큰의 조건이 아니다.
            logp_all = torch.log_softmax(logits.float() / self.cfg.temperature, dim=-1)
            seq = torch.multinomial(logp_all.exp(), num_samples=1).squeeze(-1)

            logps.append(logp_all.gather(-1, seq.unsqueeze(-1)).squeeze(-1).sum())
            tokens.append(seq)
            actions.append(self._tokens_to_action(seq))

        return np.stack(actions), torch.stack(tokens), torch.stack(logps)

    def _action_logits(self, image: np.ndarray, instruction: str):
        """관측 하나 → 액션 구간 로짓. 샘플링과 logp 재계산이 이걸 공유한다."""
        return action_logits(
            self.model,
            self.processor,
            image,
            instruction,
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

        logps = []
        for i in range(images.shape[0]):
            logits = self._action_logits(images[i], instructions[i])
            logp_all = torch.log_softmax(logits.float() / self.cfg.temperature, dim=-1)
            picked = logp_all.gather(-1, tokens[i].unsqueeze(-1)).squeeze(-1)
            logps.append(picked.sum())

        return torch.stack(logps)

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
        obs = self.client.reset(init_index=init_index, bank=self.cfg.bank)
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
        """collect_groups 가 요구하는 형태로 감싼다 → (보상, 나머지 페이로드)."""
        rewards, transitions, diag = self.collect_group(init_index)
        return rewards, (transitions, diag)

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
            used, payloads, attempted, self._init_cursor = collect_groups(
                collect_fn=self._collect_for_sampling,
                groups_per_step=cfg.groups_per_step,
                max_attempts=max_attempts,
                cursor=self._init_cursor,
            )
            group_transitions = [p[0] for p in payloads]
            group_diags = [p[1] for p in payloads]

            # ★ 성공률과 무신호 비율은 **버린 그룹까지 포함해** 센다.
            #   유효 그룹만 세면 전멸 그룹이 통계에서 사라져 성공률이 부풀려진다 —
            #   재샘플링을 켠 순간 로그가 조용히 낙관적으로 변하는 함정이다.
            attempted_arr = np.stack(attempted)
            degen = degenerate_fraction(attempted_arr)
            success_rate = float(attempted_arr.mean())

            # 유효 그룹이 하나도 없으면 group_transitions 도 비어 있어 아래 업데이트
            # 루프가 그냥 돌지 않는다. 그 스텝은 로그만 남는다 —
            # SFT 베이스라인이 게이트(30%)를 못 넘겼다는 신호다.
            rewards = np.stack(used) if used else attempted_arr
            advantages = compute_group_advantages(rewards)

            # --- 정책 업데이트 (PPO 클리핑) ---
            # advantage 는 궤적 단위다. 궤적 안의 모든 액션 청크가 같은 advantage 를
            # 공유한다 (sparse 0/1 보상이라 credit assignment 를 더 쪼갤 근거가 없다).
            self.optimizer.zero_grad()
            total_loss = 0.0
            num_terms = 0

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

            if num_terms:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

            mean_loss = total_loss / num_terms if num_terms else 0.0
            elapsed = time.time() - t0
            record = {
                "step": step,
                "success_rate": success_rate,
                "loss": mean_loss,
                "degenerate_group_frac": degen,
                # 진단: 성공률이 낮을 때 "블록을 집어 들지 못하는 것"인지
                # "포켓에 못 넣는 것"인지 가른다 (보상에는 섞지 않는다).
                "lifted_frac": float(
                    np.mean([d.get("lifted", np.nan) for d in group_diags])
                ),
                "yaw_err": float(
                    np.mean([d.get("yaw_err", np.nan) for d in group_diags])
                ),
                "updated_terms": num_terms,
                # 재샘플링 비용. 이게 안 보이면 상한을 조정할 근거가 없다.
                "groups_used": len(used),
                "group_attempts": len(attempted),
                "elapsed_min": elapsed / 60,
            }
            history.append(record)
            print(
                f"[grpo] step {step:4d} | 성공률 {success_rate:.1%} | "
                f"loss {mean_loss:+.4f} | 무신호그룹 {degen:.0%} | "
                f"그룹 {len(used)}/{len(attempted)} | "
                f"리프트 {record['lifted_frac']:.0%} | "
                f"{elapsed / 60:.1f}분"
            )
            if len(used) < cfg.groups_per_step:
                print(f"       ⚠ 유효 그룹 {len(used)}/{cfg.groups_per_step} — "
                      f"시도 {len(attempted)}회로 배치를 못 채웠다. 그룹이 전멸/전승으로 "
                      "쏠린다는 뜻이다. temperature 를 올리거나(탐색↑), "
                      "SFT 베이스라인이 30% 게이트를 넘는지 먼저 확인할 것.")

            if self.writer:
                self.writer.log(record)
            (self.log_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            if (step + 1) % cfg.save_every == 0:
                out = self.log_dir / f"checkpoint-{step + 1}"
                self.model.save_pretrained(out)
                self.processor.save_pretrained(out)
                print(f"       체크포인트 저장: {out}")

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

    print("자체 검사 통과 (재샘플링 / 상한 / 커서 전진 / 토큰→액션 왕복).")
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
        logits = action_logits(
            model, processor, image, instruction,
            device=device, num_patches=num_patches, n_action_tokens=n_tokens,
        )
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
        ref = np.asarray(
            model.predict_action(**inputs, unnorm_key=unnorm_key), dtype=np.float32
        ).reshape(-1, SPEC.ACTION_DIM)[: SPEC.NUM_ACTIONS_CHUNK]

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

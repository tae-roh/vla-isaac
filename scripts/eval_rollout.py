"""VLA 정책의 성공률을 측정한다 (SFT 베이스라인 / RFT 결과).

실행 위치: vla-train 또는 rft venv. **isaaclab venv 가 아니다.**
Isaac Sim 은 rft/ipc_bridge.py 를 통해 별도 프로세스로 돌아간다.

★ 이 스크립트는 Day 3 아침의 첫 통합 지점이기도 하다.
  정책을 태워 롤아웃이 한 바퀴 돌면 브리지·관측 스펙·액션 규약이 모두
  맞다는 뜻이고, 그때부터 RFT 는 GRPO 루프를 붙이는 배선 작업이 된다.
  여기서 막히면 계획서의 fallback 트리거를 당길 근거가 된다.

사용 예:
    # SFT 베이스라인 (강체)
    python scripts/eval_rollout.py --checkpoint runs/sft/step-20000 \
        --task VlaPick-v0 --num-envs 8 --num-episodes 32

    # 변형체 (out-of-distribution)
    python scripts/eval_rollout.py --checkpoint runs/sft/step-20000 \
        --task VlaPick-Deformable-v0 --num-envs 4 --num-episodes 16

    # 정책 없이 브리지만 검증 (랜덤 액션)
    python scripts/eval_rollout.py --random-policy --num-episodes 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec_path = REPO_ROOT / "configs" / "vla_spec.py"
_spec_mod = importlib.util.spec_from_file_location("vla_spec", _spec_path)
SPEC = importlib.util.module_from_spec(_spec_mod)
sys.modules["vla_spec"] = SPEC
_spec_mod.loader.exec_module(SPEC)

from rft.ipc_bridge import RolloutClient  # noqa: E402


# =============================================================================
def load_policy(args):
    """OpenVLA 체크포인트를 로드하고 `predict(images) -> (N, K, A)` 를 반환한다."""
    if args.random_policy:
        rng = np.random.default_rng(0)

        def predict(images: np.ndarray) -> np.ndarray:
            n = images.shape[0]
            a = rng.uniform(
                -0.3, 0.3, size=(n, SPEC.NUM_ACTIONS_CHUNK, SPEC.ACTION_DIM)
            ).astype(np.float32)
            a[..., -1] = np.where(a[..., -1] > 0, 1.0, -1.0)
            return a

        print("[eval] 랜덤 정책 — 브리지 검증 전용이다. 성공률은 의미 없다.")
        return predict

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"[eval] 체크포인트 로드: {args.checkpoint}")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    # 정규화 통계 — SFT 때 쓴 것과 반드시 같아야 한다.
    norm_stats = None
    stats_path = Path(args.norm_stats) if args.norm_stats else None
    if stats_path and stats_path.exists():
        norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"[eval] 정규화 통계: {stats_path}")
    else:
        print(
            "[eval] ⚠ norm_stats 를 못 찾았다. 모델 내장 통계를 쓴다. "
            "SFT 때와 다르면 액션이 통째로 어긋난다 — 반드시 확인할 것."
        )

    prompt = SPEC.build_prompt()

    def predict(images: np.ndarray) -> np.ndarray:
        chunks = []
        for i in range(images.shape[0]):
            # center crop 포함 (학습이 --image_aug True 이므로 필수).
            pil = SPEC.prepare_image_for_vla(images[i])
            inputs = processor(prompt, pil).to(args.device, dtype=torch.bfloat16)
            with torch.no_grad():
                # openvla-oft 의 병렬 디코딩 API. 체크포인트 종류에 따라
                # predict_action / generate_action_verl 로 이름이 다를 수 있다.
                if hasattr(model, "predict_action"):
                    act = model.predict_action(
                        **inputs,
                        unnorm_key=args.unnorm_key,
                        do_sample=False,
                    )
                else:
                    act = model.generate_action_verl(
                        **inputs, unnorm_key=args.unnorm_key
                    )
            act = np.asarray(act, dtype=np.float32).reshape(-1, SPEC.ACTION_DIM)
            # 청크 길이를 스펙에 맞춘다 (모자라면 마지막 액션을 반복).
            if act.shape[0] < SPEC.NUM_ACTIONS_CHUNK:
                pad = np.repeat(
                    act[-1:], SPEC.NUM_ACTIONS_CHUNK - act.shape[0], axis=0
                )
                act = np.concatenate([act, pad], axis=0)
            chunks.append(act[: SPEC.NUM_ACTIONS_CHUNK])
        return np.stack(chunks).astype(np.float32)

    return predict


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--task", default="VlaPick-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=SPEC.MAX_EPISODE_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unnorm-key", default="vla_pick")
    parser.add_argument("--norm-stats", default=str(REPO_ROOT / "datasets" / "rlds" / SPEC.NORM_STATS_FILENAME))
    parser.add_argument(
        "--isaaclab-python",
        type=Path,
        default=Path.home() / "env_isaaclab" / "bin" / "python",
    )
    parser.add_argument("--out", type=Path, default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    if not args.checkpoint and not args.random_policy:
        parser.error("--checkpoint 또는 --random-policy 중 하나가 필요하다.")

    print(SPEC.summary())
    predict = load_policy(args)

    client = RolloutClient(
        isaaclab_python=args.isaaclab_python,
        worker_script=REPO_ROOT / "rft" / "isaaclab_rollout_worker.py",
        num_envs=args.num_envs,
        task=args.task,
        device="cuda:0",
    )
    client.start()

    num_rounds = int(np.ceil(args.num_episodes / args.num_envs))
    steps_per_round = int(np.ceil(args.max_steps / SPEC.NUM_ACTIONS_CHUNK))

    successes: list[float] = []
    episode_lengths: list[int] = []
    t_start = time.time()

    try:
        for rnd in range(num_rounds):
            obs = client.reset(seeds=[rnd * 1000])
            done = np.zeros(args.num_envs, dtype=bool)
            reward = np.zeros(args.num_envs, dtype=np.float32)
            steps_taken = 0

            for _ in range(steps_per_round):
                chunk = predict(obs["image"])
                obs, reward, done = client.step(chunk)
                steps_taken += SPEC.NUM_ACTIONS_CHUNK
                if bool(done.all()):
                    break

            successes.extend(reward.tolist())
            episode_lengths.extend([steps_taken] * args.num_envs)

            rate = float(np.mean(successes))
            print(
                f"  라운드 {rnd + 1}/{num_rounds}: "
                f"이번 {reward.mean():.2f} / 누적 성공률 {rate:.1%} "
                f"({int(np.sum(successes))}/{len(successes)})"
            )
    finally:
        client.close()

    elapsed = time.time() - t_start
    success_rate = float(np.mean(successes)) if successes else 0.0

    print(f"\n{'=' * 60}")
    print(f"태스크        : {args.task}")
    print(f"정책          : {'random' if args.random_policy else args.checkpoint}")
    print(f"에피소드      : {len(successes)}")
    print(f"성공률        : {success_rate:.1%} ({int(np.sum(successes))}/{len(successes)})")
    print(f"소요          : {elapsed / 60:.1f}분 "
          f"({elapsed / max(len(successes), 1):.1f}s/에피소드)")
    print(f"{'=' * 60}")

    # 완료 기준 대비 해석을 같이 찍어 준다 — 숫자만 보고 넘어가지 않게.
    if not args.random_policy:
        if args.task == "VlaPick-v0":
            if success_rate >= 0.6:
                print("→ 계획서 완료 기준(강체 ≥60~70%) 충족. RFT 로 진행 가능.")
            elif success_rate > 0:
                print(f"→ 완료 기준(60%)에 못 미치지만 0 은 아니다. "
                      "SimpleVLA-RL 은 데모 1개 SFT(17.3%)에서도 RL 로 91.7% 까지 올렸다 "
                      "— RFT 진행 가치가 있다.")
            else:
                print("→ 성공률 0. RFT 를 시작해도 학습 신호가 없다. "
                      "SFT 데이터/설정을 먼저 점검할 것 (관측 스펙 대조부터).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "task": args.task,
                    "checkpoint": args.checkpoint,
                    "num_episodes": len(successes),
                    "success_rate": success_rate,
                    "successes": successes,
                    "elapsed_s": elapsed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n결과 저장: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""VLA 정책의 성공률을 측정한다 (SFT 베이스라인 / RFT 결과).

실행 위치: vla-train 또는 rft venv. **isaaclab venv 가 아니다.**
Isaac Sim 은 rft/ipc_bridge.py 를 통해 별도 프로세스로 돌아간다.

★ 이 스크립트는 Day 3 아침의 첫 통합 지점이기도 하다.
  정책을 태워 롤아웃이 한 바퀴 돌면 브리지·관측 스펙·액션 규약이 모두
  맞다는 뜻이고, 그때부터 RFT 는 GRPO 루프를 붙이는 배선 작업이 된다.
  여기서 막히면 계획서의 fallback 트리거를 당길 근거가 된다.

평가 프로토콜 (개정 §6): 홀드아웃 초기 상태를 **시드가 아니라 뱅크 인덱스로**
고정한다. 같은 파일을 쓰는 한 SFT 와 RFT 가 정확히 같은 씬에서 비교된다.

사용 예:
    # Base split — 기본 클리어런스
    python scripts/eval_rollout.py --checkpoint runs/sft/step-20000 \
        --task VlaPlace-v0 --num-envs 8 --num-episodes 64

    # Tolerance split — 클리어런스 곡선의 점 하나 (태스크 이름이 공차를 담는다)
    python scripts/eval_rollout.py --checkpoint runs/sft/step-20000 \
        --task VlaPlace-v0 --num-episodes 64

    # Language split — 지시문 rephrase (씬은 그대로)
    python scripts/eval_rollout.py --checkpoint runs/sft/step-20000 \
        --rephrase 0 --num-episodes 64

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
# ★ openvla-oft 체크포인트의 modeling 파일이 `import prismatic` 을 요구한다.
#   진짜 openvla-oft 를 설치하면 torch 2.2.0 을 끌고 와 Isaac Sim 이 고정한
#   2.7.0 을 깨므로, 추론에 필요한 심볼만 담은 vendor/prismatic 셰임을 쓴다.
#   PYTHONPATH 로 넘기던 것을 여기서 붙여 스크립트가 자립하게 한다.
sys.path.insert(0, str(REPO_ROOT / "vendor"))

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

        def predict(images: np.ndarray, instructions: list[str]) -> np.ndarray:
            n = images.shape[0]
            a = rng.uniform(
                -0.3, 0.3, size=(n, args.chunk_len, SPEC.ACTION_DIM)
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

    # ★ RFT 체크포인트는 **LoRA 어댑터만** 담는다 (약 465MB). 단독으로는
    #   from_pretrained 가 모델 가중치를 찾지 못하므로, 베이스(--checkpoint)를
    #   먼저 올리고 그 위에 어댑터를 얹는다.
    #     python scripts/eval_rollout.py --checkpoint ckpt/sft \
    #         --adapter logs/<run>/checkpoint-N
    #   ⚠ 베이스는 SFT 체크포인트여야 한다. RFT 어댑터는 그 위에서 학습됐다.
    if args.adapter:
        from peft import PeftModel

        print(f"[eval] LoRA 어댑터 적용: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        # PeftModel 은 알 수 없는 속성을 베이스로 넘긴다(norm_stats, vocab_size,
        # bin_centers, predict_action 모두 그대로 쓸 수 있다).

    model.eval()

    # 정규화 통계는 **체크포인트 안의 것**을 쓴다 (predict_action 이 unnorm_key 로
    # 직접 찾는다). 파일에서 따로 읽어 두면 같은 숫자의 출처가 둘이 되고, 언젠가
    # 갈라진 뒤 "평가와 RFT 의 액션 스케일이 다르다" 로만 드러난다.
    # 여기서는 키가 있는지만 미리 확인하고 없으면 즉시 멈춘다.
    if args.unnorm_key not in getattr(model, "norm_stats", {}):
        raise SystemExit(
            f"[eval] unnorm_key '{args.unnorm_key}' 가 체크포인트의 norm_stats 에 "
            f"없다. 있는 키: {list(getattr(model, 'norm_stats', {}))}\n"
            "  SFT 가 dataset_statistics.json 을 체크포인트에 남겼는지 확인할 것."
        )
    print(f"[eval] 정규화 통계: 체크포인트 내장 '{args.unnorm_key}'")

    def predict(images: np.ndarray, instructions: list[str]) -> np.ndarray:
        # ★ 프롬프트는 env 마다 다르다. 타깃 블록·슬롯이 env 마다 다르기 때문이다.
        #   지시문 문자열은 환경이 관측으로 내려 준다 (obs["instruction"]) —
        #   여기서 따로 만들면 SFT 데이터의 문장과 어긋날 여지가 생긴다.
        chunks = []
        for i in range(images.shape[0]):
            prompt = SPEC.build_prompt(instructions[i])
            # center crop 포함 (학습이 --image_aug True 이므로 필수).
            pil = SPEC.prepare_image_for_vla(images[i])
            inputs = processor(prompt, pil).to(args.device, dtype=torch.bfloat16)
            with torch.no_grad():
                # openvla-oft 의 병렬 디코딩 API — 청크 전체를 한 번의 forward 로
                # 예측하고 언노멀라이즈까지 해서 돌려준다.
                #
                # ★ do_sample 같은 샘플링 인자를 넘기지 말 것. predict_action 은
                #   그런 인자를 받지 않고 **kwargs 로 삼켜 아래 forward 로 흘려보낸다.
                #   넘겨도 조용히 무시되므로 "샘플링을 켰다" 고 착각하기 쉽다
                #   (실제로 그렇게 오독한 계측이 있었다). 평가는 greedy 가 맞고,
                #   샘플링이 필요한 곳은 RFT 뿐이다 — rft/grpo_fallback.py 의
                #   _action_logits 가 같은 입력 구성을 직접 만든다.
                act = model.predict_action(**inputs, unnorm_key=args.unnorm_key)
            # 반환 형태는 리비전마다 다르다. 이 체크포인트의 predict_action 은
            # **튜플** (액션 ndarray, 히든스테이트 cuda Tensor) 을 돌려준다.
            # 그대로 np.asarray 하면 모양이 다른 두 원소를 쌓으려다 ValueError,
            # 히든스테이트만 집으면 cuda 텐서라 TypeError 가 난다. 첫 원소만 쓴다.
            if isinstance(act, tuple):
                act = act[0]
            if isinstance(act, torch.Tensor):
                act = act.detach().float().cpu().numpy()
            act = np.asarray(act, dtype=np.float32).reshape(-1, SPEC.ACTION_DIM)
            # ★ 청크 길이. OFT 체크포인트는 8개를 한 번에 내지만, **베이스
            #   OpenVLA 는 관측마다 1개**를 내는 폐루프 제어용이다.
            #   그 1개를 8번 반복하면 델타가 8배로 적용되어 실제보다 훨씬
            #   나쁘게 나온다 — 베이스를 평가할 때는 --chunk-len 1 로 매 스텝
            #   질의해야 한다 (워커는 가변 청크 길이를 받는다).
            k = args.chunk_len
            if act.shape[0] < k:
                pad = np.repeat(act[-1:], k - act.shape[0], axis=0)
                act = np.concatenate([act, pad], axis=0)
            chunks.append(act[:k])
        return np.stack(chunks).astype(np.float32)

    return predict


# =============================================================================
def _write_video(outdir: Path, name: str, frames: list, scale: int = 2) -> None:
    """롤아웃 프레임을 mp4 로 남긴다. 224px 는 눈으로 보기 작아 정수배 확대한다."""
    import imageio.v2 as imageio

    a = np.stack(frames)
    if a.ndim == 4 and a.shape[-1] == 4:
        a = a[..., :3]
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, axis=1), scale, axis=2)
    out = outdir / f"{name}.mp4"
    imageio.mimsave(out, list(a), fps=4, macro_block_size=1)
    print(f"  [video] {out} ({len(a)}프레임)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--chunk-len", type=int, default=SPEC.NUM_ACTIONS_CHUNK,
        help="한 번의 관측으로 실행할 액션 수. OFT 체크포인트는 기본값(8)을 쓰고, "
             "관측마다 액션 1개를 내는 베이스 OpenVLA 는 1 로 줄 것.",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="RFT LoRA 어댑터 디렉터리. RFT 체크포인트는 어댑터만 담으므로 "
             "--checkpoint 에 베이스(ckpt/sft)를, 여기에 checkpoint-N 을 준다.",
    )
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--task", default="VlaPlace-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=SPEC.EVAL_HOLDOUT_SIZE)
    parser.add_argument(
        "--bank", default="eval_base",
        help="초기 상태 뱅크 이름. 평가는 반드시 홀드아웃 뱅크를 쓴다 "
             "(scripts/make_init_states.py 로 생성).",
    )
    parser.add_argument(
        "--rephrase", type=int, default=None,
        help="Language split: SPEC.INSTRUCTION_TEMPLATES_EVAL 의 인덱스",
    )
    parser.add_argument("--max-steps", type=int, default=SPEC.MAX_EPISODE_STEPS)
    parser.add_argument(
        "--livestream", type=int, default=None, choices=[0, 1, 2],
        help="롤아웃을 WebRTC 로 중계한다. ★ 이 환경에서는 2 를 쓸 것. "
             "1 은 클라이언트에 PUBLIC_IP 환경변수(기본 127.0.0.1)를 광고하므로 "
             "그걸 안 주면 조용히 연결이 안 된다. 2 는 접속해 온 주소를 그대로 "
             "쓰므로 공인 IP 로 바로 붙는다 (49100/tcp).",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=None,
        help="env 0 의 table_cam 관측을 에피소드마다 mp4 로 저장한다. "
             "★ 프레임은 청크 경계마다 하나씩만 남는다 — 브리지가 관측을 "
             "청크 단위로 돌려주기 때문이다 (실제 제어 주기의 1/8).",
    )
    parser.add_argument("--device", default="cuda:0")
    # 정규화 통계는 체크포인트 안의 것을 쓴다 (--norm-stats 플래그는 없앴다 —
    # 파일과 체크포인트 두 출처가 갈라지는 것을 막기 위해서다).
    parser.add_argument("--unnorm-key", default="vla_pick")
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

    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    # ★ --livestream 을 주면 롤아웃을 눈으로 볼 수 있다. 브리지는
    #   --headless --enable_cameras 를 항상 붙이는데, AppLauncher 는 livestream
    #   이 1/2 이면 호스트를 헤드리스로 두고 WebRTC 로만 내보내므로 둘이
    #   충돌하지 않는다 (dump_obs_reference.py 와 같은 규약).
    extra = []
    if args.livestream is not None:
        extra += ["--livestream", str(args.livestream)]

    client = RolloutClient(
        isaaclab_python=args.isaaclab_python,
        worker_script=REPO_ROOT / "rft" / "isaaclab_rollout_worker.py",
        num_envs=args.num_envs,
        task=args.task,
        device="cuda:0",
        extra_args=extra,
    )
    client.start()

    num_rounds = int(np.ceil(args.num_episodes / args.num_envs))
    # ★ 청크 길이가 1 이면 라운드당 스텝 수가 8배가 된다 (매 스텝 재질의).
    steps_per_round = int(np.ceil(args.max_steps / args.chunk_len))

    successes: list[float] = []
    episode_lengths: list[int] = []
    lifted_all: list[float] = []
    grasped_all: list[float] = []
    tray_all: list[float] = []
    rewards_all: list[float] = []
    height_all: list[float] = []
    t_start = time.time()

    template = (
        None if args.rephrase is None
        else SPEC.INSTRUCTION_TEMPLATES_EVAL[args.rephrase]
    )
    if template:
        print(f"[eval] Language split — 지시문 템플릿: {template!r}")

    try:
        for rnd in range(num_rounds):
            # 홀드아웃을 순서대로 훑는다. 시드가 아니라 인덱스로 고정하므로
            # 재실행·다른 체크포인트에서도 **정확히 같은 씬**이 나온다.
            indices = [
                (rnd * args.num_envs + i) % SPEC.EVAL_HOLDOUT_SIZE
                for i in range(args.num_envs)
            ]
            obs = client.reset(
                init_indices=indices,
                bank=args.bank,
                instruction_template=template,
                seeds=[rnd * 1000],
            )
            done = np.zeros(args.num_envs, dtype=bool)
            reward = np.zeros(args.num_envs, dtype=np.float32)
            steps_taken = 0
            frames: list[np.ndarray] = []
            # ★ lifted 는 "타깃을 쥔 채 리프트 임계를 넘겼는가" 의 래치다.
            #   성공률이 0 인 구간에서 파지 실패와 배치 실패를 가르는 유일한
            #   단서인데, 예전에는 브리지까지만 오고 출력되지 않아 진단 때마다
            #   별도 스크립트를 짜야 했다. 여기서 라운드마다 찍는다.
            lifted = 0.0
            grasped = 0.0
            hmax = 0.0
            in_tray = 0.0
            succ = 0.0

            for _ in range(steps_per_round):
                if args.video_dir is not None:
                    frames.append(np.asarray(obs["image"][0], dtype=np.uint8).copy())
                chunk = predict(obs["image"], obs["instruction"])
                obs, reward, done = client.step(chunk)
                lifted = max(lifted, float(client.last_diag.get("lifted", 0.0)))
                grasped = max(grasped, float(client.last_diag.get("grasped", 0.0)))
                hmax = max(hmax, float(client.last_diag.get("height_max", 0.0)))
                in_tray = max(in_tray, float(client.last_diag.get("in_tray", 0.0)))
                # ★ 단계형 보상을 켜면 reward 가 0.2/0.4 값을 가지므로 그대로
                #   평균 내면 "성공률" 이 아니다. outcome 성공률은 워커가
                #   diag["success"] 로 따로 보낸다. 보고는 반드시 이쪽으로.
                succ = max(succ, float(client.last_diag.get("success", 0.0)))
                steps_taken += args.chunk_len
                if bool(done.all()):
                    break
            lifted_all.append(lifted)
            grasped_all.append(grasped)
            tray_all.append(in_tray)
            height_all.append(hmax)

            if args.video_dir is not None and frames:
                _write_video(args.video_dir, f"rollout_ep{rnd:02d}", frames)

            successes.extend([succ] * args.num_envs)
            rewards_all.append(float(np.mean(reward)))
            episode_lengths.extend([steps_taken] * args.num_envs)

            rate = float(np.mean(successes))
            print(
                f"  라운드 {rnd + 1}/{num_rounds}: "
                f"보상 {reward.mean():.2f} / 성공 {succ:.2f} / 파지 {grasped:.2f} / 진입 {in_tray:.2f} / "
                f"최고높이 {hmax*1000:.0f}mm / "
                f"lifted {lifted:.2f} / 누적 성공률 {rate:.1%} "
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
    if lifted_all:
        # 성공률이 0 일 때 어느 단계에서 막히는지 가르는 값들이다.
        # 라운드마다 기록되는 값은 "그 라운드의 env 중 성공한 비율" 이므로
        # num_envs 를 곱하면 에피소드 수가 된다.
        n_grasp = sum(g * args.num_envs for g in grasped_all)
        n_lift = sum(x * args.num_envs for x in lifted_all)
        n_ep = len(successes)
        n_tray = sum(t * args.num_envs for t in tray_all)
        print(f"파지          : {n_grasp:.0f}/{n_ep} 에피소드 ({n_grasp / n_ep:.1%})")
        print(f"트레이 진입   : {n_tray:.0f}/{n_ep} 에피소드 ({n_tray / n_ep:.1%})"
              f"  ← RFT 마지막 단계에 신호가 있는지를 가른다")
        print(f"lifted        : {n_lift:.0f}/{n_ep} 에피소드 ({n_lift / n_ep:.1%}) / "
              f"라운드 평균 {np.mean(lifted_all):.2f}")
        print(f"블록 최고높이 : 평균 {np.mean(height_all) * 1000:.0f}mm / "
              f"최대 {max(height_all) * 1000:.0f}mm "
              f"(레일 {SPEC.TRAY_DEPTH * 1000:.0f}mm 를 넘으려면 중심 "
              f"{(SPEC.TRAY_DEPTH + SPEC.BLOCK_SIZE[2] / 2) * 1000:.0f}mm 필요)")
    print(f"소요          : {elapsed / 60:.1f}분 "
          f"({elapsed / max(len(successes), 1):.1f}s/에피소드)")
    print(f"{'=' * 60}")

    # 완료 기준 대비 해석을 같이 찍어 준다 — 숫자만 보고 넘어가지 않게.
    # 개정 §5 의 게이트: SFT 성공률이 낮으면 RL 이 아무것도 개선하지 못한다.
    if not args.random_policy:
        if success_rate >= 0.30:
            print("→ RL 게이트(≥30%) 통과. 이 split 에서 RFT 를 돌릴 수 있다.")
        elif success_rate > 0.05:
            print(f"→ 성공률 {success_rate:.1%} — 게이트(30%) 미만이다. "
                  "이 split 에서 RL 을 돌리면 개선폭이 나오지 않을 공산이 크다. "
                  "더 헐거운 클리어런스부터 곡선을 그릴 것.")
        else:
            print("→ 성공률 5% 미만. RL 에 학습 신호가 없다 "
                  "(개정 §5 임계값). 헐거운 공차 split 으로 내려가거나 "
                  "SFT 데이터/스펙부터 점검할 것.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "task": args.task,
                    "checkpoint": args.checkpoint,
                    # split 을 특정하는 세 값. 이게 없으면 나중에 곡선을 그릴 때
                    # 각 점이 무슨 조건이었는지 알 수 없다.
                    "bank": args.bank,
                    "rephrase": args.rephrase,
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

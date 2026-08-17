"""정책 forward 배치 처리량·메모리 벤치마크 (일회성).

`groups_per_step` 을 추정이 아니라 **실측**으로 정하기 위한 도구다.
이 프로젝트에서 추정치가 여러 번 빗나갔다 (무신호 16% 예측 → 실제 56%,
스텝 5.0분 예측 → 실제 6.2분). 배치 확대 폭도 재고 나서 정한다.

두 경로를 따로 잰다:
  no_grad  — 롤아웃·평가에서 쓰는 경로
  grad     — _recompute_logp 경로. 활성값이 배치에 비례하므로 여기가 메모리 상한이다.

사용:
    ~/env_eval/bin/python scripts/bench_policy_batch.py --checkpoint ckpt/sft
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vendor"))

from rft.grpo_fallback import SPEC, action_logits  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ckpt/sft")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", default="1,2,4,8,16")
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device)

    # LoRA 를 붙여 실제 학습과 같은 조건으로 잰다 (grad 경로의 메모리가 달라진다).
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.0, init_lora_weights="gaussian", bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj", "qkv", "proj", "fc1", "fc2", "fc3", "q", "kv", "lm_head"],
    ))

    npatch = (model.vision_backbone.get_num_patches()
              * model.vision_backbone.get_num_images_in_input())
    ntok = SPEC.NUM_ACTIONS_CHUNK * SPEC.ACTION_DIM
    rng = np.random.default_rng(0)
    sizes = [int(x) for x in args.batches.split(",")]

    def run(bs: int, grad: bool):
        imgs = rng.integers(
            0, 255, (bs, SPEC.IMAGE_HEIGHT, SPEC.IMAGE_WIDTH, 3), dtype=np.uint8
        )
        instrs = [SPEC.instruction_for(i % SPEC.NUM_BLOCKS) for i in range(bs)]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def once():
            if grad:
                lg = action_logits(model, proc, imgs, instrs, device=args.device,
                                   num_patches=npatch, n_action_tokens=ntok)
                lg.float().log_softmax(-1).sum().backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    action_logits(model, proc, imgs, instrs, device=args.device,
                                  num_patches=npatch, n_action_tokens=ntok)

        once()                      # 워밍업 (커널 컴파일·캐시)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            once()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.iters
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return dt, peak

    for label, grad in (("no_grad (롤아웃·평가)", False), ("grad (업데이트)", True)):
        print(f"\n=== {label} ===")
        print(f"{'배치':>4} {'초/회':>9} {'샘플/초':>9} {'배수(vs B=1)':>13} {'peak VRAM':>11}")
        base = None
        for bs in sizes:
            try:
                dt, peak = run(bs, grad)
            except torch.cuda.OutOfMemoryError:
                print(f"{bs:>4} {'OOM':>9}")
                torch.cuda.empty_cache()
                continue
            thr = bs / dt
            if base is None:
                base = thr
            print(f"{bs:>4} {dt:>9.3f} {thr:>9.2f} {thr / base:>12.2f}x {peak:>10.1f}GB")

    print("\n결정 규칙: T = grad/no_grad 중 **작은 쪽** 배수 (둘 다 스텝 시간에 들어간다)")
    print("  스텝 시간을 유지하려면 groups_per_step ≈ 2 × T")
    print("  grad peak 가 70GB 를 넘으면 policy_batch_size 를 낮출 것")
    return 0


if __name__ == "__main__":
    sys.exit(main())

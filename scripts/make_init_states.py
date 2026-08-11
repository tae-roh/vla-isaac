"""초기 상태 뱅크를 만들어 파일로 굳힌다 (개정 §3).

**RL 코드보다 먼저 돌려야 하는 스크립트다.** 나중에 붙이면 그 전 실험 결과가
전부 재현 불가가 된다.

Isaac Sim 이 필요 없다 — numpy 만 쓴다. 어느 venv 에서든 돈다.

사용:
    # 학습/생성용 (기본 4096개) + 평가 홀드아웃 (기본 64개)
    python scripts/make_init_states.py --name train
    python scripts/make_init_states.py --name eval_base --size 64 --seed 12345

    # 내용 확인
    python scripts/make_init_states.py --show eval_base

생성한 뱅크는 **커밋한다.** 코드가 바뀌어도 같은 s₀ 로 돌릴 수 있어야 하고,
평가 숫자를 비교하려면 같은 파일이어야 한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 패키지(vla_isaac_tasks)를 통하지 않고 모듈 파일만 직접 읽는다.
# 패키지 __init__ 은 gymnasium + isaaclab 을 끌어오는데, 뱅크를 만드는 데는
# numpy 하나면 충분하다 — 어느 venv 에서든 돌게 하려는 것이 이 우회의 목적이다.
_p = REPO_ROOT / "source" / "vla_isaac_tasks" / "init_states.py"
_s = importlib.util.spec_from_file_location("vla_init_states", _p)
init_states = importlib.util.module_from_spec(_s)
_s.loader.exec_module(init_states)
SPEC = init_states.SPEC


def show(name: str) -> int:
    bank = init_states.load_bank(name)
    n = SPEC.NUM_BLOCKS
    print(f"뱅크 '{name}': {len(bank)}개 × {bank.shape[1]}차원 "
          f"({init_states.bank_path(name)})")
    for i, row in enumerate(bank[:5]):
        blocks = ", ".join(
            f"{SPEC.BLOCK_ATTRS[b]}({row[3 * b]:.3f},{row[3 * b + 1]:.3f},"
            f"{row[3 * b + 2]:+.2f}rad)"
            for b in range(n)
        )
        tb = int(row[3 * n])
        print(f"  [{i}] {blocks}")
        print(f"       → {SPEC.instruction_for(tb)!r}")
    if len(bank) > 5:
        print(f"  ... 외 {len(bank) - 5}개")

    # 지시문 분포 — 한쪽으로 쏠리면 언어 채널이 사실상 죽는다.
    import collections

    counts = collections.Counter(int(r[3 * n]) for r in bank)
    named = {SPEC.BLOCK_ATTRS[k]: v for k, v in sorted(counts.items())}
    print(f"  타깃 블록 {len(counts)}종 {named} / "
          f"최소 {min(counts.values())}회, 최대 {max(counts.values())}회")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="train", help="뱅크 이름 (= split 이름)")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show", metavar="NAME", default=None, help="내용만 출력")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    args = parser.parse_args()

    if args.show:
        return show(args.show)

    # 평가 홀드아웃은 학습 뱅크와 시드를 반드시 달리한다. 같은 시드면 평가 씬이
    # 학습 씬과 겹쳐 성공률이 부풀려지고, 그 사실은 숫자만 봐서는 알 수 없다.
    is_eval = args.name.startswith("eval")
    size = args.size or (SPEC.EVAL_HOLDOUT_SIZE if is_eval else SPEC.TRAIN_BANK_SIZE)
    seed = args.seed if args.seed is not None else (12345 if is_eval else 0)

    path = init_states.bank_path(args.name)
    if path.exists() and not args.force:
        print(f"이미 있다: {path}\n  덮어쓰려면 --force. "
              "★ 덮어쓰면 이전 실험과 초기 상태가 달라져 비교가 깨진다.")
        return 1

    bank = init_states.sample_bank(size, seed)
    init_states.save_bank(args.name, bank, seed)
    print(f"뱅크 '{args.name}' 생성: {len(bank)}개 (시드 {seed}) → {path}")
    return show(args.name)


if __name__ == "__main__":
    sys.exit(main())

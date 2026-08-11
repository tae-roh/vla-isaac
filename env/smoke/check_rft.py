"""rft venv 스모크 테스트 — Phase 4 (RFT) 환경 검증.

확인 항목:
  1. torch / vllm / ray / verl import
  2. 공유 스펙 로드
  3. ipc_bridge 임포트 (순수 stdlib — 두 venv 양쪽에서 동작해야 한다)
  4. (--full) isaaclab venv 의 롤아웃 워커를 자식 프로세스로 띄우고
     핸드셰이크 → 리셋 → 액션 청크 1회 전송 → 관측·보상 수신 왕복

★ 4번이 이 프로젝트 최대 리스크(계획서 §6: "Isaac Lab × veRL 어댑터 난이도")의
  가장 이른 검증 지점이다. 이게 통과하면 나머지는 GRPO 루프에 붙이는 배선 작업이고,
  실패하면 Day 3 정오 fallback 트리거를 당길 근거가 된다.

실행:
    python env/smoke/check_rft.py
    python env/smoke/check_rft.py --full --isaaclab-python ~/env_isaaclab/bin/python
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import check, finish, load_vla_spec, warn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="롤아웃 워커 왕복까지 검증")
    parser.add_argument(
        "--isaaclab-python",
        type=Path,
        default=Path.home() / "env_isaaclab" / "bin" / "python",
        help="Isaac Sim 이 설치된 venv 의 python. 워커는 이 인터프리터로 실행된다.",
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="워커 기동 대기 상한 [s]. 첫 실행은 셰이더 캐시로 오래 걸린다.")
    args = parser.parse_args()

    print("=== rft 환경 스모크 테스트 ===\n")

    spec = None

    def _spec():
        nonlocal spec
        spec = load_vla_spec()
        spec.assert_consistent()
        return "configs/vla_spec.py 정합성 OK"

    check("공유 스펙 로드", _spec)

    def _torch():
        import torch

        if not torch.cuda.is_available():
            raise AssertionError("CUDA 사용 불가.")
        n = torch.cuda.device_count()
        if n < 2:
            warn(
                f"GPU {n}개 — 시뮬 렌더와 학습을 같은 카드에서 돌리게 된다. "
                "동작은 하지만 num_envs 를 줄여야 할 수 있다."
            )
        return f"torch {torch.__version__}, GPU {n}개"

    check("torch + CUDA", _torch)

    def _rl_libs():
        import ray  # noqa: F401
        import vllm

        return f"vllm {vllm.__version__}"

    check("vllm / ray", _rl_libs)

    def _verl():
        import verl  # noqa: F401
        from verl.workers.rollout.rob_rollout import RobHFRollout  # noqa: F401

        return "verl + RobHFRollout import OK"

    check("SimpleVLA-RL (verl) import", _verl)

    def _bridge():
        sys.path.insert(0, str(REPO_ROOT))
        from rft.ipc_bridge import PROTOCOL_VERSION, RolloutClient  # noqa: F401

        return f"프로토콜 v{PROTOCOL_VERSION}"

    check("ipc_bridge import", _bridge)

    if not args.full:
        print(
            "\n--full 없이 실행됨 → 롤아웃 워커 왕복 검증은 건너뛴다.\n"
            "어댑터 작업 중에는 반드시 --full 로 확인할 것."
        )
        return finish("rft")

    # -- 워커 왕복 -----------------------------------------------------------
    def _worker_roundtrip():
        import numpy as np

        sys.path.insert(0, str(REPO_ROOT))
        from rft.ipc_bridge import RolloutClient

        if not args.isaaclab_python.exists():
            raise FileNotFoundError(
                f"isaaclab venv 의 python 을 찾을 수 없다: {args.isaaclab_python}\n"
                "  --isaaclab-python 으로 경로를 지정할 것."
            )

        client = RolloutClient(
            isaaclab_python=args.isaaclab_python,
            worker_script=REPO_ROOT / "rft" / "isaaclab_rollout_worker.py",
            num_envs=args.num_envs,
            task="VlaPlace-v0",
            startup_timeout=args.timeout,
        )
        t0 = time.time()
        client.start()
        startup = time.time() - t0

        obs = client.reset(seeds=list(range(args.num_envs)))
        img = obs["image"]
        expected = (args.num_envs, spec.IMAGE_HEIGHT, spec.IMAGE_WIDTH, 3)
        if img.shape != expected:
            raise AssertionError(f"관측 이미지 shape {img.shape} != 기대 {expected}")
        if img.dtype != np.uint8:
            raise AssertionError(f"관측 이미지 dtype {img.dtype} != uint8")

        # 액션 청크 1회 — 전부 0 (제자리 유지 + 그리퍼 열림)
        chunk = np.zeros(
            (args.num_envs, spec.NUM_ACTIONS_CHUNK, spec.ACTION_DIM), dtype=np.float32
        )
        obs, reward, done = client.step(chunk)

        if reward.shape != (args.num_envs,):
            raise AssertionError(f"보상 shape {reward.shape} != {(args.num_envs,)}")
        if not set(np.unique(reward)).issubset({0.0, 1.0}):
            raise AssertionError(f"보상이 0/1 이 아니다: {np.unique(reward)}")

        client.close()
        return (
            f"워커 기동 {startup:.0f}s, 관측 {img.shape}, "
            f"보상 {reward.tolist()}, done {done.tolist()}"
        )

    check("Isaac Lab 롤아웃 워커 왕복", _worker_roundtrip)

    return finish("rft")


if __name__ == "__main__":
    sys.exit(main())

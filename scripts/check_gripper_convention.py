"""워커를 통과한 그리퍼 명령이 실제로 의도한 방향으로 동작하는지 확인한다.

정책도 체크포인트도 쓰지 않는다. 팔 델타는 0으로 고정하고 그리퍼 명령만 넣어,
손가락이 실제로 닫히는지/열리는지 관절값으로 본다.

왜 필요한가
-----------
데이터의 규약은 정의상 환경의 규약이다 (녹화된 액션 = env.step 에 들어간 값).
실측하면 RLDS 에서 -1 은 닫힘, +1 은 열림이다. 그런데 워커는
SPEC.GRIPPER_INVERT_FOR_VLA 로 부호를 한 번 뒤집는다 — 그 상수는 LIBERO
벤치마크(-1=열림)를 전제로 첫 커밋에 들어왔고, 우리 정책은 LIBERO 로 학습되지
않았다. 뒤집기가 불필요하면 접근 중에 그리퍼가 닫히고 파지 순간에 열린다.

지금까지 이 경로를 검증한 적이 없다. 데모 수집·재생은 워커를 지나지 않고,
워커 스모크는 --random-policy 라 그리퍼 부호 오류가 드러나지 않는다.

실행 (저장소 루트에서):
    ~/env_rft/bin/python scripts/check_gripper_convention.py
    # env_rft 가 없으면 numpy 가 있는 아무 venv 로도 된다 (ipc_bridge 는 stdlib)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rft.ipc_bridge import RolloutClient  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "vla_spec", REPO_ROOT / "configs" / "vla_spec.py"
)
SPEC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SPEC)

OPEN_WIDTH = SPEC.GRIPPER_MAX_WIDTH          # 0.08 m
CLOSED_THRESH = 0.25 * OPEN_WIDTH            # 0.02 m 미만이면 닫힘으로 본다


def finger_width(state: np.ndarray) -> np.ndarray:
    """proprio 6,7 = 손가락 관절 2개. 두 번째는 부호가 뒤집혀 오므로 절대값 합."""
    return np.abs(state[:, 6]) + np.abs(state[:, 7])


def run_phase(client, label: str, gripper_cmd: float, n_chunks: int) -> np.ndarray | None:
    chunk = np.zeros((client.num_envs, SPEC.NUM_ACTIONS_CHUNK, SPEC.ACTION_DIM), np.float32)
    chunk[:, :, 6] = gripper_cmd          # 팔 델타(0~5)는 0 — 그리퍼만 본다
    obs = None
    for _ in range(n_chunks):
        obs, _, _ = client.step(chunk)
    if "state" not in obs:
        print(f"  [{label}] proprio 가 관측에 없다 — 이 검사를 쓸 수 없다.")
        return None
    w = finger_width(np.asarray(obs["state"]))
    verdict = "닫힘" if w.mean() < CLOSED_THRESH else "열림"
    print(f"  명령 {gripper_cmd:+.0f} ({label:<12}) → 손가락 벌어짐 "
          f"{np.array2string(w, precision=4)} 평균 {w.mean():.4f} m  ⇒ {verdict}")
    return w


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="VlaPlace-v0")
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--bank", default="eval_base")
    p.add_argument("--chunks", type=int, default=6, help="국면당 청크 반복 횟수")
    p.add_argument("--isaaclab-python", type=Path,
                   default=Path.home() / "env_isaaclab" / "bin" / "python")
    args = p.parse_args()

    print("=== 그리퍼 규약 확인 ===")
    print(f"  SPEC.GRIPPER_INVERT_FOR_VLA = {SPEC.GRIPPER_INVERT_FOR_VLA}")
    print(f"  데이터 규약(실측)           : -1 = 닫힘 / +1 = 열림")
    print(f"  손가락 최대 벌어짐          : {OPEN_WIDTH} m\n")

    client = RolloutClient(
        isaaclab_python=args.isaaclab_python,
        worker_script=REPO_ROOT / "rft" / "isaaclab_rollout_worker.py",
        num_envs=args.num_envs,
        task=args.task,
        device="cuda:0",
    )
    client.start()
    try:
        client.reset(init_indices=list(range(args.num_envs)), bank=args.bank)
        w_open = run_phase(client, "열어라", +1.0, args.chunks)
        w_close = run_phase(client, "닫아라", -1.0, args.chunks)
    finally:
        client.close()

    if w_open is None or w_close is None:
        return 2

    print()
    if w_close.mean() < CLOSED_THRESH <= w_open.mean():
        print("[ OK ] 워커를 통과한 명령이 데이터 규약과 일치한다.")
        print("       그리퍼는 파지 실패의 원인이 아니다 — 다른 원인을 봐야 한다.")
        return 0
    if w_open.mean() < CLOSED_THRESH <= w_close.mean():
        print("[FAIL] 반전됐다. '닫아라'(-1)에 손가락이 열리고 '열어라'(+1)에 닫힌다.")
        print("       → 접근 중에는 닫혀 있고 파지 순간에 열린다. 성공률이 0 인 이유다.")
        print("       고치는 법: configs/vla_spec.py 의 GRIPPER_INVERT_FOR_VLA = False")
        return 1
    print("[????] 두 명령의 결과가 갈리지 않는다 "
          f"(열어라 {w_open.mean():.4f} / 닫아라 {w_close.mean():.4f}).")
    print("       그리퍼가 아예 안 움직이는 것일 수 있다 — 이중 반전이나 액션 항 배선을 볼 것.")
    return 3


if __name__ == "__main__":
    sys.exit(main())

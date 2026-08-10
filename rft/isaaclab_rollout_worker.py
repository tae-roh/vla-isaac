"""Isaac Lab 롤아웃 워커 — isaaclab venv 의 python 으로 실행된다.

부모(학습 프로세스, rft venv)와 stdin/stdout 으로 대화하며 배치 환경을 굴린다.
직접 실행하는 스크립트가 아니라 rft/ipc_bridge.py 의 RolloutClient 가 띄운다.

★ 구조적 차이 (계획서 §Phase4b-2)
  LIBERO  : 환경 인스턴스 = 프로세스 1개, N개 프로세스를 띄운다
  Isaac Lab: 단일 SimulationApp 안에서 num_envs 로 배치 벡터화
  → 롤아웃 워커를 "프로세스 N개" 가 아니라 "배치형 프로세스 1개" 로 재설계한 것이
    이 파일이다. SimulationApp 은 프로세스당 하나만 뜰 수 있으므로 선택지가 없다.

주의: stdout 은 프로토콜 전용이다. 워커 안에서 print() 를 쓰면 스트림이 오염되어
      부모가 메시지를 파싱하지 못한다. 로그는 반드시 stderr 로 보낼 것.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def log(msg: str) -> None:
    """stdout 은 프로토콜 전용이므로 로그는 stderr 로만."""
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="VlaPick-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    # AppLauncher 가 --headless / --enable_cameras 등을 처리한다.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # AppLauncher 이후에만 import 가능한 것들
    import gymnasium as gym
    import numpy as np
    import torch

    import vla_isaac_tasks  # noqa: F401  — gym.register
    from isaaclab_tasks.utils import parse_env_cfg
    from rft.ipc_bridge import PROTOCOL_VERSION, recv_message, send_message
    from vla_isaac_tasks.spec import SPEC

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    env = None
    try:
        log(f"환경 생성: {args.task}, num_envs={args.num_envs}, device={args.device}")
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        if args.max_episode_steps is not None:
            env_cfg.episode_length_s = (
                args.max_episode_steps * env_cfg.sim.dt * env_cfg.decimation
            )
        env = gym.make(args.task, cfg=env_cfg)
        env.unwrapped.seed(args.seed)
        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device
        log(f"환경 준비 완료. action_space={env.action_space.shape}")

        # 에피소드 단위 성공 래치.
        # Isaac Lab 은 종료된 env 를 자동 리셋하므로, 성공한 그 스텝에 값을 잡아
        # 두지 않으면 다음 관측이 이미 새 에피소드 것이라 놓친다.
        success_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        done_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # ---------------------------------------------------------------
        def extract_obs(obs_dict) -> dict:
            """VLA 가 받을 관측만 추려서 numpy 로 바꾼다."""
            policy = obs_dict["policy"]

            img = policy[SPEC.CAMERA_SENSOR_NAME]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if SPEC.ROTATE_IMAGE_180:
                # LIBERO 는 MuJoCo 출력이 뒤집혀 있어 180도 돌린다. Isaac Lab 은
                # 보통 필요 없다 — 스펙 플래그로만 제어하고, 실제 판단은
                # scripts/dump_obs_reference.py 의 PNG 를 눈으로 보고 한다.
                img = np.rot90(img, k=2, axes=(1, 2)).copy()

            state = policy.get("proprio")
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy().astype(np.float32)

            out = {"image": img}
            if state is not None:
                out["state"] = state
            return out

        def send_ok(payload: dict) -> None:
            send_message(stdout, payload)

        # ---------------------------------------------------------------
        log("명령 대기 시작")
        while True:
            msg = recv_message(stdin)
            if msg is None:
                log("stdin EOF — 종료한다")
                break

            cmd = msg.get("cmd")

            try:
                if cmd == "hello":
                    send_ok({
                        "ok": True,
                        "protocol": PROTOCOL_VERSION,
                        "num_envs": num_envs,
                        "task": args.task,
                        "action_dim": SPEC.ACTION_DIM,
                        "chunk_len": SPEC.NUM_ACTIONS_CHUNK,
                        "image_hw": [SPEC.IMAGE_HEIGHT, SPEC.IMAGE_WIDTH],
                    })

                elif cmd == "reset":
                    seeds = msg.get("seeds")
                    if seeds:
                        # 그룹 롤아웃에서 같은 초기 상태를 재현하려면 시드가 필요하다
                        # (GRPO 는 같은 상태에서 여러 번 굴려 그룹을 만든다).
                        env.unwrapped.seed(int(seeds[0]))
                    obs_dict, _ = env.reset()
                    success_latch.zero_()
                    done_latch.zero_()
                    send_ok({"obs": extract_obs(obs_dict)})

                elif cmd == "step":
                    chunk = msg["action_chunk"]           # (N, K, A)
                    actions = torch.as_tensor(
                        np.asarray(chunk, dtype=np.float32), device=device
                    )
                    if actions.ndim == 2:                 # (N, A) 도 허용
                        actions = actions.unsqueeze(1)

                    n, k, a = actions.shape
                    if a != SPEC.ACTION_DIM:
                        raise ValueError(
                            f"액션 차원 {a} != 스펙 {SPEC.ACTION_DIM}"
                        )
                    if n != num_envs:
                        raise ValueError(f"배치 {n} != num_envs {num_envs}")

                    # 그리퍼 부호 변환은 여기 한 곳에서만 한다.
                    # VLA(LIBERO 규약, -1=열림) → Isaac Lab(BinaryJoint, +1=열림)
                    # 두 군데서 뒤집으면 상쇄되어 "그리퍼가 안 움직이는" 증상이 된다.
                    if SPEC.GRIPPER_INVERT_FOR_VLA:
                        actions = actions.clone()
                        actions[:, :, -1] = -actions[:, :, -1]

                    obs_dict = None
                    for t in range(k):
                        obs_dict, reward, terminated, truncated, _ = env.step(
                            actions[:, t, :]
                        )
                        # 성공은 reward>0 으로 판정한다. RewardsCfg 의 success 항이
                        # weight=1.0 인 유일한 항이므로 reward 가 곧 0/1 성공이다.
                        # (진단용 grasp_lift_diag 는 weight=0 이라 섞이지 않는다)
                        success_latch |= reward > 0.5
                        done_latch |= terminated | truncated

                    send_ok({
                        "obs": extract_obs(obs_dict),
                        "reward": success_latch.float().cpu().numpy(),
                        "done": done_latch.cpu().numpy(),
                    })

                elif cmd == "success":
                    send_ok({"success": success_latch.cpu().numpy()})

                elif cmd == "close":
                    log("close 수신 — 종료한다")
                    send_ok({"ok": True})
                    break

                else:
                    send_ok({"error": f"알 수 없는 명령: {cmd!r}"})

            except Exception as exc:  # noqa: BLE001
                # 워커가 죽으면 부모는 EOF 만 보게 되어 원인을 알 수 없다.
                # 예외를 잡아 traceback 을 넘겨 주는 편이 디버깅에 훨씬 낫다.
                log(f"명령 {cmd!r} 처리 중 오류: {exc}")
                traceback.print_exc(file=sys.stderr)
                send_ok({"error": str(exc), "traceback": traceback.format_exc()})

    except Exception as exc:  # noqa: BLE001
        log(f"치명적 오류: {exc}")
        traceback.print_exc(file=sys.stderr)
        try:
            send_message(stdout, {"error": str(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

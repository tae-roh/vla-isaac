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
import os
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
    parser.add_argument("--task", default="VlaPlace-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    # AppLauncher 가 --headless / --enable_cameras / --device 를 처리한다.
    #
    # ★ --device 를 여기서 직접 추가하면 안 된다. 이 리비전의
    #   AppLauncher.add_app_launcher_args() 는 같은 이름이 이미 있으면
    #     ValueError: The passed ArgParser object already has the field 'device'
    #   로 죽고, 워커가 그대로 종료되어 브리지는 "응답 없이 종료" 만 보게 된다
    #   (원인이 GPU 메모리로 오인되기 쉽다).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    # ★ 프로토콜용 stdout 을 Isaac Sim 이 오염시키기 전에 떼어낸다.
    #
    #   브리지는 워커의 stdout 에서 "길이 접두어 + 본문" 을 읽는다. 그런데
    #   Isaac Sim(Kit) 은 **C++ 레벨에서 fd 1 에 직접** 배너·경고를 쏟는다.
    #   파이썬에서 sys.stdout 을 바꿔도 그건 막히지 않는다. 그 바이트가 길이
    #   접두어로 해석되면 브리지가 터무니없는 크기를 할당하려다
    #       MemoryError  (ipc_bridge._read_exactly)
    #   로 죽고, 워커는 broken pipe 만 남긴다 — 원인이 GPU 메모리로 오인되기 쉽다.
    #
    #   그래서 AppLauncher 를 띄우기 **전에** fd 1 을 복제해 프로토콜 전용으로
    #   쓰고, fd 1 자체는 stderr 로 돌려 Isaac Sim 의 출력이 로그로 가게 한다.
    #
    # ★ 이 블록은 parse_args() **보다도 앞**이어야 한다. parse_args() 는 인자
    #   정의를 위해 `from isaaclab.app import AppLauncher` 를 하는데, EULA 를
    #   수락하지 않은 환경에서는 **그 import 만으로** Kit 이 동의 프롬프트
    #   240바이트를 fd 1 에 쓴다 (실측). 예전에는 이 블록이 parse_args() 뒤에
    #   있어서 그 240바이트가 그대로 프로토콜 스트림에 섞였고, 증상은
    #   "워커 왕복 MemoryError" 하나로만 나타났다.
    _proto_fd = os.dup(1)
    os.dup2(2, 1)
    proto_out = os.fdopen(_proto_fd, "wb")

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
    from vla_isaac_tasks.mdp import observations as _obs_mod
    from vla_isaac_tasks.mdp import set_forced_indices, use_bank
    from vla_isaac_tasks.spec import SPEC

    # 평가 split 마다 지시문 템플릿을 바꿀 수 있게 둔다 (Language split, §6).
    # None 이면 학습에 쓴 기본 템플릿.
    instruction_template = {"value": None}

    stdin = sys.stdin.buffer
    stdout = proto_out

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
        # 진단 래치: "블록을 집어 들어올리는 것까지는 됐는가".
        # 커브가 평평할 때 파지 실패인지 배치 정밀도 문제인지 가르는 값이다.
        lift_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        grasp_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        height_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
        tray_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 파지 "유지" 카운터. 스치듯 잡았다 놓친 것에 크레딧을 주지 않는다.
        # ★ 놓친 것에 **패널티를 주지는 않는다**. 파지 보상 0.2 보다 큰 벌점을
        #   주면 "아예 안 잡는" 쪽이 이득이 되어 정책이 블록을 회피하게 된다
        #   (shaping 의 전형적 실패). 인정 조건만 엄격히 하고 재시도는 자유롭게.
        grasp_run = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # ---------------------------------------------------------------
        # 단계형 sparse 보상 (VLA_STAGED_REWARD=1 로 켠다. 기본은 outcome-only)
        #
        # ★ 왜 필요한가
        #   outcome-only 는 그룹 전체가 실패하면 표준편차가 0 이라 어드밴티지가
        #   0 이 된다 (grpo_fallback.compute_group_advantages). 그래서 SimpleVLA-RL
        #   은 성공률 몇 % 를 착수 조건으로 요구한다. 우리 SFT 는 최종 성공이
        #   0% 지만 파지 60% / 리프트 22.5% 로 **중간 단계에는 분산이 있다**.
        #   단계를 나누면 그 분산이 학습 신호가 된다.
        #
        # ★ 반복 수확 방지
        #   각 이벤트는 래치라 에피소드당 한 번만 켜지고, 보상은 켜진 단계들의
        #   **최댓값 하나**다. 누적이 아니므로 같은 이벤트를 반복해도 늘지 않고,
        #   한 번 얻은 단계를 잃지도 않는다(뒤로 갈 유인이 없다).
        #
        # ★ 보고는 outcome 으로
        #   diag["success"] 에 원래 기준의 성공률을 따로 실어 보낸다. 학습을
        #   단계 보상으로 하더라도 최종 숫자는 outcome 으로 보고할 것.
        STAGED = os.environ.get("VLA_STAGED_REWARD", "") in ("1", "true", "True")
        STAGE_GRASP, STAGE_LIFT, STAGE_TRAY = 0.2, 0.4, 0.7
        # 파지를 인정하는 최소 유지 스텝 (8Hz 기준 4스텝 = 0.5초)
        GRASP_HOLD_STEPS = int(os.environ.get("VLA_GRASP_HOLD_STEPS", 4))

        def _reward():
            if not STAGED:
                return success_latch.float()
            r = torch.zeros(num_envs, dtype=torch.float32, device=device)
            r = torch.maximum(r, grasp_latch.float() * STAGE_GRASP)
            r = torch.maximum(r, lift_latch.float() * STAGE_LIFT)
            r = torch.maximum(r, tray_latch.float() * STAGE_TRAY)
            return torch.maximum(r, success_latch.float())

        if STAGED:
            log(f"단계형 보상 ON — 파지 {STAGE_GRASP}({GRASP_HOLD_STEPS}스텝 유지) / "
                f"리프트 {STAGE_LIFT} / 트레이 {STAGE_TRAY}(순간값) / "
                f"성공 1.0({SPEC.SUCCESS_HOLD_STEPS}스텝 유지)")

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

            # 지시문. env 마다 타깃 블록·슬롯이 다르므로 문장도 env 마다 다르다.
            # ★ 여기가 롤아웃 쪽 지시문의 유일한 출처다. 정책 프롬프트를 다른
            #   곳에서 따로 만들면 SFT 데이터의 문장과 어긋나고, 증상은
            #   "RFT 를 켜니 성능이 무너진다" 로만 나타난다.
            ids = policy.get("target_ids")
            if ids is not None:
                ids = ids.detach().cpu().numpy().astype(int)
                out["instruction"] = [
                    SPEC.instruction_for(int(b), instruction_template["value"])
                    for (b,) in ids
                ]
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
                    # ★ 초기 상태는 시드가 아니라 **뱅크 인덱스**로 지정한다
                    #   (개정 §3). 시드만으로는 배치 안의 env 들이 서로 다른
                    #   s₀ 를 갖게 되어, GRPO 가 "같은 상태에서 G개" 라는 전제를
                    #   만족하지 못한다. 정수 하나를 주면 전 env 가 동일한 s₀ 다.
                    if msg.get("bank"):
                        use_bank(str(msg["bank"]))
                    if "instruction_template" in msg:
                        instruction_template["value"] = msg["instruction_template"]
                    if msg.get("init_index") is not None:
                        set_forced_indices(int(msg["init_index"]))
                    elif msg.get("init_indices") is not None:
                        set_forced_indices(list(msg["init_indices"]))
                    else:
                        set_forced_indices(None)

                    seeds = msg.get("seeds")
                    if seeds:
                        # 시드는 팔 관절 노이즈 등 뱅크 밖의 잔여 랜덤만 지배한다.
                        env.unwrapped.seed(int(seeds[0]))
                    obs_dict, _ = env.reset()
                    success_latch.zero_()
                    done_latch.zero_()
                    lift_latch.zero_()
                    grasp_latch.zero_()
                    height_max.zero_()
                    tray_latch.zero_()
                    grasp_run.zero_()
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
                        sub = obs_dict.get("subtask_terms")
                        if sub is not None and "grasp_lift" in sub:
                            lift_latch |= sub["grasp_lift"].bool()
                        # ★ lifted 가 0 일 때 "전혀 못 들었다" 와 "들었지만
                        #   60mm 임계에 못 미쳤다" 를 구분할 수 없어 진단이 막힌다.
                        #   실제 도달 높이와 파지 여부를 따로 기록해 둔다.
                        # ★ env 는 gym 래퍼다. observations 의 함수들은
                        #   ManagerBasedRLEnv 를 기대하므로 unwrapped 를 넘긴다.
                        #   (래퍼를 넘기면 조용히 예외가 나 계측이 0 으로 남는다)
                        _u = env.unwrapped
                        h = _obs_mod.target_height_above_table(_u).squeeze(-1)
                        height_max = torch.maximum(height_max, h)
                        # ★ 파지 판정에 "블록이 실제로 떴는가" 를 덧붙인다.
                        #   target_grasped 는 eef-블록 거리 60mm + 그리퍼 폭
                        #   60mm 미만이면 참이라, **닫힌 빈 손이 블록 옆을
                        #   지나가기만 해도** 켜진다. 보상 단계로 쓰면 거의 공짜다.
                        #   놓인 중심(블록높이/2)보다 5mm 위를 함께 요구해
                        #   "쥐고 움직였다" 는 증거를 붙인다.
                        _rest = SPEC.BLOCK_SIZE[2] / 2
                        _g_now = _obs_mod.target_grasped(_u) & (h > _rest + 0.005)
                        grasp_run = torch.where(
                            _g_now, grasp_run + 1, torch.zeros_like(grasp_run)
                        )
                        grasp_latch |= grasp_run >= GRASP_HOLD_STEPS
                        lift_latch |= _obs_mod.grasp_lift_signal(_u)
                        # 트레이 진입. 두 가지를 함께 요구한다:
                        #   settled — 튕겨 지나가는 한 프레임으로 크레딧을 주지 않는다
                        #   lift_latch — 들지 않고 **끌어서 밀어넣는** 해를 막는다
                        #     (박스가 사라져 지금은 끌기가 물리적으로 가능하다)
                        # ★ 유지 조건은 두지 않는다(순간값). settled 임계가
                        #   0.03 m/s 라 느리게 통과하는 중에도 만족되므로, 이 단계는
                        #   "스침" 을 포함한다는 것을 알고 쓸 것 — 0.7 → 1.0 의
                        #   간격이 완수를 유인하는 구조에 의존한다.
                        #   로그의 `진입` 대비 `성공` 격차가 벌어지면 해킹을 의심.
                        _settled = torch.linalg.norm(
                            _obs_mod.target_block_lin_vel(_u), dim=-1
                        ) < SPEC.SUCCESS_LIN_VEL_THRESHOLD
                        tray_latch |= (
                            _obs_mod.block_in_tray(_u) & _settled & lift_latch
                        )
                        # ★ 임계는 0 이다. 절대값(0.5 같은 것)으로 되돌리지 말 것.
                        #   Isaac Lab 의 RewardManager 는 `func() * weight * dt` 를
                        #   돌려준다. 성공해도 env 가 주는 값은 1.0 이 아니라
                        #   1.0 × step_dt 다. 0.5 로 잡으면 **성공이 영원히 안 잡히고**
                        #   RFT 학습 신호가 0 으로 고정되는데, 에러는 나지 않는다.
                        #   부호만 보면 decimation 을 바꿔도 맞는다 — 가중치 있는
                        #   보상 항이 success 하나뿐이기 때문이다.
                        success_latch |= reward > 1e-6
                        done_latch |= terminated | truncated

                    # 진단값은 reward 와 분리해 보낸다. 보상에 섞으면 0/1 이
                    # 아니게 되어 GRPO 의 그룹 정규화와 무신호 판정이 망가진다.
                    yaw = obs_dict["policy"].get("yaw_err")
                    send_ok({
                        "obs": extract_obs(obs_dict),
                        "reward": _reward().cpu().numpy(),
                        "done": done_latch.cpu().numpy(),
                        "diag": {
                            "lifted": float(lift_latch.float().mean()),
                            "grasped": float(grasp_latch.float().mean()),
                            "height_max": float(height_max.max()),
                            "in_tray": float(tray_latch.float().mean()),
                            "success": float(success_latch.float().mean()),
                            "yaw_err": (
                                float(yaw.mean()) if yaw is not None else float("nan")
                            ),
                        },
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

"""isaaclab venv 스모크 테스트 — Phase 1·2·4b 환경 검증.

확인 항목:
  1. torch 2.7.0 + CUDA 사용 가능
  2. isaacsim / isaaclab import
  3. 우리 태스크 3종이 gym 에 등록되었는지
  4. SimulationApp 기동 + 씬 생성 + 1 스텝 진행
  5. 카메라 관측 텐서 shape 가 스펙(224x224x3)과 일치하는지

실행:
    python env/smoke/check_isaaclab.py            # 1~3번만 (빠름, ~10초)
    python env/smoke/check_isaaclab.py --full     # 전체 (Isaac Sim 기동, 수 분)

계획서 §4-1: pip check 가 아니라 이 스크립트가 환경 검증 기준이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import check, finish, load_vla_spec, warn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Isaac Sim 을 실제로 기동해 씬 생성과 카메라 렌더까지 검증한다 (수 분 소요).",
    )
    args = parser.parse_args()

    print("=== isaaclab 환경 스모크 테스트 ===\n")

    # -- 1. 공유 스펙 ---------------------------------------------------------
    spec = None

    def _spec():
        nonlocal spec
        spec = load_vla_spec()
        spec.assert_consistent()
        return "configs/vla_spec.py 정합성 OK"

    check("공유 스펙 로드", _spec)

    # -- 2. torch ------------------------------------------------------------
    def _torch():
        import torch

        version = torch.__version__
        if not version.startswith("2.7.0"):
            raise AssertionError(
                f"torch {version} — 2.7.0 이어야 한다. Isaac Sim 5.1.0 이 이 버전에 "
                "고정되어 있으므로 상향 금지 (계획서 §4-3)."
            )
        if not torch.cuda.is_available():
            raise AssertionError("CUDA 사용 불가 — 드라이버/설치 상태 확인 필요.")
        if "cu128" not in version:
            warn(
                f"torch 빌드 문자열에 cu128 이 없다 ({version}). 기본 PyPI 휠이 덮어썼을 수 "
                "있다. --index-url https://download.pytorch.org/whl/cu128 로 재설치 검토."
            )
        return f"{version} / {torch.cuda.get_device_name(0)}"

    check("torch 2.7.0 + CUDA", _torch)

    # -- 3. Isaac Sim / Isaac Lab import -------------------------------------
    def _isaacsim_import():
        import isaacsim  # noqa: F401
        import isaaclab  # noqa: F401

        return f"isaaclab {getattr(__import__('isaaclab'), '__version__', 'unknown')}"

    check("isaacsim / isaaclab import", _isaacsim_import)

    # -- 3b. cuRobo (SkillGen) — 없어도 실패로 처리하지 않는다 ----------------
    # MimicGen 경로가 살아 있으므로 cuRobo 부재는 "기능 축소" 이지 "환경 파손" 이
    # 아니다. 여기서 FAIL 을 내면 lock 박제가 막혀 멀쩡한 환경을 못 쓰게 된다.
    try:
        import curobo  # noqa: F401

        print(
            "\033[1;32m[ OK ]\033[0m cuRobo (SkillGen 사용 가능) — "
            "생성 시 --use_skillgen 을 붙일 것"
        )
    except ImportError:
        warn(
            "cuRobo 없음 → SkillGen 불가. MimicGen(선형 보간) 방식으로 생성하면 된다.\n"
            "       generate_dataset.py 에서 --use_skillgen 을,\n"
            "       annotate_demos.py 에서 --annotate_subtask_start_signals 를 빼면 된다.\n"
            "       환경 코드는 양쪽을 모두 지원하므로 고칠 것이 없다.\n"
            "       설치를 재시도하려면: USE_SKILLGEN=1 ./setup/setup_isaaclab.sh"
        )

    if not args.full:
        print(
            "\n--full 없이 실행됨 → Isaac Sim 기동·씬 생성·카메라 검증은 건너뛴다.\n"
            "환경 구축 직후에는 반드시 한 번 --full 로 돌릴 것."
        )
        return finish("isaaclab")

    # -- 4. SimulationApp 기동 (여기부터 수 분 소요) --------------------------
    # AppLauncher 는 반드시 다른 isaaclab 모듈보다 먼저 실행되어야 한다.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch

        # 우리 태스크 패키지를 import 하면 gym.register 가 실행된다.
        import vla_isaac_tasks  # noqa: F401

        def _registered():
            expected = [
                "VlaPick-v0",
                "VlaPick-Mimic-v0",
                "VlaPick-Visuomotor-Mimic-v0",
            ]
            missing = [t for t in expected if t not in gym.registry]
            if missing:
                raise AssertionError(
                    f"gym 에 등록되지 않은 태스크: {missing}. "
                    "source/vla_isaac_tasks/__init__.py 의 gym.register 확인."
                )
            return ", ".join(expected)

        check("태스크 3종 gym 등록", _registered)

        # -- 5. 씬 생성 + 스텝 + 카메라 관측 ---------------------------------
        env_holder: dict = {}

        def _make_env():
            from isaaclab_tasks.utils import parse_env_cfg

            env_cfg = parse_env_cfg(
                "VlaPick-Visuomotor-Mimic-v0", device="cuda:0", num_envs=2
            )
            env = gym.make("VlaPick-Visuomotor-Mimic-v0", cfg=env_cfg)
            env_holder["env"] = env
            return f"num_envs=2, action_space={env.action_space.shape}"

        # -- 텔레옵 장치 설정 (Day 1 오후를 막는 종류의 버그) ------------------
        def _teleop_devices():
            from isaaclab_tasks.utils import parse_env_cfg

            cfg = parse_env_cfg("VlaPick-v0", device="cuda:0", num_envs=1)
            devices = getattr(cfg, "teleop_devices", None)
            if devices is None:
                raise AssertionError(
                    "teleop_devices 가 None 이다. record_demos.py 의 "
                    "setup_teleop_device() 가 hasattr 통과 후 None.devices 에 접근해 "
                    "AttributeError 로 죽는다. 속성을 아예 두지 않는 것과 None 을 "
                    "넣는 것은 다르다 — DevicesCfg(...) 를 설정할 것."
                )
            registered = set(devices.devices.keys())
            missing = {"keyboard", "gamepad"} - registered
            if missing:
                raise AssertionError(
                    f"등록되지 않은 텔레옵 장치: {missing}. "
                    "폴백 경로는 keyboard/spacemouse 만 만들어 주므로 gamepad 는 "
                    "여기에 없으면 --teleop_device gamepad 가 에러로 종료된다."
                )
            return f"등록됨: {sorted(registered)}"

        check("텔레옵 장치 설정", _teleop_devices)

        if check("Visuomotor-Mimic 환경 생성", _make_env):

            def _step_and_check_obs():
                env = env_holder["env"]
                obs, _ = env.reset()

                # 액션 차원 검증 — 스펙과 어긋나면 SFT/RFT 전체가 어긋난다.
                act_dim = env.action_space.shape[-1]
                if act_dim != spec.ACTION_DIM:
                    raise AssertionError(
                        f"액션 차원 {act_dim} != 스펙 {spec.ACTION_DIM}. "
                        "configs/vla_spec.py 와 환경 cfg 중 하나가 어긋났다."
                    )

                zero_action = torch.zeros(
                    (env.unwrapped.num_envs, act_dim), device=env.unwrapped.device
                )
                obs, _, _, _, _ = env.step(zero_action)

                # 카메라 관측 검증
                policy_obs = obs["policy"]
                cam_key = spec.CAMERA_SENSOR_NAME
                if cam_key not in policy_obs:
                    raise AssertionError(
                        f"관측에 카메라 키 '{cam_key}' 가 없다. 있는 키: "
                        f"{list(policy_obs.keys())}"
                    )
                img = policy_obs[cam_key]
                expected = (spec.IMAGE_HEIGHT, spec.IMAGE_WIDTH, 3)
                if tuple(img.shape[1:]) != expected:
                    raise AssertionError(
                        f"카메라 관측 shape {tuple(img.shape)} — 기대 (N, {expected})"
                    )
                return f"obs['{cam_key}'] {tuple(img.shape)}, dtype={img.dtype}"

            check("리셋·1스텝·카메라 관측 shape", _step_and_check_obs)

            def _close():
                env_holder["env"].close()
                return None

            check("환경 정리", _close)

    finally:
        simulation_app.close()

    return finish("isaaclab")


if __name__ == "__main__":
    sys.exit(main())

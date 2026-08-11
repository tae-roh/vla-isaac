"""[보류] 변형체 예비 테스트 — 후속 확장으로 밀렸다 (개정 8절).
VlaPick-Deformable-v0 은 gym 에 등록되어 있지 않아 지금 실행하면 NameNotFound 다.

변형체 파지 예비 테스트 — FEM 파라미터를 확보한다 (계획서 §Phase1-7).

Day 1 에 15~20분만 쓰고 넘어갈 것. 목적은 "변형체 태스크를 완성하는 것"이 아니라
**Phase 4b 스트레치에 진입할 때 쓸 파라미터를 미리 찾아 두는 것**이다.
여기서 물리 튜닝에 빠지면 3~4일 일정이 그대로 무너진다.

무엇을 보는가:
  1. 그리퍼가 물체를 잡고 들어올릴 수 있는가 (파지 안정성)
  2. FEM 이 발산하지 않는가 (노드 퍼짐 spread 가 폭증하지 않는가)
  3. 스텝 속도가 RFT 롤아웃에 감당 가능한가 (FPS)

계획서 §6 의 fallback 대로 **단단한 쪽(높은 youngs_modulus)에서 시작해 내려온다.**
물렁한 쪽에서 시작하면 관통·발산이 나서 무엇이 문제인지 구분되지 않는다.

사용 예:
    python scripts/test_deformable_grasp.py --headless \
        --youngs 5e5 1e5 5e4 --steps 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--youngs", type=float, nargs="+", default=[5e5, 1e5, 5e4],
                        help="스윕할 영률 목록 (큰 값 = 단단함)")
    parser.add_argument("--steps", type=int, default=300, help="설정당 시뮬 스텝 수")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--livestream", type=int, default=0)
    args = parser.parse_known_args()[0]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=args.headless, livestream=args.livestream, enable_cameras=True
    )
    simulation_app = app_launcher.app

    results = []
    try:
        import gymnasium as gym
        import torch

        import vla_isaac_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from vla_isaac_tasks.mdp import deformable as deform_mdp
        from vla_isaac_tasks.spec import SPEC

        for youngs in args.youngs:
            print(f"\n{'=' * 60}\n영률 {youngs:.1e} 테스트\n{'=' * 60}")

            env_cfg = parse_env_cfg(
                "VlaPick-Deformable-v0", device="cuda:0", num_envs=args.num_envs
            )
            env_cfg.scene.object.spawn.physics_material.youngs_modulus = youngs
            env = gym.make("VlaPick-Deformable-v0", cfg=env_cfg)

            obs, _ = env.reset()
            device = env.unwrapped.device

            # 스크립트된 파지 시퀀스: 내려가기 → 닫기 → 들어올리기.
            # 정책이 아니라 고정 시퀀스인 것이 의도한 바다 — 물리만 보려는 것이므로
            # 정책 품질이라는 변수를 끼워 넣지 않는다.
            def phase_action(step: int) -> torch.Tensor:
                a = torch.zeros((env.unwrapped.num_envs, SPEC.ACTION_DIM), device=device)
                frac = step / args.steps
                if frac < 0.35:            # 하강 + 그리퍼 열림
                    a[:, 2] = -0.35
                    a[:, 6] = SPEC.GRIPPER_OPEN_ISAACLAB
                elif frac < 0.55:          # 정지 + 그리퍼 닫기
                    a[:, 6] = SPEC.GRIPPER_CLOSE_ISAACLAB
                else:                      # 상승 (닫은 채로)
                    a[:, 2] = 0.35
                    a[:, 6] = SPEC.GRIPPER_CLOSE_ISAACLAB
                return a

            spread0 = None
            max_spread = 0.0
            t0 = time.time()

            for step in range(args.steps):
                obs, *_ = env.step(phase_action(step))
                spread = deform_mdp.deformable_spread(env.unwrapped).mean().item()
                if spread0 is None:
                    spread0 = spread
                max_spread = max(max_spread, spread)

            elapsed = time.time() - t0
            fps = args.steps / elapsed

            center = deform_mdp.deformable_center(env.unwrapped)
            lifted = (center[:, 2] - SPEC.TABLE_HEIGHT).mean().item()
            spread_ratio = max_spread / spread0 if spread0 else float("inf")

            # spread 가 초기의 3배를 넘으면 FEM 이 사실상 발산한 것으로 본다.
            diverged = spread_ratio > 3.0
            grasped = lifted > 0.05 and not diverged

            results.append(
                dict(youngs=youngs, lifted=lifted, spread_ratio=spread_ratio,
                     fps=fps, grasped=grasped, diverged=diverged)
            )
            print(f"  평균 상승 높이 : {lifted * 100:.1f} cm")
            print(f"  노드 퍼짐 배율 : {spread_ratio:.2f}x "
                  f"({'발산' if diverged else '안정'})")
            print(f"  속도           : {fps:.1f} FPS")
            print(f"  파지 성공      : {'○' if grasped else '✗'}")

            env.close()
    finally:
        simulation_app.close()

    # -- 요약 ---------------------------------------------------------------
    print(f"\n{'=' * 60}\n요약\n{'=' * 60}")
    print(f"{'영률':>10} {'상승(cm)':>10} {'퍼짐배율':>10} {'FPS':>8} {'파지':>6}")
    for r in results:
        print(f"{r['youngs']:10.1e} {r['lifted'] * 100:10.1f} "
              f"{r['spread_ratio']:10.2f} {r['fps']:8.1f} "
              f"{'○' if r['grasped'] else '✗':>6}")

    good = [r for r in results if r["grasped"]]
    print()
    if good:
        # 파지되는 것 중 가장 물렁한 것 = 가장 흥미로운 실험 조건
        best = min(good, key=lambda r: r["youngs"])
        print(f"★ 권장 시작값: youngs_modulus = {best['youngs']:.1e}")
        print(f"  파지되는 것 중 가장 물렁한 설정이다. "
              f"source/vla_isaac_tasks/deformable_env_cfg.py 에 반영할 것.")
        if best["fps"] < 30:
            print(f"  [WARN] {best['fps']:.0f} FPS 는 느리다. Phase 4b 에서 "
                  "num_envs 를 8 이하로 잡거나 simulation_hexahedral_resolution 을 낮출 것.")
    else:
        print("★ 어떤 설정에서도 파지에 실패했다.")
        print("  계획서 §6 의 fallback 대로 '변형 강도 축소(준강체)' 로 간다:")
        print("  더 큰 영률(1e6 이상)을 추가로 시도하고, 그래도 안 되면")
        print("  변형체는 스트레치에서 제외하고 강체 RFT 결과에 집중할 것.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

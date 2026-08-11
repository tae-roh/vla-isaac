"""관측 텐서를 PNG 로 덤프하고, 나중에 픽셀 단위로 대조한다.

두 가지 역할을 한다:

  (Phase 1) --save   씬을 만들고 VLA 가 실제로 보게 될 프레임을 PNG 로 저장.
                     카메라 위치·화각·해상도를 여기서 확정한다. 이후 페이즈에서는
                     바꿀 수 없다 — SFT 데이터를 만든 설정이 곧 RFT 설정이다.

  (Phase 4) --compare  RFT 어댑터가 만드는 관측을 Phase 1 레퍼런스와 대조.

★ 왜 이게 별도 스크립트로 있어야 하는가 (계획서 §Phase4b-3)

  라이브 뷰포트 화면과 모델이 받는 입력 텐서는 다른 것이다. 뷰포트가 멀쩡해
  보여도 텐서는 뒤집혀 있거나, 채널 순서가 다르거나, 정규화가 이중으로
  걸려 있을 수 있다. 그 불일치는 학습 커브가 오르지 않는 형태로만 드러나서
  원인 추적에 며칠이 든다. 픽셀을 직접 비교하는 것이 유일하게 확실한 방법이다.

사용 예:
    # Phase 1 — 레퍼런스 저장 (라이브로 보면서 카메라 확정)
    python scripts/dump_obs_reference.py --task VlaPlace-v0 --save \
        --out datasets/obs_reference --num-frames 8 --livestream 2

    # Phase 4 — 어댑터 관측과 대조
    python scripts/dump_obs_reference.py --compare datasets/obs_reference \
        --candidate rft/debug_frames
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="VlaPlace-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=8,
                        help="저장할 프레임 수. 리셋을 반복하며 서로 다른 형상을 담는다.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "obs_reference")
    parser.add_argument("--save", action="store_true", help="Isaac Sim 을 띄워 레퍼런스를 저장")
    parser.add_argument("--compare", type=Path, default=None, help="레퍼런스 디렉토리")
    parser.add_argument("--candidate", type=Path, default=None, help="비교 대상 디렉토리")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="채널당 평균 절대 오차 허용치 (0~255 스케일)")
    # ★ AppLauncher 로 그대로 넘긴다. 예전에는 이 인자가 parse_known_args 에 흡수되어
    #   조용히 무시됐고, --livestream 2 를 줘도 헤드리스로만 돌아 카메라를 눈으로
    #   확인할 수 없었다 (RUNBOOK §1-2 는 라이브로 보면서 확정하는 단계다).
    #   AppLauncher.add_app_launcher_args 를 쓰지 않는 이유: 이 스크립트의 --compare
    #   모드는 isaaclab 이 없는 venv(env_rft)에서도 돌아야 하므로 isaaclab.app 을
    #   모듈 최상단에서 import 할 수 없다. kwarg 로 넘기는 것도 공식 경로다.
    parser.add_argument("--livestream", type=int, default=-1, choices=[-1, 0, 1, 2],
                        help="1=WebRTC 공용망(PUBLIC_IP 필요) / 2=WebRTC 사설망. "
                             "-1 이면 LIVESTREAM 환경변수를 따른다.")
    return parser.parse_known_args()[0]


# =============================================================================
# 비교 모드 — Isaac Sim 이 필요 없다
# =============================================================================
def run_compare(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    ref_dir, cand_dir = args.compare, args.candidate
    if cand_dir is None:
        print("--compare 를 쓰려면 --candidate 도 필요하다.", file=sys.stderr)
        return 2

    meta_path = ref_dir / "spec.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"레퍼런스 스펙: {json.dumps(meta, ensure_ascii=False, indent=2)}\n")

    ref_imgs = sorted(ref_dir.glob("*.png"))
    cand_imgs = sorted(cand_dir.glob("*.png"))
    if not ref_imgs:
        print(f"레퍼런스 PNG 가 없다: {ref_dir}", file=sys.stderr)
        return 2
    if not cand_imgs:
        print(f"비교 대상 PNG 가 없다: {cand_dir}", file=sys.stderr)
        return 2

    print(f"레퍼런스 {len(ref_imgs)}장 / 비교 대상 {len(cand_imgs)}장\n")

    # 개별 프레임은 물리 랜덤성 때문에 정확히 같을 수 없다. 그래서 픽셀 대 픽셀이
    # 아니라 (1) 해상도·채널 구조, (2) 채널별 통계 분포를 본다.
    # 프레이밍/정규화/채널순서가 어긋나면 통계가 확실히 달라진다.
    def stats(paths: list[Path]) -> dict:
        arrs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) for p in paths]
        stacked = np.stack(arrs)
        return {
            "shape": arrs[0].shape,
            "mean_rgb": stacked.mean(axis=(0, 1, 2)),
            "std_rgb": stacked.std(axis=(0, 1, 2)),
        }

    ref_s, cand_s = stats(ref_imgs), stats(cand_imgs)

    ok = True
    if ref_s["shape"] != cand_s["shape"]:
        print(f"[FAIL] 해상도 불일치: 레퍼런스 {ref_s['shape']} vs 대상 {cand_s['shape']}")
        ok = False
    else:
        print(f"[ OK ] 해상도 일치: {ref_s['shape']}")

    mean_diff = np.abs(ref_s["mean_rgb"] - cand_s["mean_rgb"])
    print(f"       채널 평균  레퍼런스 {ref_s['mean_rgb'].round(1)} "
          f"vs 대상 {cand_s['mean_rgb'].round(1)}  (차이 {mean_diff.round(1)})")
    if mean_diff.max() > args.tolerance * 10:
        print(f"[FAIL] 채널 평균 차이가 크다 (최대 {mean_diff.max():.1f}).")
        print("       의심 순서: ① 카메라 위치/화각 불일치 ② RGB/BGR 채널 순서 "
              "③ 정규화 이중 적용 ④ 상하 반전(ROTATE_IMAGE_180)")
        ok = False
    else:
        print("[ OK ] 채널 평균 분포 유사")

    # 채널 순서가 바뀌면 R↔B 를 뒤집었을 때 오히려 더 잘 맞는다 — 그걸로 진단한다.
    swapped = np.abs(ref_s["mean_rgb"] - cand_s["mean_rgb"][::-1])
    if swapped.max() < mean_diff.max() * 0.5:
        print("[FAIL] R↔B 를 뒤집으면 더 잘 맞는다 → 채널 순서(RGB/BGR)가 반대다.")
        ok = False

    print()
    if ok:
        print("=== 관측 스펙 대조 통과 ===")
        return 0
    print("=== 관측 스펙 대조 실패 — 여기서 잡지 않으면 RFT 커브가 오르지 않는다 ===")
    return 1


# =============================================================================
# 저장 모드 — Isaac Sim 필요
# =============================================================================
def run_save(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    # headless=True 로 고정해도 라이브스트림은 켜진다 — AppLauncher 는 livestream 이
    # 1/2 일 때 호스트를 항상 헤드리스로 두고 WebRTC 로만 내보낸다.
    # enable_cameras 는 선택이 아니다: 씬에 TiledCamera 가 있어 없으면 센서 초기화가 실패한다.
    app_launcher = AppLauncher(
        headless=True, enable_cameras=True, livestream=args.livestream
    )
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image

        import vla_isaac_tasks  # noqa: F401  — gym.register
        from isaaclab_tasks.utils import parse_env_cfg
        from vla_isaac_tasks.spec import SPEC

        args.out.mkdir(parents=True, exist_ok=True)

        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)

        print(SPEC.summary())

        # 루프 밖에서 한 번만 만든다 — 아래 라이브스트림 유지 루프도 이 텐서를 쓴다.
        zero = torch.zeros(
            (env.unwrapped.num_envs, SPEC.ACTION_DIM), device=env.unwrapped.device
        )

        saved = 0
        obs, _ = env.reset()
        while saved < args.num_frames:
            # 몇 스텝 굴려 물체가 안착한 뒤에 찍는다. 리셋 직후 프레임은
            # 자재가 공중에 떠 있어 레퍼런스로 부적절하다.
            for _ in range(15):
                obs, *_ = env.step(zero)

            img = obs["policy"][SPEC.CAMERA_SENSOR_NAME]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.dtype != np.uint8:
                # normalize=False 로 두었으므로 uint8 이어야 정상이다.
                print(f"[WARN] 이미지 dtype 이 {img.dtype} 다. "
                      "관측 term 의 normalize 설정을 확인할 것.")
                img = np.clip(img, 0, 255).astype(np.uint8)

            for i in range(img.shape[0]):
                if saved >= args.num_frames:
                    break
                path = args.out / f"ref_{saved:03d}.png"
                Image.fromarray(img[i]).save(path)
                print(f"  저장 {path}  shape={img[i].shape}")
                saved += 1

            obs, _ = env.reset()

        # 스펙을 함께 남긴다 — Phase 4 에서 무엇과 비교하는지 명확해야 한다.
        (args.out / "spec.json").write_text(
            json.dumps(
                {
                    "task": args.task,
                    "image_hw": [SPEC.IMAGE_HEIGHT, SPEC.IMAGE_WIDTH],
                    "camera_sensor": SPEC.CAMERA_SENSOR_NAME,
                    "camera_pos": list(SPEC.CAMERA_POS),
                    "camera_rot": list(SPEC.CAMERA_ROT),
                    "camera_convention": SPEC.CAMERA_CONVENTION,
                    "rotate_image_180": SPEC.ROTATE_IMAGE_180,
                    "action_dim": SPEC.ACTION_DIM,
                    "num_actions_chunk": SPEC.NUM_ACTIONS_CHUNK,
                    "instruction_template": SPEC.INSTRUCTION_TEMPLATE,
                    "block_attrs": list(SPEC.BLOCK_ATTRS),
                    "slot_names": list(SPEC.SLOT_NAMES),
                    "pocket_clearance": SPEC.POCKET_CLEARANCE,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"\n레퍼런스 {saved}장 + spec.json → {args.out}")
        print("★ 이 PNG 들을 반드시 눈으로 확인할 것:")
        print("   - 박스(블록 3개)와 포켓 트레이가 둘 다 화면 안에 있는가")
        print("   - 블록 3색이 서로 구분되는가")
        print("   - 슬롯 좌우가 지시문의 left/right 와 맞는가 "
              "(어긋나면 SPEC.SLOT_Y_SIGN 을 -1 로)")
        print("   - 그리퍼가 블록에 닿는 순간이 가려지지 않는가")
        print("   - 이미지가 뒤집혀 있지 않은가 (뒤집혀 있으면 "
              "configs/vla_spec.py 의 ROTATE_IMAGE_180 을 켤 것)")
        print("   - Phase 3·4 로 반드시 함께 가져갈 것 (부록 A 체크리스트)")

        # 라이브스트림을 켰다면 여기서 앱을 닫으면 안 된다. PNG 저장은 몇 초면
        # 끝나서, 그대로 종료하면 WebRTC 클라이언트가 붙기도 전에 창이 사라진다.
        # 카메라 확정은 사람이 보고 판단하는 단계이므로 Ctrl-C 까지 씬을 굴려 준다.
        # --livestream 이 -1 이면 AppLauncher 는 LIVESTREAM 환경변수를 따르므로,
        # 유지 여부 판정도 같은 규칙으로 계산해야 한다.
        livestream = (
            args.livestream
            if args.livestream >= 0
            else int(os.environ.get("LIVESTREAM", 0))
        )
        if livestream >= 1 and simulation_app.is_running():
            print(f"\n라이브스트림 유지 중 (livestream={livestream}). "
                  "WebRTC 클라이언트로 접속해 씬을 확인할 것.")
            print("확인이 끝나면 Ctrl-C.")

            # SIGINT 를 직접 받는다. try/except KeyboardInterrupt 로는 안 된다 —
            # SimulationApp 이 자기 SIGINT 핸들러를 걸어 두어서, 그냥 Ctrl-C 하면
            # 파이썬까지 올라오지 않고 앱이 종료 코드 1 로 즉사한다.
            # (PNG 는 이미 저장된 뒤라 잃는 것은 없지만, 문서에 적어 둔 정상 종료
            #  경로가 에러로 끝나는 것은 다음 페이즈에서 오해를 만든다.)
            interrupted = {"flag": False}

            def _on_sigint(signum, frame):  # noqa: ARG001
                interrupted["flag"] = True

            signal.signal(signal.SIGINT, _on_sigint)
            while simulation_app.is_running() and not interrupted["flag"]:
                obs, *_ = env.step(zero)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n종료한다.")

        env.close()
    finally:
        simulation_app.close()

    return 0


def main() -> int:
    args = _parse_args()
    if args.compare is not None:
        return run_compare(args)
    if args.save:
        return run_save(args)
    print("--save 또는 --compare 중 하나를 지정할 것.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""HDF5 데모의 table_cam 관측을 mp4 로 뽑는다.

★ 왜 뷰포트 스트리밍 대신 이걸 쓰나

  WebRTC 라이브스트림은 시그널링(49100/tcp)과 미디어(47998~48020)를 서로 다른
  포트로 쓴다. NAT/터널 환경에서는 미디어 경로가 막혀 ICE 가 16초 만에 죽는다.
  그리고 진단 목적으로는 뷰포트보다 이쪽이 낫다 — **정책이 실제로 입력받는
  224x224 table_cam 프레임 그 자체**라, 보이는 것과 정책이 보는 것이 일치한다.

사용:
    python scripts/render_demo_video.py --input datasets/generated_8hz.hdf5 \
        --episodes 3 --outdir videos/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """(T,H,W,C) 로 정규화. float 이면 0~255 로 올리고 알파는 버린다."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.ndim == 4 and a.shape[-1] == 4:
        a = a[..., :3]
    return a


def render(src: Path, outdir: Path, n_ep: int, fps: int, scale: int) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))[:n_ep]
        for k in keys:
            d = f["data"][k]
            if "table_cam" not in d["obs"]:
                print(f"[SKIP] {k}: table_cam 없음")
                continue
            frames = _to_uint8_rgb(d["obs"]["table_cam"][:])
            if scale > 1:                     # 224px 는 눈으로 보기엔 작다
                frames = np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
            out = outdir / f"{src.stem}_{k}.mp4"
            imageio.mimsave(out, list(frames), fps=fps, macro_block_size=1)
            print(f"{out}  ({len(frames)}프레임, {len(frames) / fps:.1f}초, "
                  f"{out.stat().st_size / 2**20:.1f}MB)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "videos")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=8, help="8Hz 데이터면 8 이 실시간")
    p.add_argument("--scale", type=int, default=2, help="정수배 확대")
    a = p.parse_args()
    return render(a.input, a.outdir, a.episodes, a.fps, a.scale)


if __name__ == "__main__":
    sys.exit(main())

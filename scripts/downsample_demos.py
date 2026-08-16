"""생성된 데모를 시간축으로 솎아 제어 주기를 낮춘다 (24Hz → 8Hz 등).

★ 왜 필요한가

  우리 제어 주기는 24Hz 인데, VLA 학습 데이터로는 높은 편이다 (OXE 는 대체로
  3~15Hz, LIBERO 는 20Hz). 24Hz 에서 인접 프레임은 거의 동일해서 —
  스텝당 평균 이동이 2mm 남짓 — 정보량 대비 프레임 수가 과하다.

  결과적으로 두 가지가 동시에 걸린다:
    - 궤적이 평균 598스텝(24.9초)이라 SPEC.MAX_EPISODE_STEPS(300) 예산의 2배
    - 897,712 프레임인데 SFT 예산이 320,000 샘플 → 0.36 에폭, 다 못 본다

  k=3 (8Hz) 으로 솎으면 평균 199스텝 / 299,237 프레임 / 1.07 에폭이 되어
  둘 다 해소된다. 액션 청크 8 이 덮는 시간도 0.33초 → 1.0초로 늘어 OXE·LIBERO
  관행 범위에 들어온다.

★ 액션은 델타 포즈다 — 단순히 k번째만 남기면 안 된다

  ACTION_LAYOUT = (dx, dy, dz, drx, dry, drz, gripper) 는 **상대** 명령이다.
  k번째 프레임만 골라내면 나머지 k-1 개의 이동이 사라져 정책이 k배 느려진다.
  그래서 팔 6축은 구간 내 델타를 **합산**한다.

  회전(drx..drz)의 축각 합산은 엄밀하지 않지만, 스텝당 회전이 작아
  (평균 |drz| ~0.004 rad) 실용적으로 무시할 수 있는 오차다.

  그리퍼는 이진(-1/+1)이라 합산하면 안 되고 구간의 **마지막 값**을 쓴다.
  열림→닫힘 전환이 구간 안에 있으면 그 구간 끝의 상태가 남는다.

★ 결과 파일은 재생용이 아니다

  합산된 델타를 한 스텝에 적용하면 IK 추종 한계를 넘는다. 이 파일은 **학습
  데이터 변환(RLDS)용**이고, replay_demos.py / annotate_demos.py 에 넣으면
  안 된다. 그 용도로는 원본 HDF5 를 쓸 것.

사용:
    python scripts/downsample_demos.py --input datasets/generated.hdf5 \
        --output datasets/generated_8hz.hdf5 --factor 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 빌더가 읽는 관측만 옮긴다. states/initial_state 는 재생용인데 이 파일은
# 재생하지 않으므로 넣지 않는다 (용량도 아낀다).
_OBS_KEEP = ("table_cam", "proprio", "target_ids")


def _downsample_actions(actions: np.ndarray, k: int) -> np.ndarray:
    """팔 6축은 구간 합, 그리퍼는 구간 마지막 값."""
    n = len(actions) // k * k          # 끝의 나머지는 버린다 (최대 k-1 스텝)
    a = actions[:n].reshape(-1, k, actions.shape[1])
    out = np.empty((a.shape[0], actions.shape[1]), dtype=np.float32)
    out[:, :6] = a[:, :, :6].sum(axis=1)     # 델타 누적
    out[:, 6] = a[:, -1, 6]                  # 이진 그리퍼
    return out


def downsample(src: Path, dst: Path, k: int) -> int:
    if dst.exists():
        print(f"출력 파일이 이미 있다: {dst} (덮어쓰지 않는다)", file=sys.stderr)
        return 2

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for key, val in fin.attrs.items():
            fout.attrs[key] = val
        gout = fout.create_group("data")
        for key, val in fin["data"].attrs.items():
            gout.attrs[key] = val

        keys = sorted(fin["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        n_in = n_out = 0
        for i, k_ep in enumerate(keys):
            d = fin["data"][k_ep]
            actions = np.asarray(d["actions"], dtype=np.float32)
            obs = d["obs"]
            steps = min(len(actions), len(obs["table_cam"]))
            if steps < k:
                print(f"[SKIP] {k_ep}: {steps}스텝 < factor {k}")
                continue

            new_actions = _downsample_actions(actions[:steps], k)
            m = len(new_actions)
            idx = np.arange(m) * k        # 각 구간의 **첫** 프레임을 관측으로

            go = gout.create_group(f"demo_{i}")
            for a_key, a_val in d.attrs.items():
                go.attrs[a_key] = a_val
            go.attrs["num_samples"] = m
            go.create_dataset("actions", data=new_actions, compression="gzip")

            oo = go.create_group("obs")
            for name in _OBS_KEEP:
                if name not in obs:
                    continue
                arr = np.asarray(obs[name][: steps])[idx]
                oo.create_dataset(name, data=arr, compression="gzip")

            n_in += steps
            n_out += m
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(keys)} …")

        gout.attrs["num_episodes"] = len(gout.keys())

    print(f"\n에피소드 {len(keys)} → {len(keys)}")
    print(f"프레임   {n_in:,} → {n_out:,}  (1/{k})")
    print(f"출력     {dst}  ({dst.stat().st_size / 2**30:.2f} GB)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--factor", type=int, default=3,
                   help="솎을 배수. 24Hz 기준 3 이면 8Hz")
    args = p.parse_args()
    if args.factor < 2:
        p.error("--factor 는 2 이상이어야 한다.")
    return downsample(args.input, args.output, args.factor)


if __name__ == "__main__":
    sys.exit(main())

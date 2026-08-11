"""HDF5 데모 데이터셋에서 원하는 에피소드만 골라 새 파일로 뽑는다.

record_demos.py 는 성공한 데모를 **한 파일에** demo_0, demo_1, ... 로 쌓는다.
성공했다고 품질이 좋은 것은 아니다 — 궤적이 길거나, 중간에 멈췄거나, 손이
헤맨 데모는 Mimic 생성 성공률을 직접 깎는다. 그런 것을 골라내는 도구다.

Isaac Lab 에는 파일을 **합치는** 도구(merge_hdf5_datasets.py)는 있어도 고르는
도구가 없어서 여기에 둔다. Isaac Sim 이 필요 없다 — h5py 만 쓴다.

전형적인 흐름:

    # 1) 무엇이 들어 있나 (길이·정지구간 통계까지)
    python scripts/filter_demos.py --inspect datasets/source.hdf5

    # 2) 눈으로 확인 — 의심스러운 것만 골라 재생
    python $ISAACLAB_DIR/scripts/tools/replay_demos.py \
        --task VlaPlace-v0 --enable_cameras \
        --dataset_file ./datasets/source.hdf5 --select_episodes 3 7 11

    # 3) 좋은 것만 남긴다 (원본은 건드리지 않는다)
    python scripts/filter_demos.py --input datasets/source.hdf5 \
        --output datasets/source_clean.hdf5 --keep 0 1 2 4 5 6 8 9 10

    # 버릴 것을 지정하는 편이 짧으면 --drop 을 쓴다
    python scripts/filter_demos.py --input datasets/source.hdf5 \
        --output datasets/source_clean.hdf5 --drop 3 7 11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _episode_keys(f: h5py.File) -> list[str]:
    """demo_0, demo_1, ... 을 숫자 순으로. 문자열 정렬이면 demo_10 이 demo_2 앞에 온다."""
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def _still_fraction(actions: np.ndarray, eps: float = 1e-3) -> float:
    """액션이 사실상 0 인 스텝의 비율 = '멈춰 있던' 시간.

    키보드 텔레옵의 최대 함정이 일시정지이고, 그 구간은 정책이 배우기 어렵다.
    데모를 눈으로 다 보기 전에 의심 대상을 좁히는 용도다.
    """
    if len(actions) == 0:
        return 0.0
    # 그리퍼(마지막 차원)는 유지 중에도 값이 있으므로 팔 6축만 본다.
    arm = actions[:, :6]
    return float((np.abs(arm).max(axis=1) < eps).mean())


def inspect(path: Path) -> int:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            print(f"{path}: 'data' 그룹이 없다 — 데모 파일이 아니다.", file=sys.stderr)
            return 2
        keys = _episode_keys(f)
        print(f"{path}  —  에피소드 {len(keys)}개\n")
        print(f"  {'idx':>4}  {'키':<12} {'스텝':>6} {'초':>6} {'정지비율':>8}  지시문")
        print("  " + "-" * 74)

        lengths, stills = [], []
        for k in keys:
            d = f["data"][k]
            actions = np.asarray(d["actions"])
            n = len(actions)
            still = _still_fraction(actions)
            lengths.append(n)
            stills.append(still)

            instr = ""
            obs = d.get("obs", {})
            if "target_ids" in obs:
                from configs import vla_spec as SPEC  # noqa: N812

                tb = int(np.asarray(obs["target_ids"])[0][0])
                instr = SPEC.instruction_for(tb)

            idx = int(k.split("_")[-1])
            flag = "  ←?" if (still > 0.30 or n > 400) else ""
            print(f"  {idx:>4}  {k:<12} {n:>6} {n / 24.0:>6.1f} "
                  f"{still * 100:>7.0f}%  {instr}{flag}")

        if lengths:
            print()
            print(f"  스텝  중앙값 {int(np.median(lengths))} / 최소 {min(lengths)} / "
                  f"최대 {max(lengths)}  (24Hz 기준 "
                  f"{np.median(lengths) / 24.0:.1f}s / {min(lengths) / 24.0:.1f}s / "
                  f"{max(lengths) / 24.0:.1f}s)")
            print(f"  정지비율 중앙값 {np.median(stills) * 100:.0f}%")
            print()
            print("  ← ? 표시는 '정지비율 30% 초과 또는 400스텝 초과' — 재생해 볼 후보다.")
            print("  RFT 롤아웃은 300스텝에서 잘리므로, 그보다 긴 데모는 정책이")
            print("  따라 할 시간 예산을 넘는다는 뜻이기도 하다.")
    return 0


def filter_file(src: Path, dst: Path, keep: list[int]) -> int:
    if dst.exists():
        print(f"출력 파일이 이미 있다: {dst}  (덮어쓰지 않는다)", file=sys.stderr)
        return 2

    with h5py.File(src, "r") as fin:
        keys = _episode_keys(fin)
        available = {int(k.split("_")[-1]): k for k in keys}
        missing = [i for i in keep if i not in available]
        if missing:
            print(f"없는 에피소드 인덱스: {missing}  (있는 것: "
                  f"{sorted(available)})", file=sys.stderr)
            return 2

        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, "w") as fout:
            # ★ 파일·data 그룹의 attrs 를 반드시 복사할 것. env_args 가 여기 들어
            #   있고, replay_demos.py / annotate_demos.py 가 그걸 읽어 환경을
            #   만든다. 빠뜨리면 "태스크를 알 수 없다" 로 죽는다.
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            gout = fout.create_group("data")
            for k, v in fin["data"].attrs.items():
                gout.attrs[k] = v

            for new_i, old_i in enumerate(keep):
                fin.copy(fin["data"][available[old_i]], gout, name=f"demo_{new_i}")

            gout.attrs["num_episodes"] = len(keep)

    print(f"{len(keep)}개 에피소드 → {dst}")
    print(f"  원본 인덱스 {keep}")
    print(f"  새 인덱스   {list(range(len(keep)))}   (연속으로 다시 매긴다)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", type=Path, help="내용만 출력하고 끝낸다")
    p.add_argument("--input", type=Path, help="원본 HDF5")
    p.add_argument("--output", type=Path, help="추려서 저장할 HDF5")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--keep", type=int, nargs="+", help="남길 에피소드 인덱스")
    g.add_argument("--drop", type=int, nargs="+", help="버릴 에피소드 인덱스")
    args = p.parse_args()

    if args.inspect:
        return inspect(args.inspect)

    if not (args.input and args.output):
        p.error("--inspect 또는 (--input + --output) 이 필요하다.")
    if args.keep is None and args.drop is None:
        p.error("--keep 또는 --drop 중 하나를 지정할 것.")

    if args.drop is not None:
        with h5py.File(args.input, "r") as f:
            all_idx = [int(k.split("_")[-1]) for k in _episode_keys(f)]
        keep = [i for i in all_idx if i not in set(args.drop)]
    else:
        keep = list(args.keep)

    if not keep:
        print("남길 에피소드가 없다.", file=sys.stderr)
        return 2
    return filter_file(args.input, args.output, keep)


if __name__ == "__main__":
    sys.exit(main())

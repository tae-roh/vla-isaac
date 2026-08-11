"""Isaac Lab Mimic HDF5 → RLDS 변환 + 액션 정규화 통계 산출.

두 가지를 만든다:
  1. RLDS/TFDS 데이터셋  (SFT 학습 입력)
  2. norm_stats.json      (액션 디토크나이저 bin 범위 — SFT 와 RFT 가 공유)

★ 2번이 조용히 중요하다 (계획서 §Phase3-2)
  OpenVLA 의 이산 액션 토큰은 [q01, q99] 구간을 256 bin 으로 자른 것이다.
  SFT 때 쓴 구간과 RFT 때 쓴 구간이 다르면, 같은 토큰이 다른 물리량을 뜻하게
  되어 정책이 통째로 어긋난다. 증상은 "RFT 를 켜자 성공률이 0 이 된다" 이고,
  로그만 봐서는 원인을 찾기 어렵다. 그래서 통계를 파일로 못 박아 양쪽이
  같은 파일을 읽게 한다.

사용 예:
    # 데이터 점검만 (변환 없이)
    python scripts/convert_hdf5_to_rlds.py --inspect datasets/generated.hdf5

    # 전체 변환
    python scripts/convert_hdf5_to_rlds.py \
        --hdf5 datasets/generated.hdf5 datasets/deformable_source.hdf5 \
        --out datasets/rlds
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec_path = REPO_ROOT / "configs" / "vla_spec.py"
_spec_mod = importlib.util.spec_from_file_location("vla_spec", _spec_path)
SPEC = importlib.util.module_from_spec(_spec_mod)
sys.modules["vla_spec"] = SPEC
_spec_mod.loader.exec_module(SPEC)


# =============================================================================
def inspect_hdf5(paths: list[Path]) -> dict:
    """데이터 품질 점검 (계획서 §Phase2-7).

    생성 성공률·궤적 길이 분포를 여기서 보고, 이상하면 SFT 를 시작하기 전에
    Phase 2 로 되돌아간다. 학습을 하루 돌린 뒤에 데이터가 이상했다는 걸
    알게 되는 것이 이 일정에서 가장 비싼 실수다.
    """
    all_actions: list[np.ndarray] = []
    lengths: list[int] = []
    total_demos = 0
    instructions: list[str] = []

    for path in paths:
        with h5py.File(path, "r") as f:
            data = f["data"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
            print(f"\n--- {path.name} : 데모 {len(demo_keys)}개 ---")

            for key in demo_keys:
                demo = data[key]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                all_actions.append(actions)
                lengths.append(len(actions))
                total_demos += 1
                if "target_ids" in demo["obs"]:
                    tb = np.asarray(demo["obs"]["target_ids"])[0][0]
                    instructions.append(SPEC.instruction_for(int(tb)))

            if demo_keys:
                sample = data[demo_keys[0]]
                print(f"  obs 키: {sorted(sample['obs'].keys())}")
                print(f"  액션 shape: {np.asarray(sample['actions']).shape}")
                for k in sample["obs"]:
                    print(f"    obs/{k}: {np.asarray(sample['obs'][k]).shape}")

    if not all_actions:
        raise RuntimeError("데모를 하나도 읽지 못했다.")

    lengths_arr = np.array(lengths)
    stacked = np.concatenate(all_actions, axis=0)

    print(f"\n=== 전체 요약 ===")
    print(f"  데모 수      : {total_demos}")
    print(f"  총 스텝      : {len(stacked)}")
    print(f"  궤적 길이    : 평균 {lengths_arr.mean():.0f}, 중앙 "
          f"{np.median(lengths_arr):.0f}, 범위 {lengths_arr.min()}~{lengths_arr.max()}")

    if lengths_arr.max() > SPEC.MAX_EPISODE_STEPS:
        print(f"  [WARN] 최대 길이가 스펙 상한({SPEC.MAX_EPISODE_STEPS})을 넘는다. "
              "RFT 롤아웃 비용이 예상보다 커진다.")
    if lengths_arr.mean() > SPEC.MAX_EPISODE_STEPS * 0.6:
        print("  [WARN] 궤적이 전반적으로 길다. 계획서 §Phase2 품질 체크리스트의 "
              "'궤적이 짧은가 / 일시정지가 없는가' 를 다시 볼 것 — "
              "긴 궤적은 RFT 스텝 시간을 그대로 늘린다.")

    # 지시문 분포 — 언어 채널이 살아 있는지 여기서 본다.
    # 한 문장만 나오면 모델은 지시문을 무시하는 법을 배운다. 증강이 원본 데모의
    # 타깃만 따라가면 이런 데이터가 나온다 (SubTaskConfig.object_ref 확인).
    if not instructions:
        print("\n  [WARN] obs 에 target_ids 가 없다 — 개정 이전 데이터셋이다. "
              "RLDS 변환이 거부하므로 다시 생성할 것.")
    else:
        import collections

        counts = collections.Counter(instructions)
        print(f"\n  지시문 {len(counts)}종 / 데모 {len(instructions)}개")
        for text, c in counts.most_common(10):
            print(f"    {c:5d}× {text}")
        if len(counts) < 2:
            print("  [WARN] 지시문이 한 종류뿐이다. 이 데이터로 학습하면 정책이 "
                  "언어를 무시해도 손해가 없다 — 증강 설정을 다시 볼 것.")

    if stacked.shape[-1] != SPEC.ACTION_DIM:
        raise ValueError(
            f"액션 차원 {stacked.shape[-1]} != 스펙 {SPEC.ACTION_DIM}"
        )

    print(f"\n  액션 차원별 통계 ({SPEC.ACTION_DIM}차원):")
    print(f"    {'차원':<10} {'min':>8} {'q01':>8} {'평균':>8} {'q99':>8} {'max':>8}")
    for i, name in enumerate(SPEC.ACTION_LAYOUT):
        col = stacked[:, i]
        print(f"    {name:<10} {col.min():8.3f} {np.quantile(col, 0.01):8.3f} "
              f"{col.mean():8.3f} {np.quantile(col, 0.99):8.3f} {col.max():8.3f}")

    # 항상 0 인 차원은 텔레옵에서 그 축을 아예 안 움직였다는 뜻이다.
    # 키보드 조작에서 흔하고, 그대로 두면 정책이 그 축을 못 배운다.
    dead = [
        SPEC.ACTION_LAYOUT[i]
        for i in range(SPEC.ACTION_DIM)
        if np.abs(stacked[:, i]).max() < 1e-6
    ]
    if dead:
        print(f"\n  [WARN] 한 번도 움직이지 않은 액션 차원: {dead}")
        print("         텔레옵에서 해당 축을 쓰지 않았다는 뜻이다. 태스크상 "
              "정말 불필요한 축이면 괜찮지만, 아니라면 데모를 다시 수집할 것.")

    return {"num_demos": total_demos, "actions": stacked}


# =============================================================================
def compute_norm_stats(actions: np.ndarray, states: np.ndarray | None) -> dict:
    """BOUNDS_Q99 정규화 통계 (openvla-oft LIBERO 설정과 동일)."""

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    out = {
        "action": _stats(actions),
        "num_transitions": int(len(actions)),
        "normalization_type": SPEC.ACTION_PROPRIO_NORMALIZATION_TYPE,
        "action_layout": list(SPEC.ACTION_LAYOUT),
    }
    if states is not None:
        out["proprio"] = _stats(states)
        out["proprio_layout"] = list(SPEC.PROPRIO_LAYOUT)

    # 그리퍼 차원은 이진(-1/+1)이라 q01/q99 정규화를 걸면 오히려 망가진다.
    # openvla 도 마지막 차원은 정규화에서 제외한다 → mask 를 명시해 둔다.
    mask = [True] * SPEC.ACTION_DIM
    mask[-1] = False
    out["action"]["mask"] = mask
    return out


def collect_states(paths: list[Path]) -> np.ndarray | None:
    states = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for key in f["data"]:
                obs = f["data"][key]["obs"]
                if "proprio" not in obs:
                    return None
                states.append(np.asarray(obs["proprio"], dtype=np.float32))
    return np.concatenate(states, axis=0) if states else None


# =============================================================================
def build_rlds(hdf5_paths: list[Path], out_dir: Path) -> None:
    builder_dir = REPO_ROOT / "scripts" / "rlds" / "vla_pick"
    env = os.environ.copy()
    env["VLA_PICK_HDF5"] = ":".join(str(p.resolve()) for p in hdf5_paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tfds", "build",
        "--data_dir", str(out_dir.resolve()),
        "--overwrite",
    ]
    print(f"\n실행: {' '.join(cmd)}\n  (cwd={builder_dir})")
    print(f"  VLA_PICK_HDF5={env['VLA_PICK_HDF5']}\n")
    subprocess.run(cmd, cwd=builder_dir, env=env, check=True)


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, nargs="+", help="입력 HDF5 (여러 개 가능)")
    parser.add_argument("--inspect", type=Path, nargs="+",
                        help="점검만 하고 변환하지 않는다")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "rlds")
    parser.add_argument("--skip-build", action="store_true",
                        help="norm_stats 만 갱신하고 RLDS 빌드는 건너뛴다")
    args = parser.parse_args()

    paths = args.inspect or args.hdf5
    if not paths:
        parser.error("--hdf5 또는 --inspect 중 하나가 필요하다.")

    info = inspect_hdf5(paths)

    if args.inspect:
        print("\n점검만 수행했다 (--inspect). 변환하려면 --hdf5 로 다시 실행할 것.")
        return 0

    # --- 정규화 통계 ---
    states = collect_states(paths)
    stats = compute_norm_stats(info["actions"], states)

    args.out.mkdir(parents=True, exist_ok=True)
    stats_path = args.out / SPEC.NORM_STATS_FILENAME
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n정규화 통계 저장: {stats_path}")
    print("  ★ 이 파일은 SFT 와 RFT 가 같은 것을 읽어야 한다. "
          "Phase 3→4 전환 시 체크포인트와 함께 반드시 옮길 것.")

    # --- RLDS 빌드 ---
    if not args.skip_build:
        build_rlds(paths, args.out)
        print(f"\nRLDS 데이터셋 → {args.out}")

    print("\n다음 단계:")
    print(f"  python scripts/upload_hub.py --path {args.out} --repo <user>/vla-pick-rlds")
    return 0


if __name__ == "__main__":
    sys.exit(main())

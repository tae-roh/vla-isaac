# =============================================================================
# scripts/rlds/vla_pick/vla_pick_dataset_builder.py
#
# Isaac Lab Mimic 이 만든 HDF5 → RLDS(TFDS) 변환기.
#
# 형식은 rlds_dataset_builder 템플릿(OpenVLA 계열의 사실상 표준)을 따른다.
# 임의 형식을 발명하지 않은 이유: openvla-oft 의 데이터 로더가 이 구조를
# 전제로 하고 있어, 벗어나면 로더까지 고쳐야 한다.
#
# 실행 (converter 가 대신 호출해 준다):
#     cd scripts/rlds/vla_pick
#     tfds build --data_dir <출력경로> \
#         --imports vla_pick_dataset_builder \
#         --overwrite
#
# 입력 HDF5 경로는 환경변수 VLA_PICK_HDF5 로 넘긴다 (tfds build 가 임의
# CLI 인자를 빌더에 전달하지 못하기 때문).
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import tensorflow_datasets as tfds

# 스펙은 저장소 루트의 configs/vla_spec.py 하나만 본다.
import importlib.util
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_spec_path = _REPO_ROOT / "configs" / "vla_spec.py"
_spec_mod = importlib.util.spec_from_file_location("vla_spec", _spec_path)
SPEC = importlib.util.module_from_spec(_spec_mod)
sys.modules["vla_spec"] = SPEC
_spec_mod.loader.exec_module(SPEC)


class VlaPick(tfds.core.GeneratorBasedBuilder):
    """형상 랜덤화 자재 픽앤플레이스 — Isaac Lab Mimic 증강 데이터셋."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "최초 릴리스 (강체 증강 + 변형체 원본)."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    # 3인칭 단일 뷰. 손목캠 없음 (스펙 고정).
                                    "image": tfds.features.Image(
                                        shape=(SPEC.IMAGE_HEIGHT, SPEC.IMAGE_WIDTH, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="3인칭 카메라 RGB 224x224",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(SPEC.PROPRIO_DIM,),
                                        dtype=np.float32,
                                        doc=(
                                            "eef 위치(3) + axis-angle(3) + 그리퍼(2). "
                                            "LIBERO PROPRIO_DIM=8 과 동일 레이아웃."
                                        ),
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(SPEC.ACTION_DIM,),
                                dtype=np.float32,
                                doc="[dx,dy,dz,drx,dry,drz,gripper] — LIBERO 와 동일",
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "demo_key": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        hdf5_paths = os.environ.get("VLA_PICK_HDF5")
        if not hdf5_paths:
            raise RuntimeError(
                "환경변수 VLA_PICK_HDF5 가 없다. 변환할 HDF5 경로를 콜론(:)으로 "
                "이어서 지정할 것.\n"
                "  예: VLA_PICK_HDF5=datasets/generated.hdf5:datasets/deformable.hdf5"
            )
        paths = [Path(p) for p in hdf5_paths.split(":") if p]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"HDF5 를 찾을 수 없다: {p}")
        return {"train": self._generate_examples(paths)}

    # -------------------------------------------------------------------------
    def _generate_examples(self, paths: list[Path]) -> Iterator[tuple[str, Any]]:
        for path in paths:
            with h5py.File(path, "r") as f:
                data = f["data"]
                # demo_0, demo_1, ... 순서가 문자열 정렬로 뒤섞이지 않게 숫자 정렬.
                demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
                for demo_key in demo_keys:
                    demo = data[demo_key]
                    episode = self._build_episode(demo, path, demo_key)
                    if episode is None:
                        continue
                    yield f"{path.stem}/{demo_key}", episode

    def _build_episode(self, demo, path: Path, demo_key: str):
        actions = np.asarray(demo["actions"], dtype=np.float32)
        obs = demo["obs"]

        images = np.asarray(obs[SPEC.CAMERA_SENSOR_NAME])
        if images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)

        num_steps = min(len(actions), len(images))
        if num_steps == 0:
            print(f"[SKIP] {path.name}/{demo_key}: 스텝이 없다")
            return None

        if actions.shape[-1] != SPEC.ACTION_DIM:
            raise ValueError(
                f"{path.name}/{demo_key}: 액션 차원 {actions.shape[-1]} != "
                f"스펙 {SPEC.ACTION_DIM}. 환경 cfg 와 스펙이 어긋났다."
            )

        state = self._build_state(obs, num_steps)

        steps = []
        for t in range(num_steps):
            steps.append(
                {
                    "observation": {"image": images[t], "state": state[t]},
                    "action": actions[t],
                    "discount": np.float32(1.0),
                    # Mimic 이 생성한 데모는 전부 성공 궤적이다 (실패는
                    # generation_keep_failed=False 로 버려진다) → 마지막 스텝에 1.0.
                    "reward": np.float32(1.0 if t == num_steps - 1 else 0.0),
                    "is_first": t == 0,
                    "is_last": t == num_steps - 1,
                    "is_terminal": t == num_steps - 1,
                    "language_instruction": SPEC.TASK_INSTRUCTION,
                }
            )

        return {
            "steps": steps,
            "episode_metadata": {"file_path": str(path), "demo_key": demo_key},
        }

    def _build_state(self, obs, num_steps: int) -> np.ndarray:
        """proprio 8차원을 만든다.

        환경이 proprio 항목을 직접 기록해 두었으면 그대로 쓰고, 없으면
        eef_pos / eef_quat / gripper_pos 로부터 조립한다. 후자는 구버전
        데이터셋과의 호환용 경로다.
        """
        if "proprio" in obs:
            state = np.asarray(obs["proprio"], dtype=np.float32)[:num_steps]
            if state.shape[-1] != SPEC.PROPRIO_DIM:
                raise ValueError(
                    f"proprio 차원 {state.shape[-1]} != 스펙 {SPEC.PROPRIO_DIM}"
                )
            return state

        eef_pos = np.asarray(obs["eef_pos"], dtype=np.float32)[:num_steps]
        eef_quat = np.asarray(obs["eef_quat"], dtype=np.float32)[:num_steps]
        grip = np.asarray(obs["gripper_pos"], dtype=np.float32)[:num_steps]
        axis_angle = _quat_to_axis_angle(eef_quat)
        return np.concatenate([eef_pos, axis_angle, grip], axis=-1).astype(np.float32)


def _quat_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz 쿼터니언 → (N, 3) axis-angle 지수좌표.

    Isaac Lab 은 wxyz 순서를 쓴다. LIBERO 의 quat2axisangle 은 xyzw 를 받으므로
    순서를 헷갈리면 회전이 통째로 틀어진다 — 여기서 명시적으로 wxyz 를 가정한다.
    """
    w = np.clip(quat_wxyz[:, 0], -1.0, 1.0)
    xyz = quat_wxyz[:, 1:]
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 1e-12))
    axis = xyz / sin_half[:, None]
    # 각이 0 에 가까우면 축이 정의되지 않는다 → 0 벡터로 눌러 준다.
    small = angle < 1e-6
    out = axis * angle[:, None]
    out[small] = 0.0
    return out.astype(np.float32)

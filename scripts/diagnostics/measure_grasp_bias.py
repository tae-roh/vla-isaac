"""파지 국면 오차가 '계통 편향'인지 '무작위 산포'인지 가른다.

  |평균 오차| >> 표준편차  → 항상 같은 쪽으로 빗나간다 = 오프셋/좌표계 문제.
                            상수 하나로 고쳐질 수 있고, 학습을 더 해도 안 없어진다.
  |평균 오차| << 표준편차  → 매번 다른 쪽으로 빗나간다 = 정밀도/다중성 문제.
                            더 많은 학습이나 더 나은 관측이 필요하다.
"""

import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

tf.config.set_visible_devices([], "GPU")
sys.path.insert(0, "/home/shadeform/vla-isaac/env/smoke")
from _bootstrap import load_vla_spec  # noqa: E402

SPEC = load_vla_spec()
CKPT, NEP = sys.argv[1], int(sys.argv[2])

proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to("cuda:0").eval()

b = tfds.builder_from_directory("/home/shadeform/vla-isaac/datasets/rlds/vla_pick/2.0.0")
K = SPEC.NUM_ACTIONS_CHUNK
err_grasp, err_other = [], []

for ep in b.as_dataset(split=f"train[:{NEP}]", shuffle_files=False):
    steps = list(ep["steps"])
    T = len(steps)
    acts = np.stack([s["action"].numpy() for s in steps])
    lang = steps[0]["language_instruction"].numpy().decode()
    closed = acts[:, 6] < 0
    gi = next((t for t in range(T - 10) if closed[t:t + 10].all()), int(T * 0.3))

    for tag, t in [("grasp", max(0, gi - 4)), ("other", min(T - K - 1, gi + 30))]:
        t = int(np.clip(t, 0, T - K - 1))
        img = steps[t]["observation"]["image"].numpy()
        pil = SPEC.prepare_image_for_vla(img)
        inputs = proc(SPEC.build_prompt(lang), pil).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            pred, _ = model.predict_action(**inputs, unnorm_key="vla_pick")
        P = np.asarray(pred, np.float32).reshape(-1, 7)[:K]
        G = acts[t:t + K]
        e = P[:, :3].sum(0) - G[:, :3].sum(0)      # 청크 누적 변위의 오차 벡터
        (err_grasp if tag == "grasp" else err_other).append(e)

for name, arr in [("파지 직전", np.array(err_grasp)), ("운반 중", np.array(err_other))]:
    mu, sd = arr.mean(0), arr.std(0)
    print(f"\n=== {name} (n={len(arr)}) ===")
    for j, ax in enumerate("xyz"):
        print(f"  d{ax}: 평균 오차 {mu[j]:+.4f} | 표준편차 {sd[j]:.4f} | "
              f"|평균|/표준편차 = {abs(mu[j])/(sd[j]+1e-9):.2f}")
    r = np.linalg.norm(mu) / (np.linalg.norm(sd) + 1e-9)
    print(f"  전체 |평균벡터|/|표준편차벡터| = {r:.2f}  "
          f"→ {'계통 편향 우세' if r > 1 else '무작위 산포 우세'}")

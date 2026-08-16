"""국면별로 정책의 예측을 정답과 비교한다.

"파지에서 실패" 라는 증상은 두 가지로 갈린다:
  (a) 방향은 맞는데 정밀도가 부족 → 청크 누적 변위가 정답과 같은 쪽을 향한다
  (b) 구조적으로 틀림      → 방향 자체가 어긋난다 (코사인 유사도가 0 근처/음수)
둘의 처방이 완전히 다르므로 여기서 가른다.
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
CKPT = sys.argv[1]
NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 12

proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to("cuda:0").eval()

b = tfds.builder_from_directory("/home/shadeform/vla-isaac/datasets/rlds/vla_pick/2.0.0")
ds = b.as_dataset(split=f"train[:{NEP}]", shuffle_files=False)

K = SPEC.NUM_ACTIONS_CHUNK
rows = []
for ep in ds:
    steps = list(ep["steps"])
    T = len(steps)
    imgs = np.stack([s["observation"]["image"].numpy() for s in steps])
    acts = np.stack([s["action"].numpy() for s in steps])
    lang = steps[0]["language_instruction"].numpy().decode()
    grip = acts[:, 6]
    # 첫 '닫기'(-1) 가 유지되기 시작하는 지점 ≈ 파지 순간
    closed = grip < 0
    gi = None
    for t in range(T - 10):
        if closed[t:t + 10].all():
            gi = t
            break
    if gi is None:
        gi = int(T * 0.3)

    for tag, t in [("접근 초기", int(T * 0.05)), ("파지 직전", max(0, gi - 8)),
                   ("파지 순간", gi), ("운반", min(T - K - 1, gi + 25)),
                   ("배치 직전", int(T * 0.85))]:
        t = int(np.clip(t, 0, T - K - 1))
        pil = SPEC.prepare_image_for_vla(imgs[t])
        inputs = proc(SPEC.build_prompt(lang), pil).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            pred, _ = model.predict_action(**inputs, unnorm_key="vla_pick")
        P = np.asarray(pred, dtype=np.float32).reshape(-1, 7)[:K]
        G = acts[t:t + K]
        dp, dg = P[:, :3].sum(0), G[:, :3].sum(0)     # 청크 누적 병진
        cos = float(dp @ dg / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-9))
        rows.append({
            "tag": tag,
            "cos": cos,
            "mag_p": float(np.linalg.norm(dp)),
            "mag_g": float(np.linalg.norm(dg)),
            "grip_match": float((np.sign(P[:, 6]) == np.sign(G[:, 6])).mean()),
            "l1": float(np.abs(P[:, :3] - G[:, :3]).mean()),
        })

print(f"에피소드 {NEP}개 × 5국면\n")
print(f"{'국면':<10} {'방향 코사인':>10} {'예측크기':>9} {'정답크기':>9} {'그리퍼일치':>10} {'병진L1':>9}")
print("-" * 62)
for tag in ["접근 초기", "파지 직전", "파지 순간", "운반", "배치 직전"]:
    r = [x for x in rows if x["tag"] == tag]
    print(f"{tag:<10} {np.mean([x['cos'] for x in r]):10.3f} "
          f"{np.mean([x['mag_p'] for x in r]):9.4f} {np.mean([x['mag_g'] for x in r]):9.4f} "
          f"{np.mean([x['grip_match'] for x in r]):10.3f} {np.mean([x['l1'] for x in r]):9.4f}")

allc = np.array([x["cos"] for x in rows])
print(f"\n전체 방향 코사인: 평균 {allc.mean():.3f} | 음수(반대 방향) 비율 {(allc<0).mean()*100:.1f}%")
print(f"예측 크기 / 정답 크기 비: {np.mean([x['mag_p'] for x in rows])/np.mean([x['mag_g'] for x in rows]):.3f}")

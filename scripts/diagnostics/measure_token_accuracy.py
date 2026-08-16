"""액션 토큰 정확도 0.3 이 실제로 무엇을 뜻하는지 실측한다.

토큰 정확도는 256개 bin 중 **정확히 같은 bin** 을 맞췄을 때만 1점이다.
한 칸 옆을 찍어도 0점이라, 물리적으로 무의미한 차이가 점수를 크게 깎는다.
그래서 bin 거리 분포와 L1 을 함께 보고, 자명한 베이스라인과 비교한다.

주의: RLDS 에 train split 밖에 없고 모델이 이 데이터를 약 1 epoch 봤다.
따라서 이 수치는 "학습 분포 적합도" 이지 일반화 성능이 아니다.
"""

import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, "/home/shadeform/vla-isaac")
sys.path.insert(0, "/home/shadeform/vla-isaac/env/smoke")
from _bootstrap import load_vla_spec  # noqa: E402

from prismatic.vla.datasets import RLDSDataset  # noqa: E402

SPEC = load_vla_spec()
CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64

# --- 모델 ---------------------------------------------------------------
proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to("cuda:0").eval()

stats = model.norm_stats["vla_pick"]["action"]
q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])

# 모델이 쓰는 것과 같은 이산화 격자
bins = np.linspace(-1, 1, 256)
def to_bin(x):
    return np.clip(np.digitize(x, bins) - 1, 0, 254)

def normalize(a):
    return np.clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)

# --- 데이터 (증강 끔 = 평가 조건) ----------------------------------------
def peek(b):
    return {
        "img": b["observation"]["image_primary"][0],       # (224,224,3) uint8
        "lang": b["task"]["language_instruction"].decode(),
        "act": b["action"],                                 # (8,7) 정규화됨
    }

ds = RLDSDataset("/home/shadeform/vla-isaac/datasets/rlds", "vla_pick", peek,
                 resize_resolution=(224, 224), shuffle_buffer_size=8192,
                 image_aug=False, train=True)

gts, preds = [], []
it = iter(ds)
for i in range(N):
    s = next(it)
    pil = SPEC.prepare_image_for_vla(np.asarray(s["img"]))   # center crop 포함
    inputs = proc(SPEC.build_prompt(s["lang"]), pil).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        actions, _ = model.predict_action(**inputs, unnorm_key="vla_pick")
    p = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
    g = np.asarray(s["act"], dtype=np.float32).reshape(-1, 7)
    n = min(len(p), len(g))
    preds.append(normalize(p[:n]))
    gts.append(g[:n])
    if (i + 1) % 16 == 0:
        print(f"  {i+1}/{N}")

P = np.concatenate(preds)          # 정규화 공간
G = np.concatenate(gts)
print(f"\n비교 대상: {P.shape[0]} 스텝 × 7차원 = {P.size} 토큰\n")


def report(name, pred):
    bp, bg = to_bin(pred), to_bin(G)
    d = np.abs(bp - bg)
    l1 = np.abs(pred - G).mean()
    print(f"{name:<22} 정확도 {(d==0).mean():6.3f} | ±1bin {(d<=1).mean():6.3f} | "
          f"±5bin {(d<=5).mean():6.3f} | 평균 bin거리 {d.mean():6.2f} | L1 {l1:6.4f}")
    return d


print(f"{'':<22} {'exact':>9}   {'±1':>8}   {'±5':>8}   {'bin거리':>10}   {'L1':>8}")
d_model = report("SFT 20000 스텝", P)
report("베이스라인: 항상 0", np.zeros_like(G))
report("베이스라인: 차원별 평균", np.tile(G.mean(0), (len(G), 1)))
rng = np.random.default_rng(0)
report("베이스라인: 무작위", rng.uniform(-1, 1, G.shape))

print("\n=== 차원별 (모델) ===")
names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
bp, bg = to_bin(P), to_bin(G)
for j, nm in enumerate(names):
    dj = np.abs(bp[:, j] - bg[:, j])
    print(f"  {nm:>4}: 정확도 {(dj==0).mean():6.3f} | ±1bin {(dj<=1).mean():6.3f} | "
          f"평균 bin거리 {dj.mean():6.2f} | L1 {np.abs(P[:,j]-G[:,j]).mean():6.4f}")

print("\n=== bin 거리 분포 (모델) ===")
for lo, hi in [(0, 0), (1, 1), (2, 5), (6, 20), (21, 60), (61, 255)]:
    m = ((d_model >= lo) & (d_model <= hi)).mean()
    print(f"  {lo:>3}~{hi:<3} bin: {m*100:5.1f}%  {'█'*int(m*50)}")

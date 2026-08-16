"""오프라인 wandb 트랜잭션 로그에서 학습 지표 추세를 뽑는다.

wandb offline 은 files/wandb-summary.json 을 run 종료 시에만 쓰므로,
학습 중에는 .wandb 데이터스토어를 직접 스캔해야 한다.
"""

import glob
import os
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

run_dir = sorted(glob.glob(os.path.expanduser("~/openvla-oft/wandb/offline-run-*")))[-1]
files = glob.glob(os.path.join(run_dir, "*.wandb"))
if not files:
    print("  지표: .wandb 없음")
    sys.exit(0)

ds = datastore.DataStore()
ds.open_for_scan(files[0])
hist = []
while True:
    try:
        data = ds.scan_data()
    except Exception:
        break
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    if rec.WhichOneof("record_type") == "history":
        hist.append({it.key: it.value_json for it in rec.history.item})

if not hist:
    print("  지표: 아직 없음")
    sys.exit(0)


def g(h, k, default=float("nan")):
    try:
        return float(h[k])
    except Exception:
        return default


def line(tag, h):
    return (f"    {tag:>7}: loss {g(h,'VLA Train/Loss'):5.3f} | "
            f"액션토큰정확도 {g(h,'VLA Train/Curr Action Accuracy'):5.3f} | "
            f"L1 {g(h,'VLA Train/Curr Action L1 Loss'):5.3f} | "
            f"lr {g(h,'VLA Train/Learning Rate'):.2e}")


# 최근 20개 평균으로 노이즈를 걷어낸다 (스텝별 정확도는 배치마다 크게 흔들린다)
def avg(hs, k):
    vs = [g(h, k) for h in hs if k in h]
    return sum(vs) / len(vs) if vs else float("nan")


first, last = hist[0], hist[-1]
recent = hist[-20:]
early = hist[:20]
print(f"  지표 (history {len(hist)}개, _step {last.get('_step')})")
print(line("최초", first))
print(line("최신", last))
print(f"    추세  : 정확도 초기{len(early)}평균 {avg(early,'VLA Train/Curr Action Accuracy'):5.3f}"
      f" → 최근{len(recent)}평균 {avg(recent,'VLA Train/Curr Action Accuracy'):5.3f}"
      f" | loss {avg(early,'VLA Train/Loss'):5.3f} → {avg(recent,'VLA Train/Loss'):5.3f}")

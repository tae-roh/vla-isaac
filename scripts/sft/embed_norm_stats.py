"""체크포인트의 dataset_statistics.json 을 config.json 의 norm_stats 에 심는다.

왜 필요한가
-----------
openvla-oft 의 체크포인트 저장은 두 가지를 따로 한다:
  1. save_dataset_statistics()  → dataset_statistics.json (별도 파일)
  2. merged_vla.save_pretrained() → config.json

그런데 2번의 config 는 **베이스 모델(openvla/openvla-7b)에서 온 것**이라,
norm_stats 에 원본 OXE 25종만 들어 있고 우리 데이터셋(vla_pick)이 없다.

한편 modeling_prismatic.py 는 `self.norm_stats = config.norm_stats` 로 읽고,
predict_action(unnorm_key=...) 도 그 딕셔너리에서 찾는다. 우리 eval_rollout.py 와
rft/grpo_fallback.py 역시 `model.norm_stats[unnorm_key]` 를 전제한다
("같은 숫자의 출처가 둘이면 언젠가 갈라진다"는 이유로 파일을 따로 읽지 않는다).

→ 그대로 두면 평가와 RFT 가 시작하자마자 죽는다:
     unnorm_key 'vla_pick' 가 체크포인트의 norm_stats 에 없다

이 스크립트가 1번의 내용을 2번에 합쳐 그 전제를 실제로 성립시킨다. 멱등이다.

사용:
    python scripts/sft/embed_norm_stats.py runs/<체크포인트 디렉토리> [...]
    python scripts/sft/embed_norm_stats.py runs/*_chkpt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def embed(ckpt: Path) -> bool:
    cfg_path = ckpt / "config.json"
    stats_path = ckpt / "dataset_statistics.json"

    if not cfg_path.exists():
        print(f"  [건너뜀] config.json 없음: {ckpt}")
        return False
    if not stats_path.exists():
        print(f"  [건너뜀] dataset_statistics.json 없음: {ckpt}")
        return False

    cfg = json.loads(cfg_path.read_text())
    stats = json.loads(stats_path.read_text())
    norm = cfg.setdefault("norm_stats", {})

    added = []
    for key, value in stats.items():
        if norm.get(key) == value:
            continue
        norm[key] = value
        added.append(key)

    if not added:
        print(f"  [이미 반영됨] {ckpt.name}")
        return False

    # 원본을 남겨 둔다 — 되돌릴 수 있어야 한다.
    backup = ckpt / "config.json.pre_norm_stats"
    if not backup.exists():
        backup.write_text(json.dumps(json.loads(cfg_path.read_text()), indent=2))

    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  [반영] {ckpt.name} ← {', '.join(added)}")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    changed = 0
    for arg in argv:
        ckpt = Path(arg)
        if not ckpt.is_dir():
            print(f"  [건너뜀] 디렉토리가 아님: {ckpt}")
            continue
        changed += embed(ckpt)

    print(f"\n{changed}개 체크포인트 갱신.")
    print("확인:")
    print('  python -c "import json;'
          " print(list(json.load(open('<ckpt>/config.json'))['norm_stats']))\"")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

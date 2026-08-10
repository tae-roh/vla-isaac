# =============================================================================
# vla_isaac_tasks/spec.py
#
# configs/vla_spec.py 를 패키지 안에서 쓰기 위한 얇은 로더.
#
# 왜 그냥 import 하지 않는가: configs/ 는 저장소 루트에 있고 source/ 패키지 바깥이다.
# 일부러 그렇게 두었다 — 같은 스펙 파일을 isaaclab / vla-train / rft 세 venv 가
# 모두 읽어야 하는데, 그중 둘은 이 패키지를 설치하지 않기 때문이다.
# (계획서 §6: "SFT-RFT 스펙 불일치 → 원인불명 성능 붕괴" 를 막는 장치)
# =============================================================================

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_PATH = _REPO_ROOT / "configs" / "vla_spec.py"


def _load():
    if "vla_spec" in sys.modules:
        return sys.modules["vla_spec"]
    if not _SPEC_PATH.exists():
        raise FileNotFoundError(
            f"스펙 파일이 없다: {_SPEC_PATH}\n"
            "  이 패키지는 저장소 안에서 `pip install -e source` 로 설치되어야 하고,\n"
            "  configs/vla_spec.py 가 저장소 루트에 있어야 한다."
        )
    spec = importlib.util.spec_from_file_location("vla_spec", _SPEC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vla_spec"] = module
    spec.loader.exec_module(module)
    return module


SPEC = _load()
SPEC.assert_consistent()

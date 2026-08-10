# =============================================================================
# env/smoke/_bootstrap.py
#
# 스모크 테스트 공용 유틸. 세 venv 어디에서든 의존성 없이 동작해야 하므로
# 순수 stdlib 만 쓴다.
# =============================================================================

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# 저장소 루트 (env/smoke/_bootstrap.py 로부터 두 단계 위)
REPO_ROOT = Path(__file__).resolve().parents[2]

_GREEN = "\033[1;32m"
_RED = "\033[1;31m"
_YELLOW = "\033[1;33m"
_RESET = "\033[0m"

_failures: list[str] = []


def load_vla_spec():
    """configs/vla_spec.py 를 패키지 설치 없이 직접 로드한다.

    세 venv 가 서로 다른 패키지를 갖고 있어도 스펙 파일만은 공유해야 하므로,
    import 경로에 의존하지 않고 파일 경로로 직접 로드한다.
    """
    spec_path = REPO_ROOT / "configs" / "vla_spec.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"스펙 파일을 찾을 수 없다: {spec_path}")
    spec = importlib.util.spec_from_file_location("vla_spec", spec_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vla_spec"] = module
    spec.loader.exec_module(module)
    return module


def check(label: str, fn):
    """개별 항목을 실행하고 성공/실패를 기록한다. 실패해도 즉시 중단하지 않는다.

    한 번의 실행으로 깨진 항목을 모두 보고하는 편이, 하나 고치고 다시 돌리기를
    반복하는 것보다 훨씬 빠르다 (특히 Isaac Sim 은 기동에만 수 분이 걸린다).
    """
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - 스모크 테스트는 모든 예외를 보고한다
        _failures.append(label)
        print(f"{_RED}[FAIL]{_RESET} {label}\n        {type(exc).__name__}: {exc}")
        return False
    suffix = f" — {detail}" if detail else ""
    print(f"{_GREEN}[ OK ]{_RESET} {label}{suffix}")
    return True


def warn(message: str) -> None:
    print(f"{_YELLOW}[WARN]{_RESET} {message}")


def finish(name: str) -> int:
    """결과를 요약하고 종료 코드를 반환한다."""
    print()
    if _failures:
        print(f"{_RED}=== {name} 스모크 테스트 실패 ({len(_failures)}건) ==={_RESET}")
        for item in _failures:
            print(f"  - {item}")
        print("\n환경이 준비되지 않았다. 위 항목을 해결한 뒤 다시 실행할 것.")
        return 1
    print(f"{_GREEN}=== {name} 스모크 테스트 전체 통과 ==={_RESET}")
    print("이제 pip freeze 로 lock 을 박제할 수 있다:")
    print(f"  pip freeze > env/locks/{name}.lock.txt")
    return 0

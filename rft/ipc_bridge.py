# =============================================================================
# rft/ipc_bridge.py
#
# 학습 프로세스(rft venv) ↔ Isaac Lab 롤아웃 워커(isaaclab venv) 사이의 통신.
#
# ★ 왜 프로세스를 나누는가
#   Isaac Sim 5.1.0 은 torch 2.7.0 에 고정되어 있고, veRL/vLLM 은 자체 요구사항이
#   있다. 한 venv 에 넣으면 둘 중 하나가 깨진다. 계획서 §Phase4b-1 은 이를
#   "이 페이즈 최대의 의존성 난점" 으로 꼽으며 프로세스 분리를 대안으로 두었는데,
#   SimpleVLA-RL 의 rob_rollout.py 를 실제로 읽어 보면 LIBERO 를
#   multiprocessing.Process + Queue 로 이미 격리해서 돌리고 있다.
#   즉 분리는 우회로가 아니라 상류가 이미 쓰는 구조다 → 1순위로 채택했다.
#
# ★ 설계 원칙: 순수 stdlib 만 쓴다
#   이 파일은 서로 다른 두 venv 에서 동시에 import 된다. numpy 조차 버전이
#   다를 수 있으므로, 배열은 numpy 의 pickle 에 맡기지 않고 (dtype, shape, bytes)
#   로 직접 분해해서 보낸다. 이렇게 하면 양쪽 numpy 버전이 달라도 안전하다.
#
# 프로토콜: [8바이트 길이 (big-endian)] + [pickle 페이로드]
#   요청/응답 모두 dict. stdin/stdout 을 파이프로 쓴다.
#   워커의 stderr 는 그대로 흘려보낸다 — Isaac Sim 로그를 봐야 디버깅이 된다.
# =============================================================================

from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1

_LEN = struct.Struct(">Q")
_ARRAY_TAG = "__ndarray__"


# -----------------------------------------------------------------------------
# 배열 직렬화 — numpy 버전 독립
# -----------------------------------------------------------------------------
def encode_array(arr) -> dict:
    """numpy 배열을 stdlib 타입만으로 이루어진 dict 로 바꾼다."""
    contiguous = arr if arr.flags["C_CONTIGUOUS"] else arr.copy(order="C")
    return {
        _ARRAY_TAG: True,
        "dtype": contiguous.dtype.str,      # 예: '<f4', '|u1'
        "shape": tuple(int(s) for s in contiguous.shape),
        "data": contiguous.tobytes(),
    }


def decode_array(obj: dict):
    import numpy as np

    return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def _walk(obj: Any, fn) -> Any:
    """dict/list 안의 배열을 재귀적으로 변환한다."""
    if isinstance(obj, dict):
        if obj.get(_ARRAY_TAG):
            return fn(obj)
        return {k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_walk(v, fn) for v in obj)
    return obj


def encode_payload(obj: Any) -> Any:
    """페이로드 안의 모든 numpy 배열을 인코딩한다."""
    try:
        import numpy as np
    except ImportError:
        return obj

    def _enc(o):
        if isinstance(o, np.ndarray):
            return encode_array(o)
        if isinstance(o, dict):
            return {k: _enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_enc(v) for v in o)
        if isinstance(o, np.generic):
            return o.item()
        return o

    return _enc(obj)


def decode_payload(obj: Any) -> Any:
    return _walk(obj, decode_array)


# -----------------------------------------------------------------------------
# 프레이밍
# -----------------------------------------------------------------------------
def send_message(stream, msg: dict) -> None:
    payload = pickle.dumps(encode_payload(msg), protocol=4)
    stream.write(_LEN.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def recv_message(stream, timeout: float | None = None) -> dict | None:
    """길이 프리픽스를 읽고 그만큼 본문을 읽는다. EOF 면 None.

    timeout 은 "첫 바이트가 오기까지" 만 적용한다. 본문 수신 중에는 끝까지 기다린다
    (중간에 끊으면 스트림이 어긋나서 이후 모든 메시지가 깨진다).
    """
    header = _read_exactly(stream, _LEN.size, timeout=timeout)
    if header is None:
        return None
    (length,) = _LEN.unpack(header)
    body = _read_exactly(stream, length, timeout=None)
    if body is None:
        raise EOFError("본문 수신 중 스트림이 끊겼다 — 워커가 죽었을 가능성이 높다.")
    return decode_payload(pickle.loads(body))


def _read_exactly(stream, n: int, timeout: float | None) -> bytes | None:
    chunks: list[bytes] = []
    remaining = n
    deadline = None if timeout is None else time.time() + timeout

    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks and deadline is not None and time.time() > deadline:
                raise TimeoutError(f"{timeout}s 안에 응답이 오지 않았다.")
            if not chunk:
                return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# -----------------------------------------------------------------------------
# 클라이언트 (학습 프로세스 쪽에서 사용)
# -----------------------------------------------------------------------------
class RolloutClient:
    """Isaac Lab 롤아웃 워커를 자식 프로세스로 띄우고 대화한다.

    사용:
        client = RolloutClient(isaaclab_python, worker_script, num_envs=16)
        client.start()
        obs = client.reset()
        obs, reward, done = client.step(action_chunk)   # (N, K, 7)
        client.close()
    """

    def __init__(
        self,
        isaaclab_python: Path | str,
        worker_script: Path | str,
        num_envs: int = 8,
        task: str = "VlaPlace-v0",
        device: str = "cuda:0",
        seed: int = 0,
        startup_timeout: float = 900.0,
        extra_args: list[str] | None = None,
    ) -> None:
        # ★ `~` 를 반드시 편다. YAML 설정들이 `~/env_isaaclab/bin/python` 로 적어
        #   두었는데, subprocess.Popen 은 셸이 아니라 `~` 를 리터럴 디렉터리명으로
        #   본다 → FileNotFoundError 로 워커가 아예 안 뜬다. 학습 스크립트는 정책을
        #   다 로드한 **뒤** 이 지점에서 죽으므로, 모델 로드가 성공하는 것만 보고
        #   "환경은 됐다" 고 오해하기 쉽다.
        self.isaaclab_python = Path(isaaclab_python).expanduser()
        self.worker_script = Path(worker_script).expanduser()
        self.num_envs = num_envs
        self.task = task
        self.device = device
        self.seed = seed
        self.startup_timeout = startup_timeout
        self.extra_args = extra_args or []
        self._proc: subprocess.Popen | None = None
        self.last_diag: dict = {}

    # -- 수명 관리 ----------------------------------------------------------
    def start(self) -> None:
        if self._proc is not None:
            return

        cmd = [
            str(self.isaaclab_python),
            str(self.worker_script),
            "--task", self.task,
            "--num-envs", str(self.num_envs),
            "--device", self.device,
            "--seed", str(self.seed),
            "--headless",
            # ★ 라이브스트림이 아니라 이 플래그다. 오프스크린 렌더에 필요 (계획서 §Phase4b).
            "--enable_cameras",
            *self.extra_args,
        ]
        # ★ EULA 를 미리 수락해 둔다. 수락돼 있지 않으면 Kit 이 동의 프롬프트를
        #   띄우고 **stdin 에서 답을 기다린다** — 그런데 워커의 stdin 은 프로토콜
        #   파이프다. Kit 이 프로토콜 바이트를 EULA 답변으로 먹어 버리거나
        #   그대로 멎어, 증상은 "워커가 응답 없이 정지" 로만 나타난다.
        #   setup/setup_isaaclab.sh 가 이미 ~/.bashrc 에 같은 값을 넣지만,
        #   학습을 띄운 셸이 그걸 안 읽었을 수 있으므로 여기서 못 박는다.
        #   (이미 설정돼 있으면 그 값을 존중한다)
        env = dict(os.environ)
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr 는 부모로 그대로 흘린다 — Isaac Sim 로그가 유일한 디버깅 단서다.
            stderr=None,
            bufsize=0,
            env=env,
        )

        # 첫 실행은 extension 다운로드 + 셰이더 캐시로 10분 이상 걸릴 수 있다.
        print(
            f"[bridge] 워커 기동 중 (최대 {self.startup_timeout:.0f}s 대기) — "
            "첫 실행은 셰이더 캐시 때문에 오래 걸린다...",
            file=sys.stderr,
        )
        reply = self._request({"cmd": "hello"}, timeout=self.startup_timeout)
        if reply.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"프로토콜 불일치: 워커 v{reply.get('protocol')} vs "
                f"클라이언트 v{PROTOCOL_VERSION}. 양쪽이 같은 저장소를 보는지 확인할 것."
            )
        print(f"[bridge] 워커 준비 완료: {reply}", file=sys.stderr)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            send_message(self._proc.stdin, {"cmd": "close"})
            self._proc.wait(timeout=60)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- 명령 ---------------------------------------------------------------
    def reset(
        self,
        seeds: list[int] | None = None,
        env_ids: list[int] | None = None,
        init_index: int | None = None,
        init_indices: list[int] | None = None,
        bank: str | None = None,
        instruction_template: str | None = None,
    ):
        """환경을 리셋하고 관측을 받는다.

        Args:
            init_index: 초기 상태 뱅크 인덱스 하나. **전 env 가 같은 s₀ 로
                시작한다 — GRPO 그룹은 반드시 이 경로를 쓸 것.** 시드만으로는
                배치 안의 env 들이 서로 다른 배치를 받아, advantage 가
                "정책이 잘했는가"가 아니라 "이 env 가 쉬웠는가"를 재게 된다.
            init_indices: env 별 인덱스. 평가 홀드아웃을 순서대로 돌 때.
            bank: 초기 상태 뱅크 이름 (평가 split 전환).
            instruction_template: Language split 용 rephrase 템플릿.

        Returns:
            dict — {"image": (N,H,W,3) uint8, "state": (N,8) float32,
                    "instruction": [N개 문자열]}
        """
        reply = self._request(
            {
                "cmd": "reset",
                "seeds": seeds,
                "env_ids": env_ids,
                "init_index": init_index,
                "init_indices": init_indices,
                "bank": bank,
                "instruction_template": instruction_template,
            }
        )
        return reply["obs"]

    def step(self, action_chunk):
        """액션 청크를 실행한다.

        Args:
            action_chunk: (num_envs, chunk_len, action_dim) float32.
                청크 안의 액션들을 순서대로 env.step() 에 넣는다.

        Returns:
            (obs, reward, done)
              obs    — 청크 실행 후의 관측
              reward — (N,) float32. 청크 구간 중 성공이 한 번이라도 있었으면 1.0
              done   — (N,) bool

        진단값은 `self.last_diag` 에 남긴다 (보상에 섞으면 0/1 이 아니게 되어
        GRPO 의 그룹 정규화가 망가진다):
          lifted  — 블록을 집어 들어올리는 데까지 성공한 env 비율
          yaw_err — 타깃 블록의 (대칭 접힌) yaw 오차 평균 [rad]
        """
        reply = self._request({"cmd": "step", "action_chunk": action_chunk})
        self.last_diag = reply.get("diag", {})
        return reply["obs"], reply["reward"], reply["done"]

    def get_success(self):
        """에피소드 단위 성공 플래그 (N,) bool — 평가에 쓴다."""
        return self._request({"cmd": "success"})["success"]

    # -- 내부 ---------------------------------------------------------------
    def _request(self, msg: dict, timeout: float | None = 600.0) -> dict:
        if self._proc is None:
            raise RuntimeError("워커가 시작되지 않았다. start() 를 먼저 호출할 것.")
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"워커가 종료되었다 (exit {self._proc.returncode}). "
                "위쪽 stderr 로그에서 Isaac Sim 오류를 확인할 것."
            )

        send_message(self._proc.stdin, msg)
        reply = recv_message(self._proc.stdout, timeout=timeout)
        if reply is None:
            raise EOFError(
                "워커가 응답 없이 종료되었다. stderr 로그를 확인할 것 "
                "(GPU 메모리 부족이나 --enable_cameras 누락이 흔한 원인이다)."
            )
        if reply.get("error"):
            raise RuntimeError(f"워커 오류: {reply['error']}\n{reply.get('traceback', '')}")
        return reply

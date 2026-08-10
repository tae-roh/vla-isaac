# env/locks/

성공한 환경을 박제해 두는 곳이다 (계획서 §4-1 원칙 3).

**lock 은 스모크 테스트를 통과한 뒤에만 갱신한다.** 통과 전에 박제하면
깨진 환경을 재현 가능하게 만드는 셈이 된다.

```bash
# isaaclab venv 에서
python env/smoke/check_isaaclab.py --full && pip freeze > env/locks/isaaclab.lock.txt

# vla-train venv 에서 (H100 인스턴스)
python env/smoke/check_vla_train.py --full && pip freeze > env/locks/vla-train.lock.txt

# rft venv 에서
python env/smoke/check_rft.py --full && pip freeze > env/locks/rft.lock.txt
```

`&&` 로 이어 붙인 것이 의도한 바다 — 스모크가 실패하면 lock 이 갱신되지 않는다.

lock 파일은 인스턴스에서 생성되므로 이 저장소에는 커밋되지 않은 상태로 시작한다.
생성 후에는 커밋해서 인스턴스가 사라져도 환경을 복원할 수 있게 할 것.

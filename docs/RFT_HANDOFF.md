# RFT 인수인계 (2026-08-16)

다른 머신(RTX 6000 PRO)에서 RFT 를 이어받는 사람을 위한 문서다.
**실행 전에 이 문서를 끝까지 읽을 것.** 조용히 실패하는 함정이 여러 개 있고,
전부 증상이 "성공률 0" 하나로만 나타난다.

---

## 0. 한 줄 요약

SFT 체크포인트(`tae-roh/vla-pick-sft`, 8Hz, 20k 스텝)는 **최종 성공률 0%** 지만
파지·리프트·트레이 진입에는 분산이 있다. outcome-only GRPO 로는 착수할 수 없어
**단계형 sparse 보상 + 커리큘럼 초기 상태**로 RFT 가 성립하도록 만들어 두었다.

---

## 1. 실행 전 세팅

### 환경변수 — 이제 대부분 불필요하다

과거 문서/로그에 `VLA_DECIMATION=15` 를 export 하라는 지시가 있는데,
**지금은 필요 없다.** `configs/vla_spec.py` 의 `DECIMATION` 을 15(8Hz)로 바꿔
기본값이 실제 운용값과 같아졌다.

| 변수 | 필요? | 설명 |
|---|---|---|
| `VLA_DECIMATION` | ❌ | 기본 15(8Hz). 되돌릴 일 없으면 건드리지 말 것 |
| `VLA_STAGED_REWARD` | ✅ **RFT 에 필수** | `1` 로 켠다. 안 켜면 outcome-only 라 전 그룹 무신호 |
| `VLA_GRASP_HOLD_STEPS` | ❌ | 기본 4 (0.5초). 파지 인정 최소 유지 스텝 |
| `VLA_ANNOUNCE_TARGET` | ❌ | 데모 수집용 |
| `VLA_KEEP_FAILED`, `VLA_TRIALS_ARE_ATTEMPTS`, `VLA_ACTION_NOISE` | ❌ | 데이터 생성용 |

```bash
export VLA_STAGED_REWARD=1     # RFT 에서 이것 하나만 있으면 된다
```

### venv 2개 — 섞으면 안 된다

```
~/env_isaaclab   Isaac Sim 5.1 + torch 2.7.0+cu128   (롤아웃 워커 전용)
~/env_eval       transformers 4.40.1 + timm 0.9.16   (정책 추론/학습)
```

프로세스를 분리한 이유는 **torch 버전 충돌**이다. openvla-oft 를 제대로 설치하면
torch 2.2.0 을 끌고 와 Isaac Sim 이 고정한 2.7.0 을 깬다. `rft/ipc_bridge.py` 가
stdout 길이 접두어 프로토콜로 두 프로세스를 잇는다.

`import prismatic` 은 `vendor/prismatic` 셰임으로 해결한다 — 스크립트가 스스로
`sys.path` 에 붙이므로 `PYTHONPATH` 를 줄 필요 없다.

---

## 2. RFT 실행

```bash
cd vla-isaac
export VLA_STAGED_REWARD=1
~/env_eval/bin/python -u rft/grpo_fallback.py \
    --config rft/configs/grpo_7h.yaml \
    --checkpoint ckpt/sft
```

설정 파일 2종을 준비해 두었다. RTX 6000 PRO(Blackwell 96GB) 기준이다.

| | `grpo_7h.yaml` | `grpo_15h.yaml` |
|---|---|---|
| group_size | 8 | 8 |
| groups_per_step | 2 | 4 |
| max_steps_per_episode | 200 | 200 |
| temperature | 1.2 | 1.4 |
| total_steps | 320 | 360 |
| 스텝당 추정 | 65s | 130s |
| 총 추정 | 5.8h | 13h |

**시간 추정의 근거와 한계**: 실측 앵커는 L40S / `num_envs=8` / 300스텝에서
**6.0초/에피소드**다. RTX 6000 PRO 를 추론 1.6배로 가정했고, 학습 스텝 0.5초도
가정이다. ⚠ **첫 10스텝의 실제 소요를 재서 `total_steps` 를 조정할 것.**
크게 어긋나면 `groups_per_step` 을 먼저 줄인다.

`group_size` 를 16 으로 키우는 대신 `groups_per_step` 을 늘린 이유: 단계형
보상 덕에 G=8 에서도 그룹 내 분산이 충분하다. 남는 예산은 더 다양한 초기
상태에 쓰는 편이 낫다.

**온도를 1.2~1.4 로 낮춘 이유**: SimpleVLA-RL 은 1.6 을 쓰지만 그건 성공률이
이미 높은 정책 기준이다. 우리 정책은 궤적이 쉽게 망가진다. 로그의 무신호그룹
비율이 계속 높으면 그때 올린다.

### 샘플링은 어디서 일어나나

평가는 greedy, RFT 롤아웃은 샘플링이다. 경로가 완전히 다르다.

| 경로 | 방식 | 용도 |
|---|---|---|
| `eval_rollout.py` → `predict_action` | greedy (argmax) | 평가. 재현 가능해야 한다 |
| `grpo_fallback.py` → `_action_logits` | **multinomial 샘플링** | RFT 롤아웃. 탐색 |

```python
# rft/grpo_fallback.py
logp_all = torch.log_softmax(logits.float() / self.cfg.temperature, dim=-1)
seq = torch.multinomial(logp_all.exp(), num_samples=1).squeeze(-1)
```

`temperature` 는 샘플링과 log-prob 계산에 **둘 다** 적용된다. 정책 π 의 정의가
"temperature 로 스케일된 분포" 이기 때문이고, 한쪽만 적용하면 첫 스텝의 중요도
비가 1 이 되지 않는다. 그래서 `eval_rollout.py` 에 `do_sample` 을 넘길 이유가
없다 (넘겨도 조용히 무시된다).

---

## 3. 보상 설계 — 왜 outcome-only 가 아닌가

`grpo_fallback.compute_group_advantages` 는 그룹 내 표준편차가 0 이면
어드밴티지를 0 으로 둔다. 성공률이 0 이면 **모든 그룹이 무신호**다.
그래서 단계를 나눴다 (`VLA_STAGED_REWARD=1`).

| 단계 | 보상 | 술어 |
|---|---|---|
| 파지 | 0.2 | `target_grasped` + 블록이 5mm 이상 뜸 + **연속 4스텝 유지** |
| 리프트 | 0.4 | `grasp_lift_signal` (쥔 채 22mm 이상) |
| 트레이 진입 | 0.7 | `block_in_tray` + 정지 + **리프트 이력 필수** |
| 성공 | 1.0 | 2초 유지 |

**각 이벤트는 래치라 에피소드당 한 번만 켜지고, 보상은 최댓값 하나다.**
누적이 아니므로 같은 이벤트를 반복 수확할 수 없다.

### 막아 둔 해킹 3종 — 되돌리지 말 것

1. **파지가 거의 공짜였다.** `target_grasped` 는 거리 60mm + 그리퍼 폭 60mm 면
   참이라 **닫힌 빈 손이 옆을 지나가기만 해도** 켜진다. "블록 5mm 이상 뜸" +
   "연속 4스텝" 을 추가하니 파지율이 60% → 15% 로 떨어졌다. 대부분 허위였다.
2. **끌어서 밀어넣기.** 박스를 제거해 물리적으로 가능해졌다. 트레이 크레딧에
   `lift_latch` 를 AND 로 걸어 막는다.
3. **튕겨 지나가기.** 블록 최고높이 286mm 가 관측된다(던지는 동작). 정지 조건을
   AND 로 걸어 안착한 경우만 인정한다.

### 패널티를 넣지 않은 이유

"파지 후 놓치면 감점" 은 넣지 않았다. 파지 보상 0.2 보다 큰 벌점을 주면
**아예 안 잡는 쪽이 이득**이 되어 정책이 블록을 회피한다. 재파지는 정상적인
복구 행동이라 억제하면 안 된다. 인정 조건만 엄격히 하는 방향으로 갔다.

---

## 4. 초기 상태 뱅크

```
datasets/init_states/train.npz      원래 분포 (학습용)
datasets/init_states/train_mix.npz  원래 70% + 커리큘럼 30%   ← RFT 는 이것
datasets/init_states/eval_base.npz  평가 홀드아웃 (원래 분포만)
```

커리큘럼 행은 **타깃 블록만** 트레이 앞으로 옮긴다. 근거:

| 뱅크 | 트레이 진입률 (SFT, 실측) |
|---|---|
| 원래 분포 | 1/40 (2.5%) |
| 커리큘럼 | 1/2 (50%) |

⚠ 커리큘럼 행은 SFT 가 **초기 상태로는 본 적 없는** 배치다. 데모에서 블록이
트레이 근처인 순간은 언제나 "그리퍼에 들려 공중에" 있었고, "테이블에 놓인 채 +
팔은 홈" 조합은 새롭다. 그래서 30% 로 제한했다.

**평가는 반드시 `eval_base` 로만 한다.** 커리큘럼 성능이 올라도 원래 분포에서
떨어지면 의미가 없다. 원래 분포 성능이 드리프트하면 비율을 20% 로 낮출 것.

---

## 5. 조용히 실패하는 함정들 (전부 실제로 당했다)

| 함정 | 증상 | 상태 |
|---|---|---|
| `success_latch |= reward > 0.5` | 성공이 **영원히** 안 잡힘. RewardManager 는 `func × weight × dt` ≈ 0.042 를 돌려준다 | ✅ `> 1e-6` 로 수정됨 |
| 그리퍼 부호 역전 | 정책이 "닫아라" 할 때 열림 → 파지 불가 | ✅ `GRIPPER_INVERT_FOR_VLA=False` |
| `VLA_DECIMATION` 누락 | 정책이 3배 빠르게 움직임 | ✅ 기본값 15 로 해소 |
| `SUCCESS_HOLD_STEPS` 불일치 | 유지 요구가 2초가 아니라 6초였음 | ✅ SSOT 정합으로 해소 |
| `predict_action(do_sample=…)` | **조용히 무시된다.** `**kwargs` 로 삼켜짐. "샘플링 켰다" 고 착각하기 쉽다 | ✅ 인자 제거 |
| 단계형 보상 켜고 성공률 보고 | `reward` 평균(0.2/0.4)을 "성공률" 로 오독 | ✅ `diag["success"]` 에서 읽음 |
| 로그 버퍼링 | 크래시 시 결과가 통째로 유실 | `python -u` 로 실행할 것 |
| Isaac Sim assertion | 헤드리스에서 입력 대기로 **무한 정지** | 타임아웃 미구현. 장시간 배치는 감시 필요 |

---

## 6. 정규화 통계 — 체크포인트 내장을 쓴다

`--norm-stats` 플래그는 **없다** (병합으로 제거됐다). 파일과 체크포인트라는 두
출처가 갈라지는 것을 막기 위해서다. `predict_action` 이 `unnorm_key` 로 모델
안의 통계를 찾아 쓰고, 키가 없으면 명시적으로 에러를 낸다.

```
--unnorm-key vla_pick     (기본값)
```

`ckpt/sft/config.json` 의 `norm_stats["vla_pick"]` 에 8Hz 통계가 심겨 있음을
확인했다 (q99[0] = 0.1467). 체크포인트를 새로 만들면
`scripts/sft/embed_norm_stats.py` 로 심어야 한다.

⚠ 참고: 학습 머신은 `datasets/rlds/` 에 8Hz 데이터를 뒀지만 이 저장소는
`rlds/`=24Hz, `rlds_8hz/`=8Hz 로 나눈다. 파일을 직접 읽는 코드를 새로 쓸 때만
주의하면 된다 (평가 경로는 이제 파일을 안 읽는다).

---

## 7. 현재 성능 — 확정 기준선 (2026-08-16, 40 에피소드)

`reward > 1e-6` 수정과 유지시간 2초 정정이 **모두 반영된 뒤** 측정한 값이다.
버그가 아니라 실제 성능이다.

```
성공         0/40   (0.0%)
트레이 진입  0/40   (0.0%)
lifted       3/40   (7.5%)
파지         2/40   (5.0%)
블록 최고높이 평균 111mm / 최대 414mm
소요 3.7분 (5.5s/에피소드, num_envs=8)
```

명령 (환경변수는 VLA_STAGED_REWARD 하나면 된다):

```
export VLA_STAGED_REWARD=1
~/env_eval/bin/python -u scripts/eval_rollout.py --checkpoint ckpt/sft --num-episodes 40 --num-envs 8
```

### 실행 간 편차가 크다 — 40 에피소드로는 부족하다

같은 체크포인트·같은 조건에서 파지율이 실행마다 크게 흔들렸다:

```
5.0%  /  7.5%  /  15.0%
```

물리 시뮬레이션이 결정론적이지 않아서다. **단일 40 에피소드 측정으로 RFT 전후를
비교하면 안 된다.** 최소 100 에피소드, 가능하면 시드를 바꿔 3회 반복할 것.
"파지율이 5% -> 15% 로 올랐다" 는 이 편차 안에 완전히 묻힌다.

### RFT 착수 판단

- **outcome-only 는 불가.** 성공 0% 라 전 그룹 무신호다.
- **단계형 보상으로는 가능.** 파지 5% / lifted 7.5% 에 분산이 있다.
- **마지막 단계는 커리큘럼 뱅크가 필수.** 원래 분포에서 트레이 진입이 0/40 인데,
  커리큘럼 초기 상태에서는 1/2 (50%) 였다. `bank: train_mix` 를 반드시 쓸 것.
- 목표는 최종 성공률이 아니라 **단계 도달률 개선**으로 잡는 편이 현실적이다.
  진입이 원래 분포에서 0 인 이상, 성공률이 오르려면 커리큘럼에서 얻은 신호가
  원래 분포로 전이되어야 하는데 그건 보장되지 않는다.

### 태스크 완화 이력 (헐거운 split)

원래 값에서 다음을 바꿨다. 보고 시 **조건을 반드시 명시할 것.**

| 항목 | 원래 | 현재 | 근거 |
|---|---|---|---|
| `TRAY_DEPTH` | 20mm | 10mm | 정책의 리프트 높이로는 20mm 레일을 물리적으로 못 넘었다 |
| `LIFT_HEIGHT_THRESHOLD` | 60mm | 22mm | 60mm 는 **이미 제거된 박스 벽** 기준이라 근거가 없었다 |
| `TRAY_INNER_SIZE` | 1.2× | 2.0× | 진입률을 0 에서 띄우기 위한 완화 |

⚠ `TRAY_DEPTH` 와 `TRAY_INNER_SIZE` 변경은 **화면에 보이는 트레이 모양을 바꾼다**
— 학습 데이터와 관측 분포가 다르다(OOD). 커브가 올라도 "정책 개선" 인지 "낯선
화면 적응" 인지 구분이 어렵다. 시간이 허락하면 형상은 원복하고 성공 판정만
근접 기반으로 바꾸는 쪽이 깨끗하다.

---

## 8. 미해결 / 판단이 필요한 것

- **SFT 가 학습은 됐다.** teacher forcing 에서 베이스 openvla-7b 대비 압도적이다
  (dx 상관 0.048 → 0.448, 그리퍼 0.457 → 0.811). 다만 회전 차원의 예측 std 가
  정답의 2.4배로 **과분산**이다. 학습 부족이면 보통 평균으로 수축하는데 반대
  방향이라, 데이터 다중성(같은 화면에 여러 정답)을 의심하고 있다.
- **스텝 연장의 가치는 미확정.** 15k/17.5k/20k 지표가 평평하지만 셋 다 LR 감쇠
  (13,333) 이후 구간이라 정체의 증거가 못 된다. 재학습한다면
  `--max_steps 100000 --num_steps_before_decay 66000 --save_freq 5000` 로,
  **감쇠 이전 체크포인트를 반드시 남길 것.**
- **SkillGen 0%** — `docs/DATASET.md` 참조. MimicGen 으로 우회 중.
- **Isaac Sim assertion 무한 정지** — 워커에 스텝 타임아웃이 없다.

---

## 9. 참고 문서

- `docs/DATASET.md` — 데이터 파이프라인, 8Hz 다운샘플 근거, 미해결 2건
- `docs/RUNBOOK.md` — 환경 구축
- `configs/vla_spec.py` — **관측/액션 스펙의 SSOT.** 여기만 고치면 env·SFT·RFT 가 따라온다

"""
[공용]  공유 상수 일치 검사 — CI 와 로컬에서 같이 쓴다.  python3 scripts/check_constants.py

대시보드(index.html)와 시뮬레이터는 서버 상수를 복사해 쓴다. 한 곳만 고치면
대시보드 버튼 · 시뮬레이션이 예전 값으로 돈다 (docs/ROLES.md 3장).
"""
import re
import sys


def grab(path, pattern, required=True):
    m = re.search(pattern, open(path, encoding="utf-8").read(), re.M)
    if not m:
        if not required:
            return None
        sys.exit(f"[check_constants] {path} 에서 {pattern!r} 를 찾지 못했습니다")
    return m.group(1)


HTML = "serving_app/static/index.html"
NUM = r"([0-9.]+)"
GROUPS = {
    "입력 길이 SEQ_LEN": [
        ("data/features.py", rf"^SEQ_LEN = {NUM}"),
        (HTML, rf"const SEQ_LEN = {NUM}"),
    ],
    "판정 윈도우 WINDOW_SIZE": [
        ("serving_app/monitoring/drift_detector.py", rf"^WINDOW_SIZE = {NUM}"),
        (HTML, rf"const WINDOW_SIZE = {NUM}"),
    ],
    "RMSE 임계값 (게이트 = 드리프트)": [
        ("serving_app/train_and_register.py", rf"^RMSE_GATE = {NUM}"),
        ("serving_app/monitoring/drift_detector.py", rf"^RMSE_THRESHOLD = {NUM}"),
        (HTML, rf"const RMSE_THRESHOLD = {NUM}"),
    ],
    # 대시보드 표시는 D 가 붙이는 중 — index.html 에 아직 없으면 건너뛰고, 생기면 같이 검사 (#102)
    "MAPE 게이트 MAPE_GATE (%)": [
        ("serving_app/train_and_register.py", rf"^MAPE_GATE = {NUM}"),
        (HTML, rf"const MAPE_GATE = {NUM}", False),
    ],
    "모델 이름": [
        ("serving_app/train_and_register.py", r'^MODEL_NAME = "(\w+)"'),
        ("serving_app/model_loader.py", r'"models:/(\w+)/Production"'),
    ],
}

failed = False
for name, places in GROUPS.items():
    values = {path: grab(path, pat, *opt) for path, pat, *opt in places}
    values = {p: v for p, v in values.items() if v is not None}
    norm = {float(v) if re.fullmatch(NUM, v) else v for v in values.values()}
    ok = len(norm) == 1
    failed |= not ok
    print(f"{'OK ' if ok else 'BAD'} {name}: " + " · ".join(f"{p}={v}" for p, v in values.items()))

seq = int(float(grab("data/features.py", rf"^SEQ_LEN = {NUM}")))
win = int(float(grab("serving_app/monitoring/drift_detector.py", rf"^WINDOW_SIZE = {NUM}")))
batch = int(grab("scripts/simulate_drift.py", r"^BATCH_N = (\d+)"))
ok = batch == seq + win
failed |= not ok
print(f"{'OK ' if ok else 'BAD'} simulate_drift BATCH_N={batch} (= SEQ_LEN {seq} + WINDOW_SIZE {win})")

sys.exit(1 if failed else 0)

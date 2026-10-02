"""
[소유: B 서빙 · 박유진]  Lazy / Eager 지연 측정 (기획서 ② 응답 시간 "첫 요청 포함 1초 이내")

작성자 : 박유진 (B 서빙 · Git)
버전   : 1.0.0
작성일 : 2026-10-01
변경 이력
  1.0.0  2026-10-01  최초 작성 — Lazy / Eager 3회 측정 (#22)

모드마다 서버를 새 프로세스로 띄워(캐시 없는 상태) 아래를 잰다. 프로젝트 루트에서:
    python scripts/measure_latency.py                    # lazy · eager 각 3회, MODEL_SOURCE=mlflow
    python scripts/measure_latency.py --runs 5 --source local

  - 기동 시간   : 프로세스 시작 → /health 가 200 을 줄 때까지
  - model_loaded: 기동 직후 /health 의 값 (eager 면 true 여야 함)
  - 첫 요청     : 첫 /predict 왕복 시간 (lazy 는 여기서 모델을 불러옴)
  - 두 번째 요청: 캐시된 모델로 /predict

전제: 해당 MODEL_SOURCE 로 불러올 모델이 있어야 한다 (local: serving_app/models/*, mlflow: Production 등록).
측정 중 이력 로그 · aiops.log 는 실제 logs/ 에 쌓이지 않도록 임시 폴더에서 돌린다.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.getcwd())
from data.features import SEQ_LEN  # noqa: E402

PORT = 8079


def _req(path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


def measure_once(mode: str, source: str) -> dict:
    body = {"sequence": [{"close": 160 + i * 0.3, "volume": 1_200_000} for i in range(SEQ_LEN)]}
    env = {**os.environ, "LOADING_MODE": mode, "MODEL_SOURCE": source}
    root = os.getcwd()
    work = tempfile.mkdtemp()
    # 코드 · 모델 · mlflow 는 링크로 두고, logs/ 만 임시 폴더에 새로 만든다
    for name in os.listdir(root):
        if name not in ("logs", ".git"):
            os.symlink(os.path.join(root, name), os.path.join(work, name))
    os.mkdir(os.path.join(work, "logs"))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving_app.main:app", "--port", str(PORT)],
                            cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.perf_counter()
        while True:
            try:
                health = _req("/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("서버가 종료됨 — 모델 · MLflow 준비 여부 확인")
                time.sleep(0.05)
        startup = time.perf_counter() - t0
        t1 = time.perf_counter()
        first = _req("/predict", body)
        first_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        _req("/predict", body)
        second_s = time.perf_counter() - t2
        return {"startup_s": startup, "model_loaded": health["model_loaded"], "first_s": first_s,
                "second_s": second_s, "model_version": first["model_version"]}
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--source", default="mlflow", choices=["mlflow", "local"])
    args = ap.parse_args()

    results = {}
    for mode in ("lazy", "eager"):
        runs = [measure_once(mode, args.source) for _ in range(args.runs)]
        results[mode] = runs
        for i, r in enumerate(runs, 1):
            print(f"[{mode} #{i}] 기동 {r['startup_s']:.2f}s · loaded={r['model_loaded']} · "
                  f"첫 요청 {r['first_s']:.3f}s · 두 번째 {r['second_s']:.3f}s · {r['model_version']}")

    print(f"\n| 모드 | 기동 | 기동 직후 model_loaded | 첫 요청 | 두 번째 요청 |  (평균 {args.runs}회, MODEL_SOURCE={args.source}, 입력 {SEQ_LEN}일)")
    print("|---|---|---|---|---|")
    for mode, runs in results.items():
        avg = lambda k: statistics.mean(r[k] for r in runs)  # noqa: E731
        print(f"| {mode} | {avg('startup_s'):.2f}초 | {runs[0]['model_loaded']} | "
              f"{avg('first_s'):.3f}초 | {avg('second_s'):.3f}초 |")
    print("\n" + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

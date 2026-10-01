<!-- [소유: A 데이터 · 심준용]  보고서 프롬프트. {중괄호} 는 generator.py 가 채운다. -->

TODO: 점주 · 본사가 읽을 일일 보고서 프롬프트

- 입력: {item_name} {date} {predicted_price} {last_price} {drift_status} {rmse_recent} {retrain_events} {model_version}
- 규칙: 숫자는 입력값만 쓴다 · 드리프트면 첫 줄에 "가격 급변 감지 — 오늘 예측 신뢰도 낮음"

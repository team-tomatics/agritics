# 데이터 상태

## `tomato_prices.csv`

- 기간: 2023-10-02 ~ 2026-09-30, 908거래일
- `Close`: KAMIS 가락시장 토마토(등급 상) 경락가, 원/kg
- `Volume`: **반입량 API 승인 전 임시값 `0`**

현재 `Volume=0`은 가격 단일 피처 모델로 학습·컨테이너·드리프트 파이프라인을
먼저 검증하기 위한 값이다. 실측 반입량이나 최종 모델 성능 근거로 사용하지 않는다.

반입량 CSV를 받으면 다음 명령으로 날짜별 누락·중복을 검증해 교체한다.

```bash
python scripts/fetch_kamis.py merge \
  --prices data/tomato_prices.csv \
  --volumes /path/to/volume.csv \
  --volume-date-column Date \
  --volume-column Volume \
  --output data/tomato_prices.csv
```

실측 Volume을 넣은 뒤에는 기존 0 기반 스케일러와 모델을 이어서 사용하지 않는다.
스케일러를 다시 fit하고 모델을 처음부터 학습·등록해야 한다.

## `sample_haic_prices.csv`

토마토 실측 Volume이 들어오기 전 기존 파이프라인 비교·비상 복구용 샘플이다.
Dockerfile과 드리프트 시뮬레이터가 `tomato_prices.csv`로 전환된 뒤 제거한다.

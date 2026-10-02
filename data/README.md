# 데이터 상태

## `tomato_prices.csv`

- 기간: 2023-10-02 ~ 2026-09-30, 903거래일
- `Close`: KAMIS 가락시장 토마토(등급 상) 경락가, 원/kg
- `Volume`: 제공받은 일별 반입량을 날짜 기준으로 결합한 값, kg

가격 원본은 908거래일이지만 반입량이 없는 아래 5일은 학습 데이터에서 제외했다.

- 2023-11-04
- 2023-12-02
- 2024-03-02
- 2025-01-02
- 2025-11-01

반입량 데이터를 다시 교체할 때는 다음 명령으로 날짜별 누락·중복을 검증한다.

```bash
python scripts/fetch_kamis.py merge \
  --prices data/tomato_prices.csv \
  --volumes /path/to/volume.csv \
  --volume-date-column Date \
  --volume-column Volume \
  --output data/tomato_prices.csv
```

Volume 데이터가 바뀌면 기존 스케일러와 모델을 이어서 사용하지 않는다. 스케일러를
다시 fit하고 모델을 처음부터 학습·등록해야 한다.

# OHT 속도 분석 시뮬레이터

## 실행 방법

```bash
cd oht_system
python simulator_server.py
```

브라우저에서 http://localhost:8080 열기

## CSV 데이터 수정

### data/vehicles.csv
Vehicle 초기 데이터 (수정 가능)

| 필드 | 설명 | 예시 |
|------|------|------|
| vehicle_id | Vehicle ID | V00001 |
| current_address | 시작 주소 | 1 |
| speed_mpm | 속도 (m/min) | 280 |
| state | 상태 (1=RUN, 6=OBS_STOP, 7=JAM) | 1 |
| run_cycle | 실행 Cycle (3=DEPOSIT, 4=ACQUIRE) | 4 |
| vhl_cycle | Vhl Cycle (2=DEPOSIT_MOVING, 4=ACQUIRE_MOVING) | 4 |
| is_loaded | 적재 여부 (0/1) | 0 |
| destination | 목적지 | 100 |
| bay_name | Bay 이름 | B01 |

### data/rail_edges.csv
RailEdge 데이터 (수정 가능)

| 필드 | 설명 | 예시 |
|------|------|------|
| from_address | 시작 주소 | 1 |
| to_address | 끝 주소 | 2 |
| length_mm | 길이 (mm) | 2000 |
| max_speed | 최대 속도 (m/min) | 320 |
| speed_level | 속도 레벨 | 64 |

## 속도 계산 (지침 섹션 2)

```
속도(v) = 거리(Δx) / 시간(Δt) × 60.0
```

### 5가지 필수 조건 (지침 섹션 3)
1. 시간 차이 < 60초
2. 상태: RUN(1), OBS_STOP(6), JAM(7), E84_TIMEOUT(9)
3. Cycle 일치
4. run_cycle: DEPOSIT(3) 또는 ACQUIRE(4)
5. vhl_cycle: DEPOSIT_MOVING(2) 또는 ACQUIRE_MOVING(4)

## API 엔드포인트

- GET /api/vehicles - 모든 Vehicle
- GET /api/edges - 모든 RailEdge
- GET /api/stats - 통계
- GET /api/all - 전체 데이터
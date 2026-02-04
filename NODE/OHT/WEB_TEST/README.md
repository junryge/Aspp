# MES/MCS/SECS-GEM/OHT 통합 테스트 환경

## 구성

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│     MES     │────▶│     MCS     │────▶│  OHT Simulator  │
│  Port:10010 │     │  Port:10011 │     │   Port:10003    │
└─────────────┘     └─────────────┘     └─────────────────┘
                           │                     │
                           ▼                     ▼
                    ┌─────────────┐       ┌───────────┐
                    │  SECS/GEM   │       │  WebSocket │
                    │  Port:10012 │       │  (실시간)   │
                    └─────────────┘       └───────────┘
```

## 파일 구조

| 파일 | 설명 | 포트 |
|------|------|------|
| `mes_server.py` | MES - Transport 입력 UI | 10010 |
| `mcs_server.py` | MCS - 배차 로직 | 10011 |
| `secs_gem_mock.py` | SECS/GEM - 설비 Mock | 10012 |
| `simulator_server_3D_B_TEST.py` | OHT Simulator | 10003 |
| `run_all.py` | 전체 서버 실행 | - |

## 실행 방법

### 전체 서버 한번에 실행
```bash
cd OHT2/MES_MCS_TEST
python run_all.py
```

### 개별 서버 실행
```bash
# 터미널 1 - SECS/GEM
python secs_gem_mock.py

# 터미널 2 - OHT
python simulator_server_3D_B_TEST.py

# 터미널 3 - MCS
python mcs_server.py

# 터미널 4 - MES
python mes_server.py
```

## 테스트 시나리오

1. **MES에서 Transport 입력** (http://localhost:10010)
   - Lot ID, Carrier ID, From/To Station 입력
   - [MCS로 전송] 버튼 클릭

2. **MCS에서 배차 확인** (http://localhost:10011)
   - Transport 수신 로그 확인
   - OHT 배차 상태 확인

3. **OHT Simulator 확인** (http://localhost:10003)
   - 배차된 차량 이동 확인

4. **SECS/GEM 이벤트 확인** (http://localhost:10012)
   - Load/Unload 이벤트 로그 확인

## API 명세

### MES → MCS
```
POST http://localhost:10011/api/transport
{
    "requestId": "TR00001",
    "lotId": "LOT001",
    "carrierId": "FOUP001",
    "fromStation": 14901,
    "toStation": 25001,
    "priority": 2
}
```

### MCS → OHT
```
POST http://localhost:10003/api/dispatch
{
    "requestId": "TR00001",
    "carrierId": "FOUP001",
    "fromStation": 14901,
    "toStation": 25001,
    "priority": 2
}
```

### MCS → SECS/GEM
```
POST http://localhost:10012/api/load
{
    "carrierId": "FOUP001",
    "portId": 14901,
    "vehicleId": "V00001"
}

POST http://localhost:10012/api/unload
{
    "carrierId": "FOUP001",
    "portId": 25001,
    "vehicleId": "V00001"
}
```

## Transport 상태 흐름

```
QUEUED → DISPATCHED → TO_PICKUP → PICKING → CARRYING → COMPLETE
```

| 상태 | 설명 |
|------|------|
| QUEUED | MES에서 요청 생성 |
| DISPATCHED | OHT 배차 완료 |
| TO_PICKUP | 픽업 위치로 이동 중 |
| PICKING | 설비에서 Carrier 픽업 중 |
| CARRYING | 목적지로 이동 중 |
| COMPLETE | 배송 완료 |
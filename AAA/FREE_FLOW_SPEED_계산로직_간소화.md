# FREE_FLOW_SPEED 계산 로직 (간소화)

## 한줄 요약

**FREE_FLOW_SPEED = HID 내 모든 RailEdge의 maxVelocity 단순 평균**

---

## 계산 흐름

```
layout 설정파일 [MCP75_VEHICLE_SPEED]
    → 속도 등급별 속도값 테이블 (예: 등급1=2000.0, 등급2=1800.0)

src/util/DataService.java:700~708
    → 각 RailEdge마다 속도 등급(leftSpeed)으로 테이블 조회
    → railEdge.maxVelocity에 세팅

src/batch/HidEdgeInOutUpdateMasterBatch.java:193~228
    → 같은 HID 내 maxVelocity > 0인 RailEdge 전부 수집
    → 합계 / 개수 = FREE_FLOW_SPEED
```

---

## 코드 위치

| 순서 | 파일 | 행 | 내용 |
|------|------|------|------|
| 1 | `src/data/raw/Mcp75Config.java` | 592~622 | 속도 테이블 파싱 (등급 → 속도값) |
| 2 | `src/util/DataService.java` | 700~708 | RailEdge별 maxVelocity 세팅 |
| 3 | `src/batch/HidEdgeInOutUpdateMasterBatch.java` | 193~197 | HID별 maxVelocity 수집 |
| 4 | `src/batch/HidEdgeInOutUpdateMasterBatch.java` | 218~228 | 단순 평균 계산 → FREE_FLOW_SPEED |

---

## 주의사항

- 기존 원본 코드에는 FREE_FLOW_SPEED 계산 로직이 없음 (신규 로직)
- RailEdge.java:7 주석에 `// 분속 단위`로 기재됨
- maxVelocity <= 0인 RailEdge는 평균 계산에서 제외됨

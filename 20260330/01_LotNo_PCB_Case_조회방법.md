# Lot No.를 이용한 PCB Case 가져오기

## 현재 Manual 조회 Sequence

| Step | 시스템 | 작업 |
|------|--------|------|
| 1 | DS Smart OIP (ICRNDDS_MES) | INQUIRY > View Lot History에서 Lot No. 검색 |
| 2 | MES 조회 결과 | PCB Sap Cd 채번 |
| 3 | G-MDM | Material > Search Material에서 SapCode 검색 |
| 4 | G-MDM 결과 | Remark 란의 PCB Case 확인 |

---

## SQL Lab 통합 쿼리 작성

### Step 1: Lot No. → PCB_SAP_CODE

```sql
SELECT lot.LOT_NO,
       lot.PCB_SAP_CD
FROM   LOT_HISTORY lot          -- MES 테이블 (실제 테이블명 확인 필요)
WHERE  lot.LOT_NO = 'YOUR_LOT_NO';
```

### Step 2: SAP_CODE → Material 속성 전체 조회

```sql
SELECT m.SAP_CD,
       m.MATERIAL_NM,
       a.ATTR_NM,
       a.ATTR_VAL
FROM   MATERIAL_MASTER m
       JOIN MATERIAL_ATTR a
         ON m.MATERIAL_ID = a.MATERIAL_ID
WHERE  m.SAP_CD = 'PCB_SAP_CODE_FROM_STEP1';
```

### Step 3: 통합 쿼리 (Lot No. → PCB Case 한번에)

```sql
SELECT lot.LOT_NO,
       lot.PCB_SAP_CD,
       m.MATERIAL_NM,
       a.ATTR_NM,
       a.ATTR_VAL
FROM   LOT_HISTORY lot
       JOIN MATERIAL_MASTER m
         ON lot.PCB_SAP_CD = m.SAP_CD
       LEFT JOIN MATERIAL_ATTR a
         ON m.MATERIAL_ID = a.MATERIAL_ID
WHERE  lot.LOT_NO = 'YOUR_LOT_NO';
```

> **참고**: `LEFT JOIN`을 사용하는 이유는 ATTR 데이터가 없는 경우에도 LOT/MATERIAL 정보는 출력되도록 하기 위함입니다.

---

## ATTR_NM 값이 출력 안 되는 원인 분석

### 원인 1: JOIN 키 불일치

DataLake에 적재된 테이블 간 연결 키가 다를 수 있습니다.

```
MES 원본:     MATERIAL_ID (숫자)
DataLake 적재: MATERIAL_CD (문자) 또는 SAP_CD
```

**확인 방법:**

```sql
-- 두 테이블의 키 컬럼 확인
SELECT * FROM MATERIAL_MASTER WHERE ROWNUM <= 5;
SELECT * FROM MATERIAL_ATTR   WHERE ROWNUM <= 5;
```

### 원인 2: ATTR_NM 필터 값 불일치

G-MDM UI에서 보이는 "Remark"가 DB상 다른 값일 수 있습니다.

**확인 방법 (필터 없이 전체 ATTR_NM 조회):**

```sql
SELECT DISTINCT a.ATTR_NM
FROM   MATERIAL_ATTR a
       JOIN MATERIAL_MASTER m
         ON m.MATERIAL_ID = a.MATERIAL_ID
WHERE  m.SAP_CD = 'YOUR_SAP_CODE';
```

가능한 ATTR_NM 값 예시:

- `REMARK`
- `RMK`
- `PCB_CASE`
- `REMARK1` / `REMARK2`
- `ETC_INFO`

### 원인 3: DataLake 동기화 지연

G-MDM 원본에는 데이터가 있지만 DataLake에 아직 sync되지 않은 경우입니다.

**확인 방법:**

```sql
-- DataLake 최신 적재 시점 확인
SELECT MAX(UPDATE_DT) FROM MATERIAL_ATTR;
SELECT MAX(UPDATE_DT) FROM MATERIAL_MASTER;
```

### 원인 4: SQL Lab DB 연결 확인

SQL Lab에서 연결된 DataSource가 MES DB인지, G-MDM DB인지 확인이 필요합니다.
두 시스템이 별도 DB일 경우 Cross-DB JOIN 또는 DataLake 통합 테이블을 사용해야 합니다.

---

## 디버깅 순서 (권장)

```
1. SQL Lab에서 연결된 DB/Schema 확인
2. MATERIAL_ATTR 테이블의 DISTINCT ATTR_NM 전체 조회
3. JOIN 키 컬럼 확인 (MATERIAL_ID vs MATERIAL_CD vs SAP_CD)
4. LEFT JOIN으로 변경 후 NULL 여부 확인
5. 데이터 적재 시점 확인
```

---

## 추가 참고

현재 작성한 SQL 코드를 공유해주시면 정확한 원인을 짚어드릴 수 있습니다.

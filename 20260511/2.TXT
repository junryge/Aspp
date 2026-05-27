-- =====================================================================
-- 진단용 — 어디서 ORA-01722 가 나는지 단계별 테스트
-- =====================================================================
-- 사용법: 아래 5단계를 차례로 실행해서 어디서 오류 나는지 확인.
--         오류 안 나는 단계까지는 OK. 오류 나는 단계가 문제 위치.
-- =====================================================================

-- ───────────────────────────────────────────────────────────────────
-- 【테스트 1】 가장 단순한 SELECT — 테이블 자체가 OK?
-- ───────────────────────────────────────────────────────────────────
SELECT COUNT(*) AS ROW_CNT
  FROM AWS_IDC_DATA_HIS
 WHERE CRT_TM BETWEEN TO_DATE('2026-05-13 00:00', 'YYYY-MM-DD HH24:MI')
                  AND TO_DATE('2026-05-13 23:59', 'YYYY-MM-DD HH24:MI');
-- 예상: 숫자 (행 수)
-- 오류 시: 테이블 권한 / 인덱스 / WHERE 절 문제


-- ───────────────────────────────────────────────────────────────────
-- 【테스트 2】 컬럼 타입 확인 — IDC_VAL 이 NUMBER? VARCHAR2?
-- ───────────────────────────────────────────────────────────────────
SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH
  FROM USER_TAB_COLUMNS
 WHERE TABLE_NAME = 'AWS_IDC_DATA_HIS'
 ORDER BY COLUMN_ID;
-- 보고 사항: IDC_VAL 의 DATA_TYPE 알려주세요.
--           NUMBER, VARCHAR2, FLOAT 중 어느 것?


-- ───────────────────────────────────────────────────────────────────
-- 【테스트 3】 단일 컬럼 MAX — 핵심 검증
-- ───────────────────────────────────────────────────────────────────
SELECT
  TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI') AS T,
  MAX(IDC_VAL) AS V
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-13 00:00', 'YYYY-MM-DD HH24:MI')
                 AND TO_DATE('2026-05-13 00:10', 'YYYY-MM-DD HH24:MI')
  AND IDC_NM = 'M16HUB.QUE.ALL.CURRENTQCNT'
GROUP BY TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI');
-- 예상: 10개 행 (10분간 값)
-- 오류 시: IDC_VAL 데이터에 비숫자 값이 섞여있음


-- ───────────────────────────────────────────────────────────────────
-- 【테스트 4】 PIVOT 1개 컬럼 (간단 형태)
-- ───────────────────────────────────────────────────────────────────
SELECT
  TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI') AS T,
  MAX(CASE WHEN IDC_NM='M16HUB.QUE.ALL.CURRENTQCNT' THEN IDC_VAL END) AS C001
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-13 00:00', 'YYYY-MM-DD HH24:MI')
                 AND TO_DATE('2026-05-13 00:10', 'YYYY-MM-DD HH24:MI')
  AND IDC_NM IN ('M16HUB.QUE.ALL.CURRENTQCNT')
GROUP BY TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI');
-- 예상: 10개 행 (1개 컬럼)
-- 오류 시: PIVOT 자체 문제 (희박)


-- ───────────────────────────────────────────────────────────────────
-- 【테스트 5】 PIVOT 5개 컬럼 (소형 통합)
-- ───────────────────────────────────────────────────────────────────
SELECT
  TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI') AS T,
  MAX(CASE WHEN IDC_NM='M16HUB.QUE.ALL.CURRENTQCNT' THEN IDC_VAL END) AS C001,
  MAX(CASE WHEN IDC_NM='M16HUB.QUE.TIME.AVGTOTALTIME1MIN' THEN IDC_VAL END) AS C002,
  MAX(CASE WHEN IDC_NM='M16HUB.QUE.M14TOM16.MESCURRENTQCNT' THEN IDC_VAL END) AS C003,
  MAX(CASE WHEN IDC_NM='M16HUB.STRATE.ALL.FABSTORAGERATIO' THEN IDC_VAL END) AS C004,
  MAX(CASE WHEN IDC_NM='M14.QUE.ALL.3F_TO_HUB_JOB' THEN IDC_VAL END) AS C005
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-13 00:00', 'YYYY-MM-DD HH24:MI')
                 AND TO_DATE('2026-05-13 00:10', 'YYYY-MM-DD HH24:MI')
  AND IDC_NM IN (
    'M16HUB.QUE.ALL.CURRENTQCNT',
    'M16HUB.QUE.TIME.AVGTOTALTIME1MIN',
    'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
    'M16HUB.STRATE.ALL.FABSTORAGERATIO',
    'M14.QUE.ALL.3F_TO_HUB_JOB'
  )
GROUP BY TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI');
-- 예상: 10개 행 (5개 컬럼)
-- 오류 시: 5개 컬럼 중 하나가 문제

-- ───────────────────────────────────────────────────────────────────
-- 진단 결과 알려주세요:
--   - 테스트 1: OK / 오류
--   - 테스트 2: IDC_VAL 의 DATA_TYPE
--   - 테스트 3: OK / 오류
--   - 테스트 4: OK / 오류
--   - 테스트 5: OK / 오류
-- 이 결과로 정확한 원인 찾을 수 있음
-- ───────────────────────────────────────────────────────────────────

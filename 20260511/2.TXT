-- =====================================================================
-- FIND_MISSING_COLUMNS.sql — 결측 컬럼의 다른 이름 찾기
-- =====================================================================
-- 결측 19개 + 일부 결측 5개의 키워드로 유사한 IDC_NM 검색
-- 같은 키워드가 다른 prefix 또는 다른 철자로 존재하는지 확인
-- 2026-05-26 데이터 행 수까지 표시
-- =====================================================================

SET PAGESIZE 100
SET LINESIZE 200
SET TRIMSPOOL ON
SET FEEDBACK ON
COLUMN IDC_NM FORMAT A75
COLUMN ROW_CNT FORMAT 999,999
COLUMN AVG_VAL FORMAT A15
COLUMN SAMPLE_VAL FORMAT A30

PROMPT
PROMPT ====================================================================
PROMPT  [1] QUETIMEDELAY  (M16HUB/M14/M14B/M16A/M16B 5개 영역 100% 빈값)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND (UPPER(IDC_NM) LIKE '%QUETIMEDELAY%'
    OR UPPER(IDC_NM) LIKE '%QUE_TIME_DELAY%'
    OR UPPER(IDC_NM) LIKE '%QUETIME%DELAY%'
    OR UPPER(IDC_NM) LIKE '%TIMEDELAY%')
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [2] CARRIERTRANSDELAY  (M14 100% 빈값)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND (UPPER(IDC_NM) LIKE '%CARRIER%DELAY%'
    OR UPPER(IDC_NM) LIKE '%CARRIERTRANS%')
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [3] LTFTOALLCURRENTQCNT (오타 의심 — LFT 인지 LTF 인지)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND (UPPER(IDC_NM) LIKE '%LTFTOALL%'
    OR UPPER(IDC_NM) LIKE '%LFTTOALL%'
    OR UPPER(IDC_NM) LIKE '%LFT.LFTTO%'
    OR UPPER(IDC_NM) LIKE '%LFT.LTFTO%'
    OR UPPER(IDC_NM) LIKE '%LFT.ALLTOLFT%'
    OR UPPER(IDC_NM) LIKE '%LFT.ALLTOLTF%')
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [4] SORTERTRANSFERFAIL  (M16A/M16B 100% 빈값)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND (UPPER(IDC_NM) LIKE '%SORTER%FAIL%'
    OR UPPER(IDC_NM) LIKE '%SORTERTRANS%'
    OR UPPER(IDC_NM) LIKE '%TRANSFERFAIL%')
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [5] SORTERWAITCOUNTOVER  (M16HUB 100% 빈값 / M14·M14B 일부 결측)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND (UPPER(IDC_NM) LIKE '%SORTERWAIT%'
    OR UPPER(IDC_NM) LIKE '%SORTER%COUNT%OVER%'
    OR UPPER(IDC_NM) LIKE '%WAITCOUNTOVER%')
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [6] AOTRANSDELAY  (M14B/M16_PKT/M16_WT 80% 결측)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND (UPPER(IDC_NM) LIKE '%AOTRANS%'
    OR UPPER(IDC_NM) LIKE '%AO_TRANS%'
    OR UPPER(IDC_NM) LIKE '%AO.TRANS%'
    OR UPPER(IDC_NM) LIKE '%ABN.AO%')
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [7] M16B.QUE.CNV.*  (6개 100% 빈값 — M16B CNV 컬럼 전체 확인)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND UPPER(IDC_NM) LIKE 'M16B.QUE.CNV.%'
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [8] M16B.QUE.LFT.*  (2개 100% 빈값 — M16B LFT 컬럼 전체 확인)
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT, MIN(IDC_VAL) AS SAMPLE_VAL
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND UPPER(IDC_NM) LIKE 'M16B.QUE.LFT.%'
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [9] M16B 전체 — 어떤 IDC_NM 들이 있는지 한번에 보기
PROMPT ====================================================================
SELECT IDC_NM, COUNT(*) AS ROW_CNT
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
  AND IDC_NM LIKE 'M16B.%'
GROUP BY IDC_NM
ORDER BY IDC_NM;

PROMPT
PROMPT ====================================================================
PROMPT  [10] BONUS — 영역별 IDC_NM 총 개수 (DB에 실제 존재 vs 우리가 요청)
PROMPT ====================================================================
SELECT 
  CASE 
    WHEN IDC_NM LIKE 'M16HUB.%' THEN 'M16HUB'
    WHEN IDC_NM LIKE 'M14B.%'   THEN 'M14B'
    WHEN IDC_NM LIKE 'M14.%'    THEN 'M14'
    WHEN IDC_NM LIKE 'M16A.%'   THEN 'M16A'
    WHEN IDC_NM LIKE 'M16B.%'   THEN 'M16B'
    WHEN IDC_NM LIKE 'M16_PKT.%' THEN 'M16_PKT'
    WHEN IDC_NM LIKE 'M16_WT.%'  THEN 'M16_WT'
    WHEN IDC_NM LIKE 'M16.%'    THEN 'M16'
    ELSE 'OTHER'
  END AS AREA,
  COUNT(DISTINCT IDC_NM) AS DISTINCT_IDC_COUNT
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN TO_DATE('2026-05-26 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                 AND TO_DATE('2026-05-26 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
GROUP BY 
  CASE 
    WHEN IDC_NM LIKE 'M16HUB.%' THEN 'M16HUB'
    WHEN IDC_NM LIKE 'M14B.%'   THEN 'M14B'
    WHEN IDC_NM LIKE 'M14.%'    THEN 'M14'
    WHEN IDC_NM LIKE 'M16A.%'   THEN 'M16A'
    WHEN IDC_NM LIKE 'M16B.%'   THEN 'M16B'
    WHEN IDC_NM LIKE 'M16_PKT.%' THEN 'M16_PKT'
    WHEN IDC_NM LIKE 'M16_WT.%'  THEN 'M16_WT'
    WHEN IDC_NM LIKE 'M16.%'    THEN 'M16'
    ELSE 'OTHER'
  END
ORDER BY 1;

PROMPT
PROMPT === 진단 완료 ===

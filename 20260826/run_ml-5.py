# -*- coding: utf-8 -*-
"""
M16A HUBROOM 수집 + 룰 예측 + 로그프레소 기입 + 영역분리 동시 실행
- 수집기 스레드 (백그라운드 데몬)
- 로그프레소 기입기 스레드 (백그라운드 데몬) — 발동이벤트.csv에 이상감지 4컬럼
- MAXCAPA 기입기 스레드 (백그라운드 데몬) — 발동이벤트.csv에 조작내역 4컬럼
- PIO 기입기 스레드 (백그라운드 데몬) — 발동이벤트.csv에 경로별 반송실패 12컬럼 (ORA_USER/ORA_PASS 필요)
- 영역분리 스레드 (백그라운드 데몬) — 발동이벤트.csv를 predict_tobe/fab분리/ 로 분리
- 룰 예측기 (메인 스레드)
- Ctrl+C 한 번으로 같이 종료

※ ML(Chronos-2)은 여기 없다 — 별도 파이썬이 필요해 프로세스를 분리했다.
     (룰베이스)  python run_ml.py
     (ML)        <chronos 파이썬> ml_predict_runner_v41.py
"""

import threading
import time

import aws_idc_realtime_collector as collector
import hubroom_predictor as predictor

# ML 예측기는 여기서 돌리지 않는다 — 별도 프로세스로 분리했다.
#   Chronos-2 는 chronos-forecasting / torch 가 필요한데 이 파이썬이 아니라
#   별도 파이썬에 설치돼 있어 같은 프로세스로는 import 가 안 된다.
#
#       (룰베이스)  python run_ml.py                        ← 이 파일
#       (ML)        <chronos 파이썬> ml_predict_runner_v41.py

# 로그프레소 이상감지 기입기 (발동이벤트.csv에 BOTTLENECK_/QUEUE_ 4컬럼)
try:
    import LO_LOW_AMOS
    _LP_AVAILABLE = True
except Exception as e:
    print(f'⚠ LO_LOW_AMOS 로드 실패 — 로그프레소 기입 비활성: {e}')
    _LP_AVAILABLE = False

# MAXCAPA 조작내역 기입기 (발동이벤트.csv에 MACHINE/PORT:후(after)/PROCESS/TRANSACTIONID)
try:
    import lo_mac_maxcapa
    _MC_AVAILABLE = True
except Exception as e:
    print(f'⚠ lo_mac_maxcapa 로드 실패 — MAXCAPA 기입 비활성: {e}')
    _MC_AVAILABLE = False

# PIO 반송실패 기입기 (발동이벤트.csv에 {경로}&DEPOSITED_FAIL_CNT&PIOERROR 12컬럼)
#   ICASTAR Oracle 직접 조회 — 환경변수 ORA_USER / ORA_PASS 필요 (ORA_DSN 은 기본값 있음)
try:
    import PIO_DATA_MAKE
    _PIO_AVAILABLE = True
except Exception as e:
    print(f'⚠ PIO_DATA_MAKE 로드 실패 — PIO 기입 비활성: {e}')
    _PIO_AVAILABLE = False

# 영역(FAB)별 분리기 — predict_tobe/fab분리/ 에 영역별 발동이벤트 생성
try:
    import 발동이벤트_영역분리 as area_split
    _AS_AVAILABLE = True
except Exception as e:
    print(f'⚠ 발동이벤트_영역분리 로드 실패 — 영역분리 비활성: {e}')
    _AS_AVAILABLE = False

# 하이브리드는 v4.1 호환 작업 후 별도 활성화 — 일단 비활성 유지
# import hybrid_predictor

# 수집기 (백그라운드 데몬 — 메인 종료 시 같이 죽음)
threading.Thread(target=collector.main, daemon=True).start()

# 하이브리드 예측기 (그대로 비활성 — 별도 작업)
# threading.Thread(target=hybrid_predictor.run_watch, daemon=True).start()

# 0.5초 후 예측기 watch (메인 스레드)
time.sleep(0.5)

out_dir = predictor.DEFAULT_OUTPUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)
logger = predictor.setup_logger(out_dir)

# ★ 제일 마지막: 로그프레소 기입기 (백그라운드 데몬)
#   룰 예측기가 만드는 YYYYMMDD_발동이벤트.csv 에 매분 4컬럼 기입 — 그래서 맨 끝에 시작
#   경로는 룰 예측기가 실제로 쓰는 곳(DEFAULT_OUTPUT_DIR = predict_tobe)을 그대로 사용
#   → run_ml 을 어디서 실행하든 항상 같은 폴더를 봄
if _LP_AVAILABLE:
    threading.Thread(target=LO_LOW_AMOS.run_watch,
                     kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
                     daemon=True).start()

# MAXCAPA 조작내역 기입기 (백그라운드 데몬)
#   로그프레소 dbquery(mcs_m16) 경유로 MCS DB 조회 — 별도 접속설정 불필요
#   Oracle 직접 접속이 필요하면: kwargs 에 source='db' (mcs_config.ini 필요)
if _MC_AVAILABLE:
    threading.Thread(target=lo_mac_maxcapa.run_watch,
                     kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
                     daemon=True).start()

# PIO 반송실패 기입기 (백그라운드 데몬)
#   매분 ICASTAR Oracle 에서 경로별 DEPOSITED_FAIL_CNT 를 조회해 같은 분 행에 12컬럼 기입
#   수집기(00초)·MAXCAPA(+25초)와 접속이 겹치지 않게 기본 35초 늦게 시작
#   ORA_USER / ORA_PASS 가 비어 있으면 접속 실패 로그만 남기고 다음 분에 재시도 (다른 스레드 영향 없음)
if _PIO_AVAILABLE:
    threading.Thread(target=PIO_DATA_MAKE.run_watch,
                     kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
                     daemon=True).start()

# ★★ 진짜 마지막: 영역(FAB)별 분리기 (백그라운드 데몬)
#   위 세 기입기(로그프레소 4 · MAXCAPA 4 · PIO 12컬럼)가 채운 뒤에 돌아야 분리 파일에도 그 값이 들어간다.
#   그래서 기본 45초 지연(lag — PIO 가 +35초에 쓰므로 그 뒤)을 두고, 스레드도 제일 마지막에 시작한다.
#   · 원본(YYYYMMDD_발동이벤트.csv)은 읽기만 한다 — 절대 수정·삭제하지 않음
#   · 출력: predict_tobe/fab분리/YYYYMMDD_발동이벤트_{영역}.csv (5개)
#   · 원본이 바뀐 때만 다시 쓰고, 자정에는 전날 파일도 한 번 더 마무리
if _AS_AVAILABLE:
    threading.Thread(target=area_split.run_watch,
                     kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
                     daemon=True).start()

predictor.run_watch(predictor.DEFAULT_INPUT_CSV, out_dir, logger)

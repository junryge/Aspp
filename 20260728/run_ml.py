# -*- coding: utf-8 -*-
"""
M16A HUBROOM 수집 + 룰 예측 + ML 예측 + 로그프레소 기입 + 하이브리드 동시 실행
- 수집기 스레드 (백그라운드 데몬)
- ML 예측기 스레드 (백그라운드 데몬)
- 로그프레소 기입기 스레드 (백그라운드 데몬) — 발동이벤트.csv에 이상감지 4컬럼
- 하이브리드 예측기 스레드 (백그라운드 데몬) — 룰×ML 양방향 매트릭스 융합
- 룰 예측기 (메인 스레드)
- Ctrl+C 한 번으로 같이 종료
"""

import threading
import time

import aws_idc_realtime_collector as collector
import hubroom_predictor as predictor

# v4.1 ML runner (같은 폴더 통합 — ml/ + Rulebase_prediction/ 합침)
try:
    import ml_predict_runner_v41 as ml_runner
    _ML_AVAILABLE = True
except Exception as e:
    print(f'⚠ ml_predict_runner_v41 로드 실패 — ML 비활성: {e}')
    _ML_AVAILABLE = False

# 로그프레소 이상감지 기입기 (발동이벤트.csv에 BOTTLENECK_/QUEUE_ 4컬럼)
try:
    import LO_LOW_AMOS
    _LP_AVAILABLE = True
except Exception as e:
    print(f'⚠ LO_LOW_AMOS 로드 실패 — 로그프레소 기입 비활성: {e}')
    _LP_AVAILABLE = False

# MAXCAPA 조작내역 기입기 (발동이벤트.csv에 MACHINE/PORT:후(after)/PROCESS/TRANSACTIONID)
try:
    import mcs_maxcapa
    _MC_AVAILABLE = True
except Exception as e:
    print(f'⚠ mcs_maxcapa 로드 실패 — MAXCAPA 기입 비활성: {e}')
    _MC_AVAILABLE = False

# 하이브리드는 v4.1 호환 작업 후 별도 활성화 — 일단 비활성 유지
# import hybrid_predictor

# 수집기 (백그라운드 데몬 — 메인 종료 시 같이 죽음)
threading.Thread(target=collector.main, daemon=True).start()

# ML 예측기 v4.1 (백그라운드 데몬) — 2 lead time (10/30분)
if _ML_AVAILABLE:
    threading.Thread(target=ml_runner.run_watch, daemon=True).start()

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
#   MCS Oracle 직접 조회 — 접속정보는 mcs_config.ini (같은 폴더)
#   DB 대신 수집 CSV 를 쓰려면: kwargs 에 source='csv', maxcapa=r'.\maxcapa_v3.csv'
if _MC_AVAILABLE:
    threading.Thread(target=mcs_maxcapa.run_watch,
                     kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
                     daemon=True).start()

predictor.run_watch(predictor.DEFAULT_INPUT_CSV, out_dir, logger)

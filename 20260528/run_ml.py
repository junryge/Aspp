# -*- coding: utf-8 -*-
"""
M16A HUBROOM 수집 + 룰 예측 + ML 예측 + 하이브리드 동시 실행
- 수집기 스레드 (백그라운드 데몬)
- ML 예측기 스레드 (백그라운드 데몬)
- 하이브리드 예측기 스레드 (백그라운드 데몬) — 룰×ML 양방향 매트릭스 융합
- 룰 예측기 (메인 스레드)
- Ctrl+C 한 번으로 같이 종료
"""

import threading
import time

import aws_idc_realtime_collector as collector
import hubroom_predictor as predictor
# import ml_predict_runner as ml_runner       # ML 업그레이드 중 — 임시 비활성
# import hybrid_predictor                      # ML 업그레이드 중 — 임시 비활성

# 수집기 (백그라운드 데몬 — 메인 종료 시 같이 죽음)
threading.Thread(target=collector.main, daemon=True).start()

# ML 예측기 (백그라운드 데몬) — 업그레이드 중, 임시 비활성
# threading.Thread(target=ml_runner.run_watch, daemon=True).start()

# 하이브리드 예측기 (백그라운드 데몬) — 업그레이드 중, 임시 비활성
# threading.Thread(target=hybrid_predictor.run_watch, daemon=True).start()

# 0.5초 후 예측기 watch (메인 스레드)
time.sleep(0.5)

out_dir = predictor.DEFAULT_OUTPUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)
logger = predictor.setup_logger(out_dir)
predictor.run_watch(predictor.DEFAULT_INPUT_CSV, out_dir, logger)

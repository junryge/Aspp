#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 실시간 예측기 — M16_HUBROOM_PR.csv 1분 폴링 → 날짜별 predictions CSV append

입력:  predict/M16_HUBROOM_PR.csv  (1분마다 자동 갱신되는 슬라이딩 90분 윈도우)
출력:  ml_predict/<YYYYMMDD>_predictions.csv  (날짜별 append, 1분당 1행)

사용법:
    # 모듈로 (run.py에서 import)
    import ml_predict_runner
    ml_predict_runner.run_watch()

    # 단독 실행
    python ml_predict_runner.py [--input PATH] [--out_dir PATH] [--model PATH] [--interval SECONDS]

출력 컬럼 (utf-8-sig, 엑셀 한글 OK):
    datetime, prediction_for, ml_score, ml_level, ml_level_kr,
    rule_s1, rule_s2, rule_s3
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path


# 같은 폴더 모듈 import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from feature_builder import (
    build_features, iter_star_rows, evaluate_rules, WINDOW_MIN,
)
from ml_predictor import MLPredictor


# ==== 기본 경로 (run.py 에서 override 가능) ====
DEFAULT_INPUT_CSV   = _HERE / 'predict' / 'M16_HUBROOM_PR.csv'
DEFAULT_OUTPUT_DIR  = _HERE / 'ml_predict'
DEFAULT_MODEL_PATH  = _HERE / 'model.json'
DEFAULT_INTERVAL    = 60  # 초


def _date_key(t):
    return t.strftime('%Y%m%d')


def _append_prediction(out_dir, t, pred_for, score, level, level_kr,
                       rule_s1, rule_s2, rule_s3):
    """날짜별 CSV append (없으면 헤더 포함 신규 생성)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{_date_key(t)}_predictions.csv'

    new_file = (not path.exists()) or path.stat().st_size == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                'datetime', 'prediction_for',
                'ml_score', 'ml_level', 'ml_level_kr',
                'rule_s1', 'rule_s2', 'rule_s3',
            ])
        w.writerow([
            t.strftime('%Y-%m-%d %H:%M:%S'),
            pred_for.strftime('%Y-%m-%d %H:%M:%S'),
            f'{score:.4f}', level, level_kr,
            int(rule_s1), int(rule_s2), int(rule_s3),
        ])
    return path


def _process_csv_once(input_csv, ml, last_t, log_fn=None):
    """CSV 한 번 읽어서 last_t 이후 새 행만 처리 → (last_t, [예측 dict, ...])"""
    if not os.path.exists(input_csv):
        if log_fn:
            log_fn(f'⚠ CSV 파일 없음: {input_csv}')
        return last_t, []

    t1_window  = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)
    v3_window  = deque(maxlen=WINDOW_MIN)

    new_predictions = []
    total_rows = 0
    skipped_already = 0
    skipped_window = 0

    for t, star, _ in iter_star_rows(input_csv):
        total_rows += 1
        # 윈도우는 모두 누적 (이미 본 시각 포함 — 모델 입력 컨텍스트)
        t1_window.append(star.get('avgtotal1min'))
        m14_window.append(star.get('m14_to_m16'))
        lft_window.append(star.get('lft_list') or {})
        v3_window.append({k: star.get(k) for k in (
            'fabstorage_ratio',
            'm14b_aotransdelay', 'm14b_oht_util', 'm14b_4abld122',
            'm14b_avgtotal1min', 'm14b_7f_to_hub', 'm14b_7f_to_hub_alt',
            'm14_htstop', 'm14_congested', 'm14_abnormal',
            'm16pkt_aotransdelay', 'm16wt_aotransdelay',
        )})

        # 이미 처리한 시각은 출력 스킵 (윈도우는 채워둠)
        if last_t is not None and t <= last_t:
            skipped_already += 1
            continue
        if len(t1_window) < 31:
            skipped_window += 1
            continue

        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window, v3_window)
        features = build_features(t1_window, m14_window, lft_window, v3_window, ctx)
        score = ml.predict(features)
        level = MLPredictor.level(score)
        level_kr = MLPredictor.level_korean(score)

        new_predictions.append({
            'time': t,
            'pred_for': t + timedelta(minutes=30),
            'score': score, 'level': level, 'level_kr': level_kr,
            's1': s1, 's2': s2, 's3': s3,
        })
        last_t = t

    if log_fn:
        log_fn(f'  📊 CSV 읽음: 총 {total_rows}행, '
               f'이미 처리 {skipped_already}, 윈도우 부족 {skipped_window}, '
               f'신규 예측 {len(new_predictions)}')
        if total_rows < 31:
            log_fn(f'  ⚠ CSV 행수 {total_rows} < 31 — 윈도우 채워질 때까지 대기')
        elif total_rows == 0:
            log_fn(f'  ⚠ CSV 비어있음 — iter_star_rows 가 0행 반환 (prefix 감지 실패 가능)')

    return last_t, new_predictions


def run_watch(input_csv=DEFAULT_INPUT_CSV,
              out_dir=DEFAULT_OUTPUT_DIR,
              model_path=DEFAULT_MODEL_PATH,
              interval=DEFAULT_INTERVAL,
              logger=None):
    """폴링 루프 (run.py 에서 스레드로 호출)"""
    input_csv  = Path(input_csv)
    out_dir    = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg, flush=True)

    if not model_path.exists():
        _log(f'❌ 모델 없음: {model_path} — ML 예측기 종료')
        return

    _log(f'🤖 ML 예측기 시작')
    _log(f'   입력: {input_csv}')
    _log(f'   출력: {out_dir}')
    _log(f'   모델: {model_path}')
    _log(f'   폴링: {interval}초')

    ml = MLPredictor(str(model_path))

    last_size = -1   # -1 로 시작 → 첫 루프에서 무조건 처리
    last_t = None
    first_run = True

    while True:
        try:
            if input_csv.exists():
                cur_size = input_csv.stat().st_size
                if cur_size != last_size:
                    if first_run:
                        _log(f'🔍 초기 CSV 스캔 ({cur_size:,} bytes)')
                        first_run = False
                    last_t, preds = _process_csv_once(str(input_csv), ml, last_t, log_fn=_log)
                    for p in preds:
                        path = _append_prediction(
                            out_dir, p['time'], p['pred_for'],
                            p['score'], p['level'], p['level_kr'],
                            p['s1'], p['s2'], p['s3'],
                        )
                        # 경보/위험만 로그
                        if p['score'] >= 0.7:
                            _log(f'  [{p["time"].strftime("%H:%M")}] '
                                 f'{p["level_kr"]} score={p["score"]:.3f} '
                                 f'→ 30분 뒤({p["pred_for"].strftime("%H:%M")}) 정체 예상  '
                                 f'[{path.name}]')
                    if preds:
                        _log(f'  💾 {len(preds)}건 → {out_dir}/{preds[-1]["time"].strftime("%Y%m%d")}_predictions.csv')
                    last_size = cur_size
            else:
                _log(f'⚠ 입력 CSV 없음: {input_csv} (생성 대기)')
            time.sleep(interval)
        except KeyboardInterrupt:
            _log('🛑 ML 예측기 종료')
            break
        except Exception as e:
            import traceback
            _log(f'⚠ ML 예측기 오류: {e}\n{traceback.format_exc()}')
            time.sleep(interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',    default=str(DEFAULT_INPUT_CSV),  help='입력 CSV (자동 갱신되는 90분 윈도우)')
    p.add_argument('--out_dir',  default=str(DEFAULT_OUTPUT_DIR), help='날짜별 predictions CSV 출력 폴더')
    p.add_argument('--model',    default=str(DEFAULT_MODEL_PATH), help='학습된 model.json 경로')
    p.add_argument('--interval', type=int, default=DEFAULT_INTERVAL, help='폴링 간격(초)')
    args = p.parse_args()

    run_watch(args.input, args.out_dir, args.model, args.interval)


if __name__ == '__main__':
    main()

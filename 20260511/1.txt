#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한 번에 처리: 원본 CSV → 피처 추출 → ML 예측 → predictions.csv

사용법:
    python predict_all.py --csv test_data.csv --model model.json --out predictions.csv

특징:
  · feature_builder.py + ml_predictor.py 를 내부에서 한 번에 실행
  · 중간 features.csv 파일 안 만듦 (메모리에서 처리)
  · 출력 CSV는 utf-8-sig (엑셀에서 한글 안 깨짐)
  · 룰 단계도 같이 출력 (rule_s1/s2/s3)

출력 컬럼:
  timestamp, ml_score (0~1), ml_level (OK/INFO/WARNING/CRITICAL),
  ml_level_kr (정상/주의/경보/위험), prediction_for (t+30분 시각),
  rule_s1, rule_s2, rule_s3
"""

import argparse
import csv
import os
import sys
from collections import deque
from datetime import timedelta


# 같은 폴더의 feature_builder, ml_predictor import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_builder import (
    build_features, iter_star_rows, evaluate_rules, WINDOW_MIN,
    _resolve_csv_path,
)
from ml_predictor import MLPredictor


def predict_all(input_csv, model_path, out_path):
    print(f'📥 입력: {input_csv}')
    print(f'📥 모델: {model_path}')
    print(f'📤 출력: {out_path}')
    print()

    # ML 모델 로드
    ml = MLPredictor(model_path)
    feature_names = ml.feature_names

    # 슬라이딩 윈도우 + 처리
    t1_window  = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)
    v3_window  = deque(maxlen=WINDOW_MIN)

    rows_out = []
    skipped = 0

    for t, star, _ in iter_star_rows(input_csv):
        t1_window.append(star.get('avgtotal1min'))
        m14_window.append(star.get('m14_to_m16'))
        lft_window.append(star.get('lft_list') or {})
        v3_window.append({k: star.get(k) for k in (
            'fabstorage_ratio',
            'm14b_aotransdelay', 'm14b_oht_util', 'm14b_4abld122',
            'm14b_avgtotal1min', 'm14b_7f_to_hub', 'm14b_7f_to_hub_alt',
            'm14_htstop', 'm14_congested', 'm14_abnormal',
            'm16pkt_aotransdelay', 'm16wt_aotransdelay',
            # ★ v3.1 신규 5개 컬럼 (학습된 70개 피처와 호환)
            'hub_storage_util', 'm14_inflow',
            'm16a_2f_inflow', 'm16a_6f_inflow', 'm16b_10f_inflow',
        )})

        # 윈도우 31개 미만이면 룰 평가 불가
        if len(t1_window) < 31:
            skipped += 1
            continue

        # 룰 평가
        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window, v3_window)

        # 피처 생성
        features = build_features(t1_window, m14_window, lft_window, v3_window, ctx)

        # ML 예측
        score = ml.predict(features)
        level = MLPredictor.level(score)
        level_kr = MLPredictor.level_korean(score)

        rows_out.append({
            'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_for': (t + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
            'ml_score': f'{score:.4f}',
            'ml_level': level,
            'ml_level_kr': level_kr,
            'rule_s1': int(s1),
            'rule_s2': int(s2),
            'rule_s3': int(s3),
            # ★ v3.1 위험도 평가
            'risk_score': ctx.get('risk_score', 0),
            'risk_level': ctx.get('risk_level', '정상'),
            'risk_factors': ctx.get('risk_factors', ''),
        })

    if not rows_out:
        sys.exit('❌ 예측 결과 0건 — 입력 데이터 확인 필요')

    # CSV 저장 (한글 안 깨지게 utf-8-sig)
    fieldnames = [
        'timestamp', 'prediction_for',
        'ml_score', 'ml_level', 'ml_level_kr',
        'rule_s1', 'rule_s2', 'rule_s3',
        'risk_score', 'risk_level', 'risk_factors',
    ]
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    # 통계 출력
    from collections import Counter
    levels = Counter(r['ml_level'] for r in rows_out)
    high_scores = sum(1 for r in rows_out if float(r['ml_score']) >= 0.7)
    critical = sum(1 for r in rows_out if float(r['ml_score']) >= 0.85)

    print(f'✅ {len(rows_out):,} 분 예측 완료')
    print(f'   (윈도우 부족으로 건너뛴 초반 {skipped} 분 제외)')
    print()
    print(f'📊 위험 분포:')
    for lvl in ('OK', 'INFO', 'WARNING', 'CRITICAL'):
        n = levels.get(lvl, 0)
        pct = n / len(rows_out) * 100
        print(f'   {lvl:<10} {n:>6,} ({pct:5.2f}%)')
    print()
    print(f'⚠ 경보(score≥0.7): {high_scores:,} 분')
    print(f'🚨 위험(score≥0.85): {critical:,} 분')
    print()
    print(f'💾 → {out_path} (엑셀에서 한글 정상 표시)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='원본 1분 단위 CSV (학습 안 한 새 데이터)')
    p.add_argument('--model', default='model.json', help='학습된 XGBoost 모델 (기본 model.json)')
    p.add_argument('--out', default='predictions.csv', help='출력 CSV (기본 predictions.csv)')
    args = p.parse_args()

    csv_path = _resolve_csv_path(args.csv)

    if not os.path.exists(args.model):
        sys.exit(f'❌ 모델 파일 없음: {args.model}\n   학습을 먼저 진행하세요 (train_xgboost.py)')

    predict_all(csv_path, args.model, args.out)


if __name__ == '__main__':
    main()

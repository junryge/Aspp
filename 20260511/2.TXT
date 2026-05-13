#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 피처 추출기 — 90min.csv → 피처 데이터프레임

사용법:
    python3 feature_builder.py --csv ../DATA/90min.csv --out features.csv

피처 그룹:
  · 현재값 (4가지 핵심 컬럼)
  · 슬라이딩 통계 (5/10/30분 max/mean/std)
  · 변화율 (5/10/30분 delta) — 모멘텀
  · 가속도 (변화의 변화)
  · 룰 컨텍스트 (R-A'/R-B/R-C'/R-D flag와 값)
  · 조합 피처 (interaction)

출력: features.csv (timestamp + 피처들)
"""

import argparse
import csv
import os
import sys
from collections import deque
from datetime import datetime
from statistics import mean, stdev


# 부모 디렉터리 추가하여 룰 엔진 재사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from importlib import import_module
import importlib.util

# 3단계_룰베이스_사건단위.py 위치 자동 탐색
# (같은 폴더 → 상위 폴더 → 현재 작업 디렉터리 순서)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_this_dir, '3단계_룰베이스_사건단위.py'),          # 같은 폴더
    os.path.join(_this_dir, '..', '3단계_룰베이스_사건단위.py'),     # 상위 폴더
    os.path.join(os.getcwd(), '3단계_룰베이스_사건단위.py'),         # 현재 작업 디렉터리
]
_rule_path = None
for _p in _candidates:
    if os.path.exists(_p):
        _rule_path = _p
        break

if _rule_path is None:
    raise FileNotFoundError(
        f"3단계_룰베이스_사건단위.py 를 찾을 수 없습니다.\n"
        f"다음 위치 중 하나에 두세요:\n" + '\n'.join(f"  - {p}" for p in _candidates)
    )

_spec = importlib.util.spec_from_file_location('rule_engine', _rule_path)
_rule_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rule_engine)

iter_star_rows = _rule_engine.iter_star_rows
evaluate_rules = _rule_engine.evaluate_rules
WINDOW_MIN = _rule_engine.WINDOW_MIN


def _safe_max(seq, default=0):
    vals = [v for v in seq if v is not None]
    return max(vals) if vals else default


def _safe_mean(seq, default=0):
    vals = [v for v in seq if v is not None]
    return mean(vals) if vals else default


def _safe_std(seq, default=0):
    vals = [v for v in seq if v is not None]
    return stdev(vals) if len(vals) >= 2 else default


def _safe_delta(seq, k, default=0):
    """seq[-1] - seq[-k] (둘 다 not None)"""
    if len(seq) < k:
        return default
    a, b = seq[-1], seq[-k]
    if a is None or b is None:
        return default
    return a - b


def _v3_field(v3_window, key):
    """v3 윈도우의 최신 값"""
    if not v3_window:
        return 0
    latest = v3_window[-1]
    return latest.get(key) or 0


def _v3_series(v3_window, key, n):
    """v3 윈도우 최근 n개에서 key 값 시퀀스"""
    return [(d.get(key) or 0) for d in list(v3_window)[-n:]]


def build_features(t1_window, m14_window, lft_window, v3_window, rule_ctx):
    """1 시점 피처 dict 생성. 룰과 같은 윈도우 공유."""
    f = {}

    # ──────────────────── 1. 현재값 ────────────────────
    f['1min_now']           = (t1_window[-1] or 0) if t1_window else 0
    f['m14_now']            = (m14_window[-1] or 0) if m14_window else 0
    f['fabstorage_now']     = _v3_field(v3_window, 'fabstorage_ratio')
    f['m14b_oht_util']      = _v3_field(v3_window, 'm14b_oht_util')
    f['m14b_4abld122']      = _v3_field(v3_window, 'm14b_4abld122')
    f['m14b_7f_to_hub']     = _v3_field(v3_window, 'm14b_7f_to_hub')
    f['m14b_7f_to_hub_alt'] = _v3_field(v3_window, 'm14b_7f_to_hub_alt')

    # 리프터 합/std (현재)
    lft_now = lft_window[-1] if lft_window else {}
    lft_vals = list(lft_now.values()) if lft_now else [0]
    f['lft_sum_now']    = sum(lft_vals)
    f['lft_max_now']    = max(lft_vals) if lft_vals else 0
    f['lft_std_now']    = stdev(lft_vals) if len(lft_vals) >= 2 else 0

    # ──────────────────── 2. 슬라이딩 통계 ────────────────────
    t1_list  = list(t1_window)
    m14_list = list(m14_window)

    f['1min_max_5m']    = _safe_max(t1_list[-5:])
    f['1min_max_10m']   = _safe_max(t1_list[-10:])
    f['1min_max_30m']   = _safe_max(t1_list[-30:])
    f['1min_mean_30m']  = _safe_mean(t1_list[-30:])
    f['1min_std_30m']   = _safe_std(t1_list[-30:])

    f['m14_max_30m']    = _safe_max(m14_list[-30:])
    f['m14_mean_30m']   = _safe_mean(m14_list[-30:])

    fab30 = _v3_series(v3_window, 'fabstorage_ratio', 30)
    f['fabstorage_max_30m']  = max(fab30) if fab30 else 0
    f['fabstorage_mean_30m'] = mean(fab30) if fab30 else 0
    f['fabstorage_std_30m']  = stdev(fab30) if len(fab30) >= 2 else 0

    # ──────────────────── 3. 변화율 (모멘텀) ★ ────────────────────
    f['1min_delta_5m']  = _safe_delta(t1_list, 5)
    f['1min_delta_10m'] = _safe_delta(t1_list, 10)
    f['1min_delta_30m'] = _safe_delta(t1_list, 30)

    f['m14_delta_10m']  = _safe_delta(m14_list, 11)
    f['m14_delta_30m']  = _safe_delta(m14_list, 31)

    fab10 = _v3_series(v3_window, 'fabstorage_ratio', 11)
    fab30s = _v3_series(v3_window, 'fabstorage_ratio', 31)
    f['fabstorage_delta_10m'] = (fab10[-1] - fab10[0]) if len(fab10) >= 2 else 0
    f['fabstorage_delta_30m'] = (fab30s[-1] - fab30s[0]) if len(fab30s) >= 2 else 0

    util30 = _v3_series(v3_window, 'm14b_oht_util', 31)
    f['m14b_util_delta_30m'] = (util30[-1] - util30[0]) if len(util30) >= 2 else 0

    abld122_30 = _v3_series(v3_window, 'm14b_4abld122', 31)
    f['m14b_4abld122_delta_30m'] = (abld122_30[-1] - abld122_30[0]) if len(abld122_30) >= 2 else 0

    # ──────────────────── 4. 가속도 (변화의 변화) ────────────────────
    if len(t1_list) >= 11:
        d1 = (t1_list[-1] or 0) - (t1_list[-6] or 0)
        d2 = (t1_list[-6] or 0) - (t1_list[-11] or 0)
        f['1min_accel_5m'] = d1 - d2
    else:
        f['1min_accel_5m'] = 0

    if len(fab30s) >= 11:
        d1 = fab30s[-1] - fab30s[-6]
        d2 = fab30s[-6] - fab30s[-11]
        f['fabstorage_accel_5m'] = d1 - d2
    else:
        f['fabstorage_accel_5m'] = 0

    # ──────────────────── 5. 룰 컨텍스트 ────────────────────
    f['rule_ra_count']      = rule_ctx.get('ra_count', 0)
    f['rule_ra_value']      = rule_ctx.get('ra_value') or 0
    f['rule_ra_sustained']  = int(rule_ctx.get('ra_sustained', False))
    f['rule_ra_trig']       = int(rule_ctx.get('ra_trig', False))
    f['rule_rb_diff']       = rule_ctx.get('rb_diff', 0)
    f['rule_rb_diff_10']    = rule_ctx.get('rb_diff_10', 0)
    f['rule_rb_trig']       = int(rule_ctx.get('rb_trig', False))
    f['rule_rb_fast']       = int(rule_ctx.get('rb_fast', False))
    f['rule_rc_trend']      = rule_ctx.get('rc_trend', 0)
    f['rule_rev_count']     = rule_ctx.get('rev_count', 0)
    f['rule_rc_trig']       = int(rule_ctx.get('rc_trig', False))
    f['rule_rd_fabstorage'] = rule_ctx.get('rd_fabstorage', 0)
    f['rule_rd_trig']       = int(rule_ctx.get('rd_trig', False))

    # ──────────────────── 6. 조합 (interaction) ────────────────────
    f['1min_x_fabstorage']   = f['1min_now'] * f['fabstorage_now']
    f['1min_x_revcount']     = f['1min_now'] * f['rule_rev_count']
    f['m14_x_fabstorage']    = f['m14_now'] * f['fabstorage_now']
    f['m14b_util_x_fab']     = f['m14b_oht_util'] * f['fabstorage_now']
    f['fab_x_revcount']      = f['fabstorage_now'] * f['rule_rev_count']

    return f


def process_csv(csv_path, out_path):
    """CSV 전체 처리 → 피처 데이터프레임 저장"""
    t1_window  = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)
    v3_window  = deque(maxlen=WINDOW_MIN)

    feature_rows = []

    for t, star, _ in iter_star_rows(csv_path):
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

        # 윈도우 31개 미만이면 룰 평가 보류
        if len(t1_window) < 31:
            continue

        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window, v3_window)
        features = build_features(t1_window, m14_window, lft_window, v3_window, ctx)
        features['timestamp'] = t.strftime('%Y-%m-%d %H:%M:%S')
        features['_rule_s1'] = int(s1)
        features['_rule_s2'] = int(s2)
        features['_rule_s3'] = int(s3)

        feature_rows.append(features)

    if not feature_rows:
        print('❌ 피처 0건')
        return

    fieldnames = ['timestamp'] + [k for k in feature_rows[0].keys() if k != 'timestamp']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(feature_rows)

    print(f'✅ {len(feature_rows):,} 행 → {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='입력 90min.csv 경로')
    p.add_argument('--out', default='features.csv', help='출력 피처 CSV 경로')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f'파일 없음: {args.csv}')

    process_csv(args.csv, args.out)


if __name__ == '__main__':
    main()

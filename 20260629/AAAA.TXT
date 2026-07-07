#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgb_비정상_infer — XGBoost 비정상 모델로 매분 정체확률 생성
====================================================================
학습된 XGBoost(xgb_비정상_train) 로 매분 jam_probability 를 뽑아
하이브리드 판정의 '비정상 확인관' 입력으로 사용.

★ 롤링/델타 피처는 학습과 100% 동일하게 생성 (아래 상수/함수 train 과 일치해야 함).
★ 누수 없음: 롤링/델타는 과거만 사용.

입력:
    --features  features.csv
    --model     out_ml/xgb (model.json + feature_cols.json)
    --out       기본 ./out_ml/jam_prob.csv

실행:
    python xgb_비정상_infer.py --features ./out_ml/features.csv --model ./out_ml/xgb

출력:
    jam_prob.csv — datetime, jam_probability[0~1], jam_level
    jam_level: 안전<0.3 / 관심 / 경계0.5~ / 정체≥0.7
"""
import argparse
import csv
import json
import os
import sys

# ★ train 과 반드시 동일 (60분 흐름)
ROLL_WINS = [15, 30, 60]
DELTA_LAGS = [15, 30, 60]
MAX_WINS = [60]


def _need():
    try:
        import numpy, pandas, xgboost  # noqa
        return True
    except Exception as e:
        print("⚠️ XGBoost 라이브러리 없음 — pip install xgboost pandas numpy")
        print(f"   ({type(e).__name__}: {e})")
        return False


def build_features(feat_df, base_cols):
    import pandas as pd
    df = feat_df.sort_values('datetime').reset_index(drop=True)
    df[base_cols] = df[base_cols].ffill().fillna(0.0)
    new = {}
    for c in base_cols:
        s = df[c].astype(float)
        for w in ROLL_WINS:
            new[f'{c}__rmean{w}'] = s.rolling(w, min_periods=1).mean()
            new[f'{c}__rstd{w}'] = s.rolling(w, min_periods=1).std().fillna(0.0)
        for w in MAX_WINS:
            new[f'{c}__rmax{w}'] = s.rolling(w, min_periods=1).max()
        for lag in DELTA_LAGS:
            new[f'{c}__d{lag}'] = s - s.shift(lag).fillna(s.iloc[0])
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def level(p, tag=''):
    """6월 검증 캘리브레이션(정밀도) 근거 확률컷.
    10분: 신뢰도 잘 나뉨(74/79/96%) → 3등급.
    30분: 신뢰도 ~77%에서 평평 → 단일 경보(위험)만."""
    if '30' in str(tag):
        return '위험' if p >= 0.50 else ''            # 30분: 신뢰 77% 단일 경보
    return ('초위험' if p >= 0.90                      # 10분: 신뢰 96%
            else '위험' if p >= 0.70                   #        신뢰 79%
            else '경계' if p >= 0.50 else '')          #        신뢰 74%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--model', default='./out_ml/xgb')
    ap.add_argument('--out', default='./out_ml/jam_prob.csv')
    a = ap.parse_args()

    print("=" * 60)
    print("XGBoost 비정상 추론 — 매분 정체확률")
    print("=" * 60)
    if not _need():
        sys.exit(2)

    import pandas as pd
    import xgboost as xgb

    feat_cols = json.load(open(os.path.join(a.model, 'feature_cols.json'), encoding='utf-8'))
    # 지평선별 모델 로드 (model_y_pre10.json, model_y_pre30.json) — 없으면 구 model.json
    import glob
    mfiles = sorted(glob.glob(os.path.join(a.model, 'model_y_pre*.json')))
    if not mfiles and os.path.exists(os.path.join(a.model, 'model.json')):
        mfiles = [os.path.join(a.model, 'model.json')]
    if not mfiles:
        print("⚠️ 모델 파일 없음 (model_y_pre*.json)"); sys.exit(3)
    models = {}
    for mf in mfiles:
        tag = os.path.basename(mf).replace('model_', '').replace('.json', '')  # y_pre10 / y_pre30
        m = xgb.XGBClassifier(); m.load_model(mf)
        models[tag] = m
    print(f"[모델] {list(models.keys())} 로드")

    feat = pd.read_csv(a.features, encoding='utf-8-sig')
    feat['datetime'] = pd.to_datetime(feat['datetime'])
    base_cols = [c for c in feat.columns if c != 'datetime']
    df = build_features(feat, base_cols)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"⚠️ 피처 불일치 {len(missing)}개 (train/infer 상수 확인): {missing[:5]}...")
        sys.exit(3)
    X = df[feat_cols].values
    probs = {tag: m.predict_proba(X)[:, 1] for tag, m in models.items()}
    tags = list(models.keys())
    # 지평선(분) 추출: y_pre10 → 10, y_pre30 → 30  → 예측 대상시각 = 현재 + 지평선
    import re
    from datetime import timedelta
    horizon = {tg: int(re.sub(r'\D', '', tg) or 0) for tg in tags}

    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        # datetime(예측시점) + 각 모델: 예측대상시각 + 확률 + 등급
        header = ['datetime']
        for tg in tags:
            header += [f'{tg}_예측시각', f'{tg}_prob', f'{tg}_level']
        w.writerow(header)
        for i, t in enumerate(df['datetime']):
            row = [t.strftime('%Y-%m-%d %H:%M')]
            for tg in tags:
                tgt = (t + timedelta(minutes=horizon[tg])).strftime('%Y-%m-%d %H:%M')
                row += [tgt, f'{probs[tg][i]:.4f}', level(float(probs[tg][i]), tg)]
            w.writerow(row)

    print(f"[완료] {len(df)}분 → {a.out}")
    for tg in tags:
        hi = int((probs[tg] >= 0.7).sum())
        print(f"       {tg}: 정체(≥0.7) {hi}분 ({hi/len(df)*100:.1f}%)")
    print("다음: 하이브리드_판정.py 에서 룰·정상TSPulse·이 확률 종합")


if __name__ == '__main__':
    main()

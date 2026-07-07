#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgb_비정상_train — XGBoost 비정상(정체) 판별 모델 (비정상 TSPulse 백업)
====================================================================
하이브리드 판정의 '비정상 확인관'을 XGBoost 로 구현.
비정상 TSPulse 가 정체 데이터 파편화(56건·종류 제각각)로 잘 안 될 때의 백업.

★ 판별(discriminative) 방식: 정체 vs 정상 '경계'만 학습 → 적은·파편 데이터에 강함.
★ 라벨 = 메신저 정체 30분전 윈도우 (y_pre30) → 30분 사전예측 목표와 일치.
★ XGBoost 는 시계열 네이티브 아님 → 롤링/델타 피처를 만들어 넣음 (누수 없이 과거만).

전제(회사 PC): pip install xgboost pandas numpy scikit-learn   (torch 불필요 = 경량)

입력:
    --features  features.csv (features_31.py 산출: datetime + 31피처)
    --labels    labels.csv   (labels_채점지.py 산출: y_pre30, is_normal)
    --out       모델/리포트 폴더 (기본 ./out_ml/xgb)

실행:
    python xgb_비정상_train.py --features ./out_ml/features.csv --labels ./out_ml/labels.csv

출력:
    out_ml/xgb/model.json        학습된 XGBoost
    out_ml/xgb/feature_cols.json 피처 순서(롤링/델타 포함)
    out_ml/xgb/importance.csv    피처 중요도
    콘솔: PR-AUC · 정밀도/재현율 (시간분할 검증)
"""
import argparse
import json
import os
import sys

ROLL_WINS = [15, 30, 60]    # 롤링 평균/표준편차 창(분) — 60분 흐름 포함
DELTA_LAGS = [15, 30, 60]   # 델타(현재 − N분전) — 60분 추세
MAX_WINS = [60]             # 60분 내 최대치(얼마나 튀었나 = 급증 포착)


def _need():
    try:
        import numpy, pandas, xgboost, sklearn  # noqa
        return True
    except Exception as e:
        print("=" * 60)
        print("⚠️ XGBoost 라이브러리 없음 — 회사 PC 에서:")
        print("   pip install xgboost pandas numpy scikit-learn")
        print(f"   ({type(e).__name__}: {e})")
        print("=" * 60)
        return False


def build_features(feat_df, base_cols):
    """31 원피처 → + 롤링평균/표준편차 + 델타 (전부 과거만 → 누수 없음)."""
    import pandas as pd  # noqa
    df = feat_df.sort_values('datetime').reset_index(drop=True)
    df[base_cols] = df[base_cols].ffill().fillna(0.0)
    new = {}
    for c in base_cols:
        s = df[c].astype(float)
        for w in ROLL_WINS:
            new[f'{c}__rmean{w}'] = s.rolling(w, min_periods=1).mean()
            new[f'{c}__rstd{w}'] = s.rolling(w, min_periods=1).std().fillna(0.0)
        for w in MAX_WINS:
            new[f'{c}__rmax{w}'] = s.rolling(w, min_periods=1).max()   # 60분 내 피크(급증)
        for lag in DELTA_LAGS:
            new[f'{c}__d{lag}'] = s - s.shift(lag).fillna(s.iloc[0])
    import pandas as pd
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    feat_cols = list(base_cols) + list(new.keys())
    return df, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--test_ratio', type=float, default=0.25,
                    help='뒤쪽 비율을 시간분할 검증셋으로 (기본 0.25)')
    ap.add_argument('--out', default='./out_ml/xgb')
    a = ap.parse_args()

    print("=" * 60)
    print("XGBoost 비정상(정체) 판별 모델 학습")
    print("=" * 60)
    if not _need():
        sys.exit(2)

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    os.makedirs(a.out, exist_ok=True)
    feat = pd.read_csv(a.features, encoding='utf-8-sig')
    lab = pd.read_csv(a.labels, encoding='utf-8-sig')
    feat['datetime'] = pd.to_datetime(feat['datetime'])
    lab['datetime'] = pd.to_datetime(lab['datetime'])
    base_cols = [c for c in feat.columns if c != 'datetime']

    # 롤링/델타/최대 피처 (60분 흐름) — 공통
    df, feat_cols = build_features(feat, base_cols)
    print(f"[데이터] {len(df)}분 × {len(feat_cols)}피처 (원 {len(base_cols)} + 60분 롤링/델타/최대)")

    # 학습할 타겟 자동 선택: labels.csv 에 있는 y_pre* 컬럼 전부 (10분·30분 등)
    TARGETS = [c for c in ['y_pre10', 'y_pre30'] if c in lab.columns]
    if not TARGETS:
        TARGETS = [c for c in lab.columns if c.startswith('y_pre')]
    if not TARGETS:
        print("⚠️ labels.csv 에 y_pre10/y_pre30 컬럼 없음"); sys.exit(3)
    df = df.merge(lab[['datetime'] + TARGETS], on='datetime', how='left')
    for t in TARGETS:
        df[t] = df[t].fillna(0).astype(int)

    # 공통 피처저장 (두 모델 동일 피처)
    with open(os.path.join(a.out, 'feature_cols.json'), 'w', encoding='utf-8') as f:
        json.dump(feat_cols, f, ensure_ascii=False)

    # 시간분할 (뒤쪽 = 검증) — 누수 차단, 두 타겟 공통 분할
    n = len(df)
    cut = int(n * (1 - a.test_ratio))
    tr, te = df.iloc[:cut], df.iloc[cut:]
    Xtr = tr[feat_cols].values
    Xte = te[feat_cols].values

    print("\n" + "=" * 60)
    for tgt in TARGETS:
        horizon = tgt.replace('y_pre', '') + '분 후'
        ytr, yte = tr[tgt].values, te[tgt].values
        pos, neg = int(ytr.sum()), int((ytr == 0).sum())
        spw = neg / max(pos, 1)
        print(f"\n■ [{tgt}] {horizon} 예측 모델")
        print(f"   양성 {int(df[tgt].sum())}분({df[tgt].mean()*100:.1f}%) · "
              f"학습양성{pos}/검증양성{int(yte.sum())} · scale_pos_weight={spw:.1f}")
        if pos == 0 or int(yte.sum()) == 0:
            print(f"   ⚠️ 양성 0 — {tgt} 건너뜀"); continue

        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric='aucpr', n_jobs=4, random_state=0,
        )
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

        p = model.predict_proba(Xte)[:, 1]
        prauc = average_precision_score(yte, p)
        yhat = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, yhat, average='binary', zero_division=0)
        print(f"   [검증] PR-AUC {prauc:.3f} · 정밀도(0.5) {pr:.2f} · 재현율 {rc:.2f} · F1 {f1:.2f}")

        # ── 등급 캘리브레이션: 확률컷별 '정밀도(신뢰도)'만 (등급 = 이 예측 얼마나 믿나) ──
        #    재현율은 개별 예측의 등급과 무관(예측 순간 알 수 없음) → 등급 기준에서 제외.
        print(f"   [등급 캘리브레이션] {tgt}  — 확률컷별 정밀도(신뢰도)")
        print(f"      확률컷   예측건수   정밀도(맞을확률)")
        calib = []
        for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            yh = (p >= thr)
            npred = int(yh.sum())
            tp = int((yh & (yte == 1)).sum())
            prc = tp / npred if npred else 0.0
            calib.append((thr, npred, prc))
            print(f"      ≥{thr:.2f}    {npred:6d}      {prc*100:5.1f}%")
        # 정밀도(신뢰도) 기준 등급 자동 제안 (경계≥60% / 위험≥80% / 초위험≥95%)
        def cut_for(target_pr):
            for thr, npred, prc in calib:
                if prc >= target_pr and npred > 0:
                    return thr
            return None
        c_b, c_w, c_c = cut_for(0.60), cut_for(0.80), cut_for(0.95)
        print(f"      → 제안 등급컷: 경계 p≥{c_b}(신뢰60%) / 위험 p≥{c_w}(80%) / 초위험 p≥{c_c}(95%)")

        model.save_model(os.path.join(a.out, f'model_{tgt}.json'))
        imp = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
        with open(os.path.join(a.out, f'importance_{tgt}.csv'), 'w', encoding='utf-8-sig') as f:
            f.write('feature,importance\n')
            for name, v in imp:
                f.write(f'{name},{v:.5f}\n')
        print(f"   ▶ top5: " + ", ".join(f"{name}({v:.2f})" for name, v in imp[:5]))
        print(f"   → {a.out}/model_{tgt}.json")

    print(f"\n🎉 완료 → {a.out}/  (model_y_pre10.json, model_y_pre30.json)")
    print("다음: xgb_비정상_infer.py 로 10분·30분 확률 생성 → 하이브리드 판정")


if __name__ == '__main__':
    main()

import argparse
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# 🎯 V9 — HUB 총큐(M16HUB.QUE.ALL.CURRENTQCNT) 10분후 변화량 예측
# ------------------------------------------------------------
# V8 대비 변경:
#   · 타겟: CURRENT_M16A_3F_JOB_2 → M16HUB.QUE.ALL.CURRENTQCNT (HUB 총큐)
#   · 피처: 원본(M16A_HUBROOM_PR, 266컬럼)에 실존하는 컬럼으로 재구성
#           (V8의 HUBROOMTOTAL/M16A_3F_* 등 18개는 이 원본에 없음 → 제외)
#   · 검증: 랜덤분할 → 시간분할(뒤 20%)  ← 시계열 누수 차단
#   · 입력: 일별 원본 폴더(M16A_HUBROOM_PR_*.csv) 통째로 (2달치)
#   · 목적: CUSUM 경보 시점에 "10분 뒤 큐가 더 오르나?"를 예측 → 오탐 필터
# ============================================================

TARGET_COL = 'M16HUB.QUE.ALL.CURRENTQCNT'
LOOKBACK = 30        # 과거 창(분)
HORIZON = 10         # 예측 지평(분)
BIG_CHANGE = 50      # 가중치 3배 기준 (V8 동일; 4/1 실측 8.3% 해당)

FEATURE_COLS = {
    # 타겟 자신 + HUB 큐 수급 (핵심)
    'que_hub': ['M16HUB.QUE.ALL.CURRENTQCNT',
                'M16HUB.QUE.ALL.CURRENTQCREATED',
                'M16HUB.QUE.ALL.CURRENTQCOMPLETED',
                'M16HUB.QUE.ALL.FABTRANSJOBCNT',
                'M16HUB.QUE.TIME.AVGTOTALTIME',
                'M16HUB.QUE.TIME.AVGTOTALTIME1MIN'],
    # HUB OHT
    'oht_hub': ['M16HUB.QUE.OHT.CURRENTOHTQCNT',
                'M16HUB.QUE.OHT.OHTUTIL'],
    # M16A 큐
    'que_m16a': ['M16A.QUE.ALL.CURRENTQCOMPLETED',
                 'M16A.QUE.ALL.CURRENTQCREATED',
                 'M16A.QUE.OHT.CURRENTOHTQCNT',
                 'M16A.QUE.OHT.OHTUTIL',
                 'M16A.QUE.LOAD.AVGLOADTIME1MIN',
                 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT'],
    # 방향성 컨베이어 (6월 밀림검증에서 확정된 신호)
    'direction': ['M14.QUE.CNV.SOUTHCURRENTQCNT',
                  'M14.QUE.CNV.NORTHCURRENTQCNT',
                  'M14.QUE.CNV.SOUTHCNVTOM14TIME1MIN',
                  'M14.QUE.CNV.NORTHCNVTOM14TIME1MIN'],
    # 저장/SLA
    'storage': ['M16HUB.STRATE.ALL.FABSTORAGERATIO',
                'M16HUB.STRATE.STK.STORAGERATIO',
                'M16HUB.STRATE.STB.3F_STORAGE_UTIL'],
    'sla': ['M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO'],
}
# 30분 창 풀통계(mean/std/max/min/last/last5/slope) 그룹 — 나머지는 경량(mean/last/slope)
FULL_STAT_GROUPS = {'que_hub', 'oht_hub', 'direction', 'storage'}

# Boolean 임계 (4/1 실측 분포 기준: 중앙값909·최대1075 — 필요시 조정)
TH_Q_HIGH, TH_Q_CRIT = 1000, 1100
TH_BR_HIGH, TH_BR_CRIT = 8.0, 9.0        # AVGTOTALTIME1MIN(분)
TH_OHTUTIL_HIGH, TH_OHTUTIL_CRIT = 85.0, 90.0


def load_raw(path):
    """일별 원본 폴더(M16A_HUBROOM_PR_*.csv) 또는 단일 CSV → 시간정렬 DataFrame."""
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, '*.csv'))
                         + glob.glob(os.path.join(path, '*.CSV'))))
    if not files:
        raise SystemExit(f'❌ 원본 없음: {path}')
    dfs = []
    for fp in files:
        for enc in ('utf-8-sig', 'utf-8', 'cp949'):
            try:
                dfs.append(pd.read_csv(fp, encoding=enc, on_bad_lines='skip', low_memory=False))
                break
            except UnicodeDecodeError:
                continue
    df = pd.concat(dfs, ignore_index=True)
    tcol = 'CRT_TM' if 'CRT_TM' in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol].astype(str).str[:16], errors='coerce')
    df = (df.dropna(subset=[tcol]).drop_duplicates(subset=[tcol], keep='last')
            .sort_values(tcol).reset_index(drop=True))
    df = df.rename(columns={tcol: 'CRT_TM'})
    print(f'  파일 {len(files)}개 → {len(df)}분 ({df.CRT_TM.iloc[0]} ~ {df.CRT_TM.iloc[-1]})')
    return df


def create_features_v9(df, available_cols, stride=1):
    """V8 스타일 창 피처 (30분 과거 → 10분후 MIN/MAX/AVG 변화량 라벨)."""
    # 숫자화 + ffill
    use_cols = [c for cols in FEATURE_COLS.values() for c in cols if c in available_cols]
    num = {c: pd.to_numeric(df[c], errors='coerce').ffill().fillna(0.0).values for c in use_cols}
    tgt = num[TARGET_COL]
    hours = df['CRT_TM'].dt.hour.values
    n = len(df)

    features_list, labels, indices = [], [], []
    rng = range(LOOKBACK, n - HORIZON, stride)
    total = len(rng)
    for k, i in enumerate(rng):
        if k % 5000 == 0:
            print(f'  Feature 생성: {k}/{total} ({k/total*100:.1f}%)')
        seq_t = tgt[i - LOOKBACK:i]
        cur = seq_t[-1]
        fut = tgt[i:i + HORIZON]
        labels.append([fut.min() - cur, fut.max() - cur, fut.mean() - cur])

        f = {
            # 타겟 통계 (V8 동일)
            'target_mean': seq_t.mean(), 'target_std': seq_t.std(),
            'target_max': seq_t.max(), 'target_min': seq_t.min(),
            'target_last_value': cur, 'target_last_5_mean': seq_t[-5:].mean(),
            'target_last_10_mean': seq_t[-10:].mean(),
            'target_slope': np.polyfit(np.arange(LOOKBACK), seq_t, 1)[0],
            'target_acceleration': (seq_t[-5:].mean() - seq_t[-10:-5].mean()) / 5,
            'target_is_rising': 1 if seq_t[-1] > seq_t[-5] else 0,
            'target_rapid_rise': 1 if (seq_t[-1] - seq_t[-5] > 30) else 0,
            # 시간대 (큐는 일주기 강함)
            'hour_sin': np.sin(2 * np.pi * hours[i - 1] / 24),
            'hour_cos': np.cos(2 * np.pi * hours[i - 1] / 24),
            'is_day': 1 if 8 <= hours[i - 1] <= 19 else 0,
        }
        for group, cols in FEATURE_COLS.items():
            for c in cols:
                if c not in num or c == TARGET_COL:
                    continue
                seq = num[c][i - LOOKBACK:i]
                if group in FULL_STAT_GROUPS:
                    f[f'{c}_mean'] = seq.mean(); f[f'{c}_std'] = seq.std()
                    f[f'{c}_max'] = seq.max(); f[f'{c}_min'] = seq.min()
                    f[f'{c}_last_value'] = seq[-1]
                    f[f'{c}_last_5_mean'] = seq[-5:].mean()
                    f[f'{c}_slope'] = np.polyfit(np.arange(LOOKBACK), seq, 1)[0]
                else:
                    f[f'{c}_mean'] = seq.mean()
                    f[f'{c}_last_value'] = seq[-1]
                    f[f'{c}_slope'] = np.polyfit(np.arange(LOOKBACK), seq, 1)[0]

        # 수지(유입-처리) — HUB 큐 증감의 원인
        if 'M16HUB.QUE.ALL.CURRENTQCREATED' in num and 'M16HUB.QUE.ALL.CURRENTQCOMPLETED' in num:
            bal = num['M16HUB.QUE.ALL.CURRENTQCREATED'][i - LOOKBACK:i] \
                  - num['M16HUB.QUE.ALL.CURRENTQCOMPLETED'][i - LOOKBACK:i]
            f['hub_balance_last'] = bal[-1]
            f['hub_balance_10_mean'] = bal[-10:].mean()
            f['hub_balance_slope'] = np.polyfit(np.arange(LOOKBACK), bal, 1)[0]

        # Boolean (V8 스타일, 이 원본 분포 기준)
        f['currentq_high'] = 1 if cur >= TH_Q_HIGH else 0
        f['currentq_critical'] = 1 if cur >= TH_Q_CRIT else 0
        if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in num:
            br = num['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'][i - 1]
            f['avgtime1min_high'] = 1 if br >= TH_BR_HIGH else 0
            f['avgtime1min_critical'] = 1 if br >= TH_BR_CRIT else 0
            f['que_severe_bottleneck'] = 1 if (cur >= TH_Q_HIGH and br >= TH_BR_HIGH) else 0
        if 'M16HUB.QUE.OHT.OHTUTIL' in num:
            u = num['M16HUB.QUE.OHT.OHTUTIL'][i - 1]
            f['m16hub_ohtutil_high'] = 1 if u >= TH_OHTUTIL_HIGH else 0
            f['m16hub_ohtutil_critical'] = 1 if u >= TH_OHTUTIL_CRIT else 0
        # ── V8 Boolean 재사용 ──
        if 'M16HUB.QUE.TIME.AVGTOTALTIME' in num:
            at = num['M16HUB.QUE.TIME.AVGTOTALTIME'][i - 1]
            f['m16hub_avgtime_high'] = 1 if at >= 5.0 else 0
            f['m16hub_avgtime_critical'] = 1 if at >= 6.0 else 0
            if 'M16HUB.QUE.OHT.OHTUTIL' in num:
                f['m16hub_severe_bottleneck'] = 1 if (u >= 85.0 and at >= 5.0) else 0
        if 'M16A.QUE.LOAD.AVGLOADTIME1MIN' in num:
            lt = num['M16A.QUE.LOAD.AVGLOADTIME1MIN'][i - 1]
            f['m16a_loadtime_high'] = 1 if lt >= 2.5 else 0
            f['m16a_loadtime_critical'] = 1 if lt >= 2.8 else 0
        if 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in num:
            t4 = num['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'][i - 1]
            f['m16a_transport4min_high'] = 1 if t4 >= 40 else 0
            f['m16a_transport4min_critical'] = 1 if t4 >= 50 else 0
            if 'M16A.QUE.OHT.OHTUTIL' in num:
                ou = num['M16A.QUE.OHT.OHTUTIL'][i - 1]
                f['m16a_severe_bottleneck'] = 1 if (ou >= 85.0 and t4 >= 40) else 0

        features_list.append(f)
        indices.append(i)
    return pd.DataFrame(features_list), np.array(labels), indices


def train_v9(train_path, out_pkl='xgboost_수치형_V9.pkl', stride=1):
    print('=' * 80)
    print('🎯 V9 — HUB 총큐(M16HUB.QUE.ALL.CURRENTQCNT) 10분후 변화량 예측 (50+ 가중치)')
    print('=' * 80)

    print('\n[STEP 1] 원본 로드 (2달치 권장)')
    df = load_raw(train_path)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    available_cols = set(df.columns)
    want = [c for cols in FEATURE_COLS.values() for c in cols]
    have = [c for c in want if c in available_cols]
    print(f'\n📋 피처 컬럼: {len(have)}/{len(want)}개 사용 가능')
    for c in want:
        if c not in available_cols:
            print(f'  ⚠️ 없음(제외): {c}')

    print('\n[STEP 2] Feature 생성 (30분 창 → 10분후 MIN/MAX/AVG 변화량)')
    X, y, _ = create_features_v9(df, available_cols, stride=stride)
    print(f'✅ Feature {len(X.columns)}개 × 샘플 {len(X)}개')

    # 가중치 (V8 동일: |MAX변화| ≥ 50 → 3배)
    large = np.abs(y[:, 1]) >= BIG_CHANGE
    w = np.where(large, 3.0, 1.0)
    print(f'🔥 큰 변화(±{BIG_CHANGE}) 샘플: {large.sum()}개 ({large.mean()*100:.1f}%) → 가중치 3배')

    # ★ 시간분할 (뒤 20% 검증 — 랜덤분할은 시계열 누수)
    cut = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_val = y[:cut], y[cut:]
    w_tr = w[:cut]
    print(f'📐 시간분할: 학습 {len(X_tr)} / 검증 {len(X_val)} (뒤 20%)')

    # GPU 감지 (V8 동일)
    use_gpu = False
    try:
        m = xgb.XGBRegressor(n_estimators=5, max_depth=3, tree_method='gpu_hist', gpu_id=0)
        m.fit(X_tr.iloc[:100], y_tr[:100, 0], verbose=False)
        use_gpu = True
        print('🔍 GPU 모드')
    except Exception:
        print('🔍 CPU 모드')

    params = dict(n_estimators=250, max_depth=8, learning_rate=0.03,
                  subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
                  gamma=0.05, reg_alpha=0.05, reg_lambda=0.8, random_state=42)
    if use_gpu:
        params.update(tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor')
    else:
        params.update(tree_method='hist', n_jobs=-1)

    models = []
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        print(f'\n[{name}] 학습 중...')
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr[:, idx], sample_weight=w_tr, verbose=False)
        models.append(m)
        print(f'✓ {name} 완료')

    print('\n📊 검증 성능 (시간분할 뒤 20%):')
    y_pred = np.column_stack([m.predict(X_val) for m in models])
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_val[:, idx], y_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_val[:, idx], y_pred[:, idx]))
        r2 = r2_score(y_val[:, idx], y_pred[:, idx])
        print(f'  {name} - MAE {mae:.2f} · RMSE {rmse:.2f} · R² {r2:.3f}')

    # 오탐필터 관점 성능: "10분 뒤 +50 이상 오른다" 분류로 봤을 때
    print(f'\n🎯 급등(+{BIG_CHANGE}↑) 방향 예측 (오탐필터 용도):')
    act = y_val[:, 1] >= BIG_CHANGE
    for thr in [20, 30, 40, 50]:
        pred = y_pred[:, 1] >= thr
        tp = (pred & act).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max(act.sum(), 1)
        print(f'  예측MAX≥{thr}: 정밀도 {prec*100:.0f}% · 재현율 {rec*100:.0f}% (예측 {pred.sum()}건/실제 {act.sum()}건)')

    with open(out_pkl, 'wb') as f:
        pickle.dump({'models': models, 'feature_names': X.columns.tolist(),
                     'target': TARGET_COL, 'lookback': LOOKBACK, 'horizon': HORIZON,
                     'version': 'V9'}, f)
    print(f'\n✅ 모델 저장: {out_pkl}')

    imp = pd.DataFrame({'feature': X.columns, 'importance': models[1].feature_importances_}) \
        .sort_values('importance', ascending=False).head(20)
    print('\n🔥 Feature 중요도 Top 20 (MAX 모델):')
    for _, r in imp.iterrows():
        print(f"  {r['feature']}: {r['importance']:.4f}")
    print('\n🎉 V9 학습 완료!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='원본 폴더(2달치 M16A_HUBROOM_PR_*.csv) 또는 단일 CSV')
    ap.add_argument('--out', default='xgboost_수치형_V9.pkl')
    ap.add_argument('--stride', type=int, default=1, help='샘플 간격(분). 2로 하면 학습 2배 빠름')
    a = ap.parse_args()
    train_v9(a.train, a.out, a.stride)

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

# ★ train 과 반드시 동일 (60분 흐름 + CUSUM)
ROLL_WINS = [15, 30, 60]
DELTA_LAGS = [15, 30, 60]
MAX_WINS = [60]
CUSUM_BASE_WIN = 120
CUSUM_K = 0.5


def _need():
    try:
        import numpy, pandas, xgboost  # noqa
        return True
    except Exception as e:
        print("⚠️ XGBoost 라이브러리 없음 — pip install xgboost pandas numpy")
        print(f"   ({type(e).__name__}: {e})")
        return False


def _cusum(np, s, base_win=CUSUM_BASE_WIN, k=CUSUM_K):
    """한쪽(상승) CUSUM — train 과 100% 동일. 값이 평소+여유를 지속적으로 넘으면 누적."""
    import pandas as pd
    x = pd.Series(s).astype(float)
    base = x.shift(1).rolling(base_win, min_periods=15).median()
    sd = x.shift(1).rolling(base_win, min_periods=15).std()
    base = base.bfill().fillna(x.iloc[0] if len(x) else 0.0).values
    sd = sd.fillna(0.0).values
    xv = x.values
    n = len(xv)
    C = np.zeros(n, dtype='float64')
    prev = 0.0
    for i in range(n):
        prev = max(0.0, prev + (xv[i] - base[i] - k * sd[i]))
        C[i] = prev
    return C


def build_features(feat_df, base_cols):
    import numpy as np
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
        new[f'{c}__cusum'] = _cusum(np, s.values)                    # ★ CUSUM (train 과 동일)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


# ── 저장룰 임계 (hubroom_predictor 동일) + 실제정체 확인 ──
TH_RD_FAB = 25.0      # FAB저장률 ≥ 25% (6/4형: 평소 튈 수 있어 SLA 확인 필요)
TH_RD_STB = 99.0      # STB이용률 ≥ 99% (평소에도 100 → SLA 확인 필요)
TH_RD_STK = 10.0      # ★ STK 스토커 저장률 ≥ 10% (평소 0 → 단독 하드경보, 저장Full 직결)
TH_SLA_UP = 5.0       # 4분초과율(SLA) ≥ 5% = 실제 정체 (6/4는 0 → 차단)
# ── ★ CUSUM 밀림룰 (M16→M14 국소 밀림 = 6/22·24·29형, hubroom·XGBoost 못 잡음) ──
#    큐/저장 CUSUM(누적 상승)이 임계 넘으면 밀림 경보. 6월검증: 3/3 잡음(리드 13~23분).
TH_CUSUM_Q = 600.0    # SouthQ/NorthQ CUSUM (컨베이어 큐 지속 상승)
TH_CUSUM_FAB = 300.0  # FAB저장 CUSUM (저장 지속 상승 = 6/29형)
RANK = {'': 0, '경계': 1, '위험': 2, '초위험': 3}
INV = {v: k for k, v in RANK.items()}


def level(p, tag=''):
    """6월 검증 캘리브레이션(정밀도) 근거 확률컷.
    10분: 신뢰도 잘 나뉨(74/79/96%) → 3등급.
    30분: 신뢰도 ~77%에서 평평 → 단일 경보(위험)만."""
    if '30' in str(tag):
        return '위험' if p >= 0.50 else ''            # 30분: 신뢰 77% 단일 경보
    return ('초위험' if p >= 0.90                      # 10분: 신뢰 96%
            else '위험' if p >= 0.70                   #        신뢰 79%
            else '경계' if p >= 0.50 else '')          #        신뢰 74%


def storage_alarm(rd_fab, rd_stb, sla, rd_stk=None):
    """저장룰:
       ① STK 스토커 저장률 ≥ 10% → 단독 하드경보 (평소 0이라 튀면 곧 저장Full = 6/16형)
       ② FAB≥25 OR STB≥99 → 4분초과(SLA)≥5 확인돼야 경보 (6/4형 저장만 튐은 차단)
    """
    stk_hard = (rd_stk is not None and rd_stk >= TH_RD_STK)
    soft_hi = ((rd_fab is not None and rd_fab >= TH_RD_FAB)
               or (rd_stb is not None and rd_stb >= TH_RD_STB))
    congested = (sla is not None and sla >= TH_SLA_UP)
    alarm = stk_hard or (soft_hi and congested)
    return alarm, (stk_hard or soft_hi)


def cusum_alarm(sq_cu, nq_cu, fab_cu):
    """CUSUM 밀림룰: 큐/저장 CUSUM(누적 상승)이 클수록 높은 등급.
       임계 대비 배수로 경계/위험/초위험 구분 (심할수록 위험). 반환: (등급, 사유)."""
    # 큐(남/북)는 TH_CUSUM_Q 기준, 저장은 TH_CUSUM_FAB 기준 → 각자 '초과비율' 계산
    cands = []
    if sq_cu is not None and sq_cu >= TH_CUSUM_Q:
        cands.append((sq_cu / TH_CUSUM_Q, f"남측큐 지속상승 CUSUM {sq_cu:.0f}"))
    if nq_cu is not None and nq_cu >= TH_CUSUM_Q:
        cands.append((nq_cu / TH_CUSUM_Q, f"북측큐 지속상승 CUSUM {nq_cu:.0f}"))
    if fab_cu is not None and fab_cu >= TH_CUSUM_FAB:
        cands.append((fab_cu / TH_CUSUM_FAB, f"저장 지속상승 CUSUM {fab_cu:.0f}"))
    if not cands:
        return '', ''
    ratio, why = max(cands)                       # 가장 심한 신호 채택
    # 임계의 1.0~1.5배=경계, 1.5~2.5배=위험, 2.5배↑=초위험
    grade = '초위험' if ratio >= 2.5 else '위험' if ratio >= 1.5 else '경계'
    return grade, f"밀림경보({why}, 임계 {ratio:.1f}배)"


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

    # 저장룰용 원신호 (features 에 있으면 사용)
    def col(name):
        return df[name].values if name in df.columns else [None] * len(df)
    rd_fab_v, rd_stb_v, sla_v = col('RD_FAB'), col('RD_STB'), col('SLA_M16HUB')
    rd_stk_v = col('RD_STK')
    # CUSUM 밀림룰용 (build_features 가 만든 {col}__cusum)
    sq_cu_v = col('DIR_SouthCNV_Q__cusum')
    nq_cu_v = col('DIR_NorthCNV_Q__cusum')
    fab_cu_v = col('RD_FAB__cusum')

    def fnum(v):
        try:
            import math
            x = float(v)
            return None if math.isnan(x) else x
        except (TypeError, ValueError):
            return None

    # 지평선 정렬 (10분 먼저, 30분 뒤)
    tags = sorted(tags, key=lambda tg: horizon[tg])
    n_final = {tg: 0 for tg in tags}
    n_stor = n_block = n_mil = 0
    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        # 지평선별로 예측시각·확률·최종등급·사유 나눠서 출력
        header = ['datetime']
        for tg in tags:
            m = tg.replace('y_pre', '')
            header += [f'{m}분_예측시각', f'{m}분_확률%', f'{m}분_최종등급', f'{m}분_사유']
        header += ['저장경보', '밀림경보',
                   'RD_FAB', 'RD_STB', 'RD_STK', 'SLA_4분초과',
                   '남큐CUSUM', '북큐CUSUM', '저장CUSUM']
        w.writerow(header)
        for i, t in enumerate(df['datetime']):
            # 저장룰 (지평선 공통) — STK 하드 + (FAB/STB AND 4분초과)
            rd_fab, rd_stb, sla = fnum(rd_fab_v[i]), fnum(rd_stb_v[i]), fnum(sla_v[i])
            rd_stk = fnum(rd_stk_v[i])
            s_alarm, s_hi = storage_alarm(rd_fab, rd_stb, sla, rd_stk)
            if s_hi and not s_alarm:
                n_block += 1
            if s_alarm:
                n_stor += 1
                if rd_stk is not None and rd_stk >= TH_RD_STK:
                    s_reason = f"저장경보(STK스토커{rd_stk:.0f}%≥{TH_RD_STK:.0f})"
                else:
                    s_reason = f"저장경보(FAB{rd_fab or 0:.0f}%/STB{rd_stb or 0:.0f}%,4분초과{sla or 0:.0f}%)"
            # ★ CUSUM 밀림룰 (M16→M14 국소 밀림 = 6/22·24·29) — 심할수록 높은 등급
            sq_cu, nq_cu, fab_cu = fnum(sq_cu_v[i]), fnum(nq_cu_v[i]), fnum(fab_cu_v[i])
            m_grade, m_reason = cusum_alarm(sq_cu, nq_cu, fab_cu)
            m_alarm = bool(m_grade)
            if m_alarm:
                n_mil += 1
            # 룰 등급 = 저장경보(위험) vs 밀림등급 중 높은 것
            rule_rank = max(RANK['위험'] if s_alarm else 0, RANK[m_grade])

            row = [t.strftime('%Y-%m-%d %H:%M')]
            for tg in tags:
                m = tg.replace('y_pre', '')
                p = float(probs[tg][i])
                g_xgb = level(p, tg)
                # 지평선별 최종 = max(그 지평선 XGBoost, 저장경보, 밀림경보)
                final = INV[max(RANK[g_xgb], rule_rank)]
                why = []
                if g_xgb:
                    why.append(f"XGBoost {m}분 {p*100:.0f}%")
                if s_alarm:
                    why.append(s_reason)
                if m_alarm:
                    why.append(m_reason)
                if final:
                    n_final[tg] += 1
                tgt = (t + timedelta(minutes=horizon[tg])).strftime('%Y-%m-%d %H:%M')
                row += [tgt, f'{p*100:.1f}%', final, ' | '.join(why)]
            row += ['예' if s_alarm else '', '예' if m_alarm else '',
                    '' if rd_fab is None else f'{rd_fab:.1f}',
                    '' if rd_stb is None else f'{rd_stb:.1f}',
                    '' if rd_stk is None else f'{rd_stk:.1f}',
                    '' if sla is None else f'{sla:.1f}',
                    '' if sq_cu is None else f'{sq_cu:.0f}',
                    '' if nq_cu is None else f'{nq_cu:.0f}',
                    '' if fab_cu is None else f'{fab_cu:.0f}']
            w.writerow(row)

    print(f"[완료] {len(df)}분 → {a.out}")
    for tg in tags:
        m = tg.replace('y_pre', '')
        print(f"       {m}분 최종경보(위험+): {n_final[tg]}분 ({n_final[tg]/len(df)*100:.1f}%)")
    print(f"       저장경보 {n_stor}분 · 밀림경보(CUSUM) {n_mil}분 · 차단(6/4형) {n_block}분")


if __name__ == '__main__':
    main()

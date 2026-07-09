#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
밀림_방향_CUSUM — 순수 CUSUM 방향 밀림 감지기 (학습·모델·XGBoost 불필요)
====================================================================
룰베이스 사각지대(국소 컨베이어 밀림)를 큐의 '지속 상승'으로 감지.
방향당 큐 1개만 CUSUM → 어느 컨베이어(남측 4AFC3201 / 북측 4AFC3301 / 허브)가 밀리는지.
6월 4개 사건(6/11·22·24·29) 4/4 포착 확인.

입력 컬럼(전부 features.csv 에 이미 있음):
  CUSUM  : DIR_SouthCNV_Q(남측) · DIR_NorthCNV_Q(북측) · RD_FAB(허브저장)
  하드경보: RD_STK(≥10 저장Full) · RD_STB(≥99) · SLA_M16HUB(4분초과)

판정:
  CUSUM/임계  1.0~1.5배=경계 / 1.5~2.5배=위험 / 2.5배↑=초위험
  게이트     초위험=밤낮 항상 / 위험=주간(08~19) / 경계=미예측
  출력       예측결과 = '예측' / '미예측' (등급 대신 2값)

입력:
  --features  features.csv (features_31.py 산출)
  --out       밀림CUSUM_결과.csv

실행:
  python 밀림_방향_CUSUM.py --features .\out_ml_june\features.csv --out .\out_ml\밀림CUSUM_결과.csv
"""
import argparse, csv, os, sys

# ── 파라미터 (기존 infer 와 동일) ──
CUSUM_BASE_WIN = 120      # 기준선 창(분, 과거만)
CUSUM_K = 0.5             # 여유 = K × 과거표준편차
TH_CUSUM_Q = 600.0       # 남측/북측 큐 CUSUM 임계
TH_CUSUM_FAB = 300.0     # 허브 FAB저장 CUSUM 임계
TH_RD_STK = 10.0         # STK 스토커 저장률 하드경보(≥10%)
TH_RD_STB = 99.0         # STB 이용률
DIR_LABEL = {'남측': '남측(4AFC3201)', '북측': '북측(4AFC3301)', '허브': '허브(몰림/저장)'}
GRADE_ORD = {'': 0, '경계': 1, '위험': 2, '초위험': 3}


def _cusum(np, s):
    """한쪽(상승) CUSUM: 값이 평소+여유를 지속적으로 넘으면 누적. 순간 튐 무시."""
    import pandas as pd
    x = pd.Series(s).astype(float)
    base = x.shift(1).rolling(CUSUM_BASE_WIN, min_periods=15).median().bfill().fillna(x.iloc[0] if len(x) else 0.0)
    sd = x.shift(1).rolling(CUSUM_BASE_WIN, min_periods=15).std().fillna(0.0).values
    base = base.values; xv = x.values
    C = np.zeros(len(xv)); prev = 0.0
    for i in range(len(xv)):
        prev = max(0.0, prev + (xv[i] - base[i] - CUSUM_K * sd[i])); C[i] = prev
    return C


def grade_by_ratio(cu, thr):
    """CUSUM/임계 비율 → 등급. 임계 미달이면 ''."""
    if cu < thr:
        return '', 0.0
    r = cu / thr
    g = '초위험' if r >= 2.5 else '위험' if r >= 1.5 else '경계'
    return g, r


def gated(grade, hour):
    """초위험=항상 / 위험=주간(08~19) / 경계·''=미예측."""
    if grade == '초위험':
        return grade
    if grade == '위험' and 8 <= hour <= 19:
        return grade
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', default='./out_ml/밀림CUSUM_결과.csv')
    a = ap.parse_args()
    try:
        import numpy as np, pandas as pd
    except Exception as e:
        print(f"⚠️ numpy/pandas 필요: pip install numpy pandas ({e})"); sys.exit(2)

    df = pd.read_csv(a.features, encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    def col(name):
        return df[name].astype(float).ffill().fillna(0.0).values if name in df.columns else np.zeros(len(df))

    # 없으면 0 (경고)
    for c in ['DIR_SouthCNV_Q', 'DIR_NorthCNV_Q', 'RD_FAB', 'RD_STK', 'RD_STB', 'SLA_M16HUB']:
        if c not in df.columns:
            print(f"   ⚠️ features.csv 에 '{c}' 없음 → 0 처리")

    # 방향별 CUSUM
    cu_S = _cusum(np, col('DIR_SouthCNV_Q'))
    cu_N = _cusum(np, col('DIR_NorthCNV_Q'))
    cu_F = _cusum(np, col('RD_FAB'))
    rd_stk = col('RD_STK'); rd_stb = col('RD_STB'); sla = col('SLA_M16HUB')
    hours = df['datetime'].dt.hour.values

    cols = ['datetime',
            '남큐CUSUM', '남측_예측결과', '북큐CUSUM', '북측_예측결과',
            '저장CUSUM', '허브_예측결과', 'RD_STK', '저장하드경보',
            '밀림방향', '예측결과', '사유']
    rows = []
    n = {'남측': 0, '북측': 0, '허브': 0}
    for i, t in enumerate(df['datetime']):
        # 방향별 등급 → 게이트
        gS, rS = grade_by_ratio(cu_S[i], TH_CUSUM_Q)
        gN, rN = grade_by_ratio(cu_N[i], TH_CUSUM_Q)
        gF, rF = grade_by_ratio(cu_F[i], TH_CUSUM_FAB)
        # 저장 하드경보 (STK ≥10% → 즉시 위험, 허브에 반영)
        stk_hard = rd_stk[i] >= TH_RD_STK
        hard_txt = f"STK{rd_stk[i]:.0f}%≥{TH_RD_STK:.0f}" if stk_hard else ''
        if stk_hard and GRADE_ORD['위험'] > GRADE_ORD[gF]:
            gF, rF = '위험', max(rF, 1.5)

        ggS, ggN, ggF = gated(gS, hours[i]), gated(gN, hours[i]), gated(gF, hours[i])
        cand = []
        if ggS: cand.append(('남측', ggS, rS, f"남측큐 지속상승 CUSUM {cu_S[i]:.0f}({rS:.1f}배)"))
        if ggN: cand.append(('북측', ggN, rN, f"북측큐 지속상승 CUSUM {cu_N[i]:.0f}({rN:.1f}배)"))
        if ggF: cand.append(('허브', ggF, rF, f"허브저장 지속상승 CUSUM {cu_F[i]:.0f}({rF:.1f}배){' +'+hard_txt if hard_txt else ''}"))
        best = max(cand, key=lambda x: (GRADE_ORD[x[1]], x[2])) if cand else None
        if best:
            n[best[0]] += 1
        rows.append({
            'datetime': t.strftime('%Y-%m-%d %H:%M'),
            '남큐CUSUM': f'{cu_S[i]:.0f}', '남측_예측결과': '예측' if ggS else '미예측',
            '북큐CUSUM': f'{cu_N[i]:.0f}', '북측_예측결과': '예측' if ggN else '미예측',
            '저장CUSUM': f'{cu_F[i]:.0f}', '허브_예측결과': '예측' if ggF else '미예측',
            'RD_STK': f'{rd_stk[i]:.0f}', '저장하드경보': hard_txt,
            '밀림방향': DIR_LABEL[best[0]] if best else '',
            '예측결과': '예측' if best else '미예측',
            '사유': best[3] if best else '',
        })

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    tot = sum(n.values())
    print(f"[완료] {len(df)}분 → {a.out}")
    print(f"       밀림예측(게이트후): 남측 {n['남측']} / 북측 {n['북측']} / 허브 {n['허브']}  (총 {tot}분, {tot/len(df)*100:.1f}%)")
    print("       룰베이스와 병행: 룰 조용(50미만)한데 여기 '예측' 뜨면 = 국소밀림")
    print("       다음: python 밀림방향_평가.py --result " + a.out)


if __name__ == '__main__':
    main()

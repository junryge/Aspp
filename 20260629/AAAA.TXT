#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tspulse_infer — 학습된 TSPulse R1 로 매분 이상점수 생성 (Phase 2)  ★회사 PC 전용
====================================================================
매분 t: 과거 context 창(t 까지만) → 재구성오차 → anomaly_score(t).
정상 재구성 학습본이라, 평소와 다르면(급증=정체 전조) 오차가 커짐.

★ 누수 차단: 창은 항상 t 까지의 과거만 사용 (미래정보 X).
★ 점수 정규화: 정상구간 오차분포의 로버스트 z → [0,1] 시그모이드.

입력:
    --features  features.csv (features_31.py 산출, 추론 대상 전체 구간 가능)
    --model     out_ml/tspulse (tspulse_train.py 산출: model/ + scaler.json)
    --labels    labels.csv (선택 — 정상구간으로 점수 스케일 보정에 사용)
    --out       출력 (기본 ./out_ml/anomaly.csv)

실행:
    python tspulse_infer.py --features ./out_ml/features.csv --model ./out_ml/tspulse

출력:
    anomaly.csv  — datetime, recon_err, anomaly_score[0~1], ml_level
    ml_level: 안전<0.3 / 관심0.3~ / 경계0.5~ / 위험≥0.7  (룰 50/71/85 과 별개)
"""
import argparse
import csv
import json
import math
import os
import sys


def _need():
    try:
        import numpy, pandas, torch  # noqa
        from tsfm_public.models.tspulse import TSPulseForReconstruction  # noqa
        return True
    except Exception as e:
        print("⚠️ 추론 라이브러리 없음 — 회사 PC 전용")
        print("   pip install \"granite-tsfm[notebooks]\" torch pandas numpy")
        print(f"   ({type(e).__name__}: {e})")
        return False


def level(s):
    return '위험' if s >= 0.7 else '경계' if s >= 0.5 else '관심' if s >= 0.3 else '안전'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--model', default='./out_ml/tspulse')
    ap.add_argument('--labels', default=None)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--out', default='./out_ml/anomaly.csv')
    a = ap.parse_args()

    print("=" * 60)
    print("TSPulse R1 추론 — 매분 이상점수")
    print("=" * 60)
    if not _need():
        sys.exit(2)

    import numpy as np
    import pandas as pd
    import torch
    from tsfm_public.models.tspulse import TSPulseForReconstruction

    scaler = json.load(open(os.path.join(a.model, 'scaler.json'), encoding='utf-8'))
    cols, stats, C = scaler['features'], scaler['stats'], scaler['context']

    feat = pd.read_csv(a.features, encoding='utf-8-sig')
    feat['datetime'] = pd.to_datetime(feat['datetime'])
    feat = feat.sort_values('datetime').reset_index(drop=True)
    feat[cols] = feat[cols].ffill().fillna(0.0)
    # 정규화 (학습 통계 재사용)
    Xn = np.zeros((len(feat), len(cols)), dtype='float32')
    for j, c in enumerate(cols):
        Xn[:, j] = (feat[c].astype(float).values - stats[c]['median']) / stats[c]['iqr']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TSPulseForReconstruction.from_pretrained(os.path.join(a.model, 'model')).to(device)
    model.eval()

    # 매분 과거 창 → 재구성오차 (창의 마지막 시점 오차를 그 분의 점수로)
    idxs = list(range(C - 1, len(feat)))       # t=C-1 부터 과거창 성립
    errs = np.full(len(feat), np.nan, dtype='float64')
    total = len(idxs)
    print(f"[추론] {total}분 점수화 시작 (batch={a.batch}, device={device}) — 학습 아님, 1회만", flush=True)
    with torch.no_grad():
        buf_i, buf_x = [], []

        def flush():
            if not buf_x:
                return
            xb = torch.tensor(np.stack(buf_x)).to(device)       # (B,C,ch)
            out = model(past_values=xb)
            recon = out.reconstruction_outputs if hasattr(out, 'reconstruction_outputs') else out[1]
            # 창 마지막 시점의 채널평균 제곱오차
            e = ((recon[:, -1, :] - xb[:, -1, :]) ** 2).mean(dim=1).cpu().numpy()
            for k, ti in enumerate(buf_i):
                errs[ti] = float(e[k])
            buf_i.clear(); buf_x.clear()

        for n, t in enumerate(idxs, 1):
            buf_i.append(t)
            buf_x.append(Xn[t - C + 1:t + 1])
            if len(buf_x) >= a.batch:
                flush()
            if n % (a.batch * 20) == 0:
                print(f"    {n}/{total}분 ({n / total * 100:.0f}%)", flush=True)
        flush()
    print(f"    {total}/{total}분 (100%) 계산 완료", flush=True)

    # ── 점수 정규화: 정상구간 오차의 median/MAD 로 z → 시그모이드 [0,1] ──
    valid = ~np.isnan(errs)
    base = errs[valid]
    if a.labels and os.path.exists(a.labels):
        lab = pd.read_csv(a.labels, encoding='utf-8-sig')
        lab['datetime'] = pd.to_datetime(lab['datetime'])
        m = feat.merge(lab[['datetime', 'is_normal']], on='datetime', how='left')
        nb = errs[(m['is_normal'].fillna(0).values == 1) & valid]
        if len(nb) > 100:
            base = nb
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med))) or 1e-9
    scores = np.full(len(feat), np.nan)
    z = (errs[valid] - med) / (1.4826 * mad)
    scores[valid] = 1.0 / (1.0 + np.exp(-z))    # 시그모이드

    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['datetime', 'recon_err', 'anomaly_score', 'ml_level'])
        for i in range(len(feat)):
            if np.isnan(errs[i]):
                continue
            s = float(scores[i])
            w.writerow([feat['datetime'].iloc[i].strftime('%Y-%m-%d %H:%M'),
                        f'{errs[i]:.6f}', f'{s:.4f}', level(s)])

    nvalid = int(valid.sum())
    hi = int((scores[valid] >= 0.7).sum())
    print(f"[완료] {nvalid}분 점수화 (앞 {C-1}분은 창 부족으로 제외)")
    print(f"       위험(≥0.7) {hi}분 ({hi/nvalid*100:.1f}%)  → {a.out}")
    print("다음: 검증_선행성.py --anomaly anomaly.csv --episodes episodes_jam.csv")


if __name__ == '__main__':
    main()

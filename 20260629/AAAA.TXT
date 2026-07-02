#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tspulse_train — TSPulse R1 이상탐지 fine-tune (Phase 2)  ★회사 PC 전용
====================================================================
우리 '정상 구간' 시계열만 재구성(reconstruction) 학습 → 추론 시 재구성오차 = 이상점수.
정체가 30분 전에 이상으로 잡히는지는 tspulse_infer + 검증_선행성 에서 채점.

전제(회사 PC): pip install "granite-tsfm[notebooks]" torch pandas numpy scikit-learn
              (폐쇄망이면 granite-tsfm 휠 + TSPulse 가중치 오프라인 반입)

입력:
    --features   features.csv (features_31.py 산출: datetime + 31피처)
                 ★ 정상 raw(정상데이터_생성.py) 로 만든 features 면 전부 정상 → labels 불필요
    --labels     (옵션) labels.csv 의 is_normal 마스크. 없으면 features 전체를 정상으로 간주
    --context    윈도우 길이(분) 기본 512 (≈8.5h)
    --epochs     기본 10
    --out        모델/스케일러 저장 폴더 (기본 ./out_ml/tspulse)

실행:
    python tspulse_train.py --features ./out_ml/features.csv          # 정상 raw 기반 (labels X)
    python tspulse_train.py --features ./out_ml/features.csv --labels ./out_ml/labels.csv  # 마스크 사용

출력:
    out_ml/tspulse/model/        fine-tuned TSPulse 가중치
    out_ml/tspulse/scaler.json   robust 정규화 파라미터(median/iqr, 피처순서)
    out_ml/tspulse/train_meta.json

★ 누수 차단: 학습창은 '연속(1분) 정상 세그먼트' 안에서만 생성 → 삭제로 생긴 시간 구멍 안 넘음.
★ 정규화 통계도 정상 데이터에서만.
"""
import argparse
import csv
import json
import os
import sys

TSPULSE_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"


# ============================================================
# 0. 의존성 (회사 PC 에서만 존재)
# ============================================================
def _need():
    try:
        import numpy, pandas, torch  # noqa: F401
        from tsfm_public.models.tspulse import TSPulseForReconstruction  # noqa: F401
        return True
    except Exception as e:
        print("=" * 60)
        print("⚠️ 학습 라이브러리 없음 — 이 스크립트는 회사 PC 전용")
        print("   pip install \"granite-tsfm[notebooks]\" torch pandas numpy scikit-learn")
        print(f"   ({type(e).__name__}: {e})")
        print("=" * 60)
        return False


# ============================================================
# 1. 데이터 로드 (features + labels → 정상구간 행렬)
# ============================================================
def load_xy(features_fp, labels_fp=None):
    import os
    import pandas as pd
    feat = pd.read_csv(features_fp, encoding='utf-8-sig')
    feat['datetime'] = pd.to_datetime(feat['datetime'])
    feat_cols = [c for c in feat.columns if c != 'datetime']
    if labels_fp and os.path.exists(labels_fp):
        # (옵션) labels.csv 있으면 is_normal 마스크 사용
        lab = pd.read_csv(labels_fp, encoding='utf-8-sig')
        lab['datetime'] = pd.to_datetime(lab['datetime'])
        df = feat.merge(lab[['datetime', 'is_normal']], on='datetime', how='left')
        df['is_normal'] = df['is_normal'].fillna(0).astype(int)
    else:
        # labels 없음 = 입력이 이미 '정상 raw' (작업일정 제거됨) → 전부 정상
        df = feat.copy()
        df['is_normal'] = 1
    df = df.sort_values('datetime').reset_index(drop=True)
    # 결측 → 시간 전방채움 후 0 (설비 무보고=직전 유지가 도메인상 자연스러움)
    df[feat_cols] = df[feat_cols].ffill().fillna(0.0)
    return df, feat_cols


def robust_scaler(df_normal, cols):
    """정상구간에서 median/IQR 산출 (영역별 스케일 상이 → robust)."""
    import numpy as np
    stats = {}
    for c in cols:
        v = df_normal[c].astype(float).values
        med = float(np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        iqr = float(q3 - q1) or 1.0
        stats[c] = {'median': med, 'iqr': iqr}
    return stats


def apply_scaler(df, cols, stats):
    import numpy as np
    out = df.copy()
    for c in cols:
        s = stats[c]
        out[c] = (df[c].astype(float) - s['median']) / s['iqr']
    return out


# ============================================================
# 2. TSPulse fine-tune (정상 재구성)
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', default=None,
                    help='(옵션) is_normal 마스크. 정상 raw 로 만든 features 면 불필요')
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', default='./out_ml/tspulse')
    a = ap.parse_args()

    print("=" * 60)
    print(f"TSPulse R1 fine-tune — context {a.context}분 / epochs {a.epochs}")
    print("=" * 60)
    if not _need():
        sys.exit(2)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tsfm_public.models.tspulse import TSPulseForReconstruction

    os.makedirs(a.out, exist_ok=True)
    df, cols = load_xy(a.features, a.labels)
    n_ch = len(cols)
    print(f"[데이터] {len(df)}분 × {n_ch}피처")

    normal = df[df['is_normal'] == 1]
    print(f"[정상구간] {len(normal)}분 ({len(normal)/len(df)*100:.1f}%) → 정규화 통계 + 학습")
    if len(normal) < a.context * 3:
        print("⚠️ 정상구간이 너무 짧음 — context 를 줄이거나 guard 를 낮추세요")

    # 정규화(정상 통계) → 전체에 적용
    stats = robust_scaler(normal, cols)
    dfx = apply_scaler(df, cols, stats)
    with open(os.path.join(a.out, 'scaler.json'), 'w', encoding='utf-8') as f:
        json.dump({'features': cols, 'stats': stats, 'context': a.context}, f,
                  ensure_ascii=False, indent=2)

    # ── 연속(1분 간격) 정상 세그먼트 단위로만 슬라이딩 윈도우 ──
    #    작업일정 제거로 생긴 시간 '구멍'을 건너뛰는 창은 만들지 않음 (누수/불연속 차단).
    isn = df['is_normal'].values.astype(int)
    tmin = df['datetime'].values.astype('datetime64[m]').astype('int64')  # 분 단위 정수
    arr = dfx[cols].values.astype('float32')
    C = a.context
    # 정상 & 직전과 1분 연속 인 구간을 세그먼트로 분할
    segs = []
    i, n = 0, len(arr)
    while i < n:
        if isn[i] != 1:
            i += 1; continue
        j = i
        while j + 1 < n and isn[j + 1] == 1 and (tmin[j + 1] - tmin[j]) == 1:
            j += 1
        segs.append((i, j)); i = j + 1
    windows = []
    for s, e in segs:
        for k in range(s, e - C + 2):        # k..k+C-1 이 [s,e] 안일 때만
            windows.append(arr[k:k + C])
    too_short = sum(1 for s, e in segs if (e - s + 1) < C)
    print(f"[세그먼트] 연속 정상 구간 {len(segs)}개 "
          f"(그중 {C}분 미만 {too_short}개는 창 불가)")
    if not windows:
        print(f"⚠️ 정상 윈도우 0개 — 연속 구간이 모두 {C}분 미만. "
              f"context 를 줄이거나(--context) 데이터 확인"); sys.exit(3)
    X = torch.tensor(np.stack(windows))       # (N, C, ch)
    print(f"[학습창] 정상 윈도우 {len(windows)}개  shape={tuple(X.shape)}")

    # 모델: 사전학습 TSPulse R1 → 채널수 맞춰 reconstruction head
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[모델] {TSPULSE_MODEL} 로드 (device={device})")
    model = TSPulseForReconstruction.from_pretrained(
        TSPULSE_MODEL, num_input_channels=n_ch,
        context_length=C, mask_type="user", ignore_mismatched_sizes=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loader = DataLoader(TensorDataset(X), batch_size=a.batch, shuffle=True)
    model.train()
    for ep in range(1, a.epochs + 1):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)                # (B, C, ch)
            opt.zero_grad()
            out = model(past_values=xb)
            recon = out.reconstruction_outputs if hasattr(out, 'reconstruction_outputs') else out[1]
            loss = torch.nn.functional.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"  epoch {ep:2d}/{a.epochs}  recon MSE {tot/len(X):.5f}")

    model.save_pretrained(os.path.join(a.out, 'model'))
    with open(os.path.join(a.out, 'train_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': TSPULSE_MODEL, 'channels': n_ch, 'context': C,
                   'epochs': a.epochs, 'normal_windows': len(windows),
                   'features': cols, 'device': device},
                  f, ensure_ascii=False, indent=2)
    print(f"🎉 학습 완료 → {a.out}/model  (scaler.json 포함)")
    print("다음: tspulse_infer.py 로 매분 이상점수 생성 → 검증_선행성.py")


if __name__ == '__main__':
    main()

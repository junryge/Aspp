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
def _ensure_safetensors(model_dir):
    """로컬 모델 폴더에 model.safetensors 가 없고 다른 *.safetensors(예: model-1.safetensors)
       하나만 있으면 표준명으로 복사 → from_pretrained 가 인식하게. (sharded 인덱스면 손대지 않음)"""
    import glob
    import shutil
    std = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(std):
        return
    if os.path.exists(os.path.join(model_dir, 'model.safetensors.index.json')):
        return                                      # sharded → 그대로 둠
    cands = glob.glob(os.path.join(model_dir, '*.safetensors'))
    if len(cands) == 1:
        shutil.copy(cands[0], std)
        print(f"[모델] {os.path.basename(cands[0])} → model.safetensors 복사 (from_pretrained 호환)")
    elif not cands:
        # bin 형태 대비
        bins = glob.glob(os.path.join(model_dir, '*.bin'))
        if len(bins) == 1 and not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
            shutil.copy(bins[0], os.path.join(model_dir, 'pytorch_model.bin'))
            print(f"[모델] {os.path.basename(bins[0])} → pytorch_model.bin 복사")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', default=None,
                    help='(옵션) is_normal 마스크. 정상 raw 로 만든 features 면 불필요')
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--stride', type=int, default=64,
                    help='학습창 간격(분). 클수록 창 수↓·빠름. 64 권장 (1=최대중복·초느림)')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', default='./out_ml/tspulse')
    ap.add_argument('--model_dir', default=None,
                    help='로컬 모델 폴더(오프라인 반입). 없으면 HF에서 다운로드')
    ap.add_argument('--ca_bundle', default=None,
                    help='회사 CA 인증서(.pem) 경로 — SSL 검증 실패 시')
    ap.add_argument('--insecure', action='store_true',
                    help='SSL 검증 끄고 다운로드 (내부망 한정, 비권장)')
    a = ap.parse_args()

    # ── 회사망 SSL 대응 (HF 모델 다운로드용) ──
    if a.ca_bundle:
        os.environ['REQUESTS_CA_BUNDLE'] = a.ca_bundle
        os.environ['CURL_CA_BUNDLE'] = a.ca_bundle
        print(f"[SSL] CA 번들 사용: {a.ca_bundle}")
    if a.insecure:
        import requests, urllib3
        urllib3.disable_warnings()
        _orig_req = requests.Session.request
        def _noverify(self, *ar, **kw):
            kw['verify'] = False
            return _orig_req(self, *ar, **kw)
        requests.Session.request = _noverify
        print("[SSL] ⚠️ 검증 비활성화 (--insecure)")

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
        for k in range(s, e - C + 2, a.stride):   # stride 간격 (재구성 학습엔 겹침 과다 불필요)
            windows.append(arr[k:k + C])
    too_short = sum(1 for s, e in segs if (e - s + 1) < C)
    print(f"[세그먼트] 연속 정상 구간 {len(segs)}개 "
          f"(그중 {C}분 미만 {too_short}개는 창 불가) · stride {a.stride}")
    if not windows:
        print(f"⚠️ 정상 윈도우 0개 — 연속 구간이 모두 {C}분 미만. "
              f"context 를 줄이거나(--context) 데이터 확인"); sys.exit(3)
    X = torch.tensor(np.stack(windows))       # (N, C, ch)
    print(f"[학습창] 정상 윈도우 {len(windows)}개  shape={tuple(X.shape)}")

    # 모델: 사전학습 TSPulse R1 → 채널수 맞춰 reconstruction head
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_src = a.model_dir or TSPULSE_MODEL
    if a.model_dir:
        # 로컬 폴더 검증 + config.json 위치 자동 탐색 (zip 압축해제로 한 겹 더 생긴 경우 대응)
        if not os.path.isdir(a.model_dir):
            print(f"⚠️ 모델 폴더 없음: {a.model_dir}\n   → ls 로 실제 경로 확인 후 --model_dir 다시 지정")
            sys.exit(4)
        if not os.path.isfile(os.path.join(a.model_dir, 'config.json')):
            import glob as _g
            hits = _g.glob(os.path.join(a.model_dir, '**', 'config.json'), recursive=True)
            if hits:
                a.model_dir = os.path.dirname(hits[0])
                print(f"[모델] config.json 하위폴더에서 발견 → {a.model_dir}")
            else:
                print(f"⚠️ {a.model_dir} 안에 config.json 없음 — TSPulse 가중치 폴더가 맞는지 확인\n"
                      f"   (config.json + *.safetensors 가 같이 있어야 함)")
                sys.exit(4)
        model_src = a.model_dir
        _ensure_safetensors(a.model_dir)        # model-1.safetensors 등 자동 표준명 처리
        os.environ['HF_HUB_OFFLINE'] = '1'      # 로컬 폴더만 사용 (네트워크 X)
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print(f"[모델] {model_src} 로드 (device={device})")
    model = TSPulseForReconstruction.from_pretrained(
        model_src, num_input_channels=n_ch,
        context_length=C, mask_type="user", ignore_mismatched_sizes=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loader = DataLoader(TensorDataset(X), batch_size=a.batch, shuffle=True)
    nb = len(loader)
    print(f"[학습] {a.epochs} epochs × {nb} batch (batch={a.batch}, device={device})", flush=True)
    model.train()
    import time as _t
    for ep in range(1, a.epochs + 1):
        tot = 0.0
        t0 = _t.time() if hasattr(_t, 'time') else 0
        for bi, (xb,) in enumerate(loader, 1):
            xb = xb.to(device)                # (B, C, ch)
            opt.zero_grad()
            out = model(past_values=xb)
            recon = out.reconstruction_outputs if hasattr(out, 'reconstruction_outputs') else out[1]
            loss = torch.nn.functional.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            if bi % 5 == 0 or bi == nb:       # 진행상황 (조용히 안 돌게)
                print(f"    ep{ep} {bi}/{nb} loss {loss.item():.5f}", flush=True)
        print(f"  ✔ epoch {ep}/{a.epochs}  recon MSE {tot/len(X):.5f}", flush=True)

    model.save_pretrained(os.path.join(a.out, 'model'))

    # ── ★ 정상 기준선 고정 (frozen baseline) : 과탐지 근본해결 ──
    #    학습한 '정상' 창들의 재구성오차 median/MAD 를 scaler.json 에 박아둔다.
    #    추론 때 입력 자기분포로 정규화(=절반이 위험) 하는 대신 이 고정값을 쓰면
    #    정상 분=낮음, 급증(정체 전조)만=높음 으로 분리된다.
    model.eval()
    berr = []
    with torch.no_grad():
        for bi in range(0, len(X), a.batch):
            xb = X[bi:bi + a.batch].to(device)
            out = model(past_values=xb)
            recon = out.reconstruction_outputs if hasattr(out, 'reconstruction_outputs') else out[1]
            e = ((recon[:, -1, :] - xb[:, -1, :]) ** 2).mean(dim=1).cpu().numpy()
            berr.extend(e.tolist())
            if (bi // a.batch) % 20 == 0:
                print(f"    [기준선] {min(bi + a.batch, len(X))}/{len(X)}", flush=True)
    berr = np.asarray(berr, dtype='float64')
    emed = float(np.median(berr))
    emad = float(np.median(np.abs(berr - emed))) or 1e-9
    ep95 = float(np.percentile(berr, 95))
    ep99 = float(np.percentile(berr, 99))
    sc = json.load(open(os.path.join(a.out, 'scaler.json'), encoding='utf-8'))
    sc['err_median'] = emed          # 정상 중앙값 (P50)
    sc['err_mad'] = emad
    sc['err_p95'] = ep95             # ★ 경계선: 정상의 95%는 이 아래 → 추론 정규화 중심
    sc['err_p99'] = ep99
    sc['baseline_n'] = int(len(berr))
    json.dump(sc, open(os.path.join(a.out, 'scaler.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f"[기준선] 정상오차 P50={emed:.6f} P95={ep95:.6f} P99={ep99:.6f} (n={len(berr)}) → scaler.json 고정", flush=True)

    with open(os.path.join(a.out, 'train_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': TSPULSE_MODEL, 'channels': n_ch, 'context': C,
                   'epochs': a.epochs, 'normal_windows': len(windows),
                   'features': cols, 'device': device},
                  f, ensure_ascii=False, indent=2)
    print(f"🎉 학습 완료 → {a.out}/model  (scaler.json 포함)")
    print("다음: tspulse_infer.py 로 매분 이상점수 생성 → 검증_선행성.py")


if __name__ == '__main__':
    main()

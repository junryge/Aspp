#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chronos-2 covariate(다변량) → TightLoop Sentinel  (고도화 경로)
==============================================================
Chronos-2 만의 강점: past covariate 를 zero-shot 으로 예측에 반영.
EDA(FINDINGS_ml고도화.md)에서 찾은 선행지표를 예측에 직접 넣는다:

    타깃      : M16HUB.QUE.TIME.AVGTOTALTIME1MIN (반송시간)
    covariate : 리프터 큐(LFT.*.6F_TO_3F), 완료량(QUE.ALL.CURRENTQCOMPLETED),
                컨베이어/큐 등 — 정체를 '선행'하는 신호들

    [예측·Chronos-2 predict_df]  target+covariate 이력 → 미래 q10/q50/q90
             │
             ▼
    [행동·TightLoop Sentinel]     분포 → 경보·예비·center·tail·lead

⚠ 중요:
  · Chronos-2 API(predict_df)는 chronos-forecasting>=2.0 필요.
    설치 버전에 따라 인자명이 다를 수 있으니 아래 CALL 부분을 model card 로 확인.
  · 매분 전체 백테스트는 파운데이션 모델 호출이 많아 느리다 → --stride 로 간격 조절.
    실시간 운영은 1분당 1회라 문제없음.

사용:
    pip install -r requirements.txt
    python3 run_chronos2_covariates.py \
        --data JUNE.CSV --horizon 10 --stride 5 \
        --threshold 12.0 --model amazon/chronos-2 \
        --covariates auto            # auto=선행지표 자동선택 / 또는 컬럼명 나열
"""
from __future__ import annotations

import argparse

from data_loader import load_csv, load_any, CORE_SIGNALS
from calibrate import percentile
from sentinel import TightLoopSentinel, SentinelConfig

TARGET_DEFAULT = "M16HUB.QUE.TIME.AVGTOTALTIME1MIN"

# EDA에서 확인된 대표 선행지표 (없으면 자동 스킵)
LEADING_COVARIATES = [
    "M16HUB.QUE.ALL.CURRENTQCOMPLETED",
    "M16HUB.LFT.6ABL6031.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.SENDFAB.TO_M14B_CURRENTQCNT",
    "M16HUB.QUE.M16TOM14B.MESCURRENTQCNT",
    "M16HUB.QUE.M14TOM16.MESCURRENTQCNT",
    "M16HUB.QUE.ALL.CURRENTQCREATED",
]


def auto_select_covariates(sd, target, horizon, k=8):
    """대상 데이터에서 |corr(col[t], target[t+H])| 상위 k 컬럼을 covariate 로."""
    import math
    tgt = _ffill(sd.signal(target))
    N = len(tgt)

    def corr(col):
        x = _ffill(sd.signal(col))
        xs, ys = [], []
        for t in range(N - horizon):
            xs.append(x[t]); ys.append(tgt[t + horizon])
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        sy = math.sqrt(sum((b - my) ** 2 for b in ys))
        if sx < 1e-9 or sy < 1e-9:
            return 0.0
        return sum((xs[i]-mx)*(ys[i]-my) for i in range(len(xs))) / (sx*sy)

    cands = []
    for c in sd.columns:
        if c == target:
            continue
        vals = [v for v in sd.signal(c) if v is not None]
        if len(vals) < N * 0.5 or len(set(vals)) < 4:
            continue
        cands.append((abs(corr(c)), c))
    cands.sort(reverse=True)
    return [c for _, c in cands[:k]]


def _ffill(vals):
    out, last = [], None
    for v in vals:
        if v is not None:
            last = v
        out.append(last if last is not None else 0.0)
    return out


def build_context_df(pd, times, series_by_col, target, upto):
    """origin=upto 까지의 이력을 Chronos-2 long-format DataFrame 으로."""
    rows = []
    for t in range(upto + 1):
        row = {"id": "hubroom", "timestamp": times[t],
               "target": series_by_col[target][t]}
        for c, vals in series_by_col.items():
            if c != target:
                row[c] = vals[t]
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_with_covariates(pipeline, pd, ctx_df, horizon, times_future,
                             freq="min"):
    """
    Chronos-2 predict_df 호출 → 타깃의 q10/q50/q90 리스트 반환.
    freq="min" 을 명시해 주기 자동추론 실패(Could not infer frequency) 방지.
    (데일리 파일 병합 시 경계에서 추론이 실패할 수 있어 1분 주기를 못박음)
    """
    try:
        pred = pipeline.predict_df(
            ctx_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            freq=freq,
        )
    except TypeError:
        # 구버전이 freq 인자를 안 받으면 timestamp 를 규칙적 주기로 재생성해 재시도
        ctx_df = ctx_df.copy()
        ctx_df["timestamp"] = pd.date_range(
            end=ctx_df["timestamp"].iloc[-1], periods=len(ctx_df), freq=freq)
        pred = pipeline.predict_df(
            ctx_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
    # 반환 DataFrame 에서 분위수 컬럼 추출 (버전따라 '0.1' / 'q0.1' 등)
    def qcol(pred, q):
        for name in (str(q), f"q{q}", f"{q:.1f}", f"quantile_{q}"):
            if name in pred.columns:
                return pred[name].tolist()[:horizon]
        # fallback: 숫자형 컬럼 중 근사
        raise KeyError(f"분위수 컬럼 {q} 를 predict_df 결과에서 못 찾음: {list(pred.columns)}")
    return qcol(pred, 0.1), qcol(pred, 0.5), qcol(pred, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--signal", default=TARGET_DEFAULT)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--context", type=int, default=180)
    ap.add_argument("--stride", type=int, default=5,
                    help="백테스트 평가 간격(분). 실시간은 1")
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--pct", type=float, default=0.99)
    ap.add_argument("--covariates", nargs="+", default=["auto"],
                    help="'auto' 자동선택 / 또는 컬럼명 나열")
    ap.add_argument("--out", default=None, help="평가 시점별 액션 CSV 저장 경로")
    args = ap.parse_args()

    # 데이터 (글롭/여러파일 병합. auto covariate 선택 위해 전체 컬럼 로드)
    sd = load_any(args.data, None)
    if args.signal not in sd.columns:
        raise SystemExit(f"타깃 {args.signal} 없음")

    # covariate 선택
    if args.covariates == ["auto"]:
        covs = auto_select_covariates(sd, args.signal, args.horizon, k=8)
    else:
        covs = [c for c in args.covariates if c in sd.columns]
    covs = [c for c in covs if c in sd.columns]

    # 임계
    if args.threshold is not None:
        threshold = args.threshold
    else:
        sv = sorted(v for v in sd.signal(args.signal) if v is not None)
        threshold = round(percentile(sv, args.pct), 3)

    print("=" * 72)
    print(" Chronos-2 covariate(다변량) → TightLoop Sentinel")
    print(f" 타깃: {args.signal} | 지평 {args.horizon}분 | 임계 {threshold}")
    print(f" covariate {len(covs)}개: {[c.split('.')[-1][:18] for c in covs]}")
    print("=" * 72)

    # 모델 로드
    try:
        from chronos import Chronos2Pipeline
        import pandas as pd
        from forecaster import _resolve_model, _auto_device
        model_path = _resolve_model(args.model)      # 로컬 폴더 자동 감지
        dev = args.device or _auto_device()           # GPU 없으면 cpu
        print(f" 모델: {model_path} | device: {dev}")
        pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=dev)
    except Exception as e:
        print(f"⚠ Chronos-2 로드 실패: {e!r}")
        print("  → chronos-forecasting>=2.0 + torch + pandas 설치된 GPU 환경에서 실행하세요.")
        print("  (이 covariate 경로는 baseline 폴백이 없습니다 — 실모델 전용)")
        raise SystemExit(1)

    # 시리즈 정리 (타깃+covariate)
    series = {args.signal: _ffill(sd.signal(args.signal))}
    for c in covs:
        series[c] = _ffill(sd.signal(c))
    times = sd.times

    # 행동 계층
    sen = TightLoopSentinel(SentinelConfig(threshold=threshold))

    # strided 백테스트
    N = len(times)
    origins = list(range(max(10, args.context), N - args.horizon, args.stride))
    alarms = []
    records = []   # (time, signal_value, action) — CSV 저장용 전체 시점
    for oi, t in enumerate(origins):
        lo = max(0, t - args.context)
        # context: lo..t 이력만 (인과적)
        ctx_series = {c: v[lo:t + 1] for c, v in series.items()}
        ctx_df = build_context_df(pd, times[lo:t + 1], ctx_series,
                                  args.signal, t - lo)
        q10, q50, q90 = forecast_with_covariates(
            pipeline, pd, ctx_df, args.horizon, times[t + 1:t + 1 + args.horizon])
        a = sen.step(q10, q50, q90)
        records.append((times[t], series[args.signal][t], a))
        if a.stage >= 2:
            alarms.append((times[t], a))
        if oi % 50 == 0:
            print(f"  진행 {oi}/{len(origins)} ...")

    print(f"\n■ 문제 예측 경보 {len(alarms)}건 (stage≥2, stride={args.stride}분):")
    for tt, a in alarms:
        lead = f", 약 {a.lead_min}분 선제" if a.lead_min else ""
        print(f"  {tt.strftime('%m-%d %H:%M')} (초과확률 {a.exceed_prob:.2f}{lead})"
              f" → {a.recommendation}")

    # CSV 저장 (평가 시점별 전체 액션 — 메인 러너와 동일 스키마)
    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            w.writerow(["datetime", "signal_value", "stage", "stage_name",
                        "exceed_prob", "lead_min", "center_adjust",
                        "reserve_adjust", "tail_upper", "tail_lower",
                        "covariates", "recommendation"])
            covtxt = ";".join(covs)
            for tt, sval, a in records:
                w.writerow([tt.strftime("%Y-%m-%d %H:%M:%S"), sval,
                            a.stage, a.stage_name, a.exceed_prob,
                            a.lead_min if a.lead_min is not None else "",
                            a.center_adjust, a.reserve_adjust,
                            a.tail_upper, a.tail_lower, covtxt, a.recommendation])
        print(f"\n평가 시점별 액션 저장: {args.out}  ({len(records)}행)")


if __name__ == "__main__":
    main()

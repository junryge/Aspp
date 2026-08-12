#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect.py — Chronos-2 로 이동평균을 예측해 '심각한 정체'를 선제 감지
====================================================================
설계 근거 (RAW/FINDINGS_RAW분석.md + buildup 분석):
  · 순간값은 78%가 1분 블립(노이즈) → 예측 불가 ⇒ 이동평균을 타깃으로
  · 심각 사건(지속 10분+, 평균 52분) 18건 중 15건(83%)이 사전 buildup 있음
    ⇒ 예측 가능. 저부하→상승형(A)과 고부하 지속형(B) 두 패턴.

동작 (매 분, 인과적 — 직전까지의 데이터만):
    이동평균 이력 ──Chronos-2──▶ 미래 H분 이동평균 분포(q10/q50/q90)
                                        │
                     P(미래 이동평균 ≥ 임계) 계산
                                        │
    stage 3 = 지금 이미 임계 초과 (진행 중)
    stage 2 = h분 뒤 초과 예상  ← 선제 경보, lead = h
    stage 1 = 확률 중간 (관찰)
    stage 0 = 정상

Chronos-2 미설치/로드실패 시 baseline 예측기로 폴백 (로직 검증용).
"""
from __future__ import annotations

import argparse
import csv
import math

from data import load, moving_avg, learn_threshold, TARGET, LEADING, TIME_COL


# ──────────────────────────────────────────────────────────────
# 예측기
# ──────────────────────────────────────────────────────────────
def auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_model(path: str) -> str:
    """로컬 폴더가 있으면 그걸 쓰고(오프라인), 없으면 HF 식별자 그대로."""
    import os
    if os.path.isdir(path):
        return path
    base = os.path.basename(path)
    for c in (f"./models/{base}", f"./{base}", "./chronos_2", "./models/chronos_2"):
        if os.path.isdir(c):
            return c
    return path


class Forecaster:
    """Chronos-2 (배치 예측). 실패 시 baseline 폴백."""

    def __init__(self, model="chronos_2", device=None):
        self.pipe = None
        self.err = None
        self.backend = "baseline"
        try:
            from chronos import Chronos2Pipeline
            mp = resolve_model(model)
            dev = device or auto_device()
            self.pipe = Chronos2Pipeline.from_pretrained(mp, device_map=dev)
            self.backend = mp.rstrip("/").split("/")[-1]
            self.device = dev
        except Exception as e:
            self.err = repr(e)

    def predict(self, contexts: list[list[float]], horizon: int) -> list[dict]:
        """contexts 여러 개를 한 번에 → [{'q10','q50','q90'}, ...]"""
        if self.pipe is not None:
            try:
                import torch
                L = min(len(c) for c in contexts)
                ctx = torch.tensor([c[-L:] for c in contexts], dtype=torch.float32)
                qs, _ = self.pipe.predict_quantiles(
                    context=ctx, prediction_length=horizon,
                    quantile_levels=[0.1, 0.5, 0.9])
                # 반환 축 순서가 버전에 따라 다를 수 있음:
                #   [series, horizon, quantile]  또는  [series, quantile, horizon]
                out = []
                for s in range(qs.shape[0]):
                    a = qs[s]
                    if a.shape[0] == horizon and a.shape[-1] == 3:
                        arr = a.tolist()                       # [horizon][3]
                    elif a.shape[0] == 3:
                        t = a.tolist()                         # [3][horizon]
                        arr = [[t[0][h], t[1][h], t[2][h]] for h in range(horizon)]
                    else:
                        raise ValueError(
                            f"예상 못한 predict_quantiles 형태: {tuple(a.shape)} "
                            f"(horizon={horizon})")
                    out.append({"q10": [float(r[0]) for r in arr],
                                "q50": [float(r[1]) for r in arr],
                                "q90": [float(r[2]) for r in arr]})
                return out
            except Exception as e:
                # 실모델 예측 실패 → 즉시 알린다 (조용한 폴백은 결과를 오인시킴)
                self.err = repr(e)
                self.pipe = None
                self.backend = "baseline"
                print("=" * 72)
                print("⚠ 실모델 예측 실패 → baseline 으로 폴백합니다.")
                print(f"   원인: {self.err}")
                print("   이 결과는 Chronos-2 성적이 아닙니다. 원인 해결 후 재실행하세요.")
                print("=" * 72)
        return [_baseline(c, horizon) for c in contexts]


def _baseline(ctx: list[float], horizon: int) -> dict:
    """EWMA + 추세 외삽 (모델 없을 때 로직 검증용)."""
    x = [v for v in ctx if v is not None and math.isfinite(v)]
    if len(x) < 3:
        last = x[-1] if x else 0.0
        return {k: [last] * horizon for k in ("q10", "q50", "q90")}
    lvl = x[0]
    for v in x[1:]:
        lvl = 0.35 * v + 0.65 * lvl
    tail = x[-min(len(x), 10):]
    n = len(tail); xs = list(range(n))
    mx = sum(xs) / n; my = sum(tail) / n
    den = sum((i - mx) ** 2 for i in xs) or 1.0
    slope = sum((xs[i] - mx) * (tail[i] - my) for i in range(n)) / den
    d = [x[i] - x[i - 1] for i in range(1, len(x))]
    md = sum(d) / len(d)
    sd = math.sqrt(max(sum((v - md) ** 2 for v in d) / max(1, len(d) - 1), 1e-9))
    out = {"q10": [], "q50": [], "q90": []}
    for h in range(1, horizon + 1):
        c = lvl + slope * h
        s = sd * math.sqrt(h)
        out["q10"].append(c - 1.2816 * s)
        out["q50"].append(c)
        out["q90"].append(c + 1.2816 * s)
    return out


# ──────────────────────────────────────────────────────────────
# 분위수 → 초과확률
# ──────────────────────────────────────────────────────────────
def exceed_prob(q10, q50, q90, threshold) -> float:
    """(q10,.1)(q50,.5)(q90,.9) 로 CDF 근사 → P(X > threshold)."""
    a, b, c = sorted((q10, q50, q90))
    pts = [(a, 0.10), (b, 0.50), (c, 0.90)]
    x = threshold
    if x <= pts[0][0]:
        span = (pts[1][0] - pts[0][0]) or 1e-9
        p = pts[0][1] + (pts[1][1] - pts[0][1]) / span * (x - pts[0][0])
    elif x >= pts[2][0]:
        span = (pts[2][0] - pts[1][0]) or 1e-9
        p = pts[2][1] + (pts[2][1] - pts[1][1]) / span * (x - pts[2][0])
    else:
        p = 0.5
        for i in range(2):
            x0, p0 = pts[i]; x1, p1 = pts[i + 1]
            if x0 <= x <= x1:
                span = (x1 - x0) or 1e-9
                p = p0 + (p1 - p0) * (x - x0) / span
                break
    return max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, p))))


# ──────────────────────────────────────────────────────────────
# 메인 감지 루프
# ──────────────────────────────────────────────────────────────
def run(series, threshold, window=10, horizon=15, context=90,
        p_on=0.6, p_off=0.4, stride=1, model="chronos_2", device=None,
        verbose=True):
    times = series.times
    N = len(times)
    raw = series.get(TARGET)
    sm = moving_avg(series.filled(TARGET), window)

    f = Forecaster(model, device)
    if verbose:
        print("=" * 72)
        print(" detect — Chronos-2 이동평균 예측 → 심각 정체 선제 감지")
        print(f" backend={f.backend} | {window}분평균 | 지평 {horizon}분 | 임계 {threshold}")
        print(f" 데이터 {N}분  {times[0]} ~ {times[-1]}")
        if f.pipe is None:
            print(f" ⚠ Chronos-2 로드 실패 → baseline 폴백: {f.err}")
        print("=" * 72)

    rows = []
    active = False          # 히스테리시스 상태
    last = None             # stride 사이 재사용
    for t in range(N):
        cur = sm[t]

        # 1) 이미 초과 → stage 3 (진행 중)
        if cur >= threshold:
            rows.append([times[t], raw[t], 3, 1.0, "", "진행중",
                         f"이동평균 {cur:.1f} ≥ 임계 {threshold}"])
            active = True
            continue

        # 2) 예측 (stride 간격으로만 모델 호출)
        if t % max(1, stride) == 0 or last is None:
            if t < 15:
                rows.append([times[t], raw[t], 0, 0.0, "", "", "정상"])
                continue
            ctx = sm[max(0, t - context):t + 1]
            fc = f.predict([ctx], horizon)[0]
            best_p, lead = 0.0, None
            for h in range(horizon):
                p = exceed_prob(fc["q10"][h], fc["q50"][h], fc["q90"][h], threshold)
                if p > best_p:
                    best_p = p
                if lead is None and p >= p_on:
                    lead = h + 1
            last = (best_p, lead)
        best_p, lead = last

        # 3) 히스테리시스로 단계 판정
        if best_p >= p_on:
            active = True
        elif best_p < p_off:
            active = False
        if active and lead is not None:
            rows.append([times[t], raw[t], 2, best_p, lead, "선제",
                         f"약 {lead}분 뒤 임계 초과 예상 (확률 {best_p:.2f})"])
        elif best_p >= p_off:
            rows.append([times[t], raw[t], 1, best_p, "", "관찰",
                         f"상승 조짐 (확률 {best_p:.2f})"])
        else:
            rows.append([times[t], raw[t], 0, best_p, "", "", "정상"])

        if verbose and t % 20000 == 0 and t:
            print(f"  진행 {t}/{N} ... (backend={f.backend})")
    if verbose and f.backend == "baseline" and f.err:
        print(f"  ※ 실행 중 baseline 폴백됨 — 원인: {f.err}")
    return rows, f.backend


def save(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["datetime", "raw_value", "stage", "prob", "lead_min",
                    "kind", "reason"])
        for t, rv, st, p, lead, kind, why in rows:
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"),
                        "" if rv is None else rv, st, round(p, 3), lead, kind, why])


def main():
    ap = argparse.ArgumentParser(description="심각 정체 선제 감지 (Chronos-2)")
    ap.add_argument("--data", required=True, nargs="+", help="평가 대상 CSV(글롭)")
    ap.add_argument("--config", default=None,
                    help="학습 결과(model_config.json) — 임계·창을 그대로 재사용")
    ap.add_argument("--threshold", type=float, default=None,
                    help="임계(이동평균). 미지정 시 --config/--train/대상데이터")
    ap.add_argument("--train", nargs="+", default=None, help="임계 학습용 CSV")
    ap.add_argument("--pct", type=float, default=0.99)
    ap.add_argument("--window", type=int, default=10, help="이동평균 창(분)")
    ap.add_argument("--horizon", type=int, default=15, help="예측 지평(분)")
    ap.add_argument("--context", type=int, default=90,
                    help="모델이 보는 직전 이력(분). 기본 90분")
    ap.add_argument("--p-on", type=float, default=0.6)
    ap.add_argument("--p-off", type=float, default=0.4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--model", default="chronos_2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # 학습 결과 재사용 (--config) — 명시 인자가 없으면 config 값 사용
    if a.config:
        from data import load_config
        cfg = load_config(a.config)
        if a.threshold is None:
            a.threshold = cfg["threshold"]
        if ap.get_default("window") == a.window:
            a.window = cfg.get("window", a.window)
        print(f"[config] {a.config} → 임계 {a.threshold} · {a.window}분평균 "
              f"(학습 {cfg.get('train_span')}, 사건 {cfg.get('train_events')}건)")

    sd = load(a.data, [TARGET] + LEADING + [TIME_COL])
    if a.threshold is not None:
        thr = a.threshold
    elif a.train:
        tr = load(a.train, [TARGET, TIME_COL])
        thr = learn_threshold(moving_avg(tr.filled(TARGET), a.window), a.pct)
        print(f"[학습] 임계 = {thr} (학습 {len(tr)}분, p{a.pct*100:.1f})")
    else:
        thr = learn_threshold(moving_avg(sd.filled(TARGET), a.window), a.pct)
        print(f"[주의] 대상데이터에서 임계 산출 = {thr} (leakage 가능)")

    rows, backend = run(sd, thr, a.window, a.horizon, a.context,
                        a.p_on, a.p_off, a.stride, a.model, a.device)
    save(rows, a.out)
    n2 = sum(1 for r in rows if r[2] == 2)
    n3 = sum(1 for r in rows if r[2] == 3)
    print(f"\n저장: {a.out}")
    print(f"  선제경보(stage2) {n2}분 | 진행중(stage3) {n3}분")
    if backend == "baseline":
        print("  ※ baseline 폴백 결과 — 실 Chronos-2 로 다시 돌리세요.")


if __name__ == "__main__":
    main()

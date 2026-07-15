#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
결합기 — CUSUM(조기경보) + Chronos-2(정밀예측) 액션 CSV 병합
============================================================
두 접근의 장점을 합친다:
  · CUSUM   : 선행지표 buildup 을 10~30분 전에 감지 (lead 확보)
  · Chronos : 값 예측 + Sentinel 조치 (정밀·확률·예비/center)

병합 모드:
  union  : 둘 중 하나라도 경보면 경보 (recall·lead 최대) — 기본
  and    : 둘 다 경보일 때만 (precision 최대, 헛울림 최소)
  cusum_lead_chronos_confirm :
           CUSUM 이 먼저 울리고 Chronos 가 뒤이어 확인하면 유지
           (CUSUM lead + Chronos 로 헛울림 필터 — '10분전+정밀'의 핵심)

사용:
  python combine.py --a actions_cusum.csv --b actions_202606.csv \
                    --mode union --out actions_combined.csv
그다음 score.py 로 채점.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta


def read_actions(path):
    d = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                t = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            d[t] = {
                "signal_value": r.get("signal_value", ""),
                "stage": int(r.get("stage", 0) or 0),
                "exceed_prob": float(r.get("exceed_prob", 0) or 0),
                "rec": r.get("recommendation", ""),
                "dir": r.get("dir", ""),
            }
    return d


def within(t, dset, win):
    """t 기준 ±win분 내에 stage>=2 경보가 있으면 그 중 최고 stage 반환."""
    best = 0
    for dt in (t + timedelta(minutes=m) for m in range(-win, win + 1)):
        row = dset.get(dt)
        if row and row["stage"] >= 2:
            best = max(best, row["stage"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="액션 CSV A (예: CUSUM)")
    ap.add_argument("--b", required=True, help="액션 CSV B (예: Chronos)")
    ap.add_argument("--mode", default="union",
                    choices=["union", "and", "cusum_lead_chronos_confirm"])
    ap.add_argument("--win", type=int, default=10,
                    help="두 경보를 같은 사건으로 볼 시간창(분)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = read_actions(args.a)
    B = read_actions(args.b)
    all_times = sorted(set(A) | set(B))

    rows = []
    for t in all_times:
        a = A.get(t); b = B.get(t)
        sa = a["stage"] if a else 0
        sb = b["stage"] if b else 0

        if args.mode == "union":
            stage = max(sa, sb)
        elif args.mode == "and":
            # 둘 다(±win) 경보일 때만
            stage = max(sa, sb) if (sa >= 2 and within(t, B, args.win)) or \
                                    (sb >= 2 and within(t, A, args.win)) else 0
        else:  # cusum_lead_chronos_confirm
            # A(CUSUM)가 울렸고, B(Chronos)가 ±win 안에서 확인하면 유지
            if sa >= 2 and within(t, B, args.win):
                stage = max(sa, sb)
            elif sb >= 2:            # Chronos 단독도 경보(값 급변 캐치)
                stage = sb
            else:
                stage = 0

        prob = max(a["exceed_prob"] if a else 0, b["exceed_prob"] if b else 0)
        sval = (a and a["signal_value"]) or (b and b["signal_value"]) or ""
        rec = " + ".join(x for x in [(a and a["rec"]) or "", (b and b["rec"]) or ""]
                         if x and x != "정상")
        rows.append((t, sval, stage, prob, rec))

    with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["datetime", "signal_value", "stage", "stage_name",
                    "exceed_prob", "lead_min", "center_adjust", "reserve_adjust",
                    "tail_upper", "tail_lower", "recommendation"])
        for t, sval, stage, prob, rec in rows:
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), sval, stage,
                        f"{stage}단계", round(prob, 3), "", "", "", "", "",
                        rec or "정상"])
    n_alarm = sum(1 for r in rows if r[2] >= 2)
    print(f"결합({args.mode}) 완료 → {args.out}  (경보 분 {n_alarm}/{len(rows)})")


if __name__ == "__main__":
    main()

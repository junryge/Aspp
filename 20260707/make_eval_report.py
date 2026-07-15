#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
평가 데이터 생성 — 6월 정체 사건별 감지 결과표
===============================================
액션 CSV(union/confirm 등) + 실 데이터 → 사건 단위 평가표 CSV.

각 실제 정체 사건마다:
  잡았나(감지/놓침) · 몇 분 전에(lead) · 어느 감지기·사유
+ 오탐(헛울림) 목록도 함께.

사용:
  python make_eval_report.py --actions act_comb.csv --data "RAW6/*.CSV" \
        --threshold 16.7 --pre 15 --gap 10 --label union --out eval_union.csv
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta

from data_loader import load_any

SIGNAL = "M16HUB.QUE.TIME.AVGTOTALTIME1MIN"


def read_actions(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                t = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            rows.append({
                "t": t, "stage": int(r.get("stage", 0) or 0),
                "prob": float(r.get("exceed_prob", 0) or 0),
                "rec": r.get("recommendation", ""),
                "dir": r.get("dir", ""),
            })
    rows.sort(key=lambda x: x["t"])
    return rows


def episodes(times, flags, gap_min=10):
    eps, start, prev = [], None, None
    for t, fl in zip(times, flags):
        if fl:
            if start is None:
                start = t
            elif (t - prev).total_seconds() / 60 > gap_min:
                eps.append((start, prev)); start = t
            prev = t
    if start is not None:
        eps.append((start, prev))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--pre", type=int, default=15)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 실제 정체 사건 (peak 포함)
    sd = load_any(args.data, [SIGNAL, "CRT_TM"])
    times, vals = sd.times, sd.signal(SIGNAL)
    flags = [(v is not None and v >= args.threshold) for v in vals]
    gt = episodes(times, flags, args.gap)
    vmap = {times[i]: vals[i] for i in range(len(times))}

    def peak(gs, ge):
        vv = [vmap[t] for t in times if gs <= t <= ge and vmap[t] is not None]
        return max(vv) if vv else None

    # 예측 경보 사건
    acts = read_actions(args.actions)
    a_times = [a["t"] for a in acts]
    a_flags = [a["stage"] >= 2 for a in acts]
    alarms = episodes(a_times, a_flags, args.gap)
    amap = {a["t"]: a for a in acts}

    pre = timedelta(minutes=args.pre)

    def overlaps(as_, ae, gs, ge):
        return as_ <= ge and ae >= (gs - pre)

    def _dir(a):
        # dir 컬럼 있으면 사용, 없으면 사유 텍스트에서 방향 추출
        if a and a.get("dir"):
            return a["dir"]
        rec = a.get("rec", "") if a else ""
        for d in ("남측", "북측", "허브", "브릿지"):
            if d in rec:
                return d
        if "격상" in rec or "임계" in rec:   # Chronos 단독 경보
            return "Chronos"
        return ""

    def alarm_detail(as_):
        # 경보 시작 시점의 사유/방향/확률
        a = amap.get(as_)
        return (a["rec"] if a else "", _dir(a),
                round(a["prob"], 2) if a else "")

    rows = []
    matched_alarms = set()
    n = 0
    for (gs, ge) in gt:
        n += 1
        matched = [(as_, ae) for (as_, ae) in alarms if overlaps(as_, ae, gs, ge)]
        pk = peak(gs, ge)
        if matched:
            first = min(a[0] for a in matched)
            for m in matched:
                matched_alarms.add(m)
            lead = max(0, round((gs - first).total_seconds() / 60))
            pred_t = (first + timedelta(minutes=lead)).strftime("%m-%d %H:%M")
            rec, dirn, prob = alarm_detail(first)
            rows.append(["정체-감지", n, gs.strftime("%m-%d %H:%M"),
                         ge.strftime("%H:%M"),
                         round(pk, 1) if pk else "",
                         first.strftime("%m-%d %H:%M"), pred_t, lead,
                         "≥10분" if lead >= 10 else "5~9분" if lead >= 5 else "<5분",
                         dirn, prob, rec])
        else:
            rows.append(["정체-놓침", n, gs.strftime("%m-%d %H:%M"),
                         ge.strftime("%H:%M"),
                         round(pk, 1) if pk else "", "", "", "", "놓침", "", "", ""])

    # 오탐
    fa = 0
    for (as_, ae) in alarms:
        if (as_, ae) in matched_alarms:
            continue
        if any(overlaps(as_, ae, gs, ge) for (gs, ge) in gt):
            continue
        fa += 1
        rec, dirn, prob = alarm_detail(as_)
        rows.append(["오탐", "", as_.strftime("%m-%d %H:%M"),
                     ae.strftime("%H:%M"), "", as_.strftime("%m-%d %H:%M"),
                     "", "", "헛울림", dirn, prob, rec])

    caught = sum(1 for r in rows if r[0] == "정체-감지")
    leads = [r[7] for r in rows if r[0] == "정체-감지" and isinstance(r[7], int)]
    lead10 = sum(1 for l in leads if l >= 10)
    mean_lead = round(sum(leads) / len(leads), 1) if leads else 0

    with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow([f"# 평가: {args.label} | 정체 {len(gt)}건 | 감지 {caught} "
                    f"(recall {caught/len(gt)*100:.0f}%) | ≥10분전 {lead10}건 | "
                    f"평균lead {mean_lead}분 | 오탐 {fa}건"])
        w.writerow(["구분", "사건#", "정체시작", "정체종료", "최고반송시간(분)",
                    "감지시각", "예측시간", "lead(분)", "lead구간", "감지기/방향",
                    "확률", "사유"])
        for r in rows:
            w.writerow(r)

    print(f"[{args.label}] 정체 {len(gt)}건 · 감지 {caught}(recall {caught/len(gt)*100:.0f}%) "
          f"· ≥10분전 {lead10}건 · 평균lead {mean_lead}분 · 오탐 {fa}건")
    print(f"→ 저장: {args.out}")


if __name__ == "__main__":
    main()

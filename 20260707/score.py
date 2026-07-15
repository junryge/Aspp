#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
채점 — 예측 경보를 실제 정체와 대조 (hit / false alarm / lead)
================================================================
run_chronos_sentinel.py / run_chronos2_covariates.py 가 저장한 액션 CSV 를,
실제 6월 데이터의 '진짜 정체'와 대조해서 성적을 낸다.

정답(ground truth) 정체:
  · 기본: raw6 데이터에서 신호가 임계를 실제로 넘은 구간(에피소드)
  · (선택) 별도 사건 CSV 가 있으면 --events 로 지정

산출:
  · 정체 에피소드 수 / 잡은 수(hit) / 놓친 수(miss)
  · 경보 에피소드 수 / 헛울림(false alarm) 수
  · 평균 lead(분), precision/recall

사용:
  python score.py --actions actions_cov_202606.csv --data "raw6/*.csv" \
                  --signal M16HUB.QUE.TIME.AVGTOTALTIME1MIN --threshold 16.7 \
                  --pre 15 --gap 10
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime

from data_loader import load_any


def read_actions(path):
    """액션 CSV → [(datetime, stage, exceed_prob, lead_min)]"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                dt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            stage = int(r.get("stage", 0) or 0)
            p = float(r.get("exceed_prob", 0) or 0)
            lead = r.get("lead_min", "")
            lead = int(lead) if str(lead).strip().isdigit() else None
            rows.append((dt, stage, p, lead))
    rows.sort(key=lambda x: x[0])
    return rows


def episodes_from_flags(times, flags, gap_min=10):
    """True 구간을 에피소드로 묶되, gap_min 분 이내 간격은 하나로 병합."""
    eps = []
    start = prev = None
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
    ap.add_argument("--actions", required=True, help="예측 액션 CSV")
    ap.add_argument("--data", required=True, nargs="+", help="실제 raw6 CSV(정답용)")
    ap.add_argument("--signal", default="M16HUB.QUE.TIME.AVGTOTALTIME1MIN")
    ap.add_argument("--threshold", type=float, required=True,
                    help="정체 판정 임계 (학습값, 예: 16.7)")
    ap.add_argument("--pre", type=int, default=15,
                    help="정체 시작 몇 분 전까지의 경보를 '사전감지'로 인정")
    ap.add_argument("--gap", type=int, default=10, help="에피소드 병합 간격(분)")
    ap.add_argument("--min-stage", type=int, default=2, help="경보로 볼 최소 stage")
    args = ap.parse_args()

    # 1) 정답 정체 에피소드 (실제 신호가 임계 초과)
    sd = load_any(args.data, [args.signal, "CRT_TM"])
    times = sd.times
    vals = sd.signal(args.signal)
    gt_flags = [(v is not None and v >= args.threshold) for v in vals]
    gt = episodes_from_flags(times, gt_flags, args.gap)

    # 2) 예측 경보 에피소드
    acts = read_actions(args.actions)
    a_times = [a[0] for a in acts]
    a_flags = [a[1] >= args.min_stage for a in acts]
    alarms = episodes_from_flags(a_times, a_flags, args.gap)

    # 3) 매칭
    pre = _min(args.pre)

    def overlaps(as_, ae, gs, ge):
        # 경보구간이 [정체시작-pre, 정체종료] 와 겹치면 매칭
        return as_ <= ge and ae >= (gs - pre)

    # 정체 잡음/놓침 + lead
    caught, miss, leads = 0, 0, []
    for (gs, ge) in gt:
        matched = [(as_, ae) for (as_, ae) in alarms if overlaps(as_, ae, gs, ge)]
        if matched:
            caught += 1
            first_alarm = min(a[0] for a in matched)     # 가장 이른 경보 시작
            lead = (gs - first_alarm).total_seconds() / 60
            leads.append(max(0, round(lead)))
        else:
            miss += 1

    # 헛울림: 어떤 정체와도 안 겹치는 경보
    false_alarms = sum(
        0 if any(overlaps(as_, ae, gs, ge) for (gs, ge) in gt) else 1
        for (as_, ae) in alarms)

    n_gt = len(gt)
    n_al = len(alarms)
    recall = caught / n_gt if n_gt else 0
    precision = (n_al - false_alarms) / n_al if n_al else 0
    mean_lead = round(sum(leads) / len(leads), 1) if leads else None

    print("=" * 60)
    print(" 채점 결과 (실제 정체 vs 예측 경보)")
    print(f" 정답 임계 {args.threshold} | pre {args.pre}분 | 병합 {args.gap}분")
    print("=" * 60)
    print(f" 실제 정체 에피소드 : {n_gt}건")
    print(f"   └ 잡음(hit)     : {caught}건")
    print(f"   └ 놓침(miss)    : {miss}건")
    print(f" 예측 경보 에피소드 : {n_al}건")
    print(f"   └ 헛울림(false) : {false_alarms}건")
    print("-" * 60)
    print(f" Recall(재현율, 정체 잡은 비율)  : {recall:.1%}")
    print(f" Precision(정밀도, 경보 맞은 비율): {precision:.1%}")
    print(f" 평균 사전감지 lead            : {mean_lead}분")
    print("=" * 60)
    if mean_lead is not None and mean_lead < 5:
        print(" ※ lead가 짧으면 --pre 를 늘리거나(운영 허용 관점),")
        print("   horizon/p_on 튜닝 필요. 지금은 '맞추긴 하나 늦게'.")


def _min(m):
    from datetime import timedelta
    return timedelta(minutes=m)


if __name__ == "__main__":
    main()

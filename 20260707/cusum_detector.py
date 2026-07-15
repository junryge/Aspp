#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSUM 4감지기 (조기경보 계층) — 순수 파이썬 포팅
=================================================
밀림_방향_CUSUM.py 를 원본 M16A_HUBROOM_PR CSV 에서 바로 돌도록 포팅.
(features.csv·numpy·pandas 불필요 — data_loader 와 동일 순수 파이썬)

원리: 값이 아니라 '지속적 상승(누적 drift)'을 감지.
  C[i] = max(0, C[i-1] + (x - 평소 - K*표준편차))
  → 순간 튐은 무시, 꾸준히 쌓이면 누적 → 밀림 buildup 을 10~30분 전에 포착.

4감지기 (원본 컬럼 직접):
  남측    M14.QUE.CNV.SOUTHCURRENTQCNT       CUSUM 600
  북측    M14.QUE.CNV.NORTHCURRENTQCNT       CUSUM 600
  허브    M16HUB.STRATE.ALL.FABSTORAGERATIO  CUSUM 300 (+STK≥10% 하드경보)
  브릿지  M16HUB.QUE.TIME.AVGTOTALTIME1MIN   CUSUM 40

Chronos-2 는 '값 예측'이라 급변에 약하지만, CUSUM 은 'buildup 감지'라
선행지표(컨베이어 큐·저장률)의 완만한 상승을 미리 잡아 lead 를 확보한다.
"""
from __future__ import annotations

from data_loader import load_any

# ── 파라미터 (원본과 동일) ──
CUSUM_BASE_WIN = 120     # 기준선 창(분, 과거만)
CUSUM_K = 0.5            # 여유 = K × 과거표준편차
TH_Q = 600.0            # 남측/북측 큐
TH_FAB = 300.0          # 허브 FAB저장
TH_BR = 40.0            # 브릿지타임 (30=민감/40=균형/60=보수)
TH_STK = 10.0           # STK 하드경보(≥10%)

COL_SOUTH = "M14.QUE.CNV.SOUTHCURRENTQCNT"
COL_NORTH = "M14.QUE.CNV.NORTHCURRENTQCNT"
COL_FAB = "M16HUB.STRATE.ALL.FABSTORAGERATIO"
COL_BR = "M16HUB.QUE.TIME.AVGTOTALTIME1MIN"
COL_STK = "M16HUB.STRATE.STK.STORAGERATIO"


def _median(vals):
    s = sorted(vals); n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _std(vals, mean=None):
    n = len(vals)
    if n < 2:
        return 0.0
    m = mean if mean is not None else sum(vals) / n
    return (sum((v - m) ** 2 for v in vals) / n) ** 0.5


def cusum(x, win=CUSUM_BASE_WIN, K=CUSUM_K, min_periods=15):
    """한쪽(상승) CUSUM. x: [float|None]. 결측은 이전값 유지(포워드필)."""
    filled, last = [], None
    for v in x:
        if v is not None:
            last = v
        filled.append(last if last is not None else 0.0)
    n = len(filled)
    C = [0.0] * n
    prev = 0.0
    for i in range(n):
        past = filled[max(0, i - win):i]     # 현재 제외(shift 1) = 과거만
        if len(past) >= min_periods:
            base = _median(past)
            sd = _std(past)
        else:
            base = filled[i]
            sd = 0.0
        prev = max(0.0, prev + (filled[i] - base - K * sd))
        C[i] = prev
    return C


def grade_by_ratio(cu, thr):
    if cu < thr:
        return "", 0.0
    r = cu / thr
    g = "초위험" if r >= 2.5 else "위험" if r >= 1.5 else "경계"
    return g, r


def gated(grade, hour):
    """초위험=밤낮 항상 / 위험·경계=주간(08~19)만. 경계부터 예측→10~30분 lead."""
    if grade == "초위험":
        return grade
    if grade in ("위험", "경계") and 8 <= hour <= 19:
        return grade
    return ""


def run(series_data):
    """
    SeriesData → 분당 CUSUM 감지 결과 리스트.
    반환: [ {datetime, dir, grade, ratio, cu_*}... ] (예측 발동만 grade!='')
    """
    sd = series_data
    times = sd.times
    cu_S = cusum(sd.signal(COL_SOUTH))
    cu_N = cusum(sd.signal(COL_NORTH))
    cu_F = cusum(sd.signal(COL_FAB))
    cu_B = cusum(sd.signal(COL_BR))
    stk = [v if v is not None else 0.0 for v in sd.signal(COL_STK)]

    out = []
    grade_ord = {"": 0, "경계": 1, "위험": 2, "초위험": 3}
    for i, t in enumerate(times):
        gS, rS = grade_by_ratio(cu_S[i], TH_Q)
        gN, rN = grade_by_ratio(cu_N[i], TH_Q)
        gF, rF = grade_by_ratio(cu_F[i], TH_FAB)
        gB, rB = grade_by_ratio(cu_B[i], TH_BR)
        # STK 하드경보 → 허브 위험 승격
        if stk[i] >= TH_STK and grade_ord["위험"] > grade_ord[gF]:
            gF, rF = "위험", max(rF, 1.5)
        h = t.hour
        cand = [
            ("남측", gated(gS, h), rS),
            ("북측", gated(gN, h), rN),
            ("허브", gated(gF, h), rF),
            ("브릿지", gated(gB, h), rB),
        ]
        fired = [(d, g, r) for (d, g, r) in cand if g]
        if fired:
            fired.sort(key=lambda z: grade_ord[z[1]], reverse=True)
            d, g, r = fired[0]
            out.append({"i": i, "datetime": t, "dir": d, "grade": g,
                        "ratio": round(r, 2), "all": fired,
                        "cu": {"남측": cu_S[i], "북측": cu_N[i],
                               "허브": cu_F[i], "브릿지": cu_B[i]}})
    return out, {"times": times, "cu_S": cu_S, "cu_N": cu_N,
                 "cu_F": cu_F, "cu_B": cu_B}


# ── score.py 호환 액션 CSV 로 저장 ──
def to_action_csv(series_data, out_path):
    import csv
    detections, _ = run(series_data)
    by_i = {d["i"]: d for d in detections}
    times = series_data.times
    grade_stage = {"경계": 2, "위험": 3, "초위험": 3}
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["datetime", "signal_value", "stage", "stage_name",
                    "exceed_prob", "lead_min", "center_adjust",
                    "reserve_adjust", "tail_upper", "tail_lower",
                    "dir", "recommendation"])
        brv = series_data.signal(COL_BR)
        for i, t in enumerate(times):
            d = by_i.get(i)
            if d:
                stage = grade_stage[d["grade"]]
                prob = min(1.0, d["ratio"] / 2.5)     # ratio→확률 근사
                rec = f"CUSUM {d['dir']} {d['grade']}(x{d['ratio']})"
                dirn = d["dir"]
            else:
                stage, prob, rec, dirn = 0, 0.0, "정상", ""
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"),
                        brv[i] if brv[i] is not None else "",
                        stage, f"{stage}단계", round(prob, 3), "",
                        "", "", "", "", dirn, rec])
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CUSUM 4감지기 (조기경보)")
    ap.add_argument("--data", required=True, nargs="+", help="원본 CSV(글롭 가능)")
    ap.add_argument("--out", default=None, help="score.py 호환 액션 CSV 저장")
    args = ap.parse_args()

    need = [COL_SOUTH, COL_NORTH, COL_FAB, COL_BR, COL_STK, "CRT_TM"]
    sd = load_any(args.data, need)
    dets, _ = run(sd)
    print(f"데이터 {len(sd)}분 | CUSUM 예측 발동 {len(dets)}회")
    # 구간 압축 출력
    prev_i = None
    for d in dets[:60]:
        print(f"  {d['datetime'].strftime('%m-%d %H:%M')} [{d['dir']} {d['grade']} x{d['ratio']}]")
    if args.out:
        to_action_csv(sd, args.out)
        print(f"\n액션 CSV 저장: {args.out}")

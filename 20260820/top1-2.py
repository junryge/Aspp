#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
top1.py — 상위 1% 임계 · 시점 / 에피소드 / 컬럼 (+ 정체 사건 교차분석)
=====================================================================
"상위 1%"를 세 가지 형태로 동시에 뽑는다. 같은 임계에서 나오는 세 가지 뷰다.

  ① 컬럼      컬럼마다 임계(p99)가 얼마이고, 몇 분 초과했고, 몇 에피소드인지
  ② 에피소드   연속 초과 구간 (시작·종료·지속·피크) — 상위 1%가 언제 뭉쳐서 났나
  ③ 시점      초과한 "분" 목록 (어느 컬럼이 언제 얼마나 초과했나)

    시점 = 점(minute) → 에피소드 = 점을 이은 구간 → 컬럼 = 구간을 요약한 표

`--events` 를 붙이면 여기에 하나가 더 붙는다.

  ④ 사건 교차   학습기간 정체 사건(예: 23건) 각각에 대해
                "그때 어떤 컬럼이 상위 1% 였나 / 몇 분 먼저 올랐나"
                → 원인·선행지표 후보 순위표

사용:
    # 컬럼 전부, 순간값 기준 상위 1%
    python top1.py --data "RAW/*.CSV" --out-dir top1

    # 이동평균 10분 기준 (예측 시스템과 같은 기준)
    python top1.py --data "RAW/*.CSV" --window 10 --out-dir top1_ma

    # 학습기간 정체 사건 23건과 교차분석  ← 고객 요청이 여기까지면 이것
    python top1.py --data "RAW/M16A_HUBROOM_PR_202604*.CSV" \\
                   "RAW/M16A_HUBROOM_PR_202605*.CSV" \\
                   "RAW/M16A_HUBROOM_PR_202606*.CSV" \\
        --window 10 --events --events-config model_config.json \\
        --lead 30 --no-points --out-dir top1_events

    # 상위 0.5% 로 좁히기
    python top1.py --data "RAW/*.CSV" --pct 0.995

산출 (--out-dir 아래):
    top1_columns.csv          컬럼별 임계·초과통계        ← 고객이 먼저 볼 표
    top1_episodes.csv         에피소드 목록
    top1_points.csv           초과 시점 목록 (--no-points 로 생략 가능)
    top1_events.csv           사건별 요약            (--events)
    top1_event_columns.csv    사건 × 컬럼 상세        (--events)
    top1_event_ranking.csv    선행지표 후보 순위      (--events)  ← 원인 분석용
"""
from __future__ import annotations

import argparse
import csv
import os

from data import (load, moving_avg, percentile, find_events, load_config,
                  TARGET, TIME_COL)


# ──────────────────────────────────────────────────────────────
# 임계 초과 구간 → 에피소드
# ──────────────────────────────────────────────────────────────
def episodes(times, vals, threshold, min_duration=1, gap=0):
    """
    vals >= threshold 인 연속 구간을 에피소드로 묶는다.
      gap          : 이 분 수 이내로 끊긴 건 같은 에피소드로 병합 (0 = 병합 안 함)
      min_duration : 이 분 미만 지속은 버림 (1 = 다 남김)
    """
    spans, st, prev = [], None, None
    for i, v in enumerate(vals):
        if v is None:
            continue
        if v >= threshold:
            if st is None:
                st = i
            prev = i
        elif st is not None and (i - prev) > gap:
            spans.append((st, prev))
            st = None
    if st is not None:
        spans.append((st, prev))

    out = []
    for b, e in spans:
        dur = e - b + 1
        if dur < min_duration:
            continue
        seg = [vals[i] for i in range(b, e + 1) if vals[i] is not None]
        pk = max(seg)
        ip = next(i for i in range(b, e + 1) if vals[i] == pk)
        out.append({
            "no": len(out) + 1, "i0": b, "i1": e,
            "t_start": times[b], "t_end": times[e], "duration": dur,
            "peak": pk, "t_peak": times[ip], "mean": sum(seg) / len(seg),
        })
    return out


def smooth(raw, window):
    """이동평균 (결측은 직전 값으로 채운 뒤, 과거만 보는 인과적 계산)."""
    if not window or window <= 1:
        return raw
    filled, last = [], None
    for v in raw:
        if v is not None:
            last = v
        filled.append(last if last is not None else 0.0)
    return moving_avg(filled, window)


# ──────────────────────────────────────────────────────────────
# 컬럼 1개 분석
# ──────────────────────────────────────────────────────────────
def analyze(times, raw, pct, window, min_duration, gap):
    """반환: (stat, episodes, vals) — 상수/데이터부족이면 None"""
    present = [v for v in raw if v is not None]
    if len(present) < 30 or len(set(present)) < 2:
        return None

    vals = smooth(raw, window)
    base = [v for v in vals if v is not None]
    thr = percentile(base, pct)
    over_n = sum(1 for v in vals if v is not None and v >= thr)
    eps = episodes(times, vals, thr, min_duration, gap)
    durs = [e["duration"] for e in eps]

    stat = {
        "n": len(base), "missing": len(raw) - len(present),
        "min": min(base), "p50": percentile(base, 0.50),
        "p95": percentile(base, 0.95), "p99": percentile(base, 0.99),
        "threshold": thr, "max": max(base),
        "over_n": over_n,
        "over_ratio": over_n / len(base) if base else 0.0,
        "ep_n": len(eps), "ep_max": max(durs) if durs else 0,
        "ep_mean": sum(durs) / len(durs) if durs else 0.0,
    }

    # 분위수 임계가 무의미한 컬럼 판정.
    #   대부분이 같은 값(0/1 플래그, 고정 정원값)이면 p99 가 바닥값과 같아져
    #   "상위 1%" 가 전체의 100% 가 되어버린다. 버리지 않고 표시만 한다.
    expect = 1.0 - pct
    if thr <= stat["p50"]:
        stat["verdict"] = "계단형(임계≤p50)"
    elif stat["over_ratio"] > expect * 5:
        stat["verdict"] = "초과과다"
    else:
        stat["verdict"] = "정상"
    stat["ok"] = stat["verdict"] == "정상"
    return stat, eps, vals


# ──────────────────────────────────────────────────────────────
# 사건 × 컬럼 교차
# ──────────────────────────────────────────────────────────────
def cross(events, vals, thr, lead):
    """
    각 정체 사건에 대해 이 컬럼이 어땠는지.
    반환: [{no, hit_n, hit_ratio, lead_min, peak, t_peak}, ...] (겹침 있는 것만)
      lead_min : 사건 시작 기준 첫 초과가 몇 분 먼저였나 (양수 = 먼저 올라옴)
    """
    out = []
    n = len(vals)
    for ev in events:
        b, e = ev["i0"], ev["i1"]
        lo = max(0, b - lead)
        hit = [i for i in range(b, min(e + 1, n))
               if vals[i] is not None and vals[i] >= thr]
        pre = [i for i in range(lo, min(b, n))
               if vals[i] is not None and vals[i] >= thr]
        if not hit and not pre:
            continue
        first = (pre + hit)[0]
        seg = [vals[i] for i in range(lo, min(e + 1, n)) if vals[i] is not None]
        pk = max(seg) if seg else None
        out.append({
            "no": ev["no"],
            "hit_n": len(hit),
            "hit_ratio": len(hit) / ev["duration"] if ev["duration"] else 0.0,
            "pre_n": len(pre),
            "lead_min": b - first,
            "peak": pk,
        })
    return out


# ──────────────────────────────────────────────────────────────
# 출력 유틸
# ──────────────────────────────────────────────────────────────
def write_csv(path, header, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def r(x, n=3):
    return "" if x is None else round(x, n)


def main():
    ap = argparse.ArgumentParser(
        description="상위 1% 임계 → 시점 / 에피소드 / 컬럼 (+ 정체 사건 교차)")
    ap.add_argument("--data", required=True, nargs="+",
                    help='CSV 글롭 — 따옴표 필수. 예: "RAW/*.CSV"')
    ap.add_argument("--pct", type=float, default=0.99,
                    help="임계 분위수. 0.99 = 상위 1% (기본), 0.995 = 상위 0.5%")
    ap.add_argument("--window", type=int, default=0,
                    help="이동평균 창(분). 0 = 순간값(기본), 10 = 예측시스템 기준")
    ap.add_argument("--min-duration", type=int, default=1,
                    help="에피소드 최소 지속(분). 1 = 다 남김, 10 = 큰 것만")
    ap.add_argument("--gap", type=int, default=0,
                    help="이 분 이내로 끊긴 건 같은 에피소드로 병합")
    ap.add_argument("--columns", nargs="+", default=None,
                    help="특정 컬럼만 (생략하면 숫자 컬럼 전부)")
    ap.add_argument("--exclude-contains", nargs="+", default=None,
                    help="이 문자열이 들어간 컬럼 제외")
    ap.add_argument("--from", dest="d0", default=None, help="시작일 YYYY-MM-DD")
    ap.add_argument("--to", dest="d1", default=None, help="종료일 YYYY-MM-DD")
    ap.add_argument("--out-dir", default="top1", help="산출 폴더")
    ap.add_argument("--no-points", action="store_true",
                    help="시점 CSV 생략 (컬럼 많으면 매우 커짐)")
    ap.add_argument("--top", type=int, default=25, help="화면 표시 상위 N개")
    ap.add_argument("--include-degenerate", action="store_true",
                    help="분위수 임계가 무의미한 컬럼(0/1 플래그 등)도 화면에 표시")
    ap.add_argument("--only-normal", action="store_true",
                    help="판정 '정상' 컬럼만 CSV 에 쓴다 (고객 전달용)")
    ap.add_argument("--summary", type=int, default=12,
                    help="고객 전달용 요약표에 남길 컬럼 수 (기본 12, 0이면 생략)")
    ap.add_argument("--max-missing", type=float, default=None,
                    help="결측률이 이 값(%%)을 넘는 컬럼 제외. 예: --max-missing 10")
    ap.add_argument("--sort", default="ep_max",
                    choices=["ep_max", "ep_n", "over_n", "ep_mean", "name"],
                    help="화면 정렬 기준 (기본: 최장 에피소드)")
    # ── 사건 교차 ──
    ap.add_argument("--events", action="store_true",
                    help="정체 사건(예: 학습기간 23건)과 교차분석")
    ap.add_argument("--events-config", default="model_config.json",
                    help="사건 기준을 읽을 config (없으면 자동 산출)")
    ap.add_argument("--lead", type=int, default=30,
                    help="사건 시작 몇 분 전까지 함께 볼지 (선행지표 탐색)")
    ap.add_argument("--min-lift", type=float, default=2.0,
                    help="순위표에 남길 최소 lift (기대 대비 배수)")
    a = ap.parse_args()

    print("=" * 82)
    print(f"[상위 {(1 - a.pct) * 100:g}% 임계 분석]  p{a.pct * 100:g}"
          + (f" · {a.window}분 이동평균 기준" if a.window > 1 else " · 순간값 기준"))
    sd = load(a.data)
    if a.d0 or a.d1:
        sd = sd.slice_dates(a.d0 or "1900-01-01", a.d1 or "2999-12-31")
    if not len(sd):
        print("데이터 없음"); return 1
    days = max(1, (sd.times[-1] - sd.times[0]).days + 1)
    print(f"  기간   : {sd.times[0]:%Y-%m-%d %H:%M} ~ {sd.times[-1]:%Y-%m-%d %H:%M}"
          f"  ({len(sd)}행 / {days}일)")

    names = a.columns or [c for c in sd.cols if c != TIME_COL]
    if a.exclude_contains:
        names = [c for c in names if not any(x in c for x in a.exclude_contains)]
    print(f"  컬럼   : {len(names)}개 검사")
    if a.min_duration > 1 or a.gap > 0:
        print(f"  에피소드: 최소 {a.min_duration}분 지속 · gap {a.gap}분 병합")

    # ── 정체 사건 산출 (--events) ─────────────────────────
    events, ev_target = [], None
    if a.events:
        cfg = None
        if os.path.exists(a.events_config):
            cfg = load_config(a.events_config)
        ev_target = (cfg or {}).get("target", TARGET)
        ev_win = (cfg or {}).get("window", 10)
        ev_mind = (cfg or {}).get("min_duration", 10)
        ev_gap = (cfg or {}).get("gap", 10)
        if not sd.has(ev_target):
            print(f"  ✗ 사건 기준 컬럼이 데이터에 없음: {ev_target}"); return 1
        ev_sm = smooth(sd.get(ev_target), ev_win)
        ev_thr = (cfg or {}).get(
            "threshold", percentile([v for v in ev_sm if v is not None], 0.99))
        events = episodes(sd.times, ev_sm, ev_thr, ev_mind, ev_gap)
        ev_by_no = {e["no"]: e for e in events}
        src = f"config {a.events_config}" if cfg else "자동 산출"
        print(f"  사건   : {len(events)}건  ({ev_target} · {ev_win}분 이동평균 "
              f">= {ev_thr} · {ev_mind}분+ · {src})")
        if not events:
            print("  ✗ 사건이 0건 — 기간/임계 확인"); return 1
        print(f"  선행창 : 사건 시작 {a.lead}분 전까지 함께 확인")
    print("=" * 82)

    col_rows, ep_rows, pt_rows = [], [], []
    evcol_rows, rank = [], {}
    results, skipped = [], 0

    for name in names:
        got = analyze(sd.times, sd.get(name), a.pct, a.window,
                      a.min_duration, a.gap)
        if got is None:
            skipped += 1
            continue
        st, eps, vals = got
        results.append((name, st, eps))

        col_rows.append([
            name, st["n"], st["missing"],
            r(st["min"]), r(st["p50"]), r(st["p95"]), r(st["p99"]),
            r(st["threshold"]), r(st["max"]),
            st["over_n"], f'{st["over_ratio"] * 100:.2f}%',
            st["ep_n"], st["ep_max"], r(st["ep_mean"], 1),
            round(st["ep_n"] / days, 2), st["verdict"],
        ])

        for e in eps:
            ep_rows.append([
                name, e["no"],
                e["t_start"].strftime("%Y-%m-%d %H:%M"),
                e["t_end"].strftime("%Y-%m-%d %H:%M"),
                e["duration"], r(st["threshold"]), r(e["peak"]),
                e["t_peak"].strftime("%Y-%m-%d %H:%M"), r(e["mean"]),
                r(e["peak"] / st["threshold"], 2) if st["threshold"] else "",
            ])

        if not a.no_points:
            epi_of = {}
            for e in eps:
                for i in range(e["i0"], e["i1"] + 1):
                    epi_of[i] = e["no"]
            thr = st["threshold"]
            for i, v in enumerate(vals):
                if v is None or v < thr:
                    continue
                pt_rows.append([
                    name, sd.times[i].strftime("%Y-%m-%d %H:%M"),
                    r(v), r(thr), r(v - thr),
                    r(v / thr, 3) if thr else "", epi_of.get(i, ""),
                ])

        # 사건 교차 — vals 를 버리기 전에 여기서 처리 (메모리 절약)
        if events:
            hits = cross(events, vals, st["threshold"], a.lead)
            base = st["over_ratio"] or 1e-9
            for h in hits:
                ev = ev_by_no[h["no"]]
                evcol_rows.append([
                    h["no"],
                    ev["t_start"].strftime("%Y-%m-%d %H:%M"),
                    ev["t_end"].strftime("%Y-%m-%d %H:%M"), ev["duration"],
                    name, h["hit_n"], f'{h["hit_ratio"] * 100:.1f}%',
                    h["lead_min"], h["pre_n"], r(st["threshold"]), r(h["peak"]),
                    round(h["hit_ratio"] / base, 1),
                ])
            if hits:
                cov = [h for h in hits if h["hit_n"] > 0 or h["pre_n"] > 0]
                rank[name] = {
                    "cov": len(cov),
                    "nos": [h["no"] for h in cov],
                    "lead": sum(h["lead_min"] for h in cov) / len(cov),
                    "ratio": sum(h["hit_ratio"] for h in cov) / len(cov),
                    "lift": (sum(h["hit_ratio"] for h in cov) / len(cov)) / base,
                    "base": st["over_ratio"],
                    "ok": st["ok"], "verdict": st["verdict"],
                }

    if not results:
        print("분석 가능한 숫자 컬럼이 없습니다."); return 1

    # 전달용 필터 — 컬럼 하나를 빼면 그 컬럼의 에피소드·시점·순위도 같이 뺀다
    if a.only_normal or a.max_missing is not None:
        keep = set()
        for name, st, _ in results:
            if a.only_normal and not st["ok"]:
                continue
            if a.max_missing is not None:
                tot = st["n"] + st["missing"]
                if tot and st["missing"] / tot * 100 > a.max_missing:
                    continue
            keep.add(name)
        cond = ([" 판정=정상"] if a.only_normal else []) + \
               ([f" 결측≤{a.max_missing:g}%"] if a.max_missing is not None else [])
        print(f"\n[필터]{' ·'.join(cond)} → {len(keep)}개 유지 "
              f"/ {len(results) - len(keep)}개 제외")
        results = [x for x in results if x[0] in keep]
        col_rows = [r for r in col_rows if r[0] in keep]
        ep_rows = [r for r in ep_rows if r[0] in keep]
        pt_rows = [r for r in pt_rows if r[0] in keep]
        evcol_rows = [r for r in evcol_rows if r[4] in keep]
        rank = {n: v for n, v in rank.items() if n in keep}
        if not results:
            print("필터 후 남은 컬럼이 없습니다."); return 1

    # ── ① 컬럼 표 ─────────────────────────────────────────
    key = {"ep_max": lambda x: -x[1]["ep_max"],
           "ep_n": lambda x: -x[1]["ep_n"],
           "over_n": lambda x: -x[1]["over_n"],
           "ep_mean": lambda x: -x[1]["ep_mean"],
           "name": lambda x: x[0]}[a.sort]
    shown = results if a.include_degenerate else [x for x in results if x[1]["ok"]]
    bad = len(results) - len([x for x in results if x[1]["ok"]])
    ranked = sorted(shown, key=key)

    print(f"\n[① 컬럼]  분석 {len(results)}개 (제외 {skipped}개: 상수·데이터부족)"
          f" · 정렬 {a.sort} · 상위 {min(a.top, len(ranked))}개\n")
    print(f"{'컬럼':<46}{'임계':>11}{'최대':>11}{'초과분':>8}{'에피':>6}"
          f"{'최장':>6}{'평균':>7}")
    print("-" * 95)
    for name, st, _ in ranked[:a.top]:
        d = name if len(name) <= 46 else name[:43] + "..."
        print(f"{d:<46}{st['threshold']:>11.3f}{st['max']:>11.3f}"
              f"{st['over_n']:>8}{st['ep_n']:>6}{st['ep_max']:>6}"
              f"{st['ep_mean']:>7.1f}")
    if bad and not a.include_degenerate:
        print(f"\n  ※ {bad}개 컬럼은 분위수 임계가 무의미해 화면에서 뺐다 "
              f"(0/1 플래그·고정값 등 계단형 분포 → p99 가 바닥값과 같아짐).")
        print(f"    CSV 에는 '판정' 컬럼과 함께 전부 들어있다. "
              f"화면에도 보려면 --include-degenerate")

    # ── ② 고객 전달용 요약표 ──────────────────────────────
    # "상위 N% 임계가 얼마고, 몇 번, 얼마나 오래 넘었나" — 딱 6칸.
    # 정렬은 평균 지속. 같은 초과분수라도 뭉쳐서 오는 컬럼이 진짜 신호다.
    summary = []
    if a.summary:
        thr_label = f"상위{(1 - a.pct) * 100:g}% 임계"
        top = sorted([x for x in results if x[1]["ok"]] or results,
                     key=lambda x: -x[1]["ep_mean"])[:a.summary]
        print(f"\n[② 요약]  고객 전달용 — {thr_label} · 평균 지속 순\n")
        print(f"{'#':>3}  {'컬럼':<44}{thr_label:>13}{'최대':>10}"
              f"{'몇 번':>8}{'평균 지속':>10}")
        print("-" * 90)
        for k, (name, st, _) in enumerate(top, 1):
            print(f"{k:>3}  {name:<44}{st['threshold']:>13.1f}"
                  f"{st['max']:>10.1f}{st['ep_n']:>7}회{st['ep_mean']:>8.0f}분")
            summary.append([k, name, round(st["threshold"], 1),
                            round(st["max"], 1), st["ep_n"],
                            round(st["ep_mean"])])
        print(f"\n  읽는 법 — 1번은 {top[0][1]['threshold']:.0f} 넘으면 이상, "
              f"{days}일간 {top[0][1]['ep_n']}번 그랬고, "
              f"한 번 넘으면 평균 {top[0][1]['ep_mean']:.0f}분 갔다.")

    # ── ④ 사건 교차 순위 ──────────────────────────────────
    if events:
        cand = sorted(
            [(n, v) for n, v in rank.items()
             if v["lift"] >= a.min_lift and n != ev_target
             and (v["ok"] or a.include_degenerate)],
            key=lambda x: (-x[1]["cov"], -x[1]["lift"]))
        print(f"\n[④ 선행지표 후보]  정체 {len(events)}건 중 몇 건에서 "
              f"이 컬럼도 상위 {(1 - a.pct) * 100:g}% 였나 (lift ≥ {a.min_lift})\n")
        print(f"{'컬럼':<46}{'사건':>6}{'커버':>7}{'평균선행':>9}"
              f"{'겹침':>7}{'lift':>7}")
        print("-" * 95)
        for name, v in cand[:a.top]:
            d = name if len(name) <= 46 else name[:43] + "..."
            print(f"{d:<46}{v['cov']:>4}/{len(events):<2}"
                  f"{v['cov'] / len(events) * 100:>6.0f}%"
                  f"{v['lead']:>8.1f}분{v['ratio'] * 100:>6.0f}%"
                  f"{v['lift']:>7.1f}")
        if not cand:
            print("  (조건을 만족하는 컬럼 없음 — --min-lift 를 낮춰 볼 것)")
        print("\n  사건   = 이 컬럼도 상위%였던 정체 건수 / 전체 정체 건수")
        print("  평균선행 = 정체 시작보다 몇 분 먼저 상위%에 들어갔나 (양수 = 먼저)")
        print("  lift   = 정체 중 초과비율 ÷ 평상시 초과비율. 1.0 이면 우연 수준")

    # ── 파일 ──────────────────────────────────────────────
    os.makedirs(a.out_dir, exist_ok=True)
    j = lambda f: os.path.join(a.out_dir, f)

    print(f"\n[파일]")

    n1 = write_csv(j("top1_columns.csv"), [
        "컬럼", "데이터수", "결측수", "최소", "p50", "p95", "p99", "임계", "최대",
        "초과분수", "초과비율", "에피소드수", "최장(분)", "평균(분)", "에피소드/일",
        "판정"], col_rows)
    print(f"  {j('top1_columns.csv'):<34} 컬럼 {n1}행   (전체 상세)")

    if summary:
        ns = write_csv(j("top1_요약.csv"), [
            "#", "컬럼", f"상위{(1 - a.pct) * 100:g}% 임계", "최대",
            "몇 번(회)", "평균 지속(분)"], summary)
        print(f"  {j('top1_요약.csv'):<34} 요약 {ns}행   ← 고객에게 줄 표")

    ep_rows.sort(key=lambda x: (-x[4], x[0]))
    n2 = write_csv(j("top1_episodes.csv"), [
        "컬럼", "에피소드#", "시작", "종료", "지속(분)", "임계", "피크값",
        "피크시각", "구간평균", "피크/임계"], ep_rows)
    print(f"  {j('top1_episodes.csv'):<34} 에피소드 {n2}행")

    if a.no_points:
        print(f"  (시점 CSV 는 --no-points 로 생략)")
    else:
        pt_rows.sort(key=lambda x: (x[1], x[0]))
        n3 = write_csv(j("top1_points.csv"), [
            "컬럼", "시각", "값", "임계", "초과폭", "값/임계", "에피소드#"], pt_rows)
        print(f"  {j('top1_points.csv'):<34} 시점 {n3}행")

    if events:
        ev_rows = []
        by_ev = {}
        for row in evcol_rows:
            by_ev.setdefault(row[0], []).append(row)
        for ev in events:
            lst = sorted(by_ev.get(ev["no"], []), key=lambda x: -x[11])
            ev_rows.append([
                ev["no"], ev["t_start"].strftime("%Y-%m-%d %H:%M"),
                ev["t_end"].strftime("%Y-%m-%d %H:%M"), ev["duration"],
                r(ev["peak"]), ev["t_peak"].strftime("%Y-%m-%d %H:%M"),
                len(lst), " | ".join(x[4] for x in lst[:5]),
            ])
        n4 = write_csv(j("top1_events.csv"), [
            "사건#", "시작", "종료", "지속(분)", "피크", "피크시각",
            "겹친컬럼수", "상위5개 컬럼(lift순)"], ev_rows)
        print(f"  {j('top1_events.csv'):<34} 사건 {n4}행")

        evcol_rows.sort(key=lambda x: (x[0], -x[11]))
        n5 = write_csv(j("top1_event_columns.csv"), [
            "사건#", "사건시작", "사건종료", "사건지속(분)", "컬럼",
            "겹친분수", "겹침비율", "선행(분)", "사전초과분수",
            "임계", "구간피크", "lift"], evcol_rows)
        print(f"  {j('top1_event_columns.csv'):<34} 사건×컬럼 {n5}행")

        # 순위표에도 "어느 사건이었나"를 시각으로 붙인다.
        # 사건이 많으면 한 칸이 길어지므로 번호 목록과 시각 목록을 나눠 넣는다.
        def ev_times(nos):
            return " | ".join(
                f'{n}:{ev_by_no[n]["t_start"]:%m-%d %H:%M}'
                f'~{ev_by_no[n]["t_end"]:%H:%M}({ev_by_no[n]["duration"]}분)'
                for n in nos)

        rk = sorted(rank.items(), key=lambda x: (-x[1]["cov"], -x[1]["lift"]))
        n6 = write_csv(j("top1_event_ranking.csv"), [
            "컬럼", "사건커버(건)", "전체사건", "커버율", "평균선행(분)",
            "평균겹침비율", "평상시초과비율", "lift", "판정",
            "커버사건#", "커버사건 시각(시작~종료·지속)"],
            [[n, v["cov"], len(events),
              f'{v["cov"] / len(events) * 100:.0f}%', round(v["lead"], 1),
              f'{v["ratio"] * 100:.1f}%', f'{v["base"] * 100:.2f}%',
              round(v["lift"], 2), v["verdict"],
              ",".join(str(x) for x in v["nos"]), ev_times(v["nos"])]
             for n, v in rk])
        print(f"  {j('top1_event_ranking.csv'):<34} 순위 {n6}행  ← 원인 분석용")

    # ── 요약 ──────────────────────────────────────────────
    good = [x for x in results if x[1]["ok"]] or results
    tot_ep = sum(st["ep_n"] for _, st, _ in good)
    longest = max(good, key=lambda x: x[1]["ep_max"])
    print(f"\n[요약]  (판정 '정상' {len(good)}개 컬럼 기준)")
    print(f"  전체 에피소드 {tot_ep}건 (하루 {tot_ep / days:.1f}건)")
    print(f"  최장 {longest[1]['ep_max']}분 — {longest[0]}")
    if a.min_duration == 1 and tot_ep:
        one = sum(1 for _, _, eps in good for e in eps if e["duration"] == 1)
        tail = ("  ← 대부분 블립. --window 10 으로 다시 볼 것"
                if one / tot_ep > 0.5 else "")
        print(f"  1분짜리 {one}건 / {tot_ep}건 ({one / tot_ep * 100:.0f}%){tail}")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

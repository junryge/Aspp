#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features_31 — raw 265 → 핵심 31 피처 추출 (Phase 1)
====================================================================
룰이 보는 컬럼(raw_columns.py 의 RA/RB/RD/SLA/SORTER 매핑)만 뽑아
분 단위 wide 테이블로 저장. TSPulse fine-tune / XGBoost 공용 입력.

★ 표준 라이브러리만 사용 (pandas/numpy 불필요) → 회사 PC 어디서든.
★ 컬럼 정의는 raw_columns.py 매핑과 100% 일치 (자체 포함 = 폐쇄망 반입 안전).

입력:
    --raw   raw 폴더 (M16A_HUBROOM_PR_*.CSV 여러 개 또는 하나의 큰 CSV. CRT_TM 으로 분 인식)
    --start 학습 구간 시작 (기본 2026-04-01)  ※ 03-24 이전 22컬럼 NULL 회피
    --end   학습 구간 끝   (기본 2026-05-29)
    --out   출력 폴더 (기본 ./out_ml)

실행:
    python features_31.py --raw ./raw --start 2026-04-01 --end 2026-05-29 --out ./out_ml

출력:
    features.csv     — datetime + 31 피처 (분 단위, 원 스케일. 결측은 공란)
    features_meta.json — 피처 목록/커버리지/범위 요약
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

# ============================================================
# ★ 핵심 31 피처 = raw_columns.py 매핑 그대로 (룰과 동일)
#   (raw 풀네임, 그룹, 짧은이름)
# ============================================================
FEATURES_31 = [
    # ── R-A 반송시간 (7) ──
    ('M16HUB.QUE.TIME.AVGTOTALTIME1MIN',    'RA', 'RA_M16HUB'),
    ('M14.QUE.LOAD.AVGLOADTIME1MIN',        'RA', 'RA_M14'),
    ('M14B.QUE.TIME.AVGTOTALTIME1MIN',      'RA', 'RA_M14B'),
    ('M16A.QUE.LOAD.AVGLOADTIME1MIN',       'RA', 'RA_M16A'),
    ('M16B.QUE.LOAD.AVGLOADTIME1MIN',       'RA', 'RA_M16B'),
    ('M16_PKT.QUE.TIME.AVGTOTALTIME1MIN',   'RA', 'RA_PKT'),
    ('M16_WT.QUE.TIME.AVGTOTALTIME1MIN',    'RA', 'RA_WT'),
    # ── R-B 큐누적 (6) ──
    ('M16HUB.QUE.M14TOM16.MESCURRENTQCNT',  'RB', 'RB_M14toM16'),
    ('M14.QUE.ALL.3F_TO_HUB_JOB',           'RB', 'RB_M14'),
    ('M14B.QUE.ALL.7F_TO_HUB_JOB',          'RB', 'RB_M14B'),
    ('M16A.QUE.ALL.6F_TO_HUB_JOB',          'RB', 'RB_M16A'),
    ('M16B.QUE.ALL.10F_TO_HUB_JOB',         'RB', 'RB_M16B'),
    ('M16.QUE.SFAB.SENDQUEUETOTAL',         'RB', 'RB_M16send'),
    # ── R-D 저장/OHT (7) ──
    ('M16HUB.STRATE.ALL.FABSTORAGERATIO',   'RD', 'RD_FAB'),
    ('M16HUB.STRATE.STB.3F_STORAGE_UTIL',   'RD', 'RD_STB'),
    ('M16HUB.STRATE.STK.STORAGERATIO',      'RD', 'RD_STK'),   # ★ STK 스토커 저장률 (6/16 저장Full 겨냥)
    ('M14.QUE.OHT.OHTUTIL',                 'RD', 'RD_OHT_M14'),
    ('M14B.QUE.OHT.OHTUTIL',                'RD', 'RD_OHT_M14B'),
    ('M16A.QUE.OHT.OHTUTIL',                'RD', 'RD_OHT_M16A'),
    ('M16B.QUE.OHT.OHTUTIL',                'RD', 'RD_OHT_M16B'),
    # ── SLA 4분초과 (5) ──
    ('M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO','SLA', 'SLA_M16HUB'),
    ('M14.QUE.ALL.TRANSPORT4MINOVERRATIO',  'SLA', 'SLA_M14'),
    ('M14B.QUE.ALL.TRANSPORT4MINOVERRATIO', 'SLA', 'SLA_M14B'),
    ('M16A.QUE.ALL.TRANSPORT4MINOVERRATIO', 'SLA', 'SLA_M16A'),
    ('M16B.QUE.ALL.TRANSPORT4MINOVERRATIO', 'SLA', 'SLA_M16B'),
    # ── Sorter 대기 (5) ──
    ('M14.SORTER.ABN.SORTERWAITCOUNTOVER',  'SORTER', 'SRT_M14'),
    ('M14B.SORTER.ABN.SORTERWAITCOUNTOVER', 'SORTER', 'SRT_M14B'),
    ('M16A.SORTER.ABN.SORTERWAITCOUNTOVER', 'SORTER', 'SRT_M16A'),
    ('M16B.SORTER.ABN.SORTERWAITCOUNTOVER', 'SORTER', 'SRT_M16B'),
    ('M16HUB.SORTER.ABN.SORTERWAITCOUNTOVER','SORTER', 'SRT_M16HUB'),
    # ── 보조 (2) ──
    ('M16HUB.QUE.ABN.AOTRANSDELAY',         'AUX', 'AUX_AOdelay'),
    ('M16HUB.QUE.OHT.CURRENTOHTQCNT',       'AUX', 'AUX_OHTq(리프터대표)'),
    # ── ★ M16→M14 방향 국소 밀림/정체 (4AFC 컨베이어) — 고객 정체 4건 겨냥 ──
    ('M16HUB.QUE.M16TOM14.MESCURRENTQCNT',      'DIR', 'DIR_M16toM14_Q'),      # M16→M14 총 큐
    ('M16A.QUE.CNV.M16TOM14ACURRNETQCNT',       'DIR', 'DIR_M16toM14A_CNV'),   # M16A→M14A 컨베이어 큐
    ('M16A.QUE.CNV.M16TOM14BCURRNETQCNT',       'DIR', 'DIR_M16toM14B_CNV'),   # M16A→M14B 컨베이어 큐
    ('M14.QUE.CNV.NORTHCNVTOM14TIME1MIN',       'DIR', 'DIR_NorthCNVtoM14_T'), # 북측(4AFC3301) 컨베이어→M14 시간 ★밀림직결
    ('M14.QUE.CNV.SOUTHCNVTOM14TIME1MIN',       'DIR', 'DIR_SouthCNVtoM14_T'), # 남측(4AFC3201) 컨베이어→M14 시간 ★밀림직결
    ('M14.QUE.CNV.NORTHCURRENTQCNT',            'DIR', 'DIR_NorthCNV_Q'),      # 북측 컨베이어 큐
    ('M14.QUE.CNV.SOUTHCURRENTQCNT',            'DIR', 'DIR_SouthCNV_Q'),      # 남측 컨베이어 큐
]
FEATURE_COLS = [f[0] for f in FEATURES_31]
SHORT = {f[0]: f[2] for f in FEATURES_31}

# ============================================================
# ★ 파생 피처 (Bridge SOP 반영) — 방향별 ZT 리프터 큐 합산
#   SOP 발동기준: "M16 ZT 3F→6F QUE 280↑ / 6F→3F QUE 280↑ / HUB Total 600↑"
#   원본에 있는 10대 ZT(6ABL...) 방향별 큐를 합쳐 SOP 지표를 그대로 피처화.
#   (shortname, 그룹, 합산모드, 원본 접두, 원본 접미)
# ============================================================
DERIVED = [
    ('BR_ZT_3to6', 'BR', 'sum', 'M16HUB.LFT.', '.3F_TO_6F_CURRENTQCNT'),  # M16방향(3F→6F) 큐 합
    ('BR_ZT_6to3', 'BR', 'sum', 'M16HUB.LFT.', '.6F_TO_3F_CURRENTQCNT'),  # M14방향(6F→3F) 큐 합
    ('BR_HUB_totalQ', 'BR', 'sum', 'M16HUB.LFT.', '.TOTAL_CURRENTQCNT'),  # HUB Total(리프터 총큐)
]
DERIVED_SHORTS = [d[0] for d in DERIVED]
# 파생2(계산형): 방향 불균형 = 3to6 − 6to3 (양수=M16방향 정체 / 음수=M14방향 정체)
DERIVED_CALC = ['BR_dir_imbalance']


def _match_srcs(fieldnames):
    """각 파생피처가 합산할 실제 원본컬럼 목록 (파일 헤더 기준)."""
    out = {}
    for short, grp, mode, pre, suf in DERIVED:
        out[short] = [c for c in (fieldnames or []) if c.startswith(pre) and c.endswith(suf)]
    return out


def load_raw(raw_path, t_start, t_end):
    """raw CSV 들 → {datetime: {col: val}} (지정 구간만, 분 단위)."""
    files = []
    if os.path.isdir(raw_path):
        for ext in ('*.csv', '*.CSV'):
            files += glob.glob(os.path.join(raw_path, ext))
    else:
        files = [raw_path]
    if not files:
        print(f"⚠️ raw 파일 없음: {raw_path}")
        return {}
    grid = {}          # datetime → {col: float}
    seen_cols = set()
    dsrc = {s: set() for s in DERIVED_SHORTS}   # 파생피처별 합산 원본컬럼(전 파일 합집합)
    for fp in sorted(files):
        for enc in ('utf-8-sig', 'utf-8', 'cp949'):
            try:
                f = open(fp, encoding=enc)
                rd = csv.DictReader(f)
                _ = rd.fieldnames
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"⚠️ 읽기 실패: {fp}")
            continue
        present = [c for c in FEATURE_COLS if c in (rd.fieldnames or [])]
        seen_cols.update(present)
        # 파생 원본컬럼(방향별 ZT 큐) 매칭 → 저장 대상에 포함
        srcs = _match_srcs(rd.fieldnames)
        present_src = []
        for s, cols in srcs.items():
            dsrc[s].update(cols)
            present_src += cols
        wanted = present + present_src
        for row in rd:
            ts = (row.get('CRT_TM') or '').strip()
            if len(ts) < 16:
                continue
            try:
                t = datetime.strptime(ts[:16], '%Y-%m-%d %H:%M')
            except ValueError:
                continue
            if not (t_start <= t <= t_end):
                continue
            cell = grid.setdefault(t, {})
            for c in wanted:
                v = (row.get(c) or '').strip()
                if not v:
                    continue
                try:
                    cell[c] = float(v)
                except ValueError:
                    pass
        f.close()
    missing = [c for c in FEATURE_COLS if c not in seen_cols]
    if missing:
        print(f"⚠️ raw 에 없는 피처 {len(missing)}개: {[SHORT[c] for c in missing]}")
    dsrc = {s: sorted(v) for s, v in dsrc.items()}
    for s in DERIVED_SHORTS:
        print(f"[파생] {s}: {len(dsrc[s])}개 ZT큐 합산")
    return grid, dsrc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--start', default='2026-04-01')
    ap.add_argument('--end', default='2026-05-29')
    ap.add_argument('--out', default='./out_ml')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t_start = datetime.strptime(a.start, '%Y-%m-%d')
    t_end = datetime.strptime(a.end + ' 23:59', '%Y-%m-%d %H:%M')

    print("=" * 60)
    print(f"features_31 — {a.start} ~ {a.end}  (핵심 {len(FEATURES_31)}피처)")
    print("=" * 60)

    grid, dsrc = load_raw(a.raw, t_start, t_end)
    if not grid:
        print("⚠️ 데이터 없음 — 경로/구간 확인")
        return
    times = sorted(grid.keys())

    # ── features.csv 저장 (datetime + 31 원피처 + 파생 Bridge 피처) ──
    out_csv = os.path.join(a.out, 'features.csv')
    header = (['datetime'] + [SHORT[c] for c in FEATURE_COLS]
              + DERIVED_SHORTS + DERIVED_CALC)
    filled = {c: 0 for c in FEATURE_COLS}

    def dsum(cell, short):
        """파생 합산: 해당 ZT큐 원본들의 합 (하나라도 있으면 값, 전무면 None)."""
        cols = dsrc.get(short, [])
        vals = [cell[c] for c in cols if c in cell]
        return sum(vals) if vals else None

    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in times:
            cell = grid[t]
            row = [t.strftime('%Y-%m-%d %H:%M')]
            for c in FEATURE_COLS:
                v = cell.get(c)
                if v is not None:
                    filled[c] += 1
                row.append('' if v is None else f'{v:g}')
            # 파생 합산 (방향별 ZT 큐)
            d = {s: dsum(cell, s) for s in DERIVED_SHORTS}
            for s in DERIVED_SHORTS:
                row.append('' if d[s] is None else f'{d[s]:g}')
            # 방향 불균형 = 3to6 − 6to3
            a3, a6 = d.get('BR_ZT_3to6'), d.get('BR_ZT_6to3')
            imb = (a3 - a6) if (a3 is not None and a6 is not None) else None
            row.append('' if imb is None else f'{imb:g}')
            w.writerow(row)

    # ── meta 저장 ──
    n = len(times)
    cov = {SHORT[c]: round(filled[c] / n * 100, 1) for c in FEATURE_COLS}
    ncols = len(FEATURE_COLS) + len(DERIVED_SHORTS) + len(DERIVED_CALC)
    meta = {
        'rows': n,
        'range': [times[0].strftime('%Y-%m-%d %H:%M'), times[-1].strftime('%Y-%m-%d %H:%M')],
        'features': [SHORT[c] for c in FEATURE_COLS] + DERIVED_SHORTS + DERIVED_CALC,
        'raw_fullnames': FEATURE_COLS,
        'derived_sources': dsrc,
        'coverage_pct': cov,
    }
    with open(os.path.join(a.out, 'features_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[완료] {n}분 × {ncols}피처(원 {len(FEATURES_31)} + 파생 {len(DERIVED_SHORTS)+len(DERIVED_CALC)}) → {out_csv}")
    print(f"       범위 {meta['range'][0]} ~ {meta['range'][1]}")
    low = [(k, v) for k, v in cov.items() if v < 90]
    if low:
        print(f"⚠️ 커버리지 90%↓ 피처: {low}")
    print("다음: labels_채점지.py 로 라벨 만든 뒤 tspulse_train.py")


if __name__ == '__main__':
    main()

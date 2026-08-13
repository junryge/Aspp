#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 발동이벤트_영역분리 — 발동이벤트 CSV 를 영역(FAB)별 파일로 나눈다
# ====================================================================
# 발동이벤트는 8개 영역 컬럼이 한 줄에 다 들어있어 폭이 143칸이나 된다.
# 영역별로 보려면 매번 가로 스크롤을 해야 해서, 영역마다 파일을 따로 만든다.
#
#   20260812_발동이벤트.csv
#     → 20260812_발동이벤트_M16HUB.csv   (공통 34칸 + M16HUB 22칸)
#     → 20260812_발동이벤트_M14.csv      (공통 34칸 + M14 19칸)
#     → ... M14B / M16A / M16B
#
# 컬럼 분류
#   · 공통     접두사가 없는 것 — datetime, unified_risk_score, hot_area,
#              reason, propagation_chain, layer1_total …
#              (LO_LOW_AMOS 4칸·MAXCAPA 4칸도 여기 포함)
#   · 영역별   'M16HUB_' 처럼 영역 이름으로 시작하는 것
#   · sla_M14 / sorter_M16B 처럼 이름이 뒤집힌 것도 그 영역 파일에 같이 넣는다
#     (--no-suffix-cols 로 끄면 공통에만 남는다)
#
# 사용법
#   python 발동이벤트_영역분리.py 20260812_발동이벤트.csv
#   python 발동이벤트_영역분리.py .\predict_tobe\*_발동이벤트.csv -o .\영역별
#   python 발동이벤트_영역분리.py 20260812_발동이벤트.csv --summary
#   python 발동이벤트_영역분리.py 20260812_발동이벤트.csv --grade
#
# 옵션
#   -o, --out          출력 폴더 (기본: 입력 파일과 같은 폴더)
#   --areas            나눌 영역 (기본: M16HUB,M14,M14B,M16A,M16B)
#   --all-areas        M16 / M16_PKT / M16_WT 도 포함 (score 한 칸뿐)
#   --grade            영역등급.json 임계로 등급 컬럼 추가
#   --grade-config     임계 파일 경로 (기본: 스크립트 옆 영역등급.json)
#   --summary          영역별 raw 분포와 임계별 비율 출력 (임계 다시 잡을 때)
#   --strip-prefix     컬럼명에서 영역 접두사 제거 (M16B_score_raw → score_raw)
#   --no-suffix-cols   sla_M14 / sorter_M14 류를 영역 파일에 넣지 않음
#
# --grade 가 붙이는 컬럼
#   area_grade      영역등급.json 임계에 따른 등급
#   area_saturated  raw 가 50 을 넘어 융합에 다 반영되지 못한 상태면 'Y'
#   ※ 임계는 영역마다 다르다. 전체 unified_risk_score 의 50/71/85 를 raw 에
#     그대로 대면 M14B(최대 40)는 영원히 경계에 닿지 못한다.
#     기본값은 8/12 하루치로 잡은 잠정값이니 --summary 로 다시 잡을 것.
import argparse
import csv
import glob
import json
import os
import sys

csv.field_size_limit(10 ** 7)

DEFAULT_AREAS = ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B']
EXTRA_AREAS = ['M16', 'M16_PKT', 'M16_WT']
# 긴 이름을 먼저 봐야 M16B_ 가 M16_ 로 잘못 잡히지 않는다
ALL_AREAS = sorted(DEFAULT_AREAS + EXTRA_AREAS, key=len, reverse=True)

SATURATE_AT = 50        # 영역 점수가 잘리는 상한
GRADE_ORDER = ['초위험', '위험', '경계']
SUMMARY_THS = [10, 15, 20, 25, 27, 30, 32, 35, 37, 40, 42, 45, 50]


def load_grade_config(path):
    here = os.path.dirname(os.path.abspath(__file__))
    fp = path or os.path.join(here, '영역등급.json')
    if not os.path.exists(fp):
        print(f'  ⚠️ 임계 파일 없음: {fp} — 등급을 붙이지 않습니다')
        return {}
    with open(fp, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not str(k).startswith('_')}
    print(f'  [임계] {os.path.basename(fp)} — ' +
          ' · '.join(f"{a} {'/'.join(str(v.get(g,'-')) for g in GRADE_ORDER[::-1])}"
                     for a, v in cfg.items()))
    return cfg


def area_of(col):
    for a in ALL_AREAS:
        if col.startswith(a + '_'):
            return a
    return None


def suffix_area_of(col):
    for a in ALL_AREAS:
        if col.endswith('_' + a):
            return a
    return None


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def grade_of(raw, th):
    """th = {'경계':32,'위험':35,'초위험':40}"""
    v = fnum(raw)
    for g in GRADE_ORDER:                 # 초위험 → 위험 → 경계 순으로 검사
        if g in th and v >= th[g]:
            return g, ('Y' if v >= SATURATE_AT else '')
    return '', ('Y' if v >= SATURATE_AT else '')


def print_summary(header, body, areas):
    idx = {c: i for i, c in enumerate(header)}
    raws = {}
    for a in areas:
        c = a + '_score_raw'
        if c in idx:
            raws[a] = [fnum(r[idx[c]]) for r in body]
    if not raws:
        return
    n = len(body)

    def pct(v, p):
        v = sorted(v)
        k = (len(v) - 1) * p / 100
        lo = int(k)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    print('\n  ── score_raw 분포 ' + '─' * 46)
    print(f"     {'영역':<9}{'평균':>7}{'중앙':>6}{'p90':>6}{'p95':>6}{'p99':>6}{'최대':>6}")
    for a, v in raws.items():
        print(f'     {a:<9}{sum(v)/len(v):>7.1f}{pct(v,50):>6.0f}{pct(v,90):>6.0f}'
              f'{pct(v,95):>6.0f}{pct(v,99):>6.0f}{max(v):>6.0f}')

    lv = idx.get('unified_risk_level')
    if lv is not None:
        na = sum(1 for r in body if r[lv].strip())
        print(f'\n     참고 — 전체(unified) 등급 발생: {na}행 ({na/n*100:.1f}%)')

    print('\n  ── raw 임계별 해당 비율 (위 발생률과 비슷해지는 값을 고르세요) ' + '─' * 3)
    print(f"     {'raw≥':<6}" + ''.join(f'{a:>10}' for a in raws))
    for t in SUMMARY_THS:
        line = ''.join(f"{sum(1 for v in raws[a] if v >= t)/n*100:>9.1f}%" for a in raws)
        print(f'     {t:<6}{line}')
    print('  ' + '─' * 64)


def split_one(fp, out_dir, areas, use_suffix, strip_prefix, gcfg, summary):
    with open(fp, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f'  ⚠️ 빈 파일: {fp}')
        return []
    header = rows[0]
    body = [r for r in rows[1:] if any(x.strip() for x in r)]

    drift = sum(1 for r in body if len(r) != len(header))
    if drift:
        print(f'  ⚠️ 헤더({len(header)})와 칸 수가 다른 행 {drift}건 — 따옴표가 유실된 파일 같습니다.')
        print('     남는 칸은 버리고 모자란 칸은 빈칸으로 채워 진행합니다. 원본 재추출을 권합니다.')
        body = [(r + [''] * len(header))[:len(header)] for r in body]

    owner = {}
    for c in header:
        a = area_of(c)
        if a is None and use_suffix:
            a = suffix_area_of(c)
        owner[c] = a
    common = [c for c in header if owner[c] is None]

    hidx = {c: i for i, c in enumerate(header)}

    base = os.path.basename(fp)
    stem = base[:-4] if base.lower().endswith('.csv') else base
    made = []
    print(f'  {base} — {len(header)}컬럼 · {len(body)}행 (공통 {len(common)})')

    for a in areas:
        acols = [c for c in header if owner[c] == a]
        if not acols:
            print(f'     · {a:<8} 컬럼 없음 — 건너뜀')
            continue
        take = common + acols
        idxs = [header.index(c) for c in take]

        out_head = list(take)
        if strip_prefix:
            out_head = [c[len(a) + 1:] if c.startswith(a + '_') else c for c in out_head]

        raw_pos = take.index(a + '_score_raw') if (a + '_score_raw') in take else None
        th = gcfg.get(a) if gcfg else None
        do_grade = th is not None and raw_pos is not None
        if do_grade:
            out_head += ['area_grade', 'area_saturated']

        op = os.path.join(out_dir, f'{stem}_{a}.csv')
        with open(op, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(out_head)
            for r in body:
                vals = [r[i] for i in idxs]
                if do_grade:
                    vals += list(grade_of(vals[raw_pos], th))
                w.writerow(vals)
        made.append(op)
        extra = ' (+등급2)' if do_grade else ''
        print(f'     ✅ {a:<8} {len(take)}컬럼{extra} → {os.path.basename(op)}')

    if summary:
        print_summary(header, body, [a for a in areas if a + '_score_raw' in hidx])
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='발동이벤트 CSV (여러 개·와일드카드 가능)')
    ap.add_argument('-o', '--out', default=None, help='출력 폴더 (기본: 입력과 같은 폴더)')
    ap.add_argument('--areas', default=None, help='나눌 영역 (쉼표 구분)')
    ap.add_argument('--all-areas', action='store_true', help='M16/M16_PKT/M16_WT 도 포함')
    ap.add_argument('--grade', action='store_true', help='영역등급.json 임계로 등급 추가')
    ap.add_argument('--grade-config', default=None, help='임계 파일 경로')
    ap.add_argument('--summary', action='store_true', help='raw 분포·임계별 비율 출력')
    ap.add_argument('--strip-prefix', action='store_true', help='컬럼명에서 영역 접두사 제거')
    ap.add_argument('--no-suffix-cols', action='store_true',
                    help='sla_M14 / sorter_M14 류를 영역 파일에 넣지 않음')
    a = ap.parse_args()

    files = []
    for pat in a.inputs:
        hit = glob.glob(pat)
        files.extend(hit if hit else ([pat] if os.path.exists(pat) else []))
    files = sorted(set(files))
    if not files:
        sys.exit(f'❌ 파일 없음: {", ".join(a.inputs)}')

    areas = [x.strip() for x in a.areas.split(',')] if a.areas else list(DEFAULT_AREAS)
    if a.all_areas and not a.areas:
        areas += EXTRA_AREAS

    print('=' * 64)
    print(f'발동이벤트 영역분리 — 대상 {len(files)}개 파일 · 영역 {", ".join(areas)}')
    print('=' * 64)

    gcfg = load_grade_config(a.grade_config) if a.grade else {}

    total = []
    for fp in files:
        out_dir = a.out or (os.path.dirname(os.path.abspath(fp)))
        os.makedirs(out_dir, exist_ok=True)
        total += split_one(fp, out_dir, areas, not a.no_suffix_cols,
                           a.strip_prefix, gcfg, a.summary)

    print(f'\n🎉 완료 — {len(total)}개 파일 생성')
    if total:
        print(f'   위치: {os.path.dirname(os.path.abspath(total[0]))}')


if __name__ == '__main__':
    main()

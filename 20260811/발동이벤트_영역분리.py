#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 발동이벤트_영역분리 — 발동이벤트 CSV 를 영역(FAB)별 파일로 나눈다
# ====================================================================
# 발동이벤트는 8개 영역 컬럼이 한 줄에 다 들어있어 폭이 143칸이나 된다.
# 영역별로 보려면 매번 가로 스크롤을 해야 해서, 영역마다 파일을 따로 만든다.
#
#   20260812_발동이벤트.csv
#     → 20260812_발동이벤트_M16HUB.csv   (공통 44칸 + M16HUB 22칸)
#     → 20260812_발동이벤트_M14.csv      (공통 44칸 + M14 19칸)
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
#   python 발동이벤트_영역분리.py 20260812_발동이벤트.csv --grade
#
# 옵션
#   -o, --out          출력 폴더 (기본: 입력 파일과 같은 폴더)
#   --areas            나눌 영역 (기본: M16HUB,M14,M14B,M16A,M16B)
#   --all-areas        M16 / M16_PKT / M16_WT 도 포함 (score 한 칸뿐)
#   --grade            score_raw 로 등급 컬럼 2개 추가 (아래 설명)
#   --strip-prefix     컬럼명에서 영역 접두사 제거 (M16B_score_raw → score_raw)
#   --no-suffix-cols   sla_M14 / sorter_M14 류를 영역 파일에 넣지 않음
#
# --grade 가 붙이는 컬럼
#   area_grade      score_raw 기준 등급 — raw 25↑ 경계 / 36↑ 위험 / 43↑ 초위험
#   area_saturated  raw 가 50 을 넘어 융합에 다 반영되지 못한 상태면 'Y'
#   ※ 분모 50 은 영역 점수가 잘리는 상한. 전체 unified_risk_score 의 50/71/85 를
#     그대로 raw 에 대면 안 된다 — 영역마다 최대치가 달라 M14B 는 영원히
#     경계에 닿지 못한다.
import argparse
import csv
import glob
import os
import sys

csv.field_size_limit(10 ** 7)

DEFAULT_AREAS = ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B']
EXTRA_AREAS = ['M16', 'M16_PKT', 'M16_WT']
# 긴 이름을 먼저 봐야 M16B_ 가 M16_ 로 잘못 잡히지 않는다
ALL_AREAS = sorted(DEFAULT_AREAS + EXTRA_AREAS, key=len, reverse=True)

GRADE_CAP = 50          # 영역 점수 상한 = 등급 환산 분모
GRADE_BANDS = [(43, '초위험'), (36, '위험'), (25, '경계')]


def area_of(col):
    """컬럼이 어느 영역 것인지 — 'M16HUB_score' → 'M16HUB'"""
    for a in ALL_AREAS:
        if col.startswith(a + '_'):
            return a
    return None


def suffix_area_of(col):
    """'sla_M14' / 'sorter_M16B' 처럼 뒤에 영역이 붙은 컬럼"""
    for a in ALL_AREAS:
        if col.endswith('_' + a):
            return a
    return None


def grade_of(raw):
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return '', ''
    g = ''
    for th, name in GRADE_BANDS:
        if v >= th:
            g = name
            break
    return g, ('Y' if v >= GRADE_CAP else '')


def split_one(fp, out_dir, areas, use_suffix, strip_prefix, add_grade):
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

    # 컬럼 분류
    owner = {}                       # col → 영역 (없으면 공통)
    for c in header:
        a = area_of(c)
        if a is None and use_suffix:
            a = suffix_area_of(c)
        owner[c] = a
    common = [c for c in header if owner[c] is None]

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
        raw_pos = None
        if add_grade:
            rc = a + '_score_raw'
            if rc in take:
                raw_pos = take.index(rc)
                out_head += ['area_grade', 'area_saturated']

        op = os.path.join(out_dir, f'{stem}_{a}.csv')
        with open(op, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(out_head)
            for r in body:
                vals = [r[i] for i in idxs]
                if raw_pos is not None:
                    vals += list(grade_of(vals[raw_pos]))
                w.writerow(vals)
        made.append(op)
        extra = ' (+등급2)' if raw_pos is not None else ''
        print(f'     ✅ {a:<8} {len(take)}컬럼{extra} → {os.path.basename(op)}')
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='발동이벤트 CSV (여러 개·와일드카드 가능)')
    ap.add_argument('-o', '--out', default=None, help='출력 폴더 (기본: 입력과 같은 폴더)')
    ap.add_argument('--areas', default=None, help='나눌 영역 (쉼표 구분)')
    ap.add_argument('--all-areas', action='store_true', help='M16/M16_PKT/M16_WT 도 포함')
    ap.add_argument('--grade', action='store_true', help='score_raw 기준 등급 컬럼 추가')
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

    total = []
    for fp in files:
        out_dir = a.out or (os.path.dirname(os.path.abspath(fp)))
        os.makedirs(out_dir, exist_ok=True)
        total += split_one(fp, out_dir, areas, not a.no_suffix_cols,
                           a.strip_prefix, a.grade)

    print(f'\n🎉 완료 — {len(total)}개 파일 생성')
    if total:
        print(f'   위치: {os.path.dirname(os.path.abspath(total[0]))}')


if __name__ == '__main__':
    main()

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
# 원본은 절대 건드리지 않는다 — 읽기만 하고, 결과는 항상 새 파일로 쓴다.
# 출력 경로가 입력과 같아지면 그 영역은 건너뛴다(아래 safety 확인).
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
#   --all-areas        M16 / M16_WT 도 포함 (score 한 칸뿐)
#   --score            영역 점수(0~100)와 등급 컬럼 추가
#   --score-config     분모 파일 경로 (기본: 스크립트 옆 영역등급.json)
#   --summary          영역별 raw 분포와 임계별 비율 출력 (임계 다시 잡을 때)
#   --strip-prefix     컬럼명에서 영역 접두사 제거 (M16B_score_raw → score_raw)
#   --no-suffix-cols   sla_M14 / sorter_M14 류를 영역 파일에 넣지 않음
#
# 운영 (run_ml.py 에서 스레드로)
#   import 발동이벤트_영역분리 as area_split
#   threading.Thread(target=area_split.run_watch,
#                    kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
#                    daemon=True).start()
#   → 매분 predict_tobe 의 그날 발동이벤트를 읽어 predict_tobe/fab분리/ 에 5개로 쓴다.
#     LO_LOW_AMOS·lo_mac_maxcapa 가 컬럼을 채운 뒤에 돌아야 하므로 run_ml 의 맨 끝에서
#     시작하고, lag 초만큼 늦게 돈다(기본 20초).
#
# --score 가 붙이는 컬럼 — unified 와 같은 방식으로 만든다
#   area_score      min(100, round(score_raw × 100 ÷ 분모))
#   area_level      60=경계 / 71=위험 / 85=초위험  (영역등급.json 의 등급기준)
#   area_saturated  score_raw 가 50 을 넘어 융합에 다 반영되지 못했으면 'Y'
#   ※ 분모는 영역등급.json 에서 바꾼다. 8/12 기준 70 이 전체 등급 발생률
#     (1.1%)과 가장 비슷했다. 50 이면 M16HUB 가 25% 로 경계가 넘친다.
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

csv.field_size_limit(10 ** 7)

DEFAULT_AREAS = ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B']
# M16_PKT 제외 (2026-08 고객 요청) — 예측기에서 영역 자체가 빠져 컬럼도 더는 없다
EXTRA_AREAS = ['M16', 'M16_WT']
# 긴 이름을 먼저 봐야 M16B_ 가 M16_ 로 잘못 잡히지 않는다
ALL_AREAS = sorted(DEFAULT_AREAS + EXTRA_AREAS, key=len, reverse=True)

SATURATE_AT = 50        # 영역 점수가 잘리는 상한 (융합에 들어가는 최대)
DEFAULT_DENOM = 70      # 분모 기본값 — 설정 파일이 없을 때
DEFAULT_BANDS = {'경계': 60, '위험': 71, '초위험': 85}   # 영역 등급 구간
SUMMARY_THS = [10, 15, 20, 25, 27, 30, 32, 35, 37, 40, 42, 45, 50]
SUBDIR = 'fab분리'      # 운영 출력 하위 폴더
WATCH_LAG = 45          # 다른 기입기가 전부 쓴 뒤에 돌도록 늦추는 초
                        #   예측기 +5초 · LO_LOW_AMOS/MAXCAPA +25초 · PIO_DATA_MAKE +35초(+조회)
                        #   → 45초면 그 분의 12개 PIO 컬럼까지 채워진 뒤 분리된다 (20초였을 땐 한 분 늦게 들어갔다)
PIO_MARK = 'PIOERROR'   # PIO 기입기 컬럼 표식({경로}_PIOERROR_DEPOSITED, 예전 &PIOERROR 도 포함)
                        #   — 어느 영역 이름으로 시작하든 공통 컬럼으로 취급


def load_denoms(path, areas):
    """영역별 실효 분모를 만든다.

    영역등급.json
        "분모": 70                       전 영역 공통 (영역별 dict 도 허용)
        "조정": {"M16HUB": 120, ...}     영역별 가감 % — 100 그대로 / 120 20% 높임
    점수 = raw × 100 ÷ 분모 × 조정 ÷ 100  이므로, 실효 분모 = 분모 ÷ (조정/100)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    fp = path or os.path.join(here, '영역등급.json')
    base, adj, bands = DEFAULT_DENOM, {}, dict(DEFAULT_BANDS)
    if os.path.exists(fp):
        with open(fp, encoding='utf-8') as f:
            cfg = json.load(f)
        base = cfg.get('분모', cfg.get('denom', DEFAULT_DENOM))
        adj = dict(cfg.get('조정') or cfg.get('adjust') or {})
        bands = dict(cfg.get('등급기준') or cfg.get('bands') or DEFAULT_BANDS)
    else:
        print(f'  ⚠️ 설정 파일 없음: {fp} — 전 영역 분모 {DEFAULT_DENOM} · 조정 100 사용')

    out, shown = {}, []
    for a in areas:
        b = float(base[a]) if isinstance(base, dict) else float(base)   # 영역별 분모도 허용
        p = float(adj.get(a, 100)) or 100
        out[a] = b / (p / 100.0)
        shown.append(f'{a} {b:g}' + (f'×{p:g}%' if p != 100 else ''))
    print('  [점수설정] 분모 ' + ' · '.join(shown))
    print('  [등급기준] ' + ' / '.join(f'{k} {v}점↑' for k, v in
                                       sorted(bands.items(), key=lambda x: x[1])))
    return out, bands


def area_of(col):
    if PIO_MARK in col:           # PIO 12컬럼(M16HUB->M14B&…&PIOERROR 등)은 항상 공통 → 5개 파일 전부에
        return None
    for a in ALL_AREAS:
        if col.startswith(a + '_'):
            return a
    return None


def suffix_area_of(col):
    if PIO_MARK in col:
        return None
    for a in ALL_AREAS:
        if col.endswith('_' + a):
            return a
    return None


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def score_of(raw, denom, bands=None):
    """raw → 0~100 점수 · 등급 · 포화여부."""
    b = bands or DEFAULT_BANDS
    v = fnum(raw)
    s = min(100, round(v * 100 / denom)) if denom > 0 else 0
    lv = ''
    for name in ('초위험', '위험', '경계'):      # 높은 등급부터
        if name in b and s >= b[name]:
            lv = name
            break
    return s, lv, ('Y' if v >= SATURATE_AT else '')


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


def split_one(fp, out_dir, areas, use_suffix, strip_prefix, denoms, summary, bands=None):
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
    n_pio = sum(1 for c in common if PIO_MARK in c)

    hidx = {c: i for i, c in enumerate(header)}

    base = os.path.basename(fp)
    stem = base[:-4] if base.lower().endswith('.csv') else base
    made = []
    print(f'  {base} — {len(header)}컬럼 · {len(body)}행 (공통 {len(common)}'
          + (f' · PIO {n_pio}' if n_pio else '') + ')')

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
        dn = denoms.get(a) if denoms else None
        do_score = dn is not None and raw_pos is not None
        if do_score:
            out_head += ['area_score', 'area_level', 'area_saturated']

        op = os.path.join(out_dir, f'{stem}_{a}.csv')
        # ★ 원본 보호 — 출력이 입력과 같은 파일이면 절대 쓰지 않는다
        if os.path.abspath(op) == os.path.abspath(fp):
            print(f'     ⛔ {a:<8} 출력이 원본과 같아 건너뜀: {op}')
            continue
        with open(op, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(out_head)
            for r in body:
                vals = [r[i] for i in idxs]
                if do_score:
                    vals += list(score_of(vals[raw_pos], dn, bands))
                w.writerow(vals)
        made.append(op)
        extra = f' (+점수3·분모{dn:g})' if do_score else ''
        print(f'     ✅ {a:<8} {len(take)}컬럼{extra} → {os.path.basename(op)}')

    if summary:
        print_summary(header, body, [a for a in areas if a + '_score_raw' in hidx])
    return made


def resolve_event(path):
    """폴더면 그 안의 최신 *발동이벤트*.csv 를 고른다.
    파일명의 날짜(YYYYMMDD)가 큰 것 우선 — mtime 은 다른 기입기가 계속 바꾸므로 안 쓴다."""
    if os.path.isdir(path):
        cands = [f for f in os.listdir(path)
                 if f.lower().endswith('.csv') and '발동이벤트' in f and '_M1' not in f]
        if not cands:
            return None
        dated = [(m.group(1), f) for f in cands
                 for m in [re.search(r'(\d{8})', f)] if m]
        if dated:
            return os.path.join(path, max(dated)[1])
        return max((os.path.join(path, f) for f in cands), key=os.path.getmtime)
    return path if os.path.exists(path) else None


def run_watch(event='./predict_tobe', out=None, interval=60, lag=WATCH_LAG,
              areas=None, score=True, subdir=SUBDIR):
    """운영 진입점 — 매분 그날 발동이벤트를 영역별로 다시 쓴다.

        threading.Thread(target=발동이벤트_영역분리.run_watch,
                         kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)},
                         daemon=True).start()

    · 원본이 바뀐 때만 다시 쓴다(mtime+크기 비교) — 매분 통째로 쓰지 않는다
    · 자정이 지나면 새 날짜 파일로 자동 전환, 전날 파일도 한 번 더 마무리한다
    · 출력은 <event>/fab분리/ (out 을 주면 그쪽)
    """
    areas = areas or list(DEFAULT_AREAS)
    denoms, bands = load_denoms(None, areas) if score else ({}, None)
    out_dir = out or os.path.join(event, subdir)
    seen = {}          # 파일경로 → (mtime, size)
    last_day = None
    print(f'[영역분리] 감시 시작 — {event} → {out_dir} · {interval}초 · {lag}초 지연')

    while True:
        try:
            time.sleep(lag)
            fp = resolve_event(event)
            if not fp:
                print(f'  ⚠️ 발동이벤트 파일 없음: {os.path.abspath(event)} (대기)')
            else:
                # 자정 전환 — 전날 파일을 한 번 더 마무리
                m = re.search(r'(\d{8})', os.path.basename(fp))
                day = m.group(1) if m else None
                todo = [fp]
                if last_day and day and day != last_day:
                    prev = [os.path.join(event, f) for f in os.listdir(event)
                            if last_day in f and '발동이벤트' in f and '_M1' not in f]
                    todo = prev + todo
                    print(f'  🌙 날짜 전환 {last_day} → {day} — 전날 파일 마무리')
                last_day = day or last_day

                for t in todo:
                    try:
                        st = os.stat(t)
                    except OSError:
                        continue
                    key = (st.st_mtime, st.st_size)
                    if seen.get(t) == key:
                        continue          # 안 바뀜 — 건너뜀
                    os.makedirs(out_dir, exist_ok=True)
                    made = split_one(t, out_dir, areas, True, False, denoms, False, bands)
                    if made:
                        seen[t] = key
                        print(f'  [{datetime.now():%H:%M:%S}] {os.path.basename(t)} '
                              f'→ {len(made)}개')
        except PermissionError:
            pass                          # 다른 기입기가 쓰는 중 — 다음 사이클에
        except Exception as e:
            print(f'  ⚠️ 영역분리 오류: {e}')
        time.sleep(max(1, interval - lag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='발동이벤트 CSV (여러 개·와일드카드 가능)')
    ap.add_argument('-o', '--out', default=None, help='출력 폴더 (기본: 입력과 같은 폴더)')
    ap.add_argument('--areas', default=None, help='나눌 영역 (쉼표 구분)')
    ap.add_argument('--all-areas', action='store_true', help='M16/M16_WT 도 포함')
    ap.add_argument('--score', action='store_true', help='영역 점수(0~100)·등급 컬럼 추가')
    ap.add_argument('--score-config', default=None, help='분모 파일 경로')
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

    denoms, bands = load_denoms(a.score_config, areas) if a.score else ({}, None)

    total = []
    for fp in files:
        out_dir = a.out or (os.path.dirname(os.path.abspath(fp)))
        os.makedirs(out_dir, exist_ok=True)
        total += split_one(fp, out_dir, areas, not a.no_suffix_cols,
                           a.strip_prefix, denoms, a.summary, bands)

    print(f'\n🎉 완료 — {len(total)}개 파일 생성')
    if total:
        print(f'   위치: {os.path.dirname(os.path.abspath(total[0]))}')


if __name__ == '__main__':
    main()

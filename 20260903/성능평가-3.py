#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 성능평가 — 발동이벤트 + FAB별 영역분리 로 ALL·FAB 각각의 Precision/Recall/F1 를 낸다
# ====================================================================
# 왜 필요한가
#   고객 보고에 "룰 개정 이력 / 제작 데이터 기간 / 평가 데이터 기간 / 성능" 이 필요하다.
#   성능(P/R/F1)은 "실제로 정체가 났는가" 라는 정답이 있어야 계산된다.
#   발동이벤트에는 룰이 판정한 것만 있고 실제 결과가 없다.
#   그래서 여기서는 **대리 정답**을 쓴다 — 반드시 보고서에 기준을 같이 적을 것.
#
#     정답(실제 정체) : 반송시간(R-A)이 영역 임계 이상인 상태가
#                       SUSTAIN 분 이상 연속되면 그 구간을 실제 정체로 본다
#                       ALL 은 다섯 영역 중 하나라도 그러면 정체로 본다
#     경보            : ALL  = unified_risk_score >= 48
#                       FAB  = area_score >= 영역 경계 (M16HUB 40 · 그 외 36)
#     적중            : 사건 시작 LEAD 분 전 ~ 사건 종료 사이에 경보가 있었으면 잡은 것
#
#   ※ 반송시간은 룰의 입력이기도 하다. 완전히 독립된 정답이 아니므로
#     "대리 정답 기준"임을 반드시 명시할 것. 진짜 성능은
#     운영자 인지 시각·MCS 알람 이력을 확보한 뒤에 다시 재야 한다.
#
# 입력 두 가지
#   --event   발동이벤트 CSV/폴더            (필수) — unified_risk_score · *_ra · *_score_raw
#   --fab     FAB별 영역분리 CSV/폴더        (선택) — --score 로 만든 area_score 를 그대로 씀
#             안 주면 발동이벤트의 *_score_raw 로 area_score 를 직접 계산한다 (결과 동일)
#
# 사용법
#   python 성능평가.py --event .\predict_tobe --fab .\영역별 --csv 성능평가
#   python 성능평가.py --event .\predict_tobe --csv 성능평가
#   python 성능평가.py --event .\predict_tobe --days 30 --out 결과.txt
#
# --csv 를 주면 파일 네 개가 나온다
#   {접두사}_요약.csv    날짜 × 대상(ALL·FAB5) × 사건/경보/Precision/Recall/F1   ← PPT 용
#   {접두사}_사건목록.csv 사건 하나마다 시작·종료 시각, 지속, 첫 경보 시각, 선행시간 ← 시간 분석용
#   {접두사}_날짜별_가로.csv  날짜 × 시각(0~23시) 가로형 — 대상마다 정체분·경보분·점수최고
#   {접두사}_분단위.csv  분마다 unified_risk_score · area_score · 등급 · 경보 · 실제정체 ← 검토용
#
# --split 을 같이 주면 대상별로 파일을 따로 낸다 (고객이 ALL·FAB 분리 요청 시)
#   {접두사}_{대상}.csv           날짜별 성능 + 그날 사건 시각 + 선행 중앙값, 맨 아래 '전체' 행
#   {접두사}_{대상}_사건목록.csv   그 대상 사건만
#   {접두사}_{대상}_분단위.csv     그 대상 점수·등급·경보·실제정체만
#   {접두사}_{대상}_날짜별_가로.csv  날짜 × 시각 — 그 대상 정체분·경보분·점수최고
#   대상 6개 × 4 = 24개
#
# 운영 등급 컷 (2026-09 확인)
#   ALL 48/60/80 · M16HUB 40/55/75 · M14·M14B·M16A·M16B 36/52/72
#
# 옵션
#   --out       화면 내용을 텍스트로도 저장
#   --days      최근 N일만 / --since YYYYMMDD 이후만
#   --cut       ALL 경보 기준 (기본 48)
#   --sustain   실제 정체로 인정할 연속 분 (기본 5)
#   --lead      선행 인정 구간, 분 (기본 30)
#   --gap       이 분 이내로 끊긴 경보는 하나로 합침 (기본 10)
#   --mindur    이 분 미만 경보는 무시 (기본 5)
#   --fab-cut   FAB 경계를 한 값으로 통일 (기본: 운영값 M16HUB 40 · 그 외 36)
#   --denom     area_score 분모 (기본 70) — --fab 를 주면 안 쓴다
import argparse
import csv
import glob
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta

csv.field_size_limit(10 ** 7)

EVENT_KEY = '발동이벤트'
AREAS = ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B']
TH_RA_DEFAULT = {'M16HUB': 9.0, 'M14': 3.3, 'M14B': 5.0, 'M16A': 3.2, 'M16B': 3.5}

# ★ 운영 시스템에 실제로 설정된 등급 컷 — FAB 마다 점수 분포가 달라 컷도 다르다
ALL_BANDS = (48, 60, 80)
FAB_BANDS = {'M16HUB': (40, 55, 75), 'M14': (36, 52, 72), 'M14B': (36, 52, 72),
             'M16A': (36, 52, 72), 'M16B': (36, 52, 72)}
ALL_CUT_DEFAULT = ALL_BANDS[0]
FAB_CUT_DEFAULT = {a: b[0] for a, b in FAB_BANDS.items()}


def level_of(score, bands):
    """점수 → 등급 문자열. 경계 미만은 공란(정상)."""
    if score is None:
        return ''
    if score >= bands[2]:
        return '초위험'
    if score >= bands[1]:
        return '위험'
    if score >= bands[0]:
        return '경계'
    return ''


# ────────────────────────────────────────────────── 입력
def load_thresholds():
    """thresholds.json 이 옆에 있으면 그 값을 쓴다 (운영과 어긋나지 않게)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.dirname(here)):
        fp = os.path.join(d, 'thresholds.json')
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            continue
        ra = cfg.get('TH_RA') or {}
        th = dict(TH_RA_DEFAULT)
        for a in AREAS:
            if a in ra:
                try:
                    th[a] = float(ra[a])
                except (TypeError, ValueError):
                    pass
        return th, fp
    return dict(TH_RA_DEFAULT), None


def nfc(s):
    return unicodedata.normalize('NFC', s or '')


def is_event_csv(name):
    n = nfc(name)
    return n.lower().endswith('.csv') and EVENT_KEY in n and '_M1' not in n


def fab_of(name):
    """영역분리 파일명 → 영역. 긴 이름부터 봐야 M16B 가 M16 으로 안 잡힌다."""
    n = nfc(name)
    if not n.lower().endswith('.csv') or EVENT_KEY not in n:
        return None
    for a in sorted(AREAS, key=len, reverse=True):
        if f'_{a}.' in n or n.endswith(f'_{a}.csv'):
            return a
    return None


def collect(path, matcher):
    """폴더·글롭·단일파일 → matcher 를 통과한 파일 목록"""
    p = (path or '').strip().strip('"').strip("'")
    if not p:
        return []
    if any(c in p for c in '*?'):
        hits = glob.glob(p)
    elif os.path.isdir(p):
        hits = [os.path.join(p, f) for f in os.listdir(p)]
    elif os.path.exists(p):
        hits = [p]
    else:
        print(f'  ❌ 없음: {os.path.abspath(p)}')
        return []
    return sorted(f for f in hits if matcher(os.path.basename(f)))


def in_range(fp, since):
    m = re.search(r'(\d{8})', os.path.basename(fp))
    return not (since and m and m.group(1) < since)


def fl(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_dt(s):
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime((s or '').strip(), fmt)
        except ValueError:
            pass
    return None


def read_rows(fp):
    """{시각: 행}. 같은 시각이 여러 번 있으면 첫 행만."""
    out, dup = {}, 0
    with open(fp, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            t = parse_dt(r.get('datetime'))
            if t is None:
                continue
            if t in out:
                dup += 1
            else:
                out[t] = r
    return out, dup


# ────────────────────────────────────────────────── 구간 계산
def episodes(flags):
    ep, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            ep.append((s, i - 1))
            s = None
    if s is not None:
        ep.append((s, len(flags) - 1))
    return ep


def merge(ep, gap):
    if not ep:
        return ep
    out = [list(ep[0])]
    for s, e in ep[1:]:
        if s - out[-1][1] - 1 <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def sustained(over, n_min):
    """임계 초과가 n_min 분 이상 이어진 구간만 True 로 남긴다."""
    out = [False] * len(over)
    for s, e in episodes(over):
        if e - s + 1 >= n_min:
            for i in range(s, e + 1):
                out[i] = True
    return out


def counts(real, alarm, gap, mindur, lead):
    """→ (사건수, 경보수, 적중경보, 적중사건, 정리된 경보 flags)"""
    n = len(real)
    R = merge(episodes(real), gap)
    A = [(s, e) for s, e in merge(episodes(alarm), gap) if e - s + 1 >= mindur]
    kept = [False] * n
    for s, e in A:
        for i in range(s, e + 1):
            kept[i] = True
    tp_r = sum(1 for s, e in R if any(kept[max(0, s - lead):e + 1]))
    tp_a = sum(1 for s, e in A if any(real[s:min(n, e + 1 + lead)]))
    return len(R), len(A), tp_a, tp_r, kept


def prf(nR, nA, tpA, tpR):
    p = tpA / nA if nA else 0.0
    r = tpR / nR if nR else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ────────────────────────────────────────────────── 출력
class Tee:
    def __init__(self, path=None):
        self.f = open(path, 'w', encoding='utf-8') if path else None

    def __call__(self, s=''):
        print(s)
        if self.f:
            self.f.write(s + '\n')

    def close(self):
        if self.f:
            self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', required=True, help='발동이벤트 CSV 또는 폴더')
    ap.add_argument('--fab', default=None, help='FAB별 영역분리 CSV 또는 폴더 (선택)')
    ap.add_argument('--csv', default=None, metavar='접두사',
                    help='결과 CSV 저장 — 요약 · 사건목록 · 날짜별_가로 · 분단위')
    ap.add_argument('--split', action='store_true',
                    help='대상(ALL·FAB5)별로 CSV 를 따로 낸다')
    ap.add_argument('--out', default=None, help='화면 내용을 텍스트로도 저장')
    ap.add_argument('--days', type=int, default=None)
    ap.add_argument('--since', default=None, metavar='YYYYMMDD')
    ap.add_argument('--cut', type=float, default=ALL_CUT_DEFAULT)
    ap.add_argument('--sustain', type=int, default=5)
    ap.add_argument('--lead', type=int, default=30)
    ap.add_argument('--gap', type=int, default=10)
    ap.add_argument('--mindur', type=int, default=5)
    ap.add_argument('--fab-cut', type=float, default=None, dest='fabcut')
    ap.add_argument('--denom', type=float, default=70)
    a = ap.parse_args()

    since = a.since
    if a.days:
        since = max(since or '', (datetime.now() - timedelta(days=a.days - 1)).strftime('%Y%m%d'))

    th_ra, th_path = load_thresholds()
    fabcut = {x: a.fabcut for x in AREAS} if a.fabcut is not None else dict(FAB_CUT_DEFAULT)

    ev_files = [f for f in collect(a.event, is_event_csv) if in_range(f, since)]
    if not ev_files:
        print('❌ 발동이벤트 파일이 없습니다.')
        sys.exit(2)
    fab_files = defaultdict(list)
    if a.fab:
        for f in collect(a.fab, lambda n: fab_of(n) is not None):
            if in_range(f, since):
                fab_files[fab_of(os.path.basename(f))].append(f)

    o = Tee(a.out)
    o('=' * 74)
    o('룰베이스 성능평가 — ALL · FAB별 Precision / Recall / F1')
    o('=' * 74)
    o(f'  발동이벤트 {len(ev_files)}개'
      + (f' · 영역분리 {sum(len(v) for v in fab_files.values())}개' if fab_files else ' · 영역분리 없음(score_raw 로 계산)'))
    o(f'  R-A 임계   ' + ' · '.join(f'{k} {v:g}' for k, v in th_ra.items())
      + (f'   ({os.path.basename(th_path)})' if th_path else '   (코드 기본값)'))
    o(f'  경보 기준  ALL {a.cut:g}  ·  ' + ' · '.join(f'{k} {v:g}' for k, v in fabcut.items()))
    o(f'  정답 기준  반송시간 임계초과 {a.sustain}분 지속  ·  적중 = 사건 시작 {a.lead}분 전까지의 경보')
    o(f'  경보 정리  {a.gap}분 이내 병합 · {a.mindur}분 미만 무시')

    # ── 1) 원자료 모으기 (시각 → 값)
    dup_total = 0
    ev = {}
    for fp in ev_files:
        rows, dup = read_rows(fp)
        dup_total += dup
        for t, r in rows.items():
            ev.setdefault(t, r)
    # 영역분리에서 area_score 를 직접 읽는다 (있으면 이쪽이 정본)
    fab_score = {x: {} for x in AREAS}
    for x, files in fab_files.items():
        for fp in files:
            rows, _ = read_rows(fp)
            for t, r in rows.items():
                v = fl(r.get('area_score'))
                if v is not None:
                    fab_score[x].setdefault(t, v)
    missing_area = [x for x in AREAS if fab_files.get(x) and not fab_score[x]]
    if missing_area:
        o(f'  ⚠️ area_score 컬럼 없음: {", ".join(missing_area)}'
          f' → 영역분리를 --score 로 다시 돌리거나, score_raw 로 계산합니다.')

    # ── 2) 날짜별로 나눠 계산
    by_date = defaultdict(list)
    for t in ev:
        by_date[t.date()].append(t)

    summary = []          # 요약 CSV 행
    detail = []           # 분단위 CSV 행
    events = []           # 사건목록 CSV 행
    hourly = []           # 시간별 가로 CSV 행
    tot = defaultdict(lambda: [0, 0, 0, 0])   # 대상 → [사건, 경보, 적중경보, 적중사건]

    for d in sorted(by_date):
        ts = sorted(by_date[d])
        base = datetime(d.year, d.month, d.day)
        idx = {int((t - base).total_seconds() // 60): ev[t] for t in ts}
        n = 1440

        # 영역별 값·정답·경보
        area_sc, area_real, area_alarm, area_ra = {}, {}, {}, {}
        for x in AREAS:
            sc, ra, over = [], [], []
            for m in range(n):
                t = base + timedelta(minutes=m)
                r = idx.get(m)
                v = fab_score[x].get(t)
                if v is None and r is not None:
                    raw = fl(r.get(x + '_score_raw'))
                    v = min(100, round(raw * 100 / a.denom)) if raw is not None else None
                sc.append(v)
                q = fl(r.get(x + '_ra')) if r is not None else None
                ra.append(q)
                over.append(q is not None and q >= th_ra[x])
            area_sc[x] = sc
            area_ra[x] = ra
            area_real[x] = sustained(over, a.sustain)
            area_alarm[x] = [v is not None and v >= fabcut[x] for v in sc]

        # ALL
        uni = []
        for m in range(n):
            r = idx.get(m)
            uni.append(fl(r.get('unified_risk_score')) if r is not None else None)
        all_real = sustained([any(area_real[x][m] for x in AREAS) for m in range(n)], 1)
        all_alarm = [v is not None and v >= a.cut for v in uni]

        # 지표
        def add(name, real, alarm, cut, score=None):
            nR, nA, tpA, tpR, kept = counts(real, alarm, a.gap, a.mindur, a.lead)
            # ── 사건 하나마다 시각을 남긴다 (시간대 분석용)
            for k, (s0, e0) in enumerate(merge(episodes(real), a.gap), 1):
                first = None
                for m in range(max(0, s0 - a.lead), e0 + 1):
                    if kept[m]:
                        first = m
                        break
                peak = None
                if score:
                    vals = [v for v in score[s0:e0 + 1] if v is not None]
                    peak = max(vals) if vals else None
                ft = (base + timedelta(minutes=first)) if first is not None else None
                events.append({
                    '날짜': str(d), '대상': name,
                    '사건번호': k,
                    '시작시각': (base + timedelta(minutes=s0)).strftime('%H:%M'),
                    '종료시각': (base + timedelta(minutes=e0)).strftime('%H:%M'),
                    '시작시': (base + timedelta(minutes=s0)).hour,
                    '지속분': e0 - s0 + 1,
                    '최고점수': '' if peak is None else peak,
                    '경보': 1 if first is not None else 0,
                    '첫경보시각': ft.strftime('%H:%M') if ft else '',
                    '선행분': (s0 - first) if first is not None else '',
                })
            p, r, f1 = prf(nR, nA, tpA, tpR)
            t = tot[name]
            t[0] += nR; t[1] += nA; t[2] += tpA; t[3] += tpR
            summary.append(dict(날짜=str(d), 대상=name, 경보컷=f'{cut:g}',
                                실제사건=nR, 경보=nA, 적중경보=tpA, 적중사건=tpR,
                                Precision=round(p, 3), Recall=round(r, 3), F1=round(f1, 3),
                                실제정체_분=sum(real), 경보_분=sum(kept)))
            return kept

        all_kept = add('ALL', all_real, all_alarm, a.cut, uni)
        fab_kept = {x: add(x, area_real[x], area_alarm[x], fabcut[x], area_sc[x]) for x in AREAS}

        # 분단위 CSV
        for m in range(n):
            if m not in idx:
                continue
            t = base + timedelta(minutes=m)
            row = {'datetime': t.strftime('%Y-%m-%d %H:%M'),
                   'date': str(d), 'time': t.strftime('%H:%M'), '시': t.hour,
                   'unified_risk_score': '' if uni[m] is None else uni[m],
                   'ALL_level': level_of(uni[m], ALL_BANDS),
                   'ALL_경보': 1 if all_kept[m] else 0,
                   'ALL_실제정체': 1 if all_real[m] else 0}
            for x in AREAS:
                v = area_sc[x][m]
                row[f'{x}_area_score'] = '' if v is None else v
                row[f'{x}_level'] = level_of(v, FAB_BANDS[x])
                row[f'{x}_경보'] = 1 if fab_kept[x][m] else 0
                row[f'{x}_실제정체'] = 1 if area_real[x][m] else 0
                row[f'{x}_ra'] = '' if area_ra[x][m] is None else area_ra[x][m]
            detail.append(row)

        # ── 시간별 가로 (날짜 × 시)
        for hh in range(24):
            lo, hi = hh * 60, hh * 60 + 60
            if not any(m in idx for m in range(lo, hi)):
                continue
            row = {'날짜': str(d), '시': hh, '시각': f'{hh:02d}:00',
                   '데이터분': sum(1 for m in range(lo, hi) if m in idx)}
            row['ALL_정체분'] = sum(1 for m in range(lo, hi) if all_real[m])
            row['ALL_경보분'] = sum(1 for m in range(lo, hi) if all_kept[m])
            uv = [uni[m] for m in range(lo, hi) if uni[m] is not None]
            row['ALL_점수최고'] = max(uv) if uv else ''
            for x in AREAS:
                row[f'{x}_정체분'] = sum(1 for m in range(lo, hi) if area_real[x][m])
                row[f'{x}_경보분'] = sum(1 for m in range(lo, hi) if fab_kept[x][m])
                sv = [area_sc[x][m] for m in range(lo, hi) if area_sc[x][m] is not None]
                row[f'{x}_점수최고'] = max(sv) if sv else ''
            hourly.append(row)

    # ── 3) 화면 출력
    o('')
    o('─' * 74)
    o('[1] 날짜별 ALL 성능')
    o('─' * 74)
    o(f"{'날짜':<13}{'사건':>5}{'경보':>5}{'Precision':>11}{'Recall':>9}{'F1':>7}")
    for s in summary:
        if s['대상'] == 'ALL':
            o(f"{s['날짜']:<13}{s['실제사건']:>5}{s['경보']:>5}"
              f"{s['Precision']:>11.2f}{s['Recall']:>9.2f}{s['F1']:>7.2f}")

    o('')
    o('─' * 74)
    o('[2] 전체 합산 — ALL · FAB별   ★ PPT 에 쓸 숫자')
    o('─' * 74)
    o(f"{'대상':<9}{'컷':>4}{'사건':>6}{'경보':>6}{'Precision':>11}{'Recall':>9}{'F1':>7}")
    order = ['ALL'] + AREAS
    for name in order:
        nR, nA, tpA, tpR = tot[name]
        p, r, f1 = prf(nR, nA, tpA, tpR)
        cut = a.cut if name == 'ALL' else fabcut[name]
        o(f'{name:<9}{cut:>4.0f}{nR:>6}{nA:>6}{p:>11.2f}{r:>9.2f}{f1:>7.2f}')
        summary.append(dict(날짜='전체', 대상=name, 경보컷=f'{cut:g}',
                            실제사건=nR, 경보=nA, 적중경보=tpA, 적중사건=tpR,
                            Precision=round(p, 3), Recall=round(r, 3), F1=round(f1, 3),
                            실제정체_분='', 경보_분=''))
    dead = [x for x in AREAS if tot[x][1] == 0]
    if dead:
        o('')
        o(f'  ⚠️ 경보가 한 건도 안 뜬 영역: {", ".join(dead)}')
        o('     → 그 영역 경계가 너무 높거나, 그 기간이 정말 조용했던 것입니다.')

    o('')
    o('─' * 74)
    o('[3] ALL 경보 기준을 바꾸면 — 트레이드오프')
    o('─' * 74)
    o(f"{'경보기준':<10}{'경보수':>7}{'Precision':>11}{'Recall':>9}{'F1':>7}")
    for cut in (36, 40, 44, 48, 52, 56, 60):
        nR = nA = tpA = tpR = 0
        for d in sorted(by_date):
            base = datetime(d.year, d.month, d.day)
            idx = {int((t - base).total_seconds() // 60): ev[t] for t in by_date[d]}
            rr = [False] * 1440
            for x in AREAS:
                ov = []
                for m in range(1440):
                    r = idx.get(m)
                    q = fl(r.get(x + '_ra')) if r is not None else None
                    ov.append(q is not None and q >= th_ra[x])
                sx = sustained(ov, a.sustain)
                rr = [rr[m] or sx[m] for m in range(1440)]
            al = []
            for m in range(1440):
                r = idx.get(m)
                v = fl(r.get('unified_risk_score')) if r is not None else None
                al.append(v is not None and v >= cut)
            c = counts(rr, al, a.gap, a.mindur, a.lead)
            nR += c[0]; nA += c[1]; tpA += c[2]; tpR += c[3]
        p, r, f1 = prf(nR, nA, tpA, tpR)
        o(f'{cut:<10}{nA:>7}{p:>11.2f}{r:>9.2f}{f1:>7.2f}'
          + ('  ← 현재' if abs(cut - a.cut) < 0.5 else ''))

    if dup_total:
        o('')
        o(f'  ※ 시각 중복 {dup_total}행은 제거하고 계산했습니다.')

    # ── 3-b) 시간대 분포
    if events:
        o('')
        o('─' * 74)
        o('[4] 사건이 언제 나는가 — 시작 시각 분포 (ALL 기준)')
        o('─' * 74)
        hh = [0] * 24
        for e in events:
            if e['대상'] == 'ALL':
                hh[e['시작시']] += 1
        mx = max(hh) or 1
        for blk in range(0, 24, 12):
            o('  ' + ' '.join(f'{h:>2}' for h in range(blk, blk + 12)) + '   시')
            o('  ' + ' '.join(('■' * min(2, hh[h]) or ' ·').rjust(2) for h in range(blk, blk + 12)))
            o('  ' + ' '.join(f'{hh[h]:>2}' for h in range(blk, blk + 12)) + '   건')
            o('')
        lead = [e['선행분'] for e in events if e['대상'] == 'ALL' and e['경보'] == 1]
        if lead:
            o(f'  경보가 붙은 사건 {len(lead)}건 · 평균 선행 {sum(lead)/len(lead):.1f}분 '
              f'· 최대 {max(lead)}분  (0 = 사건 시작과 동시)')

    # ── 4) CSV 저장
    if a.csv:
        sp = a.csv + '_요약.csv'
        with open(sp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        hp = a.csv + '_날짜별_가로.csv'
        if hourly:
            with open(hp, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(hourly[0].keys()))
                w.writeheader()
                w.writerows(hourly)
            o(f'  📄 {os.path.abspath(hp)}   ({len(hourly)}행 — 날짜 × 시간대 가로형)')
        ep = a.csv + '_사건목록.csv'
        if events:
            with open(ep, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
                w.writeheader()
                w.writerows(events)
            o(f'  📄 {os.path.abspath(ep)}   ({len(events)}행 — 시각·선행시간)')
        # ── 대상별 분리 (고객 요청 형식)
        if a.split:
            import statistics as _st
            for t in ['ALL'] + AREAS:
                rows = [r for r in summary if r['대상'] == t and r['날짜'] != '전체']
                if not rows:
                    continue
                recs = []
                for r in rows:
                    ee = [x for x in events if x['대상'] == t and x['날짜'] == r['날짜']]
                    ld = [x['선행분'] for x in ee if x['경보'] == 1 and x['선행분'] != '']
                    recs.append({
                        '날짜': r['날짜'], '대상': t, '경보컷': r['경보컷'],
                        '실제사건': r['실제사건'], '경보': r['경보'],
                        '적중경보': r['적중경보'], '적중사건': r['적중사건'],
                        'Precision': r['Precision'], 'Recall': r['Recall'], 'F1': r['F1'],
                        '정체시간_분': r['실제정체_분'], '경보시간_분': r['경보_분'],
                        '사건시각': ' / '.join(f"{x['시작시각']}~{x['종료시각']}" for x in ee),
                        '선행_중앙분': round(_st.median(ld)) if ld else '',
                        '최장지속_분': max((x['지속분'] for x in ee), default=0),
                    })
                tot = next((r for r in summary if r['대상'] == t and r['날짜'] == '전체'), None)
                if tot:
                    al = [x['선행분'] for x in events
                          if x['대상'] == t and x['경보'] == 1 and x['선행분'] != '']
                    recs.append({
                        '날짜': '전체', '대상': t, '경보컷': tot['경보컷'],
                        '실제사건': tot['실제사건'], '경보': tot['경보'],
                        '적중경보': tot['적중경보'], '적중사건': tot['적중사건'],
                        'Precision': tot['Precision'], 'Recall': tot['Recall'], 'F1': tot['F1'],
                        '정체시간_분': sum(x['정체시간_분'] for x in recs),
                        '경보시간_분': sum(x['경보시간_분'] for x in recs),
                        '사건시각': f"{sum(1 for x in events if x['대상'] == t)}건",
                        '선행_중앙분': round(_st.median(al)) if al else '',
                        '최장지속_분': max((x['최장지속_분'] for x in recs), default=0),
                    })
                fp2 = f'{a.csv}_{t}.csv'
                with open(fp2, 'w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                    w.writeheader()
                    w.writerows(recs)
                made = [f'{t}.csv']

                # 사건목록 — 그 대상만
                ee = [x for x in events if x['대상'] == t]
                if ee:
                    fp3 = f'{a.csv}_{t}_사건목록.csv'
                    with open(fp3, 'w', newline='', encoding='utf-8-sig') as f:
                        w = csv.DictWriter(f, fieldnames=list(ee[0].keys()))
                        w.writeheader()
                        w.writerows(ee)
                    made.append(f'사건목록({len(ee)})')

                # 분단위 — 그 대상 칸만
                keep = ['datetime', 'date', 'time', '시']
                keep += ([c for c in ('unified_risk_score', 'ALL_level', 'ALL_경보', 'ALL_실제정체')]
                         if t == 'ALL' else
                         [f'{t}_area_score', f'{t}_level', f'{t}_경보', f'{t}_실제정체', f'{t}_ra'])
                if detail:
                    keep = [c for c in keep if c in detail[0]]
                    fp4 = f'{a.csv}_{t}_분단위.csv'
                    with open(fp4, 'w', newline='', encoding='utf-8-sig') as f:
                        w = csv.DictWriter(f, fieldnames=keep, extrasaction='ignore')
                        w.writeheader()
                        w.writerows(detail)
                    made.append(f'분단위({len(detail)})')
                # 날짜별 가로 — 그 대상 칸만
                if hourly:
                    hk = ['날짜', '시', '시각', '데이터분',
                          f'{t}_정체분', f'{t}_경보분', f'{t}_점수최고']
                    hk = [c for c in hk if c in hourly[0]]
                    fp5 = f'{a.csv}_{t}_날짜별_가로.csv'
                    with open(fp5, 'w', newline='', encoding='utf-8-sig') as f:
                        w = csv.DictWriter(f, fieldnames=hk, extrasaction='ignore')
                        w.writeheader()
                        w.writerows(hourly)
                    made.append(f'날짜별_가로({len(hourly)})')
                o(f'  📄 {a.csv}_{t}*.csv   — ' + ' · '.join(made))
        dp = a.csv + '_분단위.csv'
        with open(dp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
            w.writeheader()
            w.writerows(detail)
        o('')
        o(f'  📄 {os.path.abspath(sp)}   ({len(summary)}행 — PPT 용)')
        o(f'  📄 {os.path.abspath(dp)}   ({len(detail)}행 — 검토용)')

    o('')
    o('=' * 74)
    o('[보고서에 반드시 같이 적을 것 — 성능 산출 기준]')
    o('=' * 74)
    o(f'  · 정답(실제 정체) : 반송시간(R-A)이 영역 임계 이상인 상태가 {a.sustain}분 이상 지속')
    o(f'                     ALL 은 다섯 영역 중 하나라도 그러면 정체로 봄')
    o(f'  · 경보           : ALL unified_risk_score >= {a.cut:g}')
    o('                     FAB area_score >= ' + ' · '.join(f'{k} {v:g}' for k, v in fabcut.items()))
    o(f'  · 적중 판정       : 사건 시작 {a.lead}분 전 ~ 종료 사이에 경보 존재')
    o(f'  · 경보 정리       : {a.gap}분 이내 병합, {a.mindur}분 미만 제외')
    o('  · 한계           : 반송시간은 룰의 입력이기도 하므로 완전히 독립된 정답이 아님.')
    o('                     진짜 성능은 운영자 인지 시각·MCS 알람 이력 확보 후 재측정 필요.')
    o.close()


if __name__ == '__main__':
    main()

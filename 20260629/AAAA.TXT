#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전조 분석 — raw(4/1~5/31) + 운영로그 → "정체/Deadlock 전에 지표가 미리 움직이나" 전수 검증
================================================================================================
목적: ML 만들기 전, "사전신호가 실제로 존재하는가" 를 데이터로 증명.
      각 정체/Deadlock 사건 전 32개 핵심지표가 몇 분 전부터 위험선을 넘는지(리드타임) 측정.

★ 표준 라이브러리만 사용 (pandas/numpy 불필요) → 회사 PC 어디서든 돌아감.

입력:
    --raw   raw 폴더 (M16A_HUBROOM_PR_*.CSV 여러 개 또는 하나의 큰 CSV. CRT_TM 으로 날짜 자동 인식)
    --log   운영로그 폴더 (또는 파일). *.txt / *.TXT
    --lead  사전예측 기준 분 (기본 30)
    --out   출력 폴더 (기본 ./out_전조분석)

실행:
    python 전조분석_raw_운영로그.py --raw ./raw --log ./운영로그 --lead 30 --out ./out_전조분석

출력:
    1. 사건목록.csv      — 운영로그에서 추출한 정체/Deadlock 사건 (시각·타입·소스·원문)
    2. 전조지표.csv      — 사건별 -60/-45/-30/-15/0분 32지표 값
    3. 선행성요약.csv    — 사건별 각 지표 리드타임 (위험선 첫 돌파 → 사건까지 분)
    4. 콘솔 요약         — 사건 수, 평균 선행시간, 지표별 선행 순위
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# ============================================================
# 32개 핵심 지표 + 위험 임계 (룰베이스 기준)
# ============================================================
# (컬럼, 한글, 위험임계, 방향)  방향='hi' = 값이 크면 위험
METRICS = [
    ('M16HUB.QUE.ALL.CURRENTQCNT',          'HUB반송큐',     2000, 'hi'),
    ('M16HUB.QUE.TIME.AVGTOTALTIME1MIN',    'HUB반송시간',   9.0,  'hi'),
    ('M16HUB.QUE.OHT.OHTUTIL',              'HUB_OHT가동',   95.0, 'hi'),
    ('M16HUB.STRATE.ALL.FABSTORAGERATIO',   'FAB저장율',     25.0, 'hi'),
    ('M16HUB.STRATE.STB.3F_STORAGE_UTIL',   'STB저장율',     99.0, 'hi'),
    ('M16HUB.QUE.ABN.AOTRANSDELAY',         'HUB반송지연',   1,    'hi'),
    ('M16HUB.QUE.M14TOM16.MESCURRENTQCNT',  'M14→M16큐',    100,  'hi'),
    ('M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO','HUB_SLA초과',  5.0,  'hi'),
    ('M14.QUE.LOAD.AVGLOADTIME1MIN',        'M14반송시간',   3.3,  'hi'),
    ('M14.QUE.ALL.3F_TO_HUB_JOB',           'M14→HUB잡',    80,   'hi'),
    ('M14.QUE.OHT.OHTUTIL',                 'M14_OHT가동',   95.0, 'hi'),
    ('M14.QUE.ALL.TRANSPORT4MINOVERRATIO',  'M14_SLA초과',   25.0, 'hi'),
    ('M14.SORTER.ABN.SORTERWAITCOUNTOVER',  'M14소터대기',   100,  'hi'),
    ('M14B.QUE.TIME.AVGTOTALTIME1MIN',      'M14B반송시간',  5.0,  'hi'),
    ('M14B.QUE.ALL.7F_TO_HUB_JOB',          'M14B→HUB잡',   150,  'hi'),
    ('M14B.QUE.OHT.OHTUTIL',                'M14B_OHT가동',  95.0, 'hi'),
    ('M14B.SORTER.ABN.SORTERWAITCOUNTOVER', 'M14B소터대기',  75,   'hi'),
    ('M16A.QUE.LOAD.AVGLOADTIME1MIN',       'M16A반송시간',  3.2,  'hi'),
    ('M16A.QUE.ALL.6F_TO_HUB_JOB',          'M16A→HUB잡',   80,   'hi'),
    ('M16A.QUE.OHT.OHTUTIL',                'M16A_OHT가동',  95.0, 'hi'),
    ('M16A.QUE.ALL.TRANSPORT4MINOVERRATIO', 'M16A_SLA초과',  13.0, 'hi'),
    ('M16A.SORTER.ABN.SORTERWAITCOUNTOVER', 'M16A소터대기',  180,  'hi'),
    ('M16B.QUE.LOAD.AVGLOADTIME1MIN',       'M16B반송시간',  3.5,  'hi'),
    ('M16B.QUE.ALL.10F_TO_HUB_JOB',         'M16B→HUB잡',   30,   'hi'),
    ('M16B.QUE.OHT.OHTUTIL',                'M16B_OHT가동',  95.0, 'hi'),
    ('M16B.QUE.ALL.TRANSPORT4MINOVERRATIO', 'M16B_SLA초과',  18.0, 'hi'),
    ('M16B.SORTER.ABN.SORTERWAITCOUNTOVER', 'M16B소터대기',  90,   'hi'),
    ('M16.QUE.SFAB.SENDQUEUETOTAL',         'M16송신큐',     20,   'hi'),
    ('M16HUB.SORTER.ABN.SORTERWAITCOUNTOVER','HUB소터대기',  30,   'hi'),
    ('M14B.QUE.TIME.AVGTOTALTIME1MIN',      'M14B반송시간2', 5.0,  'hi'),
    ('M16_PKT.QUE.TIME.AVGTOTALTIME1MIN',   'PKT반송시간',   7.5,  'hi'),
    ('M16_WT.QUE.TIME.AVGTOTALTIME1MIN',    'WT반송시간',    2.8,  'hi'),
]
METRIC_COLS = [m[0] for m in METRICS]

# ============================================================
# 운영로그 사건 키워드 (정체/Deadlock 만. CAPA변경=조치는 제외)
# ============================================================
EVENT_RULES = [
    ('DEADLOCK',     re.compile(r'deadlock|데드락', re.I)),
    ('STORAGE_FULL', re.compile(r'storage.*(full|불안정)|full.*storage|스토리지.*(풀|불안정)', re.I)),
    ('FULL',         re.compile(r'\bfull\b|가득|포화', re.I)),
    ('JAM',          re.compile(r'정체|병목|몰림|밀림|쌓', re.I)),
    ('QUE_HIGH',     re.compile(r'que\s*가?\s*많|큐\s*가?\s*많|queue.*많', re.I)),
    ('BOT_ALARM',    re.compile(r'\[HIGH_ALARM|\[ALARM|>=\s*Set\s*Value', re.I)),
]
# 운영자 조치(사건 아님) — 이 줄은 사건에서 제외
EXCLUDE = re.compile(r'max\s*capa|맥스\s*카파|원복|변경\s*(완료|했|부탁)|오픈|open|차단', re.I)

# 로그 헤더:  이름,부서,YYYY-MM-DD HH:MM:SS
HEADER_RE = re.compile(r'^.+?,.+?,(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*$')


# ============================================================
# 1. 운영로그 파싱 → 사건 목록
# ============================================================
def parse_logs(log_path):
    files = []
    if os.path.isdir(log_path):
        for ext in ('*.txt', '*.TXT'):
            files += glob.glob(os.path.join(log_path, ext))
    else:
        files = [log_path]
    events = []
    for fp in sorted(files):
        cur_time = None
        for enc in ('utf-8-sig', 'utf-8', 'cp949'):
            try:
                lines = open(fp, encoding=enc).read().splitlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"⚠️ 읽기 실패: {fp}"); continue
        for ln in lines:
            h = HEADER_RE.match(ln)
            if h:
                try: cur_time = datetime.strptime(h.group(1), '%Y-%m-%d %H:%M:%S')
                except: cur_time = None
                body = ln  # 헤더 줄에도 봇 알람이 붙는 경우 대비
            else:
                body = ln
            if not cur_time or not body.strip():
                continue
            if EXCLUDE.search(body):
                continue  # 운영자 조치 줄 스킵
            for etype, rx in EVENT_RULES:
                if rx.search(body):
                    events.append({
                        'time': cur_time.replace(second=0),
                        'type': etype,
                        'text': body.strip()[:120],
                        'file': os.path.basename(fp),
                    })
                    break
    # 같은 분 같은 타입 중복 제거
    seen = set(); uniq = []
    for e in sorted(events, key=lambda x: x['time']):
        k = (e['time'], e['type'])
        if k in seen: continue
        seen.add(k); uniq.append(e)
    return uniq


# ============================================================
# 2. raw 로드 (핵심 32지표만, 분 단위)
# ============================================================
def load_raw(raw_path):
    files = []
    if os.path.isdir(raw_path):
        for ext in ('*.csv', '*.CSV'):
            files += glob.glob(os.path.join(raw_path, ext))
    else:
        files = [raw_path]
    data = {c: {} for c in METRIC_COLS}   # {col: {datetime: val}}
    total_min = set()
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
            print(f"⚠️ 읽기 실패: {fp}"); continue
        present = [c for c in METRIC_COLS if c in (rd.fieldnames or [])]
        for row in rd:
            ts = (row.get('CRT_TM') or '').strip()
            if len(ts) < 16: continue
            try: t = datetime.strptime(ts[:16], '%Y-%m-%d %H:%M')
            except: continue
            total_min.add(t)
            for c in present:
                v = (row.get(c) or '').strip()
                if not v: continue
                try: data[c][t] = float(v)
                except: pass
        f.close()
    return data, sorted(total_min)


# ============================================================
# 3. 사건별 전조 분석
# ============================================================
def at(data, col, t, off=0):
    return data[col].get(t + timedelta(minutes=off))


def lead_time(data, col, thr, t_event, max_look=180):
    """사건 t_event 직전 max_look분 안에서 지표가 임계 thr 를 처음 넘은 시각 → 리드타임(분).
       ★ 사건 직전 '정상(임계 미만)' 이었다가 넘은 첫 지점을 찾음 (하루종일 위험이면 max_look 상한)."""
    first = None
    for off in range(-max_look, 1):
        v = at(data, col, t_event, off)
        if v is not None and v >= thr:
            first = off; break
    return (-first) if first is not None else None  # 몇 분 전


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--lead', type=int, default=30)
    ap.add_argument('--out', default='./out_전조분석')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("전조 분석 — raw + 운영로그")
    print("=" * 60)

    # ── 사건 ──
    events = parse_logs(a.log)
    tc = Counter(e['type'] for e in events)
    print(f"[사건] 운영로그에서 {len(events)}건 추출")
    for k, v in tc.most_common():
        print(f"       {k}: {v}")
    if not events:
        print("⚠️ 사건 없음 — 로그 경로/형식 확인"); return

    # ── raw ──
    print(f"[raw] 로딩...")
    data, minutes = load_raw(a.raw)
    if not minutes:
        print("⚠️ raw 없음 — 경로 확인"); return
    have = [c for c in METRIC_COLS if data[c]]
    print(f"      {len(minutes)}분 ({minutes[0].date()}~{minutes[-1].date()}) / 지표 {len(have)}/{len(METRIC_COLS)}개")

    # ── 사건별 전조지표 + 리드타임 ──
    rows_pre = []
    rows_lead = []
    lead_by_metric = defaultdict(list)
    lead_any = []   # 사건별 '어느 지표든 가장 먼저 위험' 리드타임
    for e in events:
        t = e['time']
        # raw 범위 밖 사건 스킵
        if not (minutes[0] <= t <= minutes[-1]):
            continue
        pre = {'time': t.strftime('%Y-%m-%d %H:%M'), 'type': e['type'], 'text': e['text']}
        earliest = None
        for col, name, thr, _ in METRICS:
            if not data[col]:
                continue
            for off in (-60, -45, -30, -15, 0):
                pre[f'{name}_{off}'] = _fmt(at(data, col, t, off))
            lt = lead_time(data, col, thr)  if False else lead_time(data, col, thr, t)
            if lt is not None:
                lead_by_metric[name].append(lt)
                rows_lead.append({'time': pre['time'], 'type': e['type'],
                                  'metric': name, 'thr': thr, 'lead_min': lt})
                if earliest is None or lt > earliest:
                    earliest = lt
        if earliest is not None:
            lead_any.append(earliest)
        rows_pre.append(pre)

    # ── 저장: 사건목록 ──
    with open(os.path.join(a.out, '사건목록.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['time', 'type', 'text', 'file'])
        for e in events:
            w.writerow([e['time'].strftime('%Y-%m-%d %H:%M'), e['type'], e['text'], e['file']])

    # ── 저장: 전조지표 ──
    if rows_pre:
        keys = list(rows_pre[0].keys())
        with open(os.path.join(a.out, '전조지표.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows_pre: w.writerow(r)

    # ── 저장: 선행성 ──
    with open(os.path.join(a.out, '선행성_raw.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['time', 'type', 'metric', 'thr', 'lead_min']); w.writeheader()
        for r in rows_lead: w.writerow(r)

    # ── 지표별 선행 순위 요약 ──
    summ = []
    for name, lts in lead_by_metric.items():
        summ.append((name, len(lts), sum(lts) / len(lts), max(lts)))
    summ.sort(key=lambda x: -x[2])
    with open(os.path.join(a.out, '선행성_요약.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['지표', '발동사건수', '평균선행분', '최대선행분'])
        for name, n, avg, mx in summ:
            w.writerow([name, n, f'{avg:.1f}', mx])

    # ── 콘솔 요약 ──
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    an = [x for x in lead_any if x >= a.lead]
    if lead_any:
        print(f"raw 범위 내 사건: {len(rows_pre)}건")
        print(f"'어느 지표든' 사건 {a.lead}분+ 전 위험선 돌파: "
              f"{len(an)}/{len(lead_any)}건 ({len(an)/len(lead_any)*100:.0f}%)")
        print(f"평균 선행시간(가장 빠른 지표): {sum(lead_any)/len(lead_any):.1f}분")
    print(f"\n▶ 선행성 좋은 지표 top10 (평균 선행분):")
    print(f"  {'지표':<14}{'사건수':>6}{'평균선행':>8}{'최대':>6}")
    for name, n, avg, mx in summ[:10]:
        print(f"  {name:<14}{n:>6}{avg:>7.1f}분{mx:>5}분")
    print(f"\n🎉 완료 → {a.out}/  (사건목록·전조지표·선행성_raw·선행성_요약)")
    print("이 4개 CSV 를 주시면 ML 설계(리드타임·핵심지표·임계) 확정합니다.")


def _fmt(v):
    return '' if v is None else (f'{v:.1f}' if isinstance(v, float) else v)


if __name__ == '__main__':
    main()

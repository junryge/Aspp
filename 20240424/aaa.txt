#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 데드락 경보 룰 검증 스크립트 (단독 실행)

사용법:
    python3 3단계_경보_검증_스크립트.py <STAR_CSV_경로> [--ops-log <운영로그.txt>] [데드락시각 ...]

운영로그 옵션:
    --ops-log 운영로그.txt  (선택, 제공 시 각 CSV 에 운영이벤트 컬럼 추가)
    → 3단계 경보가 실제 운영 이슈와 얼마나 일치하는지 자동 검증.
    → '가설 입증' 용: 알고리즘이 실제 데드락 징후를 포착했다는 증거 확보.

데드락 시각 포맷 (혼용 가능):
    14:00                    # 시:분만 (CSV 1개 처리 시 그 날짜로 가정)
    20260421 14:00           # 날짜 + 시:분 (일주일 치 일괄 처리 시 필수)
    2026-04-21 14:00         # 하이픈 포맷
    2026-04-21T14:00         # ISO
    20260421_14:00           # 언더스코어 구분

예:
    # 1개 파일 + 시각
    python3 3단계_경보_검증_스크립트.py STAR_20260421.csv 14:00

    # 여러 파일 (일주일) + 날짜별 데드락
    python3 3단계_경보_검증_스크립트.py "STAR_*.csv" \\
        "20260421 14:00" "20260421 16:30" "20260423 09:15"

    # 데드락 시각 모를 때 (탐지만)
    python3 3단계_경보_검증_스크립트.py "STAR_*.csv"

출력:
    - 터미널: 날짜별 단계 전환 타임라인 + 종합 precision/recall
    - CSV 3종 자동 저장:
        · 검증결과_이벤트_YYYYMMDD_HHMMSS.csv     (모든 단계 전환 이벤트 + TP/FP 분류)
        · 검증결과_파일별요약_YYYYMMDD_HHMMSS.csv (파일별 S3/TP/FN/FP 개수)
        · 검증결과_종합지표_YYYYMMDD_HHMMSS.csv   (precision/recall/평균선행시간)
      ※ UTF-8 BOM 포함 (엑셀 한글 깨짐 방지)

룰 정의 (M16A_BR 검증):
    R-A' AVGTOTAL1MIN ≥ 9분이 10분창 1회 이상
    R-B  M14→M16 큐 30분간 +100 이상
    R-C' 리프터 합 감소 + 역증가 2개 이상

    1단계 = R-A' 2회+
    2단계 = R-B 발동
    3단계 = R-A' AND R-B AND R-C' (AND 조건 / 데드락 10분 전 목표)
"""

import csv
import sys
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict


# 리프터 ID 10개 (M16A_BR 기준)
LIFTER_IDS = [
    '6ABL6011', '6ABL6012', '6ABL6021', '6ABL6022',
    '6ABL6031', '6ABL6032', '6ABL0111', '6ABL0112',
    '6ABL0121', '6ABL0122',
]


# ────────────────────────────────────────────────────────────
# 운영 채팅 로그 파싱 (선택적, --ops-log 제공 시)
# ────────────────────────────────────────────────────────────
import re as _re

_OPS_EVENT_PATTERNS = [
    ('DEADLOCK_SIGNAL',  ['정체', '몰림', '밀림', '밀리는', '밀려', '증가하고 있', 'Queue 증가', 'QUE 증가', 'Que 증가', '쌓이고']),
    ('BRIDGE_ERROR',     ['Bridge OHT Error', 'Bridge OHT 발생', 'bridge 이상', 'Bridge 정체']),
    ('MLUD_ISSUE',       ['MLUD', 'Mlud', 'mlud']),
    ('CAPA_CHANGE_1',    ['MAX CAPA "1"', 'MAX CAPA 1', 'Max Capa 1', 'MAXCAPA 1', 'Capa 1로', '"1"로 변경']),
    ('CAPA_CHANGE_50',   ['MAX CAPA 50', 'Max Capa 50', 'MAXCAPA 50', 'Capa 50']),
    ('CAPA_CHANGE_3',    ['MAX CAPA "3"', 'MAX CAPA 3', 'Max Capa 3', 'MAXCAPA 3']),
    ('CAPA_RESTORE',     ['원복', '원위치']),
    ('LIFTER_DOWN',      ['Lifter', 'lifter', 'LIFTER', '리프터']),
    ('ERROR_OCCURRED',   ['Error 발생', 'Error발생', 'Err 발생', 'ERROR 발생', 'Alarm 발생']),
    ('ERROR_RECOVERED',  ['Error 조치', 'Error조치', 'Err 조치', 'Error 처리', 'Error Clear', 'ERROR 처리', 'ERROR 조치']),
    ('PORT_CLOSE',       ['AI Close', 'AI close', 'AI CLOSE', 'PORT Close', 'Port Close', 'Port 차단', 'PORT 차단']),
    ('PORT_OPEN',        ['AI Open', 'AI open', 'AI OPEN', 'PORT Open', 'Port Open']),
    ('MAINTENANCE',      ['작업', '교체', '점검', 'H/T', 'Handy Stop', 'HT-STOP', 'HT STOP']),
    ('ALERT_BOT',        ['통합알림센터', 'Intelligent Bot', 'LOW_ALARM', 'HIGH_ALARM']),
]

_OPS_HEADER_RE = _re.compile(
    r'^(?P<author>[^,]+),(?P<org>[^,]+),(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$'
)


def _ops_classify(text):
    for ev_type, keywords in _OPS_EVENT_PATTERNS:
        for kw in keywords:
            if kw in text:
                return ev_type, kw
    return 'OTHER', ''


def _ops_strip(text):
    text = _re.sub(r'<<이미지>>[^\n]*', '', text)
    text = _re.sub(r'<<파일>>[^\n]*', '', text)
    text = _re.sub(r'http[s]?://\S+', '', text)
    return text


def _ops_truncate(text, n=80):
    text = _re.sub(r'\s+', ' ', text).strip()
    return text if len(text) <= n else text[:n-1] + '…'


def parse_ops_log(path):
    """운영 채팅 로그 → [{dt, author, org, event_type, keyword, summary}, ...]"""
    ops = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        print(f'  ⚠️  운영로그 읽기 실패: {e}')
        return ops

    i = 0
    while i < len(lines):
        m = _OPS_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        author = m['author'].strip()
        org = m['org'].strip()
        try:
            dt = datetime.strptime(m['ts'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            i += 1
            continue
        body_lines = []
        i += 1
        while i < len(lines) and not _OPS_HEADER_RE.match(lines[i]):
            body_lines.append(lines[i].rstrip())
            i += 1
        body = _ops_strip('\n'.join(body_lines).strip())
        if not body:
            continue
        ev_type, kw = _ops_classify(body)
        ops.append({
            'dt': dt,
            'author': author,
            'org': org,
            'event_type': ev_type,
            'keyword': kw,
            'summary': _ops_truncate(body, 60),
        })
    ops.sort(key=lambda x: x['dt'])
    return ops


def find_nearby_ops(query_dt, ops_list, window_min=30):
    """query_dt ±window_min 범위의 운영 이벤트 반환 (ALERT_BOT 제외)"""
    if not ops_list or query_dt is None:
        return []
    lo = query_dt - timedelta(minutes=window_min)
    hi = query_dt + timedelta(minutes=window_min)
    result = []
    for op in ops_list:
        if op['dt'] < lo:
            continue
        if op['dt'] > hi:
            break
        if op['event_type'] == 'ALERT_BOT':
            continue
        result.append(op)
    return result


def summarize_ops(nearby_ops, query_dt):
    """근처 운영 이벤트 요약 → (count, types_str, top_msg_str)"""
    if not nearby_ops:
        return 0, '', ''
    type_counts = {}
    for op in nearby_ops:
        type_counts[op['event_type']] = type_counts.get(op['event_type'], 0) + 1
    types_str = ', '.join(f'{t}({c})' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))
    # 가장 가까운 시각의 이벤트 1개
    closest = min(nearby_ops, key=lambda x: abs((x['dt'] - query_dt).total_seconds()))
    diff_min = (closest['dt'] - query_dt).total_seconds() / 60.0
    sign = '+' if diff_min >= 0 else ''
    top_msg = f"[{sign}{diff_min:.0f}분] {closest['author']}: {closest['summary']}"
    return len(nearby_ops), types_str, top_msg


def safe_float(v):
    try:
        return float(v) if v not in (None, '', 'null') else None
    except (ValueError, TypeError):
        return None


def safe_int(v):
    try:
        return int(float(v)) if v not in (None, '', 'null') else None
    except (ValueError, TypeError):
        return None


_parse_time_failed_samples = []

def parse_time(s):
    """CRT_TM 파싱 — 다양한 포맷 대응"""
    if not s:
        return None
    raw = s
    s = s.strip().strip('"').strip("'")
    # Timezone suffix 제거
    if 'T' in s and '+' in s.split('T')[1]:
        s = s.split('+')[0]
    if s.endswith('Z'):
        s = s[:-1]
    # 밀리초 제거
    if '.' in s:
        s = s.split('.')[0]
    # Unix timestamp (숫자만)
    if s.isdigit() and len(s) in (10, 13):
        try:
            ts = int(s)
            if len(s) == 13:
                ts //= 1000
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            pass
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M', '%Y.%m.%d %H:%M:%S',
                '%Y%m%d%H%M%S', '%Y%m%d %H%M%S',
                '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M',
                '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                '%d-%m-%Y %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    if len(_parse_time_failed_samples) < 3:
        _parse_time_failed_samples.append(raw)
    return None


def parse_deadlock_arg(s, fallback_date=None):
    """데드락 시각 인자 파싱 — 날짜 포함 / 시각만 모두 지원.

    반환: datetime or None
    """
    if not s:
        return None
    s = s.strip().replace('_', ' ').replace('T', ' ')

    # 날짜+시각 포맷 우선
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M', '%Y%m%d %H:%M', '%Y%m%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # 시각만 (HH:MM) — fallback_date 필요
    try:
        hh, mm = map(int, s.split(':')[:2])
        if fallback_date is None:
            return None  # 날짜 미상
        return datetime.combine(fallback_date, datetime.min.time()).replace(hour=hh, minute=mm)
    except (ValueError, AttributeError):
        return None


def detect_prefix(fieldnames):
    """STAR 컬럼 prefix 자동 감지 (M16HUB / M14 / 기타)"""
    anchor = '.QUE.ALL.CURRENTQCNT'
    for col in fieldnames or []:
        if col and col.endswith(anchor):
            return col[:-len(anchor)]
    return None


def load_star(filepath):
    """STAR CSV 로드 → [(time, star_dict), ...]"""
    timeline = []
    missing_cols = defaultdict(int)
    total = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prefix = detect_prefix(reader.fieldnames)
        if not prefix:
            print(f"❌ {filepath}: prefix 감지 실패 (CURRENTQCNT 컬럼 없음)")
            return [], None

        def C(suffix):
            return f"{prefix}{suffix}"

        # 핵심 컬럼 존재 여부 사전 체크
        required = {
            'avgtotal1min': C('.QUE.TIME.AVGTOTALTIME1MIN'),
            'm14_to_m16':   C('.QUE.M14TOM16.MESCURRENTQCNT'),
        }
        for k, col in required.items():
            if col not in (reader.fieldnames or []):
                missing_cols[k] += 1

        for row in reader:
            total += 1
            t = parse_time(row.get('CRT_TM', ''))
            if not t:
                continue
            star = {
                'avgtotal1min': safe_float(row.get(C('.QUE.TIME.AVGTOTALTIME1MIN'))),
                'm14_to_m16':   safe_int(row.get(C('.QUE.M14TOM16.MESCURRENTQCNT'))),
                'queue_total':  safe_int(row.get(C('.QUE.ALL.CURRENTQCNT'))),
                'lft_list': {
                    lid: safe_int(row.get(C(f'.LFT.{lid}.TOTAL_CURRENTQCNT')))
                    for lid in LIFTER_IDS
                },
                # 추가: 간접 컬럼도 영향도 통계 계산용 로드
                'queue_completed':  safe_int(row.get(C('.QUE.ALL.CURRENTQCOMPLETED'))),
                'queue_oht':        safe_int(row.get(C('.QUE.OHT.CURRENTOHTQCNT'))),
                'oht_util':         safe_float(row.get(C('.QUE.OHT.OHTUTIL'))),
                'avg_load_time':    safe_float(row.get(C('.QUE.LOAD.AVGLOADTIME'))),
                'transport_4over':  safe_int(row.get(C('.QUE.ALL.TRANSPORT4MINOVERCNT'))),
                'driving':          safe_int(row.get(C('.OHT.STATECNT.DRIVING'))),
                'obs_bz_stop':      safe_int(row.get(C('.OHT.STATECNT.OBSANDBZSTOP'))),
                'congested':        safe_int(row.get(C('.OHT.STATECNT.CONGESTED'))),
                'pause':            safe_int(row.get(C('.OHT.STATECNT.PAUSE'))),
                'timeout':          safe_int(row.get(C('.OHT.STATECNT.TIMEOUT'))),
                'mlud_q':           safe_int(row.get(C('.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT'))),
                'fab_trans_job':    safe_int(row.get(C('.QUE.ALL.FABTRANSJOBCNT'))),
            }
            # 리프터 None 제거 (해당 FAB 에 없으면 빈 dict)
            star['lft_list'] = {k: v for k, v in star['lft_list'].items() if v is not None}
            timeline.append((t, star))

    timeline.sort(key=lambda x: x[0])

    if missing_cols:
        print(f"  ⚠️  누락 컬럼: {list(missing_cols.keys())} (해당 룰 평가 불가)")

    return timeline, prefix


def evaluate_timeline(timeline):
    """3단계 룰 평가 → [{time, stage, label, reason, rules_detail}, ...]"""
    events = []
    if not timeline:
        return events

    t1_hist = []
    m14_hist = []
    lft_hist = []

    last_logged_stage = -1
    last_s3_time = None

    for t, star in timeline:
        if star['avgtotal1min'] is not None:
            t1_hist.append(star['avgtotal1min'])
        if star['m14_to_m16'] is not None and star['m14_to_m16'] > 0:
            m14_hist.append(star['m14_to_m16'])
        if star['lft_list']:
            lft_hist.append(dict(star['lft_list']))

        # R-A' (기존 — S3 확정에 사용, 9분 기준 유지)
        recent_t1 = t1_hist[-10:]
        ra_count = sum(1 for v in recent_t1 if v >= 9.0)
        ra_value = recent_t1[-1] if recent_t1 else None
        ra_trig = ra_count >= 1

        # ⚡ R-A' EARLY (튜닝 ① — S1 조기 인지용, 6분 지속)
        ra_sustained = False
        if len(recent_t1) >= 5:
            last5 = recent_t1[-5:]
            ra_sustained = sum(1 for v in last5 if v >= 6.0) >= 3

        # R-B (기존 — S3 확정에 사용, +100/30분 유지)
        rb_diff = 0
        rb_trig = False
        if len(m14_hist) >= 31:
            rb_diff = m14_hist[-1] - m14_hist[-31]
            rb_trig = rb_diff >= 100

        # ⚡ R-B FAST (튜닝 ② — S2 조기 인지용, +30/10분)
        rb_fast = False
        if len(m14_hist) >= 11:
            rb_diff_fast = m14_hist[-1] - m14_hist[-11]
            rb_fast = rb_diff_fast >= 30

        # R-C' (그대로)
        rc_trend = 0
        rev_count = 0
        rev_lids = []
        rc_trig = False
        if len(lft_hist) >= 21:
            now_l = lft_hist[-1]
            prev_l = lft_hist[-21]
            rc_trend = sum(now_l.values()) - sum(prev_l.values())
            for lid in now_l:
                if now_l[lid] > prev_l.get(lid, 0):
                    rev_lids.append(lid)
                    rev_count += 1
            rc_trig = rc_trend < 0 and rev_count >= 2

        # 단계 판정 — S1/S2 는 튜닝된 조건으로 조기 발동, S3 는 원래 그대로
        s1 = (ra_count >= 2) or ra_sustained      # 9분 2회 또는 6분 3회 연속
        s2 = rb_trig or rb_fast                   # +100/30분 또는 +30/10분
        s3 = ra_trig and rb_trig and rc_trig      # 검증된 원래 룰 유지
        stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))

        record = False
        reason = ''
        if stage > last_logged_stage and stage > 0:
            record = True
            if stage == 1:
                reason = f'1MIN ≥9분이 {ra_count}회'
            elif stage == 2:
                reason = f'M14→M16 +{rb_diff} (30분간)'
            elif stage == 3:
                reason = (f'AND 만족 (1MIN {ra_value:.2f}, '
                          f'M14→M16 +{rb_diff}, 역증가 {rev_count}개)')
            last_logged_stage = stage
            if stage == 3:
                last_s3_time = t
        elif stage == 3 and last_s3_time is not None:
            diff_min = (t - last_s3_time).total_seconds() / 60.0
            if diff_min >= 10:
                record = True
                reason = f'재발동 (역증가 {rev_count}개)'
                last_s3_time = t
        elif stage == 0 and last_logged_stage >= 1:
            record = True
            reason = '정상화'
            last_logged_stage = 0
            last_s3_time = None

        if record:
            events.append({
                'time': t,
                'stage': stage,
                'reason': reason,
                'ra_value': ra_value,
                'rb_diff': rb_diff,
                'rev_count': rev_count,
                'rev_lids': list(rev_lids),
            })

    return events


def build_incidents(events):
    """3단계 이벤트를 "사건 단위" 로 묶음.

    1개 사건 = [신규 AND 만족] + 뒤따르는 [재발동 ...] 의 연속. '정상화' 나
    다른 신규 S3 를 만나면 사건 종료.

    반환: [{start, end, refire_count, max_1min, max_rb_diff, max_rev, severity}, ...]
    """
    incidents = []
    cur = None

    def _finalize(c, end_time):
        if not c:
            return
        c['end'] = end_time
        dur = (c['end'] - c['start']).total_seconds() / 60.0
        c['duration_min'] = round(dur, 1)
        # ★ 등급 판정
        rc = c['refire_count']
        m1 = c['max_1min'] or 0
        if rc >= 4 or m1 >= 20:
            c['severity'] = '★★★'
        elif rc >= 2 or m1 >= 15:
            c['severity'] = '★★'
        elif rc >= 1 or m1 >= 10:
            c['severity'] = '★'
        else:
            c['severity'] = '-'
        incidents.append(c)

    for e in events:
        if e['stage'] == 3 and e['reason'].startswith('AND 만족'):
            # 신규 S3 → 이전 사건 종료, 새 사건 시작
            _finalize(cur, e['time'])
            cur = {
                'start': e['time'],
                'end': e['time'],
                'refire_count': 0,
                'max_1min': e.get('ra_value') or 0,
                'max_rb_diff': e.get('rb_diff') or 0,
                'max_rev': e.get('rev_count') or 0,
                'rev_lids_union': set(e.get('rev_lids') or []),
                'start_reason': e['reason'],
            }
        elif e['stage'] == 3 and e['reason'].startswith('재발동'):
            if cur is not None:
                cur['refire_count'] += 1
                cur['end'] = e['time']
                cur['max_rev'] = max(cur['max_rev'], e.get('rev_count') or 0)
                cur['rev_lids_union'].update(e.get('rev_lids') or [])
        elif e['stage'] == 0 and cur is not None:
            # 정상화 → 사건 종료
            _finalize(cur, e['time'])
            cur = None

    # 파일 끝난 시점에 열려있는 사건 종료
    if cur is not None and events:
        _finalize(cur, events[-1]['time'])

    return incidents


def report(filepath, events, deadlock_datetimes=None):
    """결과 출력.

    deadlock_datetimes: 이 파일 날짜와 매칭되는 데드락 datetime 리스트 (이미 필터됨)
    """
    deadlock_datetimes = deadlock_datetimes or []
    print()
    print('═' * 78)
    print(f'📁 {os.path.basename(filepath)}')
    print('═' * 78)

    if not events:
        print('  ✅ 어느 단계도 발동 안 함 (완전 정상 / 또는 데이터 부족)')
        return [], [], []

    icons = {0: '✅', 1: '🔔', 2: '⚠️', 3: '🚨'}
    names = {0: '정상화', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}

    distinct_dates = sorted({e['time'].date() for e in events})
    multi_day = len(distinct_dates) > 1
    if multi_day:
        print(f'\n  📅 기간: {distinct_dates[0]} ~ {distinct_dates[-1]} ({len(distinct_dates)}일)')
    else:
        print(f'\n  📅 날짜: {distinct_dates[0]}')
    print(f'  📊 단계 전환 타임라인 ({len(events)}건)')
    print(f'  {"-"*74}')
    for e in events:
        ts = e['time'].strftime('%Y-%m-%d %H:%M') if multi_day else e['time'].strftime('%H:%M')
        print(f"  {ts}  {icons[e['stage']]} {names[e['stage']]:<18} {e['reason']}")

    # 통계
    counts = defaultdict(int)
    for e in events:
        counts[e['stage']] += 1
    print(f'\n  📈 단계별 발동 횟수')
    for stage in (1, 2, 3):
        print(f'    {names[stage]:<18}: {counts[stage]}회')

    # 사건 단위 묶음 + ★ 등급
    incidents = build_incidents(events)
    if incidents:
        new_count = len(incidents)
        refire_total = sum(i['refire_count'] for i in incidents)
        print(f'\n  🧩 사건 단위 분류: {new_count}개 사건 (신규 {new_count} + 재발동 {refire_total} = {counts[3]})')
        print(f'  {"-"*74}')
        # ★ 등급 높은 순 → 시각 순
        sev_rank = {'★★★': 3, '★★': 2, '★': 1, '-': 0}
        sorted_inc = sorted(incidents, key=lambda i: (-sev_rank[i['severity']], i['start']))
        for i in sorted_inc:
            stamp = i['start'].strftime('%Y-%m-%d %H:%M')
            one_min = i['max_1min']
            dur = f"{i['duration_min']:.0f}분" if i['duration_min'] > 0 else '단발'
            refire = f"재발동 {i['refire_count']}회" if i['refire_count'] else '단발'
            print(f"  {i['severity']:<4}  {stamp}  1MIN {one_min:>5.2f}  M14→M16 +{i['max_rb_diff']:<3}  {refire:<10}  {dur} 지속")

    tp_list, fn_list, fp_list = [], [], []

    # 데드락 ground truth 매칭
    if deadlock_datetimes:
        print(f'\n  🎯 이 날짜의 데드락 발생 시각 매칭 ({len(deadlock_datetimes)}건)')
        s3_events = [e for e in events if e['stage'] == 3]

        for dt in deadlock_datetimes:
            # 가장 가까운 3단계 발동 (이전 1시간 내)
            closest = None
            min_diff = None
            for s3 in s3_events:
                diff_sec = (dt - s3['time']).total_seconds()
                if 0 <= diff_sec <= 3600:
                    if min_diff is None or diff_sec < min_diff:
                        min_diff = diff_sec
                        closest = s3

            if closest:
                lead_min = min_diff / 60.0
                mark = '✅' if 5 <= lead_min <= 30 else '⚠️'
                print(f'    {mark} 데드락 {dt.strftime("%Y-%m-%d %H:%M")} ← '
                      f'3단계 {closest["time"].strftime("%H:%M")} (선행 {lead_min:.0f}분)')
                tp_list.append((dt, closest, lead_min))
            else:
                print(f'    ❌ 데드락 {dt.strftime("%Y-%m-%d %H:%M")}: 직전 1시간 내 3단계 없음 (FN)')
                fn_list.append(dt)

        # False positive 추정
        for s3 in s3_events:
            matched = False
            for dt in deadlock_datetimes:
                diff = (dt - s3['time']).total_seconds()
                if -600 <= diff <= 1800:  # 데드락 -10분 ~ +30분
                    matched = True
                    break
            if not matched:
                fp_list.append(s3)

        if fp_list:
            print(f'\n  ⚠️  False positive 후보 ({len(fp_list)}건, 데드락 시각과 무관한 3단계)')
            for e in fp_list:
                print(f'    {e["time"].strftime("%H:%M")} - {e["reason"]}')

    return tp_list, fn_list, fp_list


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pattern = sys.argv[1]
    raw_args = sys.argv[2:]

    # --ops-log <path> 파싱
    ops_log_path = None
    raw_deadlock_args = []
    idx = 0
    while idx < len(raw_args):
        a = raw_args[idx]
        if a == '--ops-log' and idx + 1 < len(raw_args):
            ops_log_path = raw_args[idx + 1]
            idx += 2
            continue
        if a.startswith('--ops-log='):
            ops_log_path = a.split('=', 1)[1]
            idx += 1
            continue
        raw_deadlock_args.append(a)
        idx += 1

    # 운영 로그 로드 (선택적)
    ops_list = []
    if ops_log_path:
        print(f'\n📜 운영 로그 로드: {ops_log_path}')
        ops_list = parse_ops_log(ops_log_path)
        print(f'   → {len(ops_list)}건 파싱')

    files = sorted(glob.glob(pattern)) if any(c in pattern for c in '*?[]') else [pattern]
    if not files:
        print(f'❌ 파일 없음: {pattern}')
        sys.exit(1)

    # 데드락 인자 1차 파싱
    dl_dated = []
    dl_time_only = []
    for arg in raw_deadlock_args:
        parsed = parse_deadlock_arg(arg)
        if parsed is not None:
            dl_dated.append(parsed)
        elif ':' in arg:
            dl_time_only.append(arg)
        else:
            print(f'  ⚠️  해석 불가 인자 무시: {arg!r}')

    print(f'\n🔍 검증 시작 — 파일 {len(files)}개')
    if dl_dated:
        print(f'   데드락 시각 (날짜 포함, {len(dl_dated)}건):')
        for dt in dl_dated:
            print(f'     · {dt.strftime("%Y-%m-%d %H:%M")}')
    if dl_time_only:
        print(f'   데드락 시각 (날짜 없음, {len(dl_time_only)}건): {dl_time_only}')

    all_summary = []
    total_tp, total_fn, total_fp = [], [], []
    all_events_rows = []      # CSV 출력용 (전체 단계 전환 이벤트)
    all_incidents_rows = []   # CSV 출력용 (사건 단위)
    all_xai_rows = []         # CSV 출력용 (이상 판단 근거 / XAI)
    all_col_rows = []         # CSV 출력용 (입력 STAR 컬럼별 기여도)
    all_timelines = []        # CSV 10 계산용: [(t, star_dict), ...]
    all_incident_starts = []  # CSV 10 계산용: [datetime, ...] 사건 시작 시각

    for fp in files:
        if not os.path.exists(fp):
            print(f'⚠️  스킵 (없음): {fp}')
            continue

        timeline, prefix = load_star(fp)
        if not timeline:
            print(f'⚠️  스킵 (데이터 없음): {fp}')
            continue

        # CSV 10 계산용 누적
        all_timelines.extend(timeline)

        dates_in_file = sorted({t.date() for t, _ in timeline})
        first_ts = timeline[0][0].strftime('%Y-%m-%d %H:%M:%S')
        last_ts = timeline[-1][0].strftime('%Y-%m-%d %H:%M:%S')
        span_info = f'{dates_in_file[0]} ~ {dates_in_file[-1]} ({len(dates_in_file)}일)' if len(dates_in_file) > 1 else str(dates_in_file[0])
        print(f'\n📥 {os.path.basename(fp)} — prefix={prefix}, 날짜={span_info}, {len(timeline)}행')
        print(f'    CRT_TM 범위: {first_ts} ~ {last_ts}')
        # 하위 호환: report 에 쓸 대표 날짜 (다일 파일일 땐 전체 범위)
        file_date = dates_in_file[0] if len(dates_in_file) == 1 else None
        if _parse_time_failed_samples:
            print(f'    ⚠️  시각 파싱 실패 샘플: {_parse_time_failed_samples[:3]}')
            _parse_time_failed_samples.clear()

        file_deadlocks = [dt for dt in dl_dated if dt.date() == file_date]
        for ts in dl_time_only:
            dt = parse_deadlock_arg(ts, fallback_date=file_date)
            if dt is not None:
                file_deadlocks.append(dt)

        events = evaluate_timeline(timeline)
        tp, fn, fp_list = report(fp, events, file_deadlocks)

        s3_count = sum(1 for e in events if e['stage'] == 3)
        date_label = span_info  # 다일 파일이면 'YYYY-MM-DD ~ YYYY-MM-DD (N일)'
        all_summary.append((os.path.basename(fp), date_label, len(events), s3_count,
                            len(tp), len(fn), len(fp_list)))
        total_tp.extend(tp)
        total_fn.extend(fn)
        total_fp.extend(fp_list)

        # CSV 행 구성: 각 이벤트 + 분류
        fp_times = {e['time'] for e in fp_list}
        tp_times = {closest['time']: (dl_dt, lead) for dl_dt, closest, lead in tp}

        for e in events:
            classification = '-'
            matched_deadlock = ''
            lead_min = ''
            if e['stage'] == 3:
                if e['time'] in tp_times:
                    dl_dt, lead = tp_times[e['time']]
                    classification = 'TP'
                    matched_deadlock = dl_dt.strftime('%Y-%m-%d %H:%M')
                    lead_min = f'{lead:.1f}'
                elif e['time'] in fp_times:
                    classification = 'FP'
            nearby = find_nearby_ops(e['time'], ops_list, 30) if ops_list else []
            op_cnt, op_types, op_top = summarize_ops(nearby, e['time'])
            # 알고리즘 vs 운영자 시차
            earliest = min(nearby, key=lambda o: o['dt']) if nearby else None
            if earliest:
                lead_vs_op = round((earliest['dt'] - e['time']).total_seconds() / 60.0, 1)
                pred_tag = '🔮 예측' if lead_vs_op > 5 else '🏷️ 라벨' if lead_vs_op < -5 else '⏱️ 동시'
            else:
                lead_vs_op = ''
                pred_tag = '❓' if ops_list else ''
            if e['stage'] == 3:
                cutoff = e['time'] - timedelta(minutes=60)
                earliest_for_this = e['time']
                for ev in events:
                    if ev['time'] >= e['time']:
                        break
                    if ev['time'] < cutoff:
                        continue
                    if ev['stage'] in (1, 2) and ev['time'] < earliest_for_this:
                        earliest_for_this = ev['time']
                predict_time_str = earliest_for_this.strftime('%Y-%m-%d %H:%M')
            else:
                predict_time_str = ''
            all_events_rows.append({
                'file': os.path.basename(fp),
                'date': e['time'].strftime('%Y-%m-%d'),
                'predict_time': predict_time_str,
                'time': e['time'].strftime('%H:%M'),
                'datetime': e['time'].strftime('%Y-%m-%d %H:%M'),
                'stage': e['stage'],
                'stage_name': {0:'정상화', 1:'1단계 조기경보', 2:'2단계 주의보', 3:'3단계 확정'}[e['stage']],
                'reason': e['reason'],
                'classification': classification,
                'matched_deadlock': matched_deadlock,
                'lead_minutes': lead_min,
                'ops_count_30min': op_cnt,
                'ops_event_types': op_types,
                'ops_top_message': op_top,
                'lead_min_vs_op': lead_vs_op,
                'prediction_type': pred_tag,
            })

        # 사건 단위 CSV (사건 시작 ~ 종료 구간에 걸친 운영 이벤트 집계)
        file_incidents = build_incidents(events)
        # CSV 10 계산용 — 사건 시작 시각 누적
        for _inc in file_incidents:
            all_incident_starts.append(_inc['start'])
        # 각 사건의 "최초 인지 시각" 계산: 3단계 이전 60분 내 최초 S1/S2
        def _find_earliest_signal(incident_start):
            earliest = incident_start
            cutoff = incident_start - timedelta(minutes=60)
            for ev in events:
                if ev['time'] >= incident_start:
                    break
                if ev['time'] < cutoff:
                    continue
                if ev['stage'] in (1, 2) and ev['time'] < earliest:
                    earliest = ev['time']
            return earliest

        for i in file_incidents:
            # 알고리즘 최초 인지 시각 (predict_time) — 비교 기준
            earliest_sig = _find_earliest_signal(i['start'])
            # 사건 기간 ±30분 윈도우
            win_ops = []
            if ops_list:
                lo = i['start'] - timedelta(minutes=30)
                hi = i['end'] + timedelta(minutes=30)
                for op in ops_list:
                    if op['dt'] < lo:
                        continue
                    if op['dt'] > hi:
                        break
                    if op['event_type'] == 'ALERT_BOT':
                        continue
                    win_ops.append(op)
            op_cnt = len(win_ops)
            op_types_ct = {}
            for op in win_ops:
                op_types_ct[op['event_type']] = op_types_ct.get(op['event_type'], 0) + 1
            op_types_str = ', '.join(f'{t}({c})' for t, c in sorted(op_types_ct.items(), key=lambda x: -x[1]))
            op_msgs = ' | '.join(f"{o['dt'].strftime('%H:%M')} {o['author']}: {o['summary']}" for o in win_ops[:3])

            # ⏱️ 시차 분석: predict_time (알고리즘 최초 인지) vs 운영자 (핵심 예측 지표)
            earliest_op = min(win_ops, key=lambda o: o['dt']) if win_ops else None
            if earliest_op:
                earliest_op_time = earliest_op['dt'].strftime('%Y-%m-%d %H:%M')
                # lead = 운영자 첫 메시지 - 알고리즘 최초 인지 시각 (earliest_sig)
                # 양수 = 알고리즘이 먼저 인지 (예측)
                # 음수 = 알고리즘이 나중 (라벨)
                lead_sec = (earliest_op['dt'] - earliest_sig).total_seconds()
                lead_min_vs_op = round(lead_sec / 60.0, 1)
                if lead_min_vs_op > 5:
                    pred_type = '🔮 예측'
                elif lead_min_vs_op < -5:
                    pred_type = '🏷️ 라벨'
                else:
                    pred_type = '⏱️ 동시'
            else:
                earliest_op_time = ''
                lead_min_vs_op = ''
                pred_type = '❓ 운영로그無'

            # 가설 검증 플래그
            has_deadlock_signal = any(o['event_type'] in ('DEADLOCK_SIGNAL', 'BRIDGE_ERROR', 'ERROR_OCCURRED') for o in win_ops)
            has_operator_action = any(o['event_type'].startswith('CAPA_CHANGE') or o['event_type'].startswith('PORT_') for o in win_ops)
            verdict = (
                '✅ 운영 이슈 일치' if has_deadlock_signal else
                '⚠️ 대응만 있음' if has_operator_action else
                '❓ 운영 로그 無' if ops_list else '-'
            )

            # 🔍 이상 판단 근거 (설명가능성 XAI)
            # 각 룰이 임계치를 얼마나 초과했는지 점수화 (초과 비율 %)
            ra_val = float(i['max_1min']) if i['max_1min'] else 0
            rb_val = int(i['max_rb_diff']) if i['max_rb_diff'] else 0
            rc_val = int(i['max_rev']) if i['max_rev'] else 0
            ra_score = round(100 * (ra_val / 9.0 - 1) * 100) / 100 if ra_val >= 9.0 else 0  # 9분 기준 초과 %
            rb_score = round(100 * (rb_val / 100.0 - 1) * 100) / 100 if rb_val >= 100 else 0  # 100 기준
            rc_score = round(100 * (rc_val / 2.0 - 1) * 100) / 100 if rc_val >= 2 else 0  # 2개 기준

            contrib_list = [
                ("R-A' 반송시간", ra_score, ra_val, f"{ra_val:.2f}분 (기준 9분)"),
                ("R-B FAB큐",    rb_score, rb_val, f"+{rb_val} (기준 +100)"),
                ("R-C' 리프터",  rc_score, rc_val, f"{rc_val}개 역증가 (기준 2개)"),
            ]
            # 임계 초과율 높은 순
            contrib_list.sort(key=lambda x: -x[1])
            primary_cause = contrib_list[0][0] if contrib_list[0][1] > 0 else '기준 미달'
            contrib_breakdown = ' | '.join(f"{name} {desc}" for name, _, _, desc in contrib_list)

            # 사람이 읽기 쉬운 설명 한 줄
            if contrib_list[0][1] > 50:
                impact = '매우 강함'
            elif contrib_list[0][1] > 20:
                impact = '강함'
            elif contrib_list[0][1] > 0:
                impact = '보통'
            else:
                impact = '약함'

            explanation_parts = []
            if ra_score > 0:
                explanation_parts.append(f"반송시간 {ra_val:.1f}분")
            if rb_score > 0:
                explanation_parts.append(f"FAB간 큐 +{rb_val}")
            if rc_score > 0:
                explanation_parts.append(f"리프터 역증가 {rc_val}개")
            if not explanation_parts:
                anomaly_explanation = '3단계 조건 일부만 부분 충족'
            else:
                anomaly_explanation = f"{primary_cause} 주도 ({impact}): " + ", ".join(explanation_parts)

            all_incidents_rows.append({
                'file': os.path.basename(fp),
                'date': i['start'].strftime('%Y-%m-%d'),
                'predict_time': earliest_sig.strftime('%H:%M'),
                'start_time': i['start'].strftime('%H:%M'),
                'end_time': i['end'].strftime('%H:%M'),
                'duration_min': i['duration_min'],
                'refire_count': i['refire_count'],
                'max_1min': round(i['max_1min'], 2),
                'max_m14_diff': i['max_rb_diff'],
                'max_reverse_lifters': i['max_rev'],
                'severity': i['severity'],
                'primary_cause': primary_cause,
                'contrib_breakdown': contrib_breakdown,
                'anomaly_explanation': anomaly_explanation,
                'ops_count_window': op_cnt,
                'ops_event_types': op_types_str,
                'ops_sample_messages': op_msgs,
                'earliest_op_time': earliest_op_time,
                'lead_min_vs_op': lead_min_vs_op,
                'prediction_type': pred_type,
                'verdict': verdict,
            })

            # 이상 판단 근거 (별도 XAI CSV)
            all_xai_rows.append({
                'file': os.path.basename(fp),
                'date': i['start'].strftime('%Y-%m-%d'),
                'start_time': i['start'].strftime('%H:%M'),
                'severity': i['severity'],
                '주요_원인': primary_cause,
                'R_A_반송시간_값': round(ra_val, 2),
                'R_A_기준': '9분',
                'R_A_초과율_pct': round(ra_score, 1),
                'R_B_FAB큐_증가': rb_val,
                'R_B_기준': '+100',
                'R_B_초과율_pct': round(rb_score, 1),
                'R_C_리프터_역증가': rc_val,
                'R_C_기준': '2개',
                'R_C_초과율_pct': round(rc_score, 1),
                '영향도': impact,
                '한줄_설명': anomaly_explanation,
                '기여도_순위': contrib_breakdown,
            })

            # 입력 STAR 컬럼별 기여도 (한 사건당 여러 행, 룰별)
            # 초과율 높은 순으로 rank 부여
            ranked = sorted(
                [('R-A\'', '반송시간 스파이크', f'{prefix}.QUE.TIME.AVGTOTALTIME1MIN', f'{ra_val:.2f}분', '9분', ra_score),
                 ('R-B', 'FAB 간 큐 누적', f'{prefix}.QUE.M14TOM16.MESCURRENTQCNT', f'+{rb_val}', '+100', rb_score),
                 ('R-C\'', '리프터 역증가', f'{prefix}.LFT.6ABL*.TOTAL_CURRENTQCNT', f'{rc_val}개 역증가', '2개', rc_score)],
                key=lambda x: -x[5]
            )
            rev_lids_str = ', '.join(sorted(i.get('rev_lids_union') or set()))
            for rank_no, (rule, rule_meaning, column_name, value_str, threshold_str, score) in enumerate(ranked, 1):
                contribution = (
                    'Primary (주요)' if rank_no == 1 and score > 0 else
                    'Secondary (보조)' if rank_no == 2 and score > 0 else
                    'Minor (경미)' if rank_no == 3 and score > 0 else
                    '-'
                )
                # R-C' 인 경우 구체 리프터 ID 를 value 에 포함
                extra_info = rev_lids_str if rule == "R-C'" and rev_lids_str else ''
                all_col_rows.append({
                    'file': os.path.basename(fp),
                    'date': i['start'].strftime('%Y-%m-%d'),
                    'start_time': i['start'].strftime('%H:%M'),
                    'severity': i['severity'],
                    'rule': rule,
                    'rule_meaning': rule_meaning,
                    'STAR_컬럼': column_name,
                    '관측값': value_str,
                    '임계값': threshold_str,
                    '초과율_pct': round(score, 1),
                    '기여도_순위': rank_no,
                    '기여도': contribution,
                    '세부_리프터': extra_info,
                })

    # 종합 요약 + CSV 저장
    if all_summary:
        print('\n' + '═' * 78)
        print('📊 전체 요약')
        print('═' * 78)
        print(f'{"파일":<45} {"날짜":<12} {"전환":>5} {"S3":>4} {"TP":>4} {"FN":>4} {"FP":>4}')
        print('-' * 78)
        for name, d, ev, s3, tp, fn, fp_ct in all_summary:
            print(f'{name[:45]:<45} {d:<12} {ev:>5} {s3:>4} {tp:>4} {fn:>4} {fp_ct:>4}')

        n_tp, n_fn, n_fp = len(total_tp), len(total_fn), len(total_fp)
        metrics = {}
        if n_tp + n_fn + n_fp > 0:
            print()
            print(f'  실제 데드락: {n_tp + n_fn}건')
            print(f'  3단계 적중 (TP): {n_tp}건')
            print(f'  3단계 놓침 (FN): {n_fn}건')
            print(f'  False positive (FP): {n_fp}건')
            metrics['total_deadlocks'] = n_tp + n_fn
            metrics['TP'] = n_tp
            metrics['FN'] = n_fn
            metrics['FP'] = n_fp
            if n_tp + n_fp > 0:
                prec = 100.0 * n_tp / (n_tp + n_fp)
                print(f'  Precision: {prec:.1f}%  ({n_tp}/{n_tp+n_fp})')
                metrics['precision_pct'] = f'{prec:.1f}'
            if n_tp + n_fn > 0:
                rec = 100.0 * n_tp / (n_tp + n_fn)
                print(f'  Recall:    {rec:.1f}%  ({n_tp}/{n_tp+n_fn})')
                metrics['recall_pct'] = f'{rec:.1f}'
            if total_tp:
                avg_lead = sum(lead for _, _, lead in total_tp) / len(total_tp)
                print(f'  평균 선행시간: {avg_lead:.1f}분')
                metrics['avg_lead_minutes'] = f'{avg_lead:.1f}'

        # === 운영 로그 대비 가설 검증 (ops log 제공 시) ===
        if ops_list and all_incidents_rows:
            n_inc = len(all_incidents_rows)
            n_match = sum(1 for r in all_incidents_rows if r.get('verdict', '').startswith('✅'))
            n_action = sum(1 for r in all_incidents_rows if r.get('verdict', '').startswith('⚠️'))
            n_noop = sum(1 for r in all_incidents_rows if r.get('verdict', '').startswith('❓'))
            high = [r for r in all_incidents_rows if r['severity'] in ('★★★', '★★')]
            n_high = len(high)
            n_high_match = sum(1 for r in high if r.get('verdict', '').startswith('✅'))
            print()
            print(f'🎯 가설 검증 (운영 로그 대비):')
            print(f'   전체 사건 {n_inc}개 중')
            print(f'     ✅ 운영 이슈 직접 일치 : {n_match}건 ({100.0*n_match/n_inc:.0f}%)')
            print(f'     ⚠️ 운영자 대응만 감지   : {n_action}건 ({100.0*n_action/n_inc:.0f}%)')
            print(f'     ❓ 운영 로그에 근거 無  : {n_noop}건 ({100.0*n_noop/n_inc:.0f}%)')
            if n_high > 0:
                print(f'   ★★ 이상 사건 {n_high}개 중 운영 이슈 일치: {n_high_match}건 ({100.0*n_high_match/n_high:.0f}%)')
            metrics['incidents_total'] = n_inc
            metrics['incidents_match_ops_issue'] = n_match
            metrics['incidents_match_operator_action'] = n_action
            metrics['incidents_no_ops_evidence'] = n_noop
            metrics['incidents_high_severity'] = n_high
            metrics['incidents_high_match_pct'] = f'{100.0*n_high_match/n_high:.1f}' if n_high else '0'
            metrics['hypothesis_precision_pct'] = f'{100.0*(n_match+n_action)/n_inc:.1f}' if n_inc else '0'

            # ⏱️ 진짜 예측 비율 (알고리즘이 운영자보다 먼저 울린 비율)
            pred_rows = [r for r in all_incidents_rows if r.get('prediction_type', '').startswith('🔮')]
            label_rows = [r for r in all_incidents_rows if r.get('prediction_type', '').startswith('🏷️')]
            concur_rows = [r for r in all_incidents_rows if r.get('prediction_type', '').startswith('⏱️')]
            n_pred = len(pred_rows)
            n_label = len(label_rows)
            n_concur = len(concur_rows)
            n_with_op = n_pred + n_label + n_concur
            # 시차 평균 (운영자 - 알고리즘, 양수 = 알고 먼저 = 예측)
            leads = [float(r['lead_min_vs_op']) for r in (pred_rows + label_rows + concur_rows) if r['lead_min_vs_op'] != '']
            avg_lead_vs_op = sum(leads) / len(leads) if leads else 0
            print()
            print(f'⏱️ 진짜 예측 vs 라벨 분석 (알고리즘 발동 시각 vs 운영자 첫 메시지):')
            print(f'   🔮 예측 (알고리즘이 5분+ 먼저) : {n_pred}건 ({100.0*n_pred/n_with_op:.0f}%)' if n_with_op else f'   🔮 예측: 0건')
            print(f'   ⏱️ 동시 (±5분 내)            : {n_concur}건')
            print(f'   🏷️ 라벨 (알고리즘이 5분+ 늦음) : {n_label}건 ({100.0*n_label/n_with_op:.0f}%)' if n_with_op else f'   🏷️ 라벨: 0건')
            print(f'   평균 시차 (운영자-알고): {avg_lead_vs_op:+.1f}분  (양수=알고 먼저, 음수=알고 늦음)')
            metrics['prediction_count'] = n_pred
            metrics['label_count'] = n_label
            metrics['concurrent_count'] = n_concur
            metrics['prediction_rate_pct'] = f'{100.0*n_pred/n_with_op:.1f}' if n_with_op else '0'
            metrics['avg_lead_min_vs_operator'] = f'{avg_lead_vs_op:+.1f}'

        # === CSV 6종 출력 (각 용도별 분리) ===
        ts_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_events  = f'검증결과_01_전체이벤트_{ts_suffix}.csv'
        out_s3_all  = f'검증결과_02_S3전체_{ts_suffix}.csv'
        out_s3_new  = f'검증결과_03_S3신규만_{ts_suffix}.csv'
        out_s3_ref  = f'검증결과_04_S3재발동만_{ts_suffix}.csv'
        out_incidents = f'검증결과_05_사건단위_{ts_suffix}.csv'
        out_summary = f'검증결과_06_파일별요약_{ts_suffix}.csv'
        out_metrics = f'검증결과_07_종합지표_{ts_suffix}.csv'
        out_xai = f'검증결과_08_이상판단근거_{ts_suffix}.csv'
        out_columns = f'검증결과_09_입력컬럼기여도_{ts_suffix}.csv'
        out_catalog = f'검증결과_10_컬럼_데드락영향_{ts_suffix}.csv'

        def write_csv(path, rows, header=None):
            if not rows:
                return
            with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                if header:
                    w = csv.writer(f)
                    w.writerows([header] + rows)
                else:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        # 1. 전체 이벤트 (모든 단계 전환)
        write_csv(out_events, all_events_rows)

        # 2/3/4. S3 관련 분리
        s3_all  = [r for r in all_events_rows if r['stage'] == 3]
        s3_new  = [r for r in s3_all if 'AND 만족' in r['reason']]
        s3_ref  = [r for r in s3_all if '재발동' in r['reason']]
        write_csv(out_s3_all, s3_all)
        write_csv(out_s3_new, s3_new)
        write_csv(out_s3_ref, s3_ref)

        # 5. 사건 단위 CSV (★ 등급 내림차순)
        if all_incidents_rows:
            sev_rank = {'★★★': 3, '★★': 2, '★': 1, '-': 0}
            all_incidents_rows.sort(key=lambda r: (-sev_rank.get(r['severity'], 0), r['date'], r['start_time']))
            write_csv(out_incidents, all_incidents_rows)

        # 6. 파일별 요약
        write_csv(out_summary, [list(row) for row in all_summary],
                  header=['file', 'date', 'transitions', 'S3_count', 'TP', 'FN', 'FP'])

        # 7. 종합 지표
        if metrics:
            write_csv(out_metrics, [[k, v] for k, v in metrics.items()],
                      header=['metric', 'value'])

        # 8. 이상 판단 근거 (XAI — 고객 설명용)
        if all_xai_rows:
            # severity 높은 순, 같으면 날짜순
            all_xai_rows.sort(key=lambda r: (-{'★★★':3,'★★':2,'★':1,'-':0}.get(r['severity'], 0), r['date'], r['start_time']))
            write_csv(out_xai, all_xai_rows)

        # 9. 입력 STAR 컬럼별 기여도 (Primary/Secondary/Minor)
        if all_col_rows:
            write_csv(out_columns, all_col_rows)

        # 10. 각 STAR 컬럼의 데드락 영향 카탈로그 (전체 input 변수 + 영향도 %)
        from collections import Counter

        primary_ct = Counter()
        any_trig_ct = Counter()
        lifter_rev_ct = Counter()  # 개별 리프터 역증가 발생 건수
        for r in all_col_rows:
            col = r['STAR_컬럼']
            if r['기여도'].startswith('Primary'):
                primary_ct[col] += 1
            if r['초과율_pct'] > 0:
                any_trig_ct[col] += 1
            # 세부 리프터 카운트
            sep_lids = r.get('세부_리프터', '')
            if sep_lids:
                for lid in sep_lids.split(','):
                    lid = lid.strip()
                    if lid:
                        lifter_rev_ct[lid] += 1

        # 분모 (전체 사건 수)
        total_incidents = len(all_incidents_rows) if all_incidents_rows else 1

        # prefix 감지
        prefix_hint = 'M16HUB'
        for r in all_col_rows:
            col = r['STAR_컬럼']
            if '.QUE.' in col or '.LFT.' in col:
                prefix_hint = col.split('.')[0]
                break

        def pct(n):
            return round(100.0 * n / total_incidents, 1) if total_incidents else 0.0

        def row(star_col, meaning, path, rule, direction, normal, threshold, cycle, unit,
                primary_pct_val, total_pct_val, note):
            return {
                'STAR_컬럼': star_col,
                '컬럼_의미': meaning,
                '데드락_영향_경로': path,
                '사용_룰': rule,
                '영향_방향': direction,
                '정상_범위': normal,
                '경보_임계': threshold,
                '측정_주기': cycle,
                '단위': unit,
                '영향도_pct': col_impact.get(star_col, 0.0),  # S3 시점 ±10분 평균 vs 전체 평균 편차 %
                '이번_데이터_Primary_기여_pct': primary_pct_val,
                '이번_데이터_어떤기여라도_pct': total_pct_val,
                '이번_데이터_Primary_횟수': primary_ct.get(star_col, 0) if primary_pct_val is not None else 0,
                '비고': note,
            }

        # 직접 사용 3종 (R-A'/R-B/R-C')
        col_ra = f'{prefix_hint}.QUE.TIME.AVGTOTALTIME1MIN'
        col_rb = f'{prefix_hint}.QUE.M14TOM16.MESCURRENTQCNT'
        col_rc = f'{prefix_hint}.LFT.6ABL*.TOTAL_CURRENTQCNT'

        # ⭐ 통계적 영향도 계산 — S3 사건 시작 ±10분 윈도우 vs 전체 평균
        # 각 컬럼이 사건 시점에 얼마나 정상 대비 변했는지 %
        incident_window_min = 10
        incident_dt_set = all_incident_starts

        def _column_values(key, sub=None):
            """timeline 에서 특정 컬럼의 값 리스트 추출"""
            vals = []
            for t, star in all_timelines:
                if sub is not None:
                    v = (star.get(key) or {}).get(sub)
                else:
                    v = star.get(key)
                if v is None:
                    continue
                try:
                    vals.append((t, float(v)))
                except (TypeError, ValueError):
                    continue
            return vals

        def _impact_pct(vals):
            """사건 시점 ±10분 평균 vs 전체 평균의 편차 %"""
            if not vals or not incident_dt_set:
                return 0.0
            all_mean = sum(v for _, v in vals) / len(vals)
            if all_mean == 0:
                return 0.0
            in_event = []
            for t, v in vals:
                for inc_t in incident_dt_set:
                    diff = abs((t - inc_t).total_seconds())
                    if diff <= incident_window_min * 60:
                        in_event.append(v)
                        break
            if not in_event:
                return 0.0
            event_mean = sum(in_event) / len(in_event)
            return round(abs(event_mean - all_mean) / abs(all_mean) * 100, 1)

        # 각 컬럼 영향도 사전 계산
        col_impact = {}
        col_impact[col_ra] = _impact_pct(_column_values('avgtotal1min'))
        col_impact[col_rb] = _impact_pct(_column_values('m14_to_m16'))
        # R-C' 집계: 10개 리프터 합의 영향도
        lft_sum_vals = []
        for t, star in all_timelines:
            lft = star.get('lft_list') or {}
            if lft:
                lft_sum_vals.append((t, float(sum(v for v in lft.values() if v is not None))))
        col_impact[col_rc] = _impact_pct(lft_sum_vals)
        # 리프터 개별
        for lid in LIFTER_IDS:
            full_col = f'{prefix_hint}.LFT.{lid}.TOTAL_CURRENTQCNT'
            col_impact[full_col] = _impact_pct(_column_values('lft_list', sub=lid))
        # 간접 컬럼들
        indirect_keys = {
            f'{prefix_hint}.QUE.ALL.CURRENTQCNT': 'queue_total',
            f'{prefix_hint}.QUE.ALL.CURRENTQCOMPLETED': 'queue_completed',
            f'{prefix_hint}.QUE.OHT.CURRENTOHTQCNT': 'queue_oht',
            f'{prefix_hint}.QUE.OHT.OHTUTIL': 'oht_util',
            f'{prefix_hint}.QUE.LOAD.AVGLOADTIME': 'avg_load_time',
            f'{prefix_hint}.QUE.ALL.TRANSPORT4MINOVERCNT': 'transport_4over',
            f'{prefix_hint}.OHT.STATECNT.DRIVING': 'driving',
            f'{prefix_hint}.OHT.STATECNT.OBSANDBZSTOP': 'obs_bz_stop',
            f'{prefix_hint}.OHT.STATECNT.CONGESTED': 'congested',
            f'{prefix_hint}.OHT.STATECNT.PAUSE': 'pause',
            f'{prefix_hint}.OHT.STATECNT.TIMEOUT': 'timeout',
            f'{prefix_hint}.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT': 'mlud_q',
            f'{prefix_hint}.QUE.ALL.FABTRANSJOBCNT': 'fab_trans_job',
        }
        for full_col, key in indirect_keys.items():
            col_impact[full_col] = _impact_pct(_column_values(key))

        catalog = []

        # === 직접 룰 사용 컬럼 (Primary 기여도 계산 가능) ===
        catalog.append(row(
            col_ra, '최근 1분 평균 반송 시간',
            '반송 시간이 길어짐 → 시스템 부하 증가 → 데드락 위험',
            "R-A' (반송시간 스파이크)", '커질수록 위험', '4~7분', '≥9분 1회+ 또는 ≥6분 3회연속', '1분', '분',
            pct(primary_ct.get(col_ra, 0)), pct(any_trig_ct.get(col_ra, 0)),
            '04-21 13:50 기록 9.27분'
        ))
        catalog.append(row(
            col_rb, 'M14 → M16 반송 대기 큐 갯수',
            'FAB 간 처리 격차 → 브릿지 누적 → 데드락 직전',
            'R-B (FAB 간 큐 누적)', '커질수록 위험 (추세)', '500 대', '30분간 +100 또는 10분간 +30', '1분', '개',
            pct(primary_ct.get(col_rb, 0)), pct(any_trig_ct.get(col_rb, 0)),
            '04-21 13:50 +112 (30분간)'
        ))
        catalog.append(row(
            col_rc, '리프터 10대 개별 큐 (집계)',
            '전체 감소 + 일부 역증가 → 경로 병목 시그너처',
            "R-C' (리프터 역증가)", '역증가 수 많을수록 위험', '합 100~200 (동기)', '합 감소 + 역증가 2개+', '1분', '개',
            pct(primary_ct.get(col_rc, 0)), pct(any_trig_ct.get(col_rc, 0)),
            'R-C\' 의 집계 지표. 아래 10개 리프터 세부 참조'
        ))

        # === 리프터 10개 개별 ===
        for lid in LIFTER_IDS:
            full_col = f'{prefix_hint}.LFT.{lid}.TOTAL_CURRENTQCNT'
            cnt = lifter_rev_ct.get(lid, 0)
            catalog.append(row(
                full_col, f'리프터 {lid} 개별 큐',
                f'데드락 시 특정 리프터에서만 작업 쌓임 (경로 병목 구체 포인트)',
                "R-C' (세부)", '역증가 발생 시 위험', '10~30', '역증가 (합 감소 중)', '1분', '개',
                None, pct(cnt),
                f'3개월 간 역증가 발생 {cnt}회'
            ))
            catalog[-1]['이번_데이터_Primary_기여_pct'] = '-'  # 리프터 개별은 집계로만
            catalog[-1]['이번_데이터_Primary_횟수'] = cnt

        # === 참고 / 간접 입력 변수 (룰 미사용, 0%) ===
        indirect = [
            ('QUE.ALL.CURRENTQCNT', '전체 반송 큐', '부하 전반 지표 (데드락 상관 있음)', '-', '커지면 부하↑', '389~574', '-', '1분', '개', '보조 모니터링'),
            ('QUE.ALL.CURRENTQCOMPLETED', '완료 반송 수', '처리율 지표', '-', '정상 운영 판별', '353~496', '-', '1분', '개', '처리량 모니터링'),
            ('QUE.OHT.CURRENTOHTQCNT', 'OHT 반송 Q', 'OHT 할당 큐', '-', '커지면 할당 많음', '161~298', '-', '1분', '개', '보조'),
            ('QUE.OHT.OHTUTIL', 'OHT 큐 할당률 (%)', '실제 주행률 아님 (주의)', '-', '해석 주의', '53~95%', '-', '1분', '%', '이름 오해 주의'),
            ('QUE.LOAD.AVGLOADTIME', '평균 적재 시간', '로딩 지연 지표', '-', '커지면 적재 지연', '-', '-', '1분', '분', '수집 불가 (DB 미존재)'),
            ('QUE.ALL.TRANSPORT4MINOVERCNT', '4분 초과 반송 수', '지연 발생 건수', '-', '커지면 지연 많음', '0', '-', '1분', '개', '집계 버그 (전부 0)'),
            ('OHT.STATECNT.DRIVING', '주행 중 OHT 수', '실제 주행 차량', '-', '증가 시 정상 운영', '150~250', '-', '1분', '대', 'UDP 복원 가능'),
            ('OHT.STATECNT.OBSANDBZSTOP', 'OBS/BZ STOP 수', '기존 OBS 배지 입력', '별도 OBS 5단계 배지', '커질수록 위험', '30~50', 'OBS_WARNING_THRESHOLD', '1분', '대', '별도 경보 시스템'),
            ('OHT.STATECNT.CONGESTED', '혼잡 상태 OHT 수', '혼잡도 지표', '-', '커지면 혼잡', '-', '-', '1분', '대', '보조'),
            ('OHT.STATECNT.PAUSE', 'Pause 상태 OHT 수', '일시정지 차량', '-', '커지면 이상', '-', '-', '1분', '대', '보조'),
            ('OHT.STATECNT.TIMEOUT', 'Timeout 상태 OHT 수', '타임아웃 차량', '-', '커지면 이상', '-', '-', '1분', '대', '보조'),
            ('QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT', 'MLUD (Manual 출고 포트) 큐', '데드락 시 "MLUD 쏠림" 현장 증언과 일치', '(참고 지표)', '대응 조치 대상', '40 대', '-', '1분', '개', '대응 컬럼 — 운영자가 50% 로 낮추는 대상'),
            ('QUE.ALL.FABTRANSJOBCNT', 'FAB 전체 작업량', '전체 부하 상위 지표', '-', '커지면 부하↑', '1085~1282', '-', '1분', '개', '모니터링'),
        ]

        for short, meaning, path, rule, direction, normal, threshold, cycle, unit, note in indirect:
            full_col = f'{prefix_hint}.{short}'
            catalog.append(row(
                full_col, meaning, path, rule, direction, normal, threshold, cycle, unit,
                pct(primary_ct.get(full_col, 0)),
                pct(any_trig_ct.get(full_col, 0)),
                note
            ))

        # 영향도 % 높은 순으로 정렬
        catalog.sort(key=lambda r: -r.get('영향도_pct', 0))
        # 순위 부여
        for idx, r in enumerate(catalog, 1):
            r['순위'] = idx

        # 컬럼 순서 조정
        catalog_ordered = [{**{'순위': r['순위']}, **{k: v for k, v in r.items() if k != '순위'}} for r in catalog]

        write_csv(out_catalog, catalog_ordered)

        print('\n💾 CSV 저장 (용도별 분리):')
        if all_events_rows: print(f'   01 · {out_events}  ({len(all_events_rows)} 전체 이벤트)')
        if s3_all:  print(f'   02 · {out_s3_all}  ({len(s3_all)} S3 전체)')
        if s3_new:  print(f'   03 · {out_s3_new}  ({len(s3_new)} S3 신규만)')
        if s3_ref:  print(f'   04 · {out_s3_ref}  ({len(s3_ref)} S3 재발동만)')
        if all_incidents_rows:
            high = sum(1 for r in all_incidents_rows if r["severity"] in ("★★★", "★★"))
            print(f'   05 · {out_incidents}  ({len(all_incidents_rows)} 사건, ★★ 이상 {high}건)')
        print(f'   06 · {out_summary}  ({len(all_summary)} 파일)')
        if metrics:
            print(f'   07 · {out_metrics}  ({len(metrics)} 지표)')
        if all_xai_rows:
            print(f'   08 · {out_xai}  ({len(all_xai_rows)} 이상 판단 근거)')
        if all_col_rows:
            print(f'   09 · {out_columns}  ({len(all_col_rows)} 입력 STAR 컬럼 기여도)')
        print(f'   10 · {out_catalog}  ({len(catalog_ordered)} 컬럼 전체 영향도 %) ⭐ 신규')
        print(f'\n👉 이 CSV들을 전달해주세요.')


if __name__ == '__main__':
    main()

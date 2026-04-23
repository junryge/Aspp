#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 데드락 경보 룰 검증 스크립트 (단독 실행)

사용법:
    python3 3단계_경보_검증_스크립트.py <STAR_CSV_경로> [데드락_발생시각_HH:MM ...]

예:
    python3 3단계_경보_검증_스크립트.py STAR_OHT_컬럼수집_DATA_20260421.csv 14:00
    python3 3단계_경보_검증_스크립트.py *.csv                      # 여러 파일 일괄

출력:
    - 단계별 발동 시점 타임라인
    - 데드락 발생 시각 대비 선행시간
    - 단계별 발동 횟수 통계
    - false positive / true positive 분류 (데드락 시각 주면)

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


def parse_time(s):
    """CRT_TM 파싱 — 다양한 포맷 대응"""
    if not s:
        return None
    s = s.strip().rstrip('Z').split('.')[0]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y%m%d%H%M%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
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

        # R-A'
        recent_t1 = t1_hist[-10:]
        ra_count = sum(1 for v in recent_t1 if v >= 9.0)
        ra_value = recent_t1[-1] if recent_t1 else None
        ra_trig = ra_count >= 1

        # R-B
        rb_diff = 0
        rb_trig = False
        if len(m14_hist) >= 31:
            rb_diff = m14_hist[-1] - m14_hist[-31]
            rb_trig = rb_diff >= 100

        # R-C'
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

        s1 = ra_count >= 2
        s2 = rb_trig
        s3 = ra_trig and rb_trig and rc_trig
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
            })

    return events


def report(filepath, events, deadlock_times=None):
    """결과 출력"""
    deadlock_times = deadlock_times or []
    print()
    print('═' * 78)
    print(f'📁 {os.path.basename(filepath)}')
    print('═' * 78)

    if not events:
        print('  ✅ 어느 단계도 발동 안 함 (완전 정상 / 또는 데이터 부족)')
        return

    icons = {0: '✅', 1: '🔔', 2: '⚠️', 3: '🚨'}
    names = {0: '정상화', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}

    print(f'\n  📊 단계 전환 타임라인 ({len(events)}건)')
    print(f'  {"-"*74}')
    for e in events:
        print(f"  {e['time'].strftime('%H:%M')}  {icons[e['stage']]} {names[e['stage']]:<18} {e['reason']}")

    # 통계
    counts = defaultdict(int)
    for e in events:
        counts[e['stage']] += 1
    print(f'\n  📈 단계별 발동 횟수')
    for stage in (1, 2, 3):
        print(f'    {names[stage]:<18}: {counts[stage]}회')

    # 데드락 ground truth 매칭
    if deadlock_times:
        print(f'\n  🎯 데드락 발생 시각과 매칭 ({len(deadlock_times)}건)')
        s3_events = [e for e in events if e['stage'] == 3]

        for dt_str in deadlock_times:
            try:
                base = events[0]['time'].date()
                hh, mm = map(int, dt_str.split(':'))
                dt = datetime.combine(base, datetime.min.time()).replace(hour=hh, minute=mm)
            except ValueError:
                print(f'    ❌ 잘못된 시각 포맷: {dt_str}')
                continue

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
                print(f'    {mark} 데드락 {dt_str} ← 3단계 {closest["time"].strftime("%H:%M")} '
                      f'(선행 {lead_min:.0f}분)')
            else:
                print(f'    ❌ 데드락 {dt_str}: 직전 1시간 내 3단계 발동 없음 (FN)')

        # False positive 추정 (데드락 시각과 30분 이상 떨어진 3단계)
        fp = []
        for s3 in s3_events:
            matched = False
            for dt_str in deadlock_times:
                try:
                    base = events[0]['time'].date()
                    hh, mm = map(int, dt_str.split(':'))
                    dt = datetime.combine(base, datetime.min.time()).replace(hour=hh, minute=mm)
                    diff = (dt - s3['time']).total_seconds()
                    if -600 <= diff <= 1800:  # 데드락 -10분 ~ +30분
                        matched = True
                        break
                except ValueError:
                    continue
            if not matched:
                fp.append(s3)

        if fp:
            print(f'\n  ⚠️  False positive 후보 ({len(fp)}건, 데드락 시각과 무관한 3단계)')
            for e in fp:
                print(f'    {e["time"].strftime("%H:%M")} - {e["reason"]}')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pattern = sys.argv[1]
    deadlock_times = sys.argv[2:]

    files = sorted(glob.glob(pattern)) if any(c in pattern for c in '*?[]') else [pattern]
    if not files:
        print(f'❌ 파일 없음: {pattern}')
        sys.exit(1)

    print(f'\n🔍 검증 시작 — 파일 {len(files)}개')
    if deadlock_times:
        print(f'   데드락 발생 시각: {deadlock_times}')

    all_summary = []
    for fp in files:
        if not os.path.exists(fp):
            print(f'⚠️  스킵 (없음): {fp}')
            continue

        timeline, prefix = load_star(fp)
        if not timeline:
            print(f'⚠️  스킵 (데이터 없음): {fp}')
            continue

        print(f'\n📥 {os.path.basename(fp)} — prefix={prefix}, {len(timeline)}행 로드')

        events = evaluate_timeline(timeline)
        report(fp, events, deadlock_times if len(files) == 1 else None)

        s3_count = sum(1 for e in events if e['stage'] == 3)
        all_summary.append((os.path.basename(fp), len(events), s3_count))

    if len(files) > 1:
        print('\n' + '═' * 78)
        print('📊 전체 요약')
        print('═' * 78)
        print(f'{"파일":<55} {"단계전환":>8} {"3단계":>6}')
        print('-' * 78)
        for name, ev, s3 in all_summary:
            print(f'{name:<55} {ev:>8} {s3:>6}')


if __name__ == '__main__':
    main()

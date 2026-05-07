#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 데드락 룰베이스 사건 단위 추출 스크립트

목적:
  운영 환경 시뮬레이션. STAR CSV 한 줄씩 처리하면서 과거 90분 데이터만 보고
  3단계 데드락 위험 사건을 사건 단위로 추출해 CSV 출력.

기존 `3단계_경보_검증_스크립트.py` 와 차이:
  - 검증 스크립트: 전체 CSV 일괄 분석 (10개 CSV 출력, 분석/검증용)
  - 본 스크립트: 슬라이딩 윈도우 (운영 시뮬용)

룰 (3차 검증된 것 — 변경 없음):
  R-A'         : 1MIN ≥ 9분이 10분창 1회+
  ra_sustained : 1MIN ≥ 6분이 5분창 3회+
  R-B          : M14→M16 +100/30분
  rb_fast      : M14→M16 +30/10분
  R-C'         : 리프터 합 감소 + 역증가 2개+

  S1 = R-A' 2회+ OR ra_sustained
  S2 = R-B OR rb_fast
  S3 = R-A' AND R-B AND R-C'  (불변 — 04-21 검증 보호)

출력 정책 (1분 단위 CSV 입력 가정):
  · 발동이벤트 CSV: 매 분 1행 기록
       - 이벤트 없음 → "이벤트 없음" 기입
       - 1·2단계 → 단계 + 사유 기입
       - 3단계   → 단계 + 사유 기입 (그리고 사건단위 CSV에도 누적)
  · 사건단위 CSV: 진짜 위험한 사건(S3)만 사건 단위로 기록

사용법:
    # 일괄 처리 (기존 CSV 전체 분석)
    python3 3단계_룰베이스_사건단위.py STAR.csv
    python3 3단계_룰베이스_사건단위.py STAR.csv -o ./out

    # 실시간 감시 모드 (1분마다 새 행 추가되는 CSV 폴링)
    python3 3단계_룰베이스_사건단위.py STAR.csv --watch
    python3 3단계_룰베이스_사건단위.py STAR.csv --watch --interval 60 -o ./out

출력 (날짜별 파일 — 같은 날짜면 기존 파일에 append, 없으면 신규 생성):
    <YYYYMMDD>_발동이벤트.csv   (매 분 1행, 이벤트 없는 분도 "이벤트없음"으로 기입)
    <YYYYMMDD>_사건단위.csv     (S3 — 진짜 위험한 사건만)
"""

import csv
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta


# ====== 상수 ======
WINDOW_MIN = 90  # 슬라이딩 윈도우 길이 (분)

LIFTER_IDS = [
    '6ABL6011', '6ABL6012', '6ABL6021', '6ABL6022',
    '6ABL6031', '6ABL6032', '6ABL0111', '6ABL0112',
    '6ABL0121', '6ABL0122',
]

# 룰 임계치 (검증된 값)
TH_RA_VALUE = 9.0     # R-A' 절대 임계 (분)
TH_RA_SUSTAINED_VALUE = 6.0
TH_RA_SUSTAINED_COUNT = 3   # 5분창 안 3회
TH_RB_DIFF_30 = 100   # R-B 30분 +100
TH_RB_DIFF_10 = 30    # R-B FAST 10분 +30
TH_RC_REVERSE = 2     # R-C' 역증가 2개+

# 사건 종료 판단: S3 신규/재발동 사이 간격
INCIDENT_END_GAP_MIN = 10
# predict_time 추적 윈도우: S3 직전 60분 내 최초 S1/S2 시각
PREDICT_LOOKBACK_MIN = 60


# ====== 유틸 ======
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
    if not s:
        return None
    s = s.strip().strip('"').strip("'")
    if 'T' in s and '+' in s.split('T')[1]:
        s = s.split('+')[0]
    if s.endswith('Z'):
        s = s[:-1]
    if '.' in s:
        s = s.split('.')[0]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y%m%d%H%M%S', '%Y%m%d %H%M%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def detect_prefix(fieldnames):
    anchor = '.QUE.ALL.CURRENTQCNT'
    for col in fieldnames or []:
        if col and col.endswith(anchor):
            return col[:-len(anchor)]
    return None


# ====== STAR 로드 (한 줄씩 yield) ======
def iter_star_rows(filepath):
    """STAR CSV → (datetime, star_dict) 제너레이터"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prefix = detect_prefix(reader.fieldnames)
        if not prefix:
            print(f"❌ {filepath}: prefix 감지 실패")
            return

        def C(suffix):
            return f"{prefix}{suffix}"

        for row in reader:
            t = parse_time(row.get('CRT_TM', ''))
            if not t:
                continue
            star = {
                'avgtotal1min': safe_float(row.get(C('.QUE.TIME.AVGTOTALTIME1MIN'))),
                'm14_to_m16':   safe_int(row.get(C('.QUE.M14TOM16.MESCURRENTQCNT'))),
                'lft_list': {
                    lid: safe_int(row.get(C(f'.LFT.{lid}.TOTAL_CURRENTQCNT')))
                    for lid in LIFTER_IDS
                },
            }
            star['lft_list'] = {k: v for k, v in star['lft_list'].items() if v is not None}
            yield t, star, prefix


# ====== 룰 평가 (90분 윈도우 데이터 기반) ======
def evaluate_rules(t1_window, m14_window, lft_window):
    """
    슬라이딩 윈도우 deque 들로부터 S1/S2/S3 평가.

    반환: (s1, s2, s3, ctx)
      ctx = {
        'ra_count', 'ra_value', 'ra_sustained',
        'rb_diff', 'rb_fast', 'rb_diff_10',
        'rc_trend', 'rev_count', 'rev_lids',
      }
    """
    t1_list = list(t1_window)
    m14_list = list(m14_window)
    lft_list = list(lft_window)

    # ── R-A' (절대) ── 최근 10개 (10분) 중 9분 이상 회수
    recent_t1 = t1_list[-10:]
    ra_count = sum(1 for v in recent_t1 if v is not None and v >= TH_RA_VALUE)
    ra_value = recent_t1[-1] if recent_t1 else None
    ra_trig = ra_count >= 1

    # ── R-A' SUSTAINED ── 최근 5개 중 6분 이상 3회+
    ra_sustained = False
    if len(recent_t1) >= 5:
        last5 = [v for v in recent_t1[-5:] if v is not None]
        if len(last5) >= 3:
            ra_sustained = sum(1 for v in last5 if v >= TH_RA_SUSTAINED_VALUE) >= TH_RA_SUSTAINED_COUNT

    # ── R-B ── 30분 전 대비 +100 (m14_window 길이 31 이상 필요)
    rb_diff = 0
    rb_trig = False
    if len(m14_list) >= 31 and m14_list[-1] is not None and m14_list[-31] is not None:
        rb_diff = m14_list[-1] - m14_list[-31]
        rb_trig = rb_diff >= TH_RB_DIFF_30

    # ── R-B FAST ── 10분 전 대비 +30
    rb_fast = False
    rb_diff_10 = 0
    if len(m14_list) >= 11 and m14_list[-1] is not None and m14_list[-11] is not None:
        rb_diff_10 = m14_list[-1] - m14_list[-11]
        rb_fast = rb_diff_10 >= TH_RB_DIFF_10

    # ── R-C' ── 20분 전 대비 합 감소 + 역증가 2개+
    rc_trend = 0
    rev_count = 0
    rev_lids = []
    rc_trig = False
    if len(lft_list) >= 21 and lft_list[-1] and lft_list[-21]:
        now_l = lft_list[-1]
        prev_l = lft_list[-21]
        rc_trend = sum(now_l.values()) - sum(prev_l.values())
        for lid in now_l:
            if now_l[lid] > prev_l.get(lid, 0):
                rev_lids.append(lid)
                rev_count += 1
        rc_trig = rc_trend < 0 and rev_count >= TH_RC_REVERSE

    # ── Stage ──
    s1 = (ra_count >= 2) or ra_sustained
    s2 = rb_trig or rb_fast
    # S3: R-A' AND (R-B 30분 OR R-B FAST 10분) AND R-C'
    # rb_fast 도 인정 — 룰 정의서의 R-B 계열은 (rb_trig OR rb_fast)이고 S2도 OR 처리.
    # S3만 rb_trig 단독이라 5/7 같은 빠른 폭주형이 누락되던 문제 해결.
    s3 = ra_trig and (rb_trig or rb_fast) and rc_trig

    ctx = {
        'ra_count': ra_count,
        'ra_value': ra_value,
        'ra_sustained': ra_sustained,
        'ra_trig': ra_trig,
        'rb_diff': rb_diff,
        'rb_diff_10': rb_diff_10,
        'rb_fast': rb_fast,
        'rb_trig': rb_trig,
        'rc_trend': rc_trend,
        'rc_trig': rc_trig,
        'rev_count': rev_count,
        'rev_lids': rev_lids,
    }
    return s1, s2, s3, ctx


# ====== 사건 추적 FSM ======
class IncidentTracker:
    """
    상태:
      - 'IDLE'        : 활성 사건 없음
      - 'IN_INCIDENT' : 사건 진행 중

    전이:
      IDLE        + s3 신규     → IN_INCIDENT
      IN_INCIDENT + s3 (10분+)  → 같은 사건 재발동
      IN_INCIDENT + 10분 무 s3  → IDLE (사건 종료)
    """
    def __init__(self):
        self.state = 'IDLE'
        self.current = None
        self.incidents = []
        self.early_signals = deque(maxlen=PREDICT_LOOKBACK_MIN)
        # 발동 이벤트 타임라인 (단계 전환 시점 기록)
        self.events = []          # [{time, stage, prev_stage, reason}]
        self.last_stage = 0

    def _record_event(self, t, stage, ctx):
        """매 분 호출. 단계 무관하게 1행씩 기록 — 이벤트 없으면 '이벤트 없음'."""
        is_transition = stage != self.last_stage
        if stage == 0:
            reason = "이벤트 없음"
        elif stage == 1:
            if ctx.get('ra_count', 0) >= 2:
                reason = f"1MIN ≥9분이 {ctx['ra_count']}회"
            elif ctx.get('ra_sustained'):
                reason = "1MIN ≥6분 지속"
            else:
                reason = "1단계 발동"
        elif stage == 2:
            if ctx.get('rb_diff', 0) >= 100:
                reason = f"M14→M16 +{ctx['rb_diff']} (30분간)"
            elif ctx.get('rb_fast'):
                reason = f"M14→M16 +{ctx.get('rb_diff_10', 0)} (10분간 fast)"
            else:
                reason = "2단계 발동"
        elif stage == 3:
            reason = (f"AND 만족 (1MIN {ctx.get('ra_value') or 0:.2f}, "
                      f"M14→M16 +{ctx.get('rb_diff', 0)}, "
                      f"역증가 {ctx.get('rev_count', 0)}개)")
        else:
            reason = "이벤트 없음"
        self.events.append({
            'time': t,
            'stage': stage,
            'prev_stage': self.last_stage,
            'reason': reason,
            'is_transition': is_transition,
            'ra_value': ctx.get('ra_value'),
            'ra_count': ctx.get('ra_count', 0),
            'ra_sustained': bool(ctx.get('ra_sustained')),
            'rb_diff': ctx.get('rb_diff', 0),
            'rb_diff_10': ctx.get('rb_diff_10', 0),
            'rb_fast': bool(ctx.get('rb_fast')),
            'rb_trig': bool(ctx.get('rb_trig')),
            'rc_trend': ctx.get('rc_trend', 0),
            'rev_count': ctx.get('rev_count', 0),
            'rev_lids': list(ctx.get('rev_lids') or []),
        })
        self.last_stage = stage

    def update(self, t, s1, s2, s3, ctx):
        if s1 or s2:
            self.early_signals.append(t)

        # 단계 변환 기록 (S1/S2/S3/정상화 전환마다 events 에 추가)
        cur_stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))
        self._record_event(t, cur_stage, ctx)

        if self.state == 'IDLE':
            if s3:
                self._start_new(t, ctx)
        else:  # IN_INCIDENT
            if s3:
                last_s3 = self.current['last_s3_time']
                if (t - last_s3).total_seconds() / 60.0 >= INCIDENT_END_GAP_MIN:
                    self.current['refire_count'] += 1
                self._update_current(t, ctx)
            else:
                last_s3 = self.current['last_s3_time']
                if (t - last_s3).total_seconds() / 60.0 >= INCIDENT_END_GAP_MIN:
                    self._end_current(t)

    def _start_new(self, t, ctx):
        cutoff = t - timedelta(minutes=PREDICT_LOOKBACK_MIN)
        early = [x for x in self.early_signals if x >= cutoff]
        predict_time = min(early) if early else t

        self.current = {
            'predict_time': predict_time,
            'start_time': t,
            'last_s3_time': t,
            'end_time': t,
            'refire_count': 0,
            'max_1min': ctx['ra_value'] or 0,
            'max_rb_diff': ctx['rb_diff'] or 0,
            'max_rev': ctx['rev_count'] or 0,
            'rev_lids_union': set(ctx.get('rev_lids') or []),
        }
        self.state = 'IN_INCIDENT'

    def _update_current(self, t, ctx):
        c = self.current
        c['last_s3_time'] = t
        c['end_time'] = t
        if ctx['ra_value'] and ctx['ra_value'] > c['max_1min']:
            c['max_1min'] = ctx['ra_value']
        if ctx['rb_diff'] and ctx['rb_diff'] > c['max_rb_diff']:
            c['max_rb_diff'] = ctx['rb_diff']
        if ctx['rev_count'] and ctx['rev_count'] > c['max_rev']:
            c['max_rev'] = ctx['rev_count']
        c['rev_lids_union'].update(ctx.get('rev_lids') or [])

    def _end_current(self, t):
        c = self.current
        c['end_time'] = c['last_s3_time']
        self.incidents.append(c)
        self.current = None
        self.state = 'IDLE'

    def finalize(self, last_t):
        if self.state == 'IN_INCIDENT':
            self.current['end_time'] = self.current['last_s3_time']
            self.incidents.append(self.current)
            self.current = None
            self.state = 'IDLE'


# ====== 사건 → CSV 행 변환 ======
def _classify_severity(refire_count, max_1min):
    """★/★★/★★★/-"""
    if refire_count >= 4 or max_1min >= 20:
        return '★★★'
    elif refire_count >= 2 or max_1min >= 15:
        return '★★'
    elif refire_count >= 1 or max_1min >= 10:
        return '★'
    return '-'


def _build_explanation(max_1min, max_rb_diff, max_rev):
    """primary_cause, contrib_breakdown, anomaly_explanation 계산"""
    ra_val = float(max_1min or 0)
    rb_val = int(max_rb_diff or 0)
    rc_val = int(max_rev or 0)

    ra_score = round(100 * (ra_val / 9.0 - 1), 1) if ra_val >= 9.0 else 0
    rb_score = round(100 * (rb_val / 100.0 - 1), 1) if rb_val >= 100 else 0
    rc_score = round(100 * (rc_val / 2.0 - 1), 1) if rc_val >= 2 else 0

    contrib = [
        ("R-A' 반송시간", ra_score, f"{ra_val:.2f}분 (기준 9분)"),
        ("R-B FAB큐",    rb_score, f"+{rb_val} (기준 +100)"),
        ("R-C' 리프터",  rc_score, f"{rc_val}개 역증가 (기준 2개)"),
    ]
    contrib.sort(key=lambda x: -x[1])

    primary_cause = contrib[0][0] if contrib[0][1] > 0 else '기준 미달'
    breakdown = ' | '.join(f"{name} {desc}" for name, _, desc in contrib)

    if contrib[0][1] > 50:
        impact = '매우 강함'
    elif contrib[0][1] > 20:
        impact = '강함'
    elif contrib[0][1] > 0:
        impact = '보통'
    else:
        impact = '약함'

    parts = []
    if ra_score > 0:
        parts.append(f"반송시간 {ra_val:.1f}분")
    if rb_score > 0:
        parts.append(f"FAB간 큐 +{rb_val}")
    if rc_score > 0:
        parts.append(f"리프터 역증가 {rc_val}개")

    if not parts:
        explanation = '3단계 조건 일부만 부분 충족'
    else:
        explanation = f"{primary_cause} 주도 ({impact}): " + ", ".join(parts)

    return primary_cause, breakdown, explanation


def incident_to_row(c, file_name):
    duration_min = round((c['end_time'] - c['start_time']).total_seconds() / 60.0, 1)
    lead_min = round((c['start_time'] - c['predict_time']).total_seconds() / 60.0)
    primary_cause, breakdown, explanation = _build_explanation(
        c['max_1min'], c['max_rb_diff'], c['max_rev']
    )
    early_warning = (
        f"{c['predict_time'].strftime('%H:%M')} 1·2단계 발동 → "
        f"{c['start_time'].strftime('%H:%M')} 3단계 확정 ({lead_min}분 먼저 인지)"
    )
    severity = '확정' if (c['max_rb_diff'] or 0) >= 100 else '주의'
    return {
        'file': file_name,
        'date': c['start_time'].strftime('%Y-%m-%d'),
        'severity': severity,
        'predict_time': c['predict_time'].strftime('%H:%M'),
        'start_time': c['start_time'].strftime('%H:%M'),
        'end_time': c['end_time'].strftime('%H:%M'),
        'lead_min': lead_min,
        'duration_min': duration_min,
        'refire_count': c['refire_count'],
        'max_1min': round(c['max_1min'], 2),
        'max_m14_diff': c['max_rb_diff'],
        'max_reverse_lifters': c['max_rev'],
        'primary_cause': primary_cause,
        'contrib_breakdown': breakdown,
        'anomaly_explanation': explanation,
        'early_warning': early_warning,
        'relation': build_incident_relation(c),
    }


# ====== 발동이벤트 CSV 헬퍼 ======
STAGE_LABEL = {0: '이벤트없음', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}

# 90min(STAR) CSV 원본 컬럼 — 룰의 근거가 되는 컬럼명 (prefix는 환경별로 다르므로 suffix만 표기)
SRC_COL_RA = 'QUE.TIME.AVGTOTALTIME1MIN'
SRC_COL_RB = 'QUE.M14TOM16.MESCURRENTQCNT'
SRC_COL_RC_TPL = 'LFT.{lid}.TOTAL_CURRENTQCNT'

EVENT_FIELDS = [
    'file', 'datetime', 'date', 'time',
    'stage', 'stage_name', 'prev_stage', 'transition', 'reason',
    'relation',
]

INCIDENT_FIELDS = [
    'file', 'date', 'severity', 'predict_time', 'start_time', 'end_time',
    'lead_min', 'duration_min', 'refire_count',
    'max_1min', 'max_m14_diff', 'max_reverse_lifters',
    'primary_cause', 'contrib_breakdown', 'anomaly_explanation', 'early_warning',
    'relation',
]


def build_event_relation(ev):
    """
    발동 사유의 근거가 된 STAR 컬럼+값. 매 분 R-A'/R-B/R-C' 3룰 상태를 모두 표기.
    사용자가 90min 어느 컬럼/룰이 부족해서 단계 진입을 못 했는지 추적 가능.
    형식: "[R-A' Y] AVGTOTALTIME1MIN=9.27분 | [R-B N] M14TOM16 +34 (<100) | [R-C' N] 역증가 1개 (<2)"
    """
    if ev['stage'] == 0:
        return ''
    ra_val = ev.get('ra_value')
    ra_cnt = ev.get('ra_count', 0) or 0
    ra_sus = bool(ev.get('ra_sustained'))
    ra_trig = ra_cnt >= 1 or ra_sus
    rb_trig = bool(ev.get('rb_trig'))
    rb_fast = bool(ev.get('rb_fast'))
    rb_any = rb_trig or rb_fast
    rev_lids = ev.get('rev_lids') or []
    rev_n = ev.get('rev_count', 0) or 0
    rc_trend = ev.get('rc_trend', 0) or 0
    rc_trig = (rc_trend < 0) and (rev_n >= 2)

    # R-A'
    ra_flag = 'Y' if ra_trig else 'N'
    ra_val_s = f"{ra_val:.2f}분" if ra_val is not None else 'N/A'
    ra_part = f"[R-A' {ra_flag}] {SRC_COL_RA}={ra_val_s} (≥9분 10분창 {ra_cnt}회"
    ra_part += f", 지속Y" if ra_sus else ", 지속N"
    ra_part += ")"

    # R-B
    rb_flag = 'Y' if rb_any else 'N'
    rb_part = (f"[R-B {rb_flag}] {SRC_COL_RB} 30분Δ={ev.get('rb_diff', 0)} (≥100), "
               f"10분Δ={ev.get('rb_diff_10', 0)} (≥30 fast)")

    # R-C'
    rc_flag = 'Y' if rc_trig else 'N'
    if rev_lids:
        lid_cols = ', '.join(SRC_COL_RC_TPL.format(lid=l) for l in rev_lids)
        rc_part = f"[R-C' {rc_flag}] 역증가 {rev_n}개 (≥2): {lid_cols}; trend={rc_trend}"
    else:
        rc_part = f"[R-C' {rc_flag}] 역증가 0개 (≥2 필요); trend={rc_trend}"

    return f"{ra_part} | {rb_part} | {rc_part}"


def build_incident_relation(c):
    """사건단위(S3) 의 근거가 된 STAR 컬럼명+값 — 누적 최대값/역증가 리프터 ID."""
    parts = []
    if c.get('max_1min'):
        parts.append(f"{SRC_COL_RA} max={float(c['max_1min']):.2f}분 (기준 9.0분)")
    if c.get('max_rb_diff'):
        parts.append(f"{SRC_COL_RB} 최대증가 +{c['max_rb_diff']} (기준 +100)")
    rev = sorted(c.get('rev_lids_union') or [])
    if rev:
        lid_cols = ', '.join(SRC_COL_RC_TPL.format(lid=l) for l in rev)
        parts.append(f"역증가 LFT({len(rev)}): {lid_cols}")
    return ' | '.join(parts) if parts else ''


def event_to_row(ev, file_name):
    """발동이벤트 1행. stage=0(이벤트없음)이면 stage/prev_stage/transition/reason/relation 모두 빈칸."""
    t_str = ev['time'].strftime('%Y-%m-%d %H:%M')
    d_str = ev['time'].strftime('%Y-%m-%d')
    hm = ev['time'].strftime('%H:%M')
    stage_name = STAGE_LABEL.get(ev['stage'], '')
    if ev['stage'] == 0:
        return [file_name, t_str, d_str, hm, '', stage_name, '', '', '', '']
    transition = f"{ev['prev_stage']}→{ev['stage']}" if ev.get('is_transition') else ''
    relation = build_event_relation(ev)
    return [
        file_name, t_str, d_str, hm,
        ev['stage'], stage_name, ev['prev_stage'], transition, ev['reason'],
        relation,
    ]


def append_rows_csv(path, fields, rows):
    """헤더가 없는 신규 파일이면 헤더 작성 후 rows append. 있으면 그대로 append."""
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(fields)
        for r in rows:
            w.writerow(r)


def write_events_by_date(out_dir, events, file_name):
    """이벤트들을 날짜별 <YYYYMMDD>_발동이벤트.csv 에 append. 파일 경로 리스트 반환."""
    by_date = {}
    for ev in events:
        key = ev['time'].strftime('%Y%m%d')
        by_date.setdefault(key, []).append(event_to_row(ev, file_name))
    paths = []
    for ymd in sorted(by_date):
        path = os.path.join(out_dir, f'{ymd}_발동이벤트.csv')
        append_rows_csv(path, EVENT_FIELDS, by_date[ymd])
        paths.append(path)
    return paths


def write_incidents_by_date(out_dir, incidents, file_name):
    """사건들을 날짜별 <YYYYMMDD>_사건단위.csv 에 append. 파일 경로 리스트 반환."""
    by_date = {}
    for c in incidents:
        ymd = c['start_time'].strftime('%Y%m%d')
        row = incident_to_row(c, file_name)
        by_date.setdefault(ymd, []).append([row[k] for k in INCIDENT_FIELDS])
    paths = []
    for ymd in sorted(by_date):
        path = os.path.join(out_dir, f'{ymd}_사건단위.csv')
        append_rows_csv(path, INCIDENT_FIELDS, by_date[ymd])
        paths.append(path)
    return paths


def append_event_row(out_dir, ev, file_name):
    """발동이벤트 — 1행을 해당 날짜 CSV에 append (실시간/스트림 모드용)."""
    ymd = ev['time'].strftime('%Y%m%d')
    path = os.path.join(out_dir, f'{ymd}_발동이벤트.csv')
    append_rows_csv(path, EVENT_FIELDS, [event_to_row(ev, file_name)])
    return path


def append_incident_row(out_dir, incident, file_name):
    """사건단위 — 1건을 해당 날짜 CSV에 append (실시간/스트림 모드용)."""
    ymd = incident['start_time'].strftime('%Y%m%d')
    path = os.path.join(out_dir, f'{ymd}_사건단위.csv')
    row = incident_to_row(incident, file_name)
    append_rows_csv(path, INCIDENT_FIELDS, [[row[k] for k in INCIDENT_FIELDS]])
    return path


# ====== 메인 (일괄 처리) ======
def process(input_csv, out_dir='.'):
    """
    입력 CSV 전체를 일괄 처리해서 날짜별 발동이벤트 / 사건단위 CSV에 append.
    out_dir 안에 같은 날짜 파일 있으면 그 파일에 추가, 없으면 새로 생성.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 90분 슬라이딩 윈도우
    t1_window = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)

    tracker = IncidentTracker()
    last_t = None
    file_name = os.path.basename(input_csv)
    rows_processed = 0
    rules_evaluated = 0

    print(f'📥 입력: {input_csv}')
    print(f'   출력 폴더: {os.path.abspath(out_dir)}')
    print(f'   윈도우: 과거 {WINDOW_MIN}분')

    for t, star, prefix in iter_star_rows(input_csv):
        t1_window.append(star.get('avgtotal1min'))
        m14_window.append(star.get('m14_to_m16'))
        lft_window.append(star.get('lft_list') or {})
        rows_processed += 1
        last_t = t

        # 90분 안 채워지면 룰 평가 보류 (R-B 30분 전 비교 위해 31개 필요)
        if len(t1_window) < 31:
            continue

        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window)
        rules_evaluated += 1
        tracker.update(t, s1, s2, s3, ctx)

    tracker.finalize(last_t)

    # ── 날짜별 CSV append ──
    event_paths = write_events_by_date(out_dir, tracker.events, file_name)
    incident_paths = write_incidents_by_date(out_dir, tracker.incidents, file_name)

    print()
    print(f'📊 처리 결과')
    print(f'   행 처리:    {rows_processed:,} 행')
    print(f'   룰 평가:    {rules_evaluated:,} 회 (90분 채워진 후)')
    print(f'   사건 추출:  {len(tracker.incidents)} 건 (S3 — 진짜 위험)')
    print(f'   발동 기록:  {len(tracker.events)} 분 (1분당 1행)')
    print()
    print(f'💾 발동이벤트 CSV ({len(event_paths)}개 날짜):')
    for p in event_paths:
        print(f'   · {p}')
    if incident_paths:
        print(f'💾 사건단위 CSV ({len(incident_paths)}개 날짜):')
        for p in incident_paths:
            print(f'   · {p}')
    else:
        print(f'💾 사건단위 CSV: 진짜 위험 사건 0건 (생성 없음)')


# ====== 실시간 감시 모드 ======
def watch(input_csv, out_dir='.', interval=60):
    """
    1분 간격(또는 interval초)으로 CSV 파일을 폴링해서 새로 추가된 행만 처리.
    매 분마다 발동이벤트 CSV에 1행 append, 사건 종료 시 사건단위 CSV에 append.
    """
    os.makedirs(out_dir, exist_ok=True)

    t1_window = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)
    tracker = IncidentTracker()
    file_name = os.path.basename(input_csv)

    last_size = 0
    last_event_count = 0
    last_incident_count = 0
    last_t = None

    print(f'👁  실시간 감시 시작')
    print(f'   입력: {input_csv}')
    print(f'   출력 폴더: {os.path.abspath(out_dir)}')
    print(f'   폴링 간격: {interval}초 (Ctrl+C로 종료)')
    print()

    while True:
        try:
            if os.path.exists(input_csv):
                cur_size = os.path.getsize(input_csv)
                if cur_size != last_size:
                    # 파일 변화 감지 — 처음부터 다시 읽되 이미 본 행은 윈도우 상태로 흘려보냄
                    seen = 0
                    for t, star, _ in iter_star_rows(input_csv):
                        seen += 1
                        # 같은 시각이거나 이전 시각이면 스킵 (이미 처리)
                        if last_t is not None and t <= last_t:
                            continue
                        t1_window.append(star.get('avgtotal1min'))
                        m14_window.append(star.get('m14_to_m16'))
                        lft_window.append(star.get('lft_list') or {})
                        last_t = t

                        if len(t1_window) < 31:
                            continue

                        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window)
                        tracker.update(t, s1, s2, s3, ctx)

                        # 매 분 발동이벤트 1행 append
                        if len(tracker.events) > last_event_count:
                            for ev in tracker.events[last_event_count:]:
                                p = append_event_row(out_dir, ev, file_name)
                                stage_disp = STAGE_LABEL.get(ev['stage'], '')
                                print(f'   [{ev["time"].strftime("%H:%M")}] {stage_disp} '
                                      f'— {ev["reason"]} → {os.path.basename(p)}')
                            last_event_count = len(tracker.events)

                        # 신규 종료된 사건이 있으면 사건단위 append
                        if len(tracker.incidents) > last_incident_count:
                            for c in tracker.incidents[last_incident_count:]:
                                p = append_incident_row(out_dir, c, file_name)
                                print(f'   ⭐ 사건 확정 → {os.path.basename(p)}')
                            last_incident_count = len(tracker.incidents)

                    last_size = cur_size

            time.sleep(interval)
        except KeyboardInterrupt:
            print('\n👋 감시 종료')
            # 미해소 사건 강제 종료
            tracker.finalize(last_t)
            if len(tracker.incidents) > last_incident_count:
                for c in tracker.incidents[last_incident_count:]:
                    p = append_incident_row(out_dir, c, file_name)
                    print(f'   ⭐ 사건 확정 (종료 시) → {os.path.basename(p)}')
            break


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_csv = sys.argv[1]
    out_dir = '.'
    watch_mode = False
    interval = 60

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == '-o' and i + 1 < len(args):
            out_dir = args[i + 1]
            i += 2
        elif a == '--watch':
            watch_mode = True
            i += 1
        elif a == '--interval' and i + 1 < len(args):
            interval = int(args[i + 1])
            i += 2
        else:
            i += 1

    if not os.path.exists(input_csv) and not watch_mode:
        print(f'❌ 파일 없음: {input_csv}')
        sys.exit(1)

    if watch_mode:
        watch(input_csv, out_dir, interval)
    else:
        process(input_csv, out_dir)


if __name__ == '__main__':
    main()

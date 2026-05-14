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
    python3 룰베이스검증.py STAR.csv
    python3 룰베이스검증.py STAR.csv -o ./out

    # 실시간 감시 모드 (1분마다 새 행 추가되는 CSV 폴링)
    python3 룰베이스검증.py STAR.csv --watch
    python3 룰베이스검증.py STAR.csv --watch --interval 60 -o ./out

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

# R-D (FAB 저장률 정체 보조 룰) — 기존 32컬럼만으로도 작동
TH_RD_FABSTORAGE = 25.0
TH_RD_7F_HUB_ALT = 20

# ★ v3.1 신규 룰 임계값 (보조 신호용 — 기존 S3 로직 불변)
TH_RE_HUB_STORAGE_LOW = 60.0    # R-E: HUB 저장 가용 ≤ 60% (= 사용율 ≥ 40%)
TH_RF_INFLOW_TOTAL    = 700     # R-F: HUB 전체 인플로 ≥ 700개 (대형 부하)
TH_RF_INFLOW_SPIKE    = 1.5     # R-F-fast: 인플로 10분 +50% 폭증

# 사건 종료 판단: S3 신규/재발동 사이 간격
INCIDENT_END_GAP_MIN = 10
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
    """STAR CSV → (datetime, star_dict, prefix) 제너레이터"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prefix = detect_prefix(reader.fieldnames)
        if not prefix:
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
                'fabstorage_ratio':  safe_float(row.get(C('.STRATE.ALL.FABSTORAGERATIO'))),
                'm14b_aotransdelay': safe_float(row.get('M14B.QUE.ABN.AOTRANSDELAY')),
                'm14b_oht_util':     safe_float(row.get('M14B.QUE.OHT.OHTUTIL')),
                'm14b_4abld122':     safe_int(row.get('M14B.LFT.4ABLD122.TOTAL_CURRENTQCNT')),
                'm14b_avgtotal1min': safe_float(row.get('M14B.QUE.TIME.AVGTOTALTIME1MIN')),
                'm14b_7f_to_hub':    safe_int(row.get('M14B.QUE.ALL.7F_TO_HUB_JOB')),
                'm14b_7f_to_hub_alt': safe_int(row.get('M14B.QUE.ALL.7F_TO_HUB_JOB_ALT')),
                'm14_htstop':        safe_int(row.get('M14.OHT.STATECNT.HTSTOP')),
                'm14_congested':     safe_int(row.get('M14.OHT.STATECNT.CONGESTED')),
                'm14_abnormal':      safe_int(row.get('M14.OHT.STATECNT.ABNORMAL')),
                'm16pkt_aotransdelay': safe_float(row.get('M16_PKT.QUE.ABN.AOTRANSDELAY')),
                'm16wt_aotransdelay':  safe_float(row.get('M16_WT.QUE.ABN.AOTRANSDELAY')),
                # ★ v3.1 신규 5개 컬럼
                'hub_storage_util':  safe_float(row.get(C('.STRATE.STB.3F_STORAGE_UTIL'))),
                'm14_inflow':        safe_int(row.get('M14.QUE.ALL.3F_TO_HUB_JOB')),
                'm16a_2f_inflow':    safe_int(row.get('M16A.QUE.ALL.2F_TO_HUB_JOB')),
                'm16a_6f_inflow':    safe_int(row.get('M16A.QUE.ALL.6F_TO_HUB_JOB')),
                'm16b_10f_inflow':   safe_int(row.get('M16B.QUE.ALL.10F_TO_HUB_JOB')),
            }
            star['lft_list'] = {k: v for k, v in star['lft_list'].items() if v is not None}
            yield t, star, prefix


# ====== 룰 평가 (90분 윈도우 데이터 기반) ======
def evaluate_rules(t1_window, m14_window, lft_window, v3_window=None):
    t1_list = list(t1_window)
    m14_list = list(m14_window)
    lft_list = list(lft_window)

    recent_t1 = t1_list[-10:]
    ra_count = sum(1 for v in recent_t1 if v is not None and v >= TH_RA_VALUE)
    ra_value = recent_t1[-1] if recent_t1 else None
    ra_trig = ra_count >= 1

    ra_sustained = False
    if len(recent_t1) >= 5:
        last5 = [v for v in recent_t1[-5:] if v is not None]
        if len(last5) >= 3:
            ra_sustained = sum(1 for v in last5 if v >= TH_RA_SUSTAINED_VALUE) >= TH_RA_SUSTAINED_COUNT

    rb_diff = 0
    rb_trig = False
    if len(m14_list) >= 31 and m14_list[-1] is not None and m14_list[-31] is not None:
        rb_diff = m14_list[-1] - m14_list[-31]
        rb_trig = rb_diff >= TH_RB_DIFF_30

    rb_fast = False
    rb_diff_10 = 0
    if len(m14_list) >= 11 and m14_list[-1] is not None and m14_list[-11] is not None:
        rb_diff_10 = m14_list[-1] - m14_list[-11]
        rb_fast = rb_diff_10 >= TH_RB_DIFF_10

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

    rd_fabstorage = 0
    rd_7f_alt = 0
    rd_trig = False
    # ★ v3.1 신규 룰 변수 (R-E: HUB저장, R-F: 인플로)
    hub_storage_util = None
    inflow_total = 0
    inflow_total_10ago = 0
    re_trig = False     # HUB 저장 가용공간 부족
    rf_trig = False     # 인플로 절대량 큼
    rf_fast = False     # 인플로 급증
    if v3_window:
        latest = v3_window[-1] if v3_window else {}
        if latest:
            rd_fabstorage = latest.get('fabstorage_ratio') or 0
            rd_7f_alt = latest.get('m14b_7f_to_hub_alt') or 0
            rd_trig = rd_fabstorage >= TH_RD_FABSTORAGE

            # R-E: HUB Storage 가용공간 부족
            hub_storage_util = latest.get('hub_storage_util')
            if hub_storage_util is not None:
                re_trig = hub_storage_util <= TH_RE_HUB_STORAGE_LOW

            # R-F: 인플로 통합 (M14 + M16A 2F+6F + M16B 10F)
            def _sum_inflow(d):
                if not d: return 0
                return ((d.get('m14_inflow') or 0)
                        + (d.get('m16a_2f_inflow') or 0)
                        + (d.get('m16a_6f_inflow') or 0)
                        + (d.get('m16b_10f_inflow') or 0))
            inflow_total = _sum_inflow(latest)
            rf_trig = inflow_total >= TH_RF_INFLOW_TOTAL

            # R-F-fast: 10분 전 대비 인플로 50% 이상 폭증
            if len(v3_window) >= 11:
                prev = v3_window[-11] if v3_window[-11] else {}
                inflow_total_10ago = _sum_inflow(prev)
                if inflow_total_10ago > 100:
                    rf_fast = inflow_total >= inflow_total_10ago * TH_RF_INFLOW_SPIKE

    s1 = (ra_count >= 2) or ra_sustained
    s2 = rb_trig or rb_fast
    # ★ S3 = 불변 (검증된 로직) — R-E/R-F 는 ctx 출력만, S3 영향 없음
    s3 = ra_trig and rc_trig and (rb_trig or rd_trig)

    # ============================================================
    # ★★★ 위험도 점수 (Risk Score) — "S3 로 갈 진짜 위험인가"
    # S1/S2 시점에도 0~100점 부여. 신호 동시 발동 많을수록 점수↑
    # ============================================================
    risk_score = 0
    risk_factors = []

    # 시간축 (R-A')
    if ra_sustained:
        risk_score += 25; risk_factors.append('ra_sustained')
    if ra_count >= 2:
        risk_score += 20; risk_factors.append(f'ra_count={ra_count}')
    if ra_value and ra_value >= 7.5:  # 9분 임계 직전 (S3 임박)
        risk_score += 15; risk_factors.append(f'ra={ra_value:.1f}분')

    # 양축 (R-B)
    if rb_fast:
        risk_score += 25; risk_factors.append('rb_fast')
    if rb_trig:
        risk_score += 15; risk_factors.append(f'rb=+{rb_diff}')

    # 위치축 (R-C')
    if rc_trig:
        risk_score += 20; risk_factors.append(f'rc_rev={rev_count}')
    elif rev_count >= 3:  # 트리거 미만이지만 역증가 많음
        risk_score += 10; risk_factors.append(f'rev={rev_count}(준위험)')

    # 공간축 (R-D)
    if rd_trig:
        risk_score += 25; risk_factors.append(f'rd={rd_fabstorage:.0f}%')
    elif rd_fabstorage >= 15:  # 임계 직전 (25% 임계 60%)
        risk_score += 10; risk_factors.append(f'rd={rd_fabstorage:.0f}%(준위험)')

    # ★ v3.1 신규 신호
    if re_trig:
        risk_score += 15; risk_factors.append('re(HUB저장부족)')
    if rf_trig:
        risk_score += 10; risk_factors.append(f'rf={inflow_total}(인플로↑)')
    if rf_fast:
        risk_score += 10; risk_factors.append('rf_fast')

    # 위험도 레벨
    if risk_score >= 70:
        risk_level = '매우위험'   # 90% 이상 S3 진행
    elif risk_score >= 45:
        risk_level = '위험'       # 60% 이상 S3 진행 가능
    elif risk_score >= 25:
        risk_level = '주의'       # 30% 이하 S3 진행
    elif risk_score > 0:
        risk_level = '관심'
    else:
        risk_level = '정상'

    ctx = {
        'ra_count': ra_count, 'ra_value': ra_value, 'ra_sustained': ra_sustained,
        'ra_trig': ra_trig,
        'rb_diff': rb_diff, 'rb_diff_10': rb_diff_10,
        'rb_fast': rb_fast, 'rb_trig': rb_trig,
        'rc_trend': rc_trend, 'rc_trig': rc_trig,
        'rev_count': rev_count, 'rev_lids': rev_lids,
        'rd_fabstorage': rd_fabstorage, 'rd_7f_alt': rd_7f_alt, 'rd_trig': rd_trig,
        # ★ v3.1 신규
        'hub_storage_util': hub_storage_util,
        'inflow_total': inflow_total, 'inflow_total_10ago': inflow_total_10ago,
        're_trig': re_trig, 'rf_trig': rf_trig, 'rf_fast': rf_fast,
        # ★★★ 위험도 평가
        'risk_score': risk_score, 'risk_level': risk_level,
        'risk_factors': ';'.join(risk_factors),
    }
    return s1, s2, s3, ctx


# ====== 사건 추적 FSM ======
class IncidentTracker:
    def __init__(self):
        self.state = 'IDLE'
        self.current = None
        self.incidents = []
        self.early_signals = deque(maxlen=PREDICT_LOOKBACK_MIN)
        self.events = []
        self.last_stage = 0

    def _record_event(self, t, stage, ctx):
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
            if ctx.get('rd_trig') and not (ctx.get('rb_trig') and ctx.get('rc_trig')):
                reason = (f"R-D FAB저장률 정체 (FABSTORAGE={ctx.get('rd_fabstorage', 0):.1f}%, "
                          f"1MIN {ctx.get('ra_value') or 0:.2f})")
            else:
                reason = (f"AND 만족 (1MIN {ctx.get('ra_value') or 0:.2f}, "
                          f"M14→M16 +{ctx.get('rb_diff', 0)}, "
                          f"역증가 {ctx.get('rev_count', 0)}개)")
        else:
            reason = "이벤트 없음"
        self.events.append({
            'time': t, 'stage': stage, 'prev_stage': self.last_stage,
            'reason': reason, 'is_transition': is_transition,
            'ra_value': ctx.get('ra_value'), 'ra_count': ctx.get('ra_count', 0),
            'ra_sustained': bool(ctx.get('ra_sustained')),
            'rb_diff': ctx.get('rb_diff', 0), 'rb_diff_10': ctx.get('rb_diff_10', 0),
            'rb_fast': bool(ctx.get('rb_fast')), 'rb_trig': bool(ctx.get('rb_trig')),
            'rc_trend': ctx.get('rc_trend', 0),
            'rev_count': ctx.get('rev_count', 0),
            'rev_lids': list(ctx.get('rev_lids') or []),
            'rd_fabstorage': ctx.get('rd_fabstorage', 0),
            'rd_7f_alt': ctx.get('rd_7f_alt', 0),
            'rd_trig': bool(ctx.get('rd_trig')),
            # ★ v3.1 신규
            'hub_storage_util': ctx.get('hub_storage_util'),
            'inflow_total': ctx.get('inflow_total', 0),
            're_trig': bool(ctx.get('re_trig')),
            'rf_trig': bool(ctx.get('rf_trig')),
            'rf_fast': bool(ctx.get('rf_fast')),
            # ★★★ 위험도 평가
            'risk_score': ctx.get('risk_score', 0),
            'risk_level': ctx.get('risk_level', '정상'),
            'risk_factors': ctx.get('risk_factors', ''),
        })
        self.last_stage = stage

    def update(self, t, s1, s2, s3, ctx):
        if s1 or s2:
            self.early_signals.append(t)
        cur_stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))
        self._record_event(t, cur_stage, ctx)

        if self.state == 'IDLE':
            if s3:
                self._start_new(t, ctx)
        else:
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
            'predict_time': predict_time, 'start_time': t,
            'last_s3_time': t, 'end_time': t, 'refire_count': 0,
            'max_1min': ctx['ra_value'] or 0, 'max_rb_diff': ctx['rb_diff'] or 0,
            'max_rev': ctx['rev_count'] or 0,
            'rev_lids_union': set(ctx.get('rev_lids') or []),
            'max_rd_fabstorage': ctx.get('rd_fabstorage', 0) or 0,
            'max_rd_7f_alt': ctx.get('rd_7f_alt', 0) or 0,
            'rd_triggered': bool(ctx.get('rd_trig')),
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
        rd_f = ctx.get('rd_fabstorage', 0) or 0
        rd_a = ctx.get('rd_7f_alt', 0) or 0
        if rd_f > c.get('max_rd_fabstorage', 0):
            c['max_rd_fabstorage'] = rd_f
        if rd_a > c.get('max_rd_7f_alt', 0):
            c['max_rd_7f_alt'] = rd_a
        if ctx.get('rd_trig'):
            c['rd_triggered'] = True

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


# ====== CSV 출력 헬퍼 (생략하지 않은 원본 그대로) ======
STAGE_LABEL = {0: '이벤트없음', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}
SRC_COL_RA = 'QUE.TIME.AVGTOTALTIME1MIN'
SRC_COL_RB = 'QUE.M14TOM16.MESCURRENTQCNT'
SRC_COL_RC_TPL = 'LFT.{lid}.TOTAL_CURRENTQCNT'

EVENT_FIELDS = [
    'file', 'datetime', 'date', 'time',
    'stage', 'stage_name', 'prev_stage', 'transition', 'reason', 'relation',
    # ★ v3.1 신규 보조 신호
    'hub_storage_util', 'inflow_total', 're_trig', 'rf_trig', 'rf_fast',
    # ★★★ 위험도 평가
    'risk_score', 'risk_level', 'risk_factors',
]
INCIDENT_FIELDS = [
    'file', 'date', 'severity', 'predict_time', 'start_time', 'end_time',
    'lead_min', 'duration_min', 'refire_count',
    'max_1min', 'max_m14_diff', 'max_reverse_lifters',
    'primary_cause', 'contrib_breakdown', 'anomaly_explanation', 'early_warning',
    'relation',
]


def _classify_severity(refire_count, max_1min):
    if refire_count >= 4 or max_1min >= 20:
        return '★★★'
    elif refire_count >= 2 or max_1min >= 15:
        return '★★'
    elif refire_count >= 1 or max_1min >= 10:
        return '★'
    return '-'


def _build_explanation(max_1min, max_rb_diff, max_rev):
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
    if ra_score > 0: parts.append(f"반송시간 {ra_val:.1f}분")
    if rb_score > 0: parts.append(f"FAB간 큐 +{rb_val}")
    if rc_score > 0: parts.append(f"리프터 역증가 {rc_val}개")
    explanation = '3단계 조건 일부만 부분 충족' if not parts else f"{primary_cause} 주도 ({impact}): " + ", ".join(parts)
    return primary_cause, breakdown, explanation


def build_event_relation(ev):
    if ev['stage'] == 0: return ''
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
    ra_flag = 'Y' if ra_trig else 'N'
    ra_val_s = f"{ra_val:.2f}분" if ra_val is not None else 'N/A'
    ra_part = f"[R-A' {ra_flag}] {SRC_COL_RA}={ra_val_s} (≥9분 10분창 {ra_cnt}회, 지속{'Y' if ra_sus else 'N'})"
    rb_flag = 'Y' if rb_any else 'N'
    rb_part = (f"[R-B {rb_flag}] {SRC_COL_RB} 30분Δ={ev.get('rb_diff', 0)} (≥100), "
               f"10분Δ={ev.get('rb_diff_10', 0)} (≥30 fast)")
    rc_flag = 'Y' if rc_trig else 'N'
    if rev_lids:
        lid_cols = ', '.join(SRC_COL_RC_TPL.format(lid=l) for l in rev_lids)
        rc_part = f"[R-C' {rc_flag}] 역증가 {rev_n}개 (≥2): {lid_cols}; trend={rc_trend}"
    else:
        rc_part = f"[R-C' {rc_flag}] 역증가 0개 (≥2 필요); trend={rc_trend}"
    return f"{ra_part} | {rb_part} | {rc_part}"


def build_incident_relation(c):
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
    t_str = ev['time'].strftime('%Y-%m-%d %H:%M')
    d_str = ev['time'].strftime('%Y-%m-%d')
    hm = ev['time'].strftime('%H:%M')
    stage_name = STAGE_LABEL.get(ev['stage'], '')
    hub_util = ev.get('hub_storage_util')
    hub_util_s = f"{hub_util:.1f}" if hub_util is not None else ''
    risk_score = ev.get('risk_score', 0)
    risk_level = ev.get('risk_level', '정상')
    risk_factors = ev.get('risk_factors', '')
    common_tail = [
        hub_util_s, ev.get('inflow_total', 0) or 0,
        int(bool(ev.get('re_trig'))), int(bool(ev.get('rf_trig'))), int(bool(ev.get('rf_fast'))),
        risk_score, risk_level, risk_factors,
    ]
    if ev['stage'] == 0:
        return [file_name, t_str, d_str, hm, '', stage_name, '', '', '', ''] + common_tail
    transition = f"{ev['prev_stage']}→{ev['stage']}" if ev.get('is_transition') else ''
    relation = build_event_relation(ev)
    return [file_name, t_str, d_str, hm, ev['stage'], stage_name, ev['prev_stage'], transition, ev['reason'], relation] + common_tail


def incident_to_row(c, file_name):
    duration_min = round((c['end_time'] - c['start_time']).total_seconds() / 60.0, 1)
    lead_min = round((c['start_time'] - c['predict_time']).total_seconds() / 60.0)
    primary_cause, breakdown, explanation = _build_explanation(c['max_1min'], c['max_rb_diff'], c['max_rev'])
    early_warning = f"{c['predict_time'].strftime('%H:%M')} 1·2단계 발동 → {c['start_time'].strftime('%H:%M')} 3단계 확정 ({lead_min}분 먼저 인지)"
    severity = '확정' if (c['max_rb_diff'] or 0) >= 100 else '주의'
    return {
        'file': file_name, 'date': c['start_time'].strftime('%Y-%m-%d'),
        'severity': severity,
        'predict_time': c['predict_time'].strftime('%H:%M'),
        'start_time': c['start_time'].strftime('%H:%M'),
        'end_time': c['end_time'].strftime('%H:%M'),
        'lead_min': lead_min, 'duration_min': duration_min,
        'refire_count': c['refire_count'],
        'max_1min': round(c['max_1min'], 2),
        'max_m14_diff': c['max_rb_diff'],
        'max_reverse_lifters': c['max_rev'],
        'primary_cause': primary_cause, 'contrib_breakdown': breakdown,
        'anomaly_explanation': explanation, 'early_warning': early_warning,
        'relation': build_incident_relation(c),
    }


def append_rows_csv(path, fields, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(fields)
        for r in rows:
            w.writerow(r)


def write_events_by_date(out_dir, events, file_name):
    by_date = {}
    for ev in events:
        key = ev['time'].strftime('%Y%m%d')
        by_date.setdefault(key, []).append(event_to_row(ev, file_name))
    for ymd in sorted(by_date):
        path = os.path.join(out_dir, f'{ymd}_발동이벤트.csv')
        append_rows_csv(path, EVENT_FIELDS, by_date[ymd])


def write_incidents_by_date(out_dir, incidents, file_name):
    by_date = {}
    for c in incidents:
        ymd = c['start_time'].strftime('%Y%m%d')
        row = incident_to_row(c, file_name)
        by_date.setdefault(ymd, []).append([row[k] for k in INCIDENT_FIELDS])
    for ymd in sorted(by_date):
        path = os.path.join(out_dir, f'{ymd}_사건단위.csv')
        append_rows_csv(path, INCIDENT_FIELDS, by_date[ymd])


# ====== 메인 (일괄 처리) ======
def process(input_csv, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)

    t1_window = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)
    v3_window = deque(maxlen=WINDOW_MIN)

    tracker = IncidentTracker()
    last_t = None
    file_name = os.path.basename(input_csv)

    for t, star, prefix in iter_star_rows(input_csv):
        t1_window.append(star.get('avgtotal1min'))
        m14_window.append(star.get('m14_to_m16'))
        lft_window.append(star.get('lft_list') or {})
        v3_window.append({k: star.get(k) for k in (
            'fabstorage_ratio',
            'm14b_aotransdelay', 'm14b_oht_util', 'm14b_4abld122',
            'm14b_avgtotal1min', 'm14b_7f_to_hub', 'm14b_7f_to_hub_alt',
            'm14_htstop', 'm14_congested', 'm14_abnormal',
            'm16pkt_aotransdelay', 'm16wt_aotransdelay',
            # ★ v3.1 신규 5개 컬럼
            'hub_storage_util', 'm14_inflow',
            'm16a_2f_inflow', 'm16a_6f_inflow', 'm16b_10f_inflow',
        )})
        last_t = t

        if len(t1_window) < 31:
            continue

        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window, v3_window)
        tracker.update(t, s1, s2, s3, ctx)

    tracker.finalize(last_t)

    write_events_by_date(out_dir, tracker.events, file_name)
    write_incidents_by_date(out_dir, tracker.incidents, file_name)
    return tracker


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    input_csv = sys.argv[1]
    out_dir = '.'
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == '-o' and i + 1 < len(args):
            out_dir = args[i + 1]
            i += 2
        else:
            i += 1

    if not os.path.exists(input_csv):
        sys.exit(f'파일 없음: {input_csv}')

    tracker = process(input_csv, out_dir)
    # 요약 출력
    n_events = len(tracker.events)
    n_s1 = sum(1 for e in tracker.events if e['stage'] == 1)
    n_s2 = sum(1 for e in tracker.events if e['stage'] == 2)
    n_s3 = sum(1 for e in tracker.events if e['stage'] == 3)
    n_idle = sum(1 for e in tracker.events if e['stage'] == 0)
    n_inc = len(tracker.incidents)
    print(f"\n=== 룰베이스 검증 결과 ===")
    print(f"총 분석 분: {n_events}")
    print(f"  정상  : {n_idle:6d} ({100*n_idle/n_events:.1f}%)")
    print(f"  S1    : {n_s1:6d} ({100*n_s1/n_events:.1f}%)")
    print(f"  S2    : {n_s2:6d} ({100*n_s2/n_events:.1f}%)")
    print(f"  S3    : {n_s3:6d} ({100*n_s3/n_events:.1f}%)")
    print(f"사건 수: {n_inc}")

    # ★ v3.1 신규 룰 통계
    have_v31 = [e for e in tracker.events if e.get('hub_storage_util') is not None]
    if have_v31:
        n_re = sum(1 for e in have_v31 if e.get('re_trig'))
        n_rf = sum(1 for e in have_v31 if e.get('rf_trig'))
        n_rf_fast = sum(1 for e in have_v31 if e.get('rf_fast'))
        print(f"\n=== v3.1 신규 보조 룰 통계 (데이터 있는 {len(have_v31):,}분 기준) ===")
        print(f"  R-E (HUB 저장 ≤{TH_RE_HUB_STORAGE_LOW}%) 발동: {n_re:6d} ({100*n_re/len(have_v31):.1f}%)")
        print(f"  R-F (인플로 ≥{TH_RF_INFLOW_TOTAL}) 발동:    {n_rf:6d} ({100*n_rf/len(have_v31):.1f}%)")
        print(f"  R-F-fast (인플로 1.5x 폭증):     {n_rf_fast:6d} ({100*n_rf_fast/len(have_v31):.1f}%)")

        # S3 시점에 신규 룰도 동시 발동 비율
        s3_events = [e for e in have_v31 if e['stage'] == 3]
        if s3_events:
            s3_re = sum(1 for e in s3_events if e.get('re_trig'))
            s3_rf = sum(1 for e in s3_events if e.get('rf_trig'))
            s3_rf_fast = sum(1 for e in s3_events if e.get('rf_fast'))
            print(f"\n=== S3 발동 시 신규 룰 동시 발동 (S3 {len(s3_events)}분 기준) ===")
            print(f"  S3 + R-E 동시: {s3_re:4d}분 ({100*s3_re/len(s3_events):.0f}%)")
            print(f"  S3 + R-F 동시: {s3_rf:4d}분 ({100*s3_rf/len(s3_events):.0f}%)")
            print(f"  S3 + R-F-fast 동시: {s3_rf_fast:4d}분 ({100*s3_rf_fast/len(s3_events):.0f}%)")

    if tracker.incidents:
        print("\n=== 사건 목록 (최근 30개) ===")
        for i, c in enumerate(tracker.incidents[-30:], 1):
            duration = (c['end_time'] - c['start_time']).total_seconds() / 60
            lead = (c['start_time'] - c['predict_time']).total_seconds() / 60
            print(f"  {i:2d}. {c['start_time']} ~ {c['end_time']} "
                  f"({duration:.0f}분, 사전인지 {lead:.0f}분, "
                  f"1MIN max={c['max_1min']:.1f}, rb_max=+{c['max_rb_diff']}, "
                  f"rev={c['max_rev']}, rd={c.get('max_rd_fabstorage', 0):.1f}%)")


if __name__ == '__main__':
    main()

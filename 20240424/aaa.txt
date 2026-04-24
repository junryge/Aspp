#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macro_predictor.py - OHT 매크로 예측기

MD 문서에서 검증된 예측 모델:
- 큐: 10분 이동평균 + 선형 외삽
- TAT: 상수 2.88분 (24시간 균일)
- 물동량: ~20,000건/시 (균일)
- 혼잡 구간: HID 저속 구간 고정
- 데드락 위험: OBS 추세 (36분 빌드업)
"""

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import (
    TAT_AVERAGE_MIN, TAT_MEDIAN_MIN,
    THROUGHPUT_PER_HOUR,
    QUEUE_NORMAL_RANGE, QUEUE_AVG,
    OBS_NORMAL_RANGE, OBS_WARNING_THRESHOLD,
    OBS_DANGER_THRESHOLD, OBS_DEADLOCK_THRESHOLD,
)


class MacroPredictor:
    """매크로 예측기"""

    def __init__(self, history_size: int = 30):
        # 최근 N분 히스토리 (이동평균용)
        self.queue_history: deque = deque(maxlen=history_size)
        self.obs_history: deque = deque(maxlen=history_size)
        self.driving_history: deque = deque(maxlen=history_size)

        # 3단계 데드락 경보용 히스토리 (M16A_BR 확장)
        self.t1min_history: deque = deque(maxlen=20)         # AVGTOTAL1MIN
        self.m14_history:   deque = deque(maxlen=35)         # M14→M16
        self.lft_history:   deque = deque(maxlen=25)         # 리프터 10개 dict
        self.mlud_history:  deque = deque(maxlen=30)         # MLUD Q
        self.warning_log:   List[dict] = []                   # 단계 발동 이력 (타임라인용)
        self._last_stage:   int = 0
        self.full_timeline: List[dict] = []                   # 사전 계산 전체 타임라인

        # 현재 값
        self.current_queue: int = 0
        self.current_obs: int = 0
        self.current_driving: int = 0
        self.current_time: Optional[datetime] = None

        # 상관관계 데이터 (전체 기간)
        self.queue_tat_pairs: List[Tuple[int, float]] = []  # (큐, TAT) 쌍
        self.queue_obs_pairs: List[Tuple[int, int]] = []

    def update(self, t: datetime, star: dict, vehicle_stats: dict):
        """스타 지표 + 차량 통계로 업데이트. STAR 없으면 vehicle_stats 기반 fallback."""
        self.current_time = t

        if star:
            q = star.get('queue_total', 0)
            obs = star.get('obs_bz_stop', 0)
            driving = star.get('driving', 0)

            if q > 0:
                self.current_queue = q
                self.queue_history.append((t, q))
            if obs >= 0:
                self.current_obs = obs
                self.obs_history.append((t, obs))
            if driving > 0:
                self.current_driving = driving
                self.driving_history.append((t, driving))

            # 상관관계 데이터 축적
            if q > 0 and obs >= 0:
                self.queue_obs_pairs.append((q, obs))

            # 3단계 경보용 히스토리
            t1min = star.get('avgtotal1min')
            m14 = star.get('m14_to_m16')
            lft = star.get('lft_list')
            mlud = star.get('mlud_q')
            if t1min is not None:
                self.t1min_history.append((t, float(t1min)))
            if m14 is not None and m14 > 0:
                self.m14_history.append((t, int(m14)))
            if isinstance(lft, dict) and lft:
                self.lft_history.append((t, dict(lft)))
            if mlud is not None:
                self.mlud_history.append((t, int(mlud)))
        elif vehicle_stats:
            # STAR 없을 때 — 차량 상태 집계에서 OBS 추출
            obs = vehicle_stats.get('obs_bz_stop', 0)
            driving = vehicle_stats.get('running', 0)
            self.current_obs = obs
            self.obs_history.append((t, obs))
            if driving > 0:
                self.current_driving = driving
                self.driving_history.append((t, driving))

        # fleet 크기 기억 (임계값 스케일용)
        if vehicle_stats and vehicle_stats.get('total', 0) > 0:
            self.current_fleet = vehicle_stats['total']

    def get_prediction(self) -> dict:
        """매크로 예측 결과"""
        return {
            'queue_10min': self._predict_queue_10min(),
            'tat': self._predict_tat(),
            'throughput': self._predict_throughput(),
            'deadlock_risk': self._predict_deadlock_risk(),
            'early_warning': self._predict_early_warning(),
            'obs_trend': self._get_obs_trend(),
            'current': {
                'queue': self.current_queue,
                'obs': self.current_obs,
                'driving': self.current_driving,
            },
        }

    def _predict_queue_10min(self) -> dict:
        """10분 후 큐 예측 — 이동평균 + 선형 외삽"""
        if len(self.queue_history) < 3:
            return {
                'value': self.current_queue,
                'confidence': 'low',
                'method': 'insufficient_data',
            }

        # 최근 10분 이동평균
        recent = list(self.queue_history)[-10:]
        avg = sum(v for _, v in recent) / len(recent)

        # 선형 추세 (최근 5분)
        if len(recent) >= 5:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            avg_first = sum(v for _, v in first_half) / len(first_half)
            avg_second = sum(v for _, v in second_half) / len(second_half)
            trend_per_min = (avg_second - avg_first) / (len(recent) / 2)

            predicted = avg + trend_per_min * 10  # 10분 외삽
        else:
            predicted = avg
            trend_per_min = 0

        # 범위 (±50)
        return {
            'value': round(predicted),
            'range_low': round(predicted - 50),
            'range_high': round(predicted + 50),
            'trend': round(trend_per_min, 1),
            'trend_direction': 'up' if trend_per_min > 5 else 'down' if trend_per_min < -5 else 'stable',
            'moving_avg': round(avg),
            'confidence': 'high' if len(recent) >= 8 else 'medium',
            'method': 'linear_extrapolation',
        }

    def _predict_tat(self) -> dict:
        """TAT 예측 — 상수 2.88분 (24시간 균일 확인됨)"""
        return {
            'average': TAT_AVERAGE_MIN,
            'median': TAT_MEDIAN_MIN,
            'prediction_10min': TAT_AVERAGE_MIN,  # 변동 없음
            'confidence': 'high',
            'method': 'constant (24h uniform)',
            'note': '큐 1000~1800 범위에서 TAT 2.4~2.6분으로 안정적',
        }

    def _predict_throughput(self) -> dict:
        """물동량 예측 — ~20,000건/시 균일"""
        return {
            'per_hour': THROUGHPUT_PER_HOUR,
            'per_minute': round(THROUGHPUT_PER_HOUR / 60),
            'confidence': 'high',
            'method': 'constant (day/night same)',
            'note': '주간 ~20,000건/시, 야간 ~19,700건/시',
        }

    def _predict_deadlock_risk(self) -> dict:
        """데드락 위험도 예측 — OBS 추세 기반"""
        if len(self.obs_history) < 3:
            return {
                'level': 'unknown',
                'obs_current': self.current_obs,
                'method': 'insufficient_data',
            }

        recent_obs = [v for _, v in list(self.obs_history)[-10:]]
        avg_obs = sum(recent_obs) / len(recent_obs)

        # 추세 분석
        if len(recent_obs) >= 5:
            first = sum(recent_obs[:len(recent_obs)//2]) / (len(recent_obs)//2)
            second = sum(recent_obs[len(recent_obs)//2:]) / (len(recent_obs) - len(recent_obs)//2)
            trend = second - first
        else:
            trend = 0

        # 위험 레벨 판단
        if avg_obs >= OBS_DEADLOCK_THRESHOLD:
            level = "CRITICAL"
            message = f"OBS {int(avg_obs)}대 — 데드락 발생 가능"
        elif avg_obs >= OBS_DANGER_THRESHOLD:
            level = "DANGER"
            message = f"OBS {int(avg_obs)}대 — 위험 수준, 빌드업 중"
        elif avg_obs >= OBS_WARNING_THRESHOLD:
            level = "WARNING"
            message = f"OBS {int(avg_obs)}대 — 주의 필요"
        elif trend > 10:
            level = "WATCH"
            message = f"OBS {int(avg_obs)}대 — 상승 추세 감지"
        else:
            level = "NORMAL"
            message = f"OBS {int(avg_obs)}대 — 정상 범위"

        return {
            'level': level,
            'message': message,
            'obs_current': self.current_obs,
            'obs_avg': round(avg_obs, 1),
            'obs_trend': round(trend, 1),
            'obs_trend_direction': 'rising' if trend > 5 else 'falling' if trend < -5 else 'stable',
            'thresholds': {
                'normal': f"{OBS_NORMAL_RANGE[0]}~{OBS_NORMAL_RANGE[1]}",
                'warning': str(OBS_WARNING_THRESHOLD),
                'danger': str(OBS_DANGER_THRESHOLD),
                'deadlock': str(OBS_DEADLOCK_THRESHOLD),
            },
        }

    def _get_obs_trend(self) -> List[dict]:
        """OBS 추이 (차트용)"""
        return [
            {'time': t.strftime("%H:%M"), 'value': v}
            for t, v in self.obs_history
        ]

    def _predict_early_warning(self) -> dict:
        """3단계 데드락 경보 (M16A_BR 룰 — R-A' + R-B + R-C' AND).

        검증 사례: 2026-04-21 14:00 데드락 체인 6대
        - 13:50 시점에 3단계 확정 발동 (10분 전)
        - 14:14 / 14:16 재발동 (진행 중 재확인)
        - 16:00 false positive 구간 완벽 차단
        """

        # R-A': 최근 10분창에 AVGTOTAL1MIN ≥ 9분 이 몇 회
        t1_values = [v for _, v in list(self.t1min_history)[-10:]]
        ra_count = sum(1 for v in t1_values if v >= 9.0)
        ra_latest = t1_values[-1] if t1_values else None
        ra_triggered = ra_count >= 1

        # ⚡ R-A' SUSTAINED (튜닝: S1 조기 인지용 — 6분이 5샘플 중 3회 이상)
        ra_sustained = False
        if len(t1_values) >= 5:
            ra_sustained = sum(1 for v in t1_values[-5:] if v >= 6.0) >= 3

        # R-B: M14→M16 30분 전 대비 +100 이상
        rb_diff = 0
        rb_triggered = False
        if len(self.m14_history) >= 31:
            rb_diff = self.m14_history[-1][1] - self.m14_history[-31][1]
            rb_triggered = rb_diff >= 100

        # ⚡ R-B FAST (튜닝: S2 조기 인지용 — 10분 +30)
        rb_fast = False
        if len(self.m14_history) >= 11:
            rb_diff_fast = self.m14_history[-1][1] - self.m14_history[-11][1]
            rb_fast = rb_diff_fast >= 30

        # R-C': 전체 리프터 합 20분 전 대비 감소 AND 역증가 2개+
        rc_trend = 0
        reverse_count = 0
        reverse_lids: List[str] = []
        rc_triggered = False
        if len(self.lft_history) >= 21:
            now_lft = self.lft_history[-1][1]
            prev_lft = self.lft_history[-21][1]
            rc_trend = sum(now_lft.values()) - sum(prev_lft.values())
            for lid in now_lft:
                if now_lft[lid] > prev_lft.get(lid, 0):
                    reverse_lids.append(lid)
                    reverse_count += 1
            rc_triggered = rc_trend < 0 and reverse_count >= 2

        # Stage 판정 (튜닝: S1/S2 조기 발동, S3 그대로 — 04-21 검증 보호)
        s1 = (ra_count >= 2) or ra_sustained         # 9분 2회 OR 6분 3회 연속
        s2 = rb_triggered or rb_fast                  # +100/30분 OR +30/10분
        s3 = ra_triggered and rb_triggered and rc_triggered  # 3단계 확정 (그대로)

        stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))

        labels = ['NORMAL', 'S1_EARLY', 'S2_WATCH', 'S3_CONFIRMED']
        messages = [
            '정상 운영 중',
            '조기경보: 반송시간 급증 감지',
            '주의보: FAB 간 큐 누적',
            '⚠️ 확정: 데드락 10분 전 임박',
        ]
        guides = [
            '',
            '대시보드 확인 빈도 상향',
            'FAB 간 물동량 점검, 관리자 알림',
            '즉시 MLUD port capa 조정 검토 / SFA 리프터 상태 확인 / 10분 내 데드락 방지 윈도우',
        ]

        return {
            'stage': stage,
            'label': labels[stage],
            'message': messages[stage],
            'guide': guides[stage],
            'rules': {
                'R-A_prime': {
                    'triggered': ra_triggered,
                    'count': ra_count,
                    'value': round(ra_latest, 2) if ra_latest is not None else None,
                    'sustained': ra_sustained,
                    'threshold': '≥9분 1회+ / S1: 9분 2회+ 또는 6분 3회연속 (튜닝)',
                },
                'R-B': {
                    'triggered': rb_triggered,
                    'diff_30min': rb_diff,
                    'fast': rb_fast,
                    'threshold': '+100/30분 또는 +30/10분 (튜닝)',
                },
                'R-C_prime': {
                    'triggered': rc_triggered,
                    'reverse_count': reverse_count,
                    'reverse_lids': reverse_lids,
                    'trend': rc_trend,
                    'threshold': '합 감소 + 역증가 2개+',
                },
            },
            'history': self.full_timeline,  # 사전 계산된 전체 타임라인 (replay 시각과 무관)
        }

    def precompute_timeline(self, star_timeline: List[Tuple[datetime, dict]]) -> None:
        """전체 star_timeline 을 미리 스캔해 경보 진행 타임라인 고정 생성.

        데이터 로드 직후 1회 호출. replay 재생 시각과 무관하게 동일 결과 반환.
        단계 전환 시점만 기록 (같은 단계 연속 유지는 무시, 하락 후 재발동은 기록).
        """
        self.full_timeline = []
        if not star_timeline:
            return

        t1_hist: List[float] = []
        m14_hist: List[int] = []
        lft_hist: List[dict] = []
        last_logged_stage = -1  # -1 = 아직 없음 (0 도 기록하기 위해)
        last_s3_time: Optional[datetime] = None

        for t, star in star_timeline:
            t1 = star.get('avgtotal1min')
            m14 = star.get('m14_to_m16')
            lft = star.get('lft_list')

            if t1 is not None:
                t1_hist.append(float(t1))
            if m14 is not None and m14 > 0:
                m14_hist.append(int(m14))
            if isinstance(lft, dict) and lft:
                lft_hist.append(dict(lft))

            # R-A'
            recent_t1 = t1_hist[-10:]
            ra_count = sum(1 for v in recent_t1 if v >= 9.0)
            ra_value = recent_t1[-1] if recent_t1 else None
            ra_trig = ra_count >= 1

            # ⚡ R-A' SUSTAINED (튜닝)
            ra_sustained = False
            if len(recent_t1) >= 5:
                ra_sustained = sum(1 for v in recent_t1[-5:] if v >= 6.0) >= 3

            # R-B
            rb_diff = 0
            rb_trig = False
            if len(m14_hist) >= 31:
                rb_diff = m14_hist[-1] - m14_hist[-31]
                rb_trig = rb_diff >= 100

            # ⚡ R-B FAST (튜닝)
            rb_fast = False
            if len(m14_hist) >= 11:
                rb_fast = (m14_hist[-1] - m14_hist[-11]) >= 30

            # R-C'
            rc_trend = 0
            rev_count = 0
            rev_lids: List[str] = []
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

            s1 = (ra_count >= 2) or ra_sustained
            s2 = rb_trig or rb_fast
            s3 = ra_trig and rb_trig and rc_trig
            stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))

            # 기록 규칙:
            #  (a) 단계 상승 (last → stage, stage > last) → 기록
            #  (b) stage==3 유지 중이나 직전 S3 와 10분 이상 차이 → 재발동 기록
            #  (c) stage==0 으로 떨어질 때 정상화 기록 (최초 한 번)
            record = False
            reason = ''
            if stage > last_logged_stage and stage > 0:
                record = True
                if stage == 1:
                    if ra_count >= 2:
                        reason = f'1MIN ≥9분이 {ra_count}회'
                    elif ra_sustained:
                        reason = '1MIN ≥6분 지속 (튜닝 조기 인지)'
                    else:
                        reason = '1단계 조기경보'
                elif stage == 2:
                    if rb_trig:
                        reason = f'M14→M16 +{rb_diff} (30분간)'
                    elif rb_fast:
                        rb_d10 = m14_hist[-1] - m14_hist[-11] if len(m14_hist) >= 11 else 0
                        reason = f'M14→M16 +{rb_d10} (10분간, 튜닝)'
                    else:
                        reason = '2단계 주의보'
                elif stage == 3:
                    reason = f'R-A+R-B+R-C AND 만족 (1MIN {ra_value:.2f}, M14→M16 +{rb_diff}, 역증가 {rev_count}개)'
                last_logged_stage = stage
                if stage == 3:
                    last_s3_time = t
            elif stage == 3 and last_s3_time is not None:
                diff_min = (t - last_s3_time).total_seconds() / 60.0
                if diff_min >= 10:
                    record = True
                    reason = f'재발동 (진행 중 재확인, 역증가 {rev_count}개)'
                    last_s3_time = t
            elif stage == 0 and last_logged_stage >= 1:
                record = True
                reason = '정상화 완료'
                last_logged_stage = 0
                last_s3_time = None

            if record:
                self.full_timeline.append({
                    'time': t.strftime('%H:%M'),
                    'stage': stage,
                    'label': ['NORMAL', 'S1_EARLY', 'S2_WATCH', 'S3_CONFIRMED'][stage],
                    'reason': reason,
                })

    def get_correlations(self) -> dict:
        """데이터 간 상관관계 (MD 문서 검증 결과 반영)"""
        # 큐↔OBS 상관계수
        queue_obs_r = self._calc_correlation(self.queue_obs_pairs) if len(self.queue_obs_pairs) > 10 else None

        return {
            'queue_tat': {
                'correlation': 'weak_positive',
                'r_value': 0.12,
                'description': '큐가 올라가면 TAT도 약간 증가 (2.4~2.6분 범위)',
                'data_source': 'OHT_데이터_연관관계_월드모델_예측.md',
                'detail': [
                    {'queue_range': '1000~1200', 'tat': 2.49},
                    {'queue_range': '1200~1400', 'tat': 2.53},
                    {'queue_range': '1600~1800', 'tat': 2.62},
                ],
            },
            'queue_obs': {
                'correlation': 'none',
                'r_value': round(queue_obs_r, 3) if queue_obs_r else 0.03,
                'description': 'OBS 정지는 큐와 거의 무관. 물리적 구간 혼잡이 원인',
                'detail': [
                    {'queue_range': '<1300', 'obs': 154},
                    {'queue_range': '>1600', 'obs': 157},
                ],
            },
            'queue_hid_speed': {
                'correlation': 'none',
                'r_value': -0.01,
                'description': '큐가 올라가도 HID 구간 속도 변화 없음. 물리적 구조에 의해 결정',
                'detail': [
                    {'queue_range': '1000~1200', 'avg_speed': 92.0},
                    {'queue_range': '1200~1400', 'avg_speed': 91.9},
                ],
            },
            'time_tat': {
                'correlation': 'none',
                'description': 'TAT는 24시간 균일 (~2.88분)',
            },
            'time_throughput': {
                'correlation': 'none',
                'description': '물동량은 24시간 균일 (~20,000건/시)',
            },
        }

    def _calc_correlation(self, pairs: List[Tuple]) -> float:
        """피어슨 상관계수 계산"""
        n = len(pairs)
        if n < 3:
            return 0.0

        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

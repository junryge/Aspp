#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 예측 결과 — API Receiver 로 샘플 데이터 전송 테스터

목적:
  api_receiver.py 가 hybrid CSV 한 줄 받을 수 있는지 확인.
  실제 데이터 아닌 **샘플 6개 시나리오** 만들어서 전송.

시나리오:
  1) 정상            — 모든 룰 OFF, risk_level=정상
  2) 관심            — 단독 약신호
  3) 주의            — S2 + ML 약
  4) 경보            — ML 단독강 (룰 OFF + ML강)
  5) 위험-예측  ★    — S2 + ML강 → 30분내 S3 진행예상
  6) 위험-확정  ★★   — S3 발동 (사건진행)

실행:
  python sample_sender.py                              # 기본 localhost:9100
  python sample_sender.py http://localhost:9200        # 다른 포트
  python sample_sender.py http://192.168.0.10:9100     # 다른 서버
  python sample_sender.py --interval 5                 # 5초 간격 전송
  python sample_sender.py --once                       # 1회만 (테스트용)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    sys.exit("❌ requests 모듈 필요: pip install requests")


DEFAULT_URL = "http://localhost:9100"
TAG = "hybrid"  # POST /api/receive/<tag>


def make_sample(level: str, dt: datetime) -> dict:
    """level 별 샘플 hybrid 데이터 1행"""
    base = {
        "datetime":       dt.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_for": (dt + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
        # 원천값
        "avgtotal1min":   3.5,
        "m14_to_m16":     150,
        "fabstorage_ratio": 1.2,
        "hub_storage_util": 95.0,
        "inflow_total":   200,
        # 룰 판정
        "rule_s1": 0, "rule_s2": 0, "rule_s3": 0,
        "ra_value": 3.5, "ra_trig": 0, "ra_sustained": False,
        "rb_diff": 5,    "rb_trig": 0, "rb_fast": False,
        "rc_trend": 0,   "rev_count": 0, "rc_trig": 0,
        "rd_fabstorage": 1.2, "rd_trig": 0,
        "re_trig": 0, "rf_trig": 0, "rf_fast": 0,
        # 룰 이벤트
        "stage": 0, "stage_name": "이벤트없음",
        "rule_reason": "", "rule_relation": "",
        # 위험도
        "risk_score": 0, "risk_level": "정상", "risk_factors": "",
        # ML
        "ml_score": 0.05, "ml_level": "OK", "ml_level_kr": "정상",
        # 융합
        "final_level": "정상", "agreement": "none",
        "direction": "-",      "final_reason": "",
    }

    if level == "정상":
        return base

    if level == "관심":
        base.update({
            "rev_count": 3, "rc_trend": 5,
            "risk_score": 10, "risk_level": "관심", "risk_factors": "rev=3(준위험)",
            "ml_score": 0.18, "ml_level": "OK", "ml_level_kr": "정상",
            "final_level": "관심", "agreement": "none", "direction": "-",
            "final_reason": "리프터 쏠림 초기",
        })
        return base

    if level == "주의":
        base.update({
            "rule_s2": 1, "rb_diff": 80, "rb_fast": True, "rb_diff_10": 35,
            "stage": 2, "stage_name": "2단계 주의보",
            "rule_reason": "M14→M16 +35 (10분간 fast)",
            "risk_score": 30, "risk_level": "주의",
            "risk_factors": "rb_fast;rb=+80",
            "ml_score": 0.42, "ml_level": "INFO", "ml_level_kr": "주의",
            "final_level": "주의", "agreement": "both",
            "direction": "sync(양합치)",
            "final_reason": "S2 일반 (ml 약 0.42)",
        })
        return base

    if level == "경보":
        base.update({
            "risk_score": 25, "risk_level": "주의",
            "risk_factors": "rev=3(준위험)",
            "ml_score": 0.88, "ml_level": "CRITICAL", "ml_level_kr": "위험",
            "final_level": "경보", "agreement": "ml_only",
            "direction": "ml→rule(예측)",
            "final_reason": "ML 단독강 → 룰 곧 발동가능 ml=0.88",
        })
        return base

    if level == "위험-예측":
        base.update({
            "avgtotal1min": 7.85,
            "rule_s2": 1, "rb_diff": 118, "rb_trig": 1,
            "rb_fast": True, "rb_diff_10": 42,
            "rev_count": 3, "rc_trend": -8, "rc_trig": 1,
            "rev_lids": "6ABL6011;6ABL6022;6ABL0121",
            "stage": 2, "stage_name": "2단계 주의보", "transition": "1→2",
            "rule_reason": "M14→M16 +118 (30분간)",
            "rule_relation": "[R-A' Y] T1MIN=7.85분 | [R-B Y] +118/30분 | [R-C' Y] 역증가 3개",
            "risk_score": 70, "risk_level": "매우위험",
            "risk_factors": "ra=7.9분;rb_fast;rb=+118;rc_rev=3",
            "ml_score": 0.8742, "ml_level": "CRITICAL", "ml_level_kr": "위험",
            "final_level": "위험-예측", "agreement": "both",
            "direction": "sync(양합치)",
            "final_reason": "★ S2+ML강 → 30분내 S3 진행예상 ml=0.87 | risk=70(매우위험)",
        })
        return base

    if level == "위험-확정":
        base.update({
            "avgtotal1min": 12.5,
            "rule_s1": 1, "rule_s2": 1, "rule_s3": 1,
            "ra_value": 12.5, "ra_count": 5, "ra_trig": 1, "ra_sustained": True,
            "rb_diff": 162, "rb_trig": 1, "rb_fast": True, "rb_diff_10": 53,
            "rev_count": 5, "rc_trend": -15, "rc_trig": 1,
            "rev_lids": "6ABL6011;6ABL6022;6ABL0121;6ABL0122;6ABL0112",
            "rd_fabstorage": 28.5, "rd_trig": 1,
            "stage": 3, "stage_name": "3단계 ⭐확정", "transition": "2→3",
            "rule_reason": "AND 만족 (1MIN 12.5, M14→M16 +162, 역증가 5개)",
            "rule_relation": "[R-A' Y] T1MIN=12.5분 | [R-B Y] +162/30분 | [R-C' Y] 역증가 5개 | [R-D Y] FAB 28.5%",
            "risk_score": 140, "risk_level": "매우위험",
            "risk_factors": "ra_sustained;ra_count=5;ra=12.5분;rb_fast;rb=+162;rc_rev=5;rd=29%",
            "ml_score": 0.9912, "ml_level": "CRITICAL", "ml_level_kr": "위험",
            "final_level": "위험-확정", "agreement": "both",
            "direction": "sync(양합치)",
            "final_reason": "S3 발동, 이미 사건진행 ml=0.99 | risk=140(매우위험)",
        })
        return base

    return base


SCENARIOS = ["정상", "관심", "주의", "경보", "위험-예측", "위험-확정"]


def send_one(url: str, payload: dict) -> bool:
    """1건 전송. 성공시 True"""
    full_url = f"{url.rstrip('/')}/api/receive/{TAG}"
    try:
        r = requests.post(full_url, json=payload, timeout=5)
        ok = r.status_code == 200
        body = r.json() if ok else r.text[:200]
        lvl = payload.get("final_level", "?")
        score = payload.get("risk_score", 0)
        ml = payload.get("ml_score", 0)
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{payload['datetime']}] {lvl:<10} risk={score:>3} ml={ml:.2f} → {r.status_code} {body}")
        return ok
    except Exception as e:
        print(f"  ✗ 전송 실패: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url", nargs="?", default=DEFAULT_URL,
                   help=f"API receiver URL (기본 {DEFAULT_URL})")
    p.add_argument("--interval", type=int, default=2,
                   help="시나리오 간격 초 (기본 2)")
    p.add_argument("--once", action="store_true",
                   help="6 시나리오 1회만 보내고 종료")
    p.add_argument("--loop", action="store_true",
                   help="시나리오 무한 반복")
    p.add_argument("--scenario", choices=SCENARIOS,
                   help="특정 시나리오 1개만 1회 전송")
    args = p.parse_args()

    print("=" * 60)
    print("  Hybrid → API Receiver 샘플 전송")
    print("=" * 60)
    print(f"  대상 URL: {args.url}")
    print(f"  엔드포인트: POST /api/receive/{TAG}")
    print(f"  간격: {args.interval}초")
    print("=" * 60)

    # 헬스 체크
    try:
        h = requests.get(f"{args.url.rstrip('/')}/api/health", timeout=3)
        if h.status_code == 200:
            print(f"  ✓ 서버 응답 정상: {h.json()}")
        else:
            print(f"  ⚠ /api/health 응답 코드 {h.status_code}")
    except Exception as e:
        print(f"  ⚠ 서버 헬스 체크 실패: {e}")
        print(f"  계속 진행은 가능하지만 서버가 안 켜져있을 수 있음")
    print()

    # 단일 시나리오
    if args.scenario:
        payload = make_sample(args.scenario, datetime.now())
        send_one(args.url, payload)
        return

    # 시나리오 순차 전송
    rounds = 1
    while True:
        print(f"━━━ Round {rounds} — 6 시나리오 전송 ━━━")
        ok_cnt = 0
        for i, lvl in enumerate(SCENARIOS):
            t = datetime.now() + timedelta(minutes=i)  # 시각 다르게
            payload = make_sample(lvl, t)
            if send_one(args.url, payload):
                ok_cnt += 1
            time.sleep(args.interval)
        print(f"  → {ok_cnt}/{len(SCENARIOS)} 성공\n")

        if args.once:
            break
        if not args.loop:
            break
        rounds += 1

    print("종료")


if __name__ == "__main__":
    main()

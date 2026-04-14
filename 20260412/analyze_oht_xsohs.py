#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OHT XSOHS 데이터 분석 스크립트
- CSV 파일을 읽어 종합 분석 리포트(MD)를 생성
- Usage: python analyze_oht_xsohs.py [csv_path] [output_report_path]
"""

import csv
import sys
import statistics
from collections import Counter, defaultdict
from datetime import datetime

# ============================================================
# Enum 매핑 (OHT2/layout/real_oht_parser.py 기준)
# ============================================================
VHL_STATE_MAP = {
    "1": "RUN", "2": "STOP", "3": "ABNORMAL", "4": "MANUAL",
    "5": "REMOVING", "6": "OBS_BZ_STOP", "7": "JAM",
    "8": "HT_STOP", "9": "E84_TIMEOUT",
}
RUN_CYCLE_MAP = {
    "0": "NONE", "1": "POSITION_DETECT", "2": "MOVING",
    "3": "ACQUIRE", "4": "DEPOSIT", "5": "SAMPLING",
    "9": "FLOOR_TRANS", "21": "WHEELDRIVE", "22": "MANUAL_CONTROL",
    "23": "DRIVE_TEACHING", "24": "TRANS_TEACHING",
    "25": "WHEELDRIVE_25",
}
VHL_CYCLE_MAP = {
    "0": "NONE", "1": "MOVING", "2": "ACQUIRE_MOVING",
    "3": "ACQUIRING", "4": "DEPOSIT_MOVING", "5": "DEPOSITING",
    "6": "MAINT_MOVING", "7": "WAITING", "8": "INPUT",
}
VHL_DET_STATE_MAP = {
    "0": "NONE", "1": "WAIT", "2": "STAGE_WAIT",
    "3": "STANDBY_WAIT", "4": "DEPOSIT_SIG_WAIT", "5": "ACQ_WAIT",
    "6": "MAP_WAIT", "101": "MOVING", "102": "PARKING_UTS_MOVING",
    "103": "STAGE_MOVING", "104": "STANDBY_MOVING",
    "105": "BALANCE_MOVING", "106": "PARKING_MOVING",
}

# ============================================================
# 데이터 로드 및 파싱
# ============================================================
def load_and_parse(csv_path):
    """CSV 파일을 읽어 메시지 타입별로 분류하여 반환"""
    type1, type2, type3, type4 = [], [], [], []
    total = 0
    first_time = None
    last_time = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            total += 1
            time_str = row[3]
            if first_time is None:
                first_time = time_str
            last_time = time_str

            if len(row) <= 4:
                continue
            parts = row[4].split(',')
            msg_type = parts[0]

            if msg_type == '2' and len(parts) == 21:
                type2.append({'time': time_str, 'fields': parts})
            elif msg_type == '3' and len(parts) == 8:
                type3.append({'time': time_str, 'fields': parts})
            elif msg_type == '1':
                type1.append({'time': time_str, 'fields': parts})
            elif msg_type == '4':
                type4.append({'time': time_str, 'fields': parts})

    return {
        'total': total,
        'first_time': first_time, 'last_time': last_time,
        'type1': type1, 'type2': type2, 'type3': type3, 'type4': type4,
    }


def parse_time(t):
    try:
        return datetime.strptime(t[:23], '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        return None


# ============================================================
# 분석 함수들
# ============================================================
def analyze_overview(data):
    lines = []
    lines.append("## 1. 데이터 개요\n")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 총 레코드 수 | {data['total']:,} |")
    lines.append(f"| 시간 범위 | {data['last_time']} ~ {data['first_time']} |")
    lines.append(f"| Type 1 (시스템 상태) | {len(data['type1']):,} ({len(data['type1'])/data['total']*100:.1f}%) |")
    lines.append(f"| Type 2 (VHL 상태 보고) | {len(data['type2']):,} ({len(data['type2'])/data['total']*100:.1f}%) |")
    lines.append(f"| Type 3 (IN SERVICE) | {len(data['type3']):,} ({len(data['type3'])/data['total']*100:.1f}%) |")
    lines.append(f"| Type 4 (MTL) | {len(data['type4']):,} ({len(data['type4'])/data['total']*100:.1f}%) |")
    lines.append("")
    return '\n'.join(lines)


def analyze_fleet(type2):
    v_set = set()
    r_set = set()
    hourly_active = defaultdict(set)

    for rec in type2:
        vid = rec['fields'][2]
        if vid.startswith('V'):
            v_set.add(vid)
        elif vid.startswith('R'):
            r_set.add(vid)
        try:
            hour = rec['time'].split(' ')[1].split(':')[0]
            hourly_active[hour].add(vid)
        except Exception:
            pass

    lines = []
    lines.append("## 2. Fleet 현황\n")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|------|-----|")
    lines.append(f"| V-Vehicle (운반차량) | {len(v_set):,}대 |")
    lines.append(f"| R-Vehicle (예비차량) | {len(r_set):,}대 |")
    lines.append(f"| 총 차량 | {len(v_set) + len(r_set):,}대 |")
    lines.append("")
    lines.append("### 시간대별 활성 차량 수\n")
    lines.append("| 시간 | 활성 차량 수 |")
    lines.append("|------|------------|")
    for h in sorted(hourly_active.keys()):
        lines.append(f"| {h}:00 | {len(hourly_active[h]):,} |")
    lines.append("")
    return '\n'.join(lines)


def analyze_states(type2):
    state_cnt = Counter()
    run_cycle_cnt = Counter()
    vhl_cycle_cnt = Counter()
    det_state_cnt = Counter()
    combo_cnt = Counter()

    for rec in type2:
        f = rec['fields']
        vid = f[2]
        if not vid.startswith('V'):
            continue
        st = f[3]
        rc = f[10]
        vc = f[11]
        ds = f[19]
        state_cnt[st] += 1
        run_cycle_cnt[rc] += 1
        vhl_cycle_cnt[vc] += 1
        det_state_cnt[ds] += 1
        combo_cnt[f"State={st}/RC={rc}/VC={vc}"] += 1

    total = sum(state_cnt.values())
    lines = []
    lines.append("## 3. 차량 상태 분석 (V-Vehicle 기준)\n")

    lines.append("### State 분포\n")
    lines.append("| State | 이름 | 건수 | 비율 |")
    lines.append("|-------|------|------|------|")
    for st, cnt in state_cnt.most_common():
        name = VHL_STATE_MAP.get(st, f"UNKNOWN({st})")
        lines.append(f"| {st} | {name} | {cnt:,} | {cnt/total*100:.1f}% |")

    lines.append("\n### RunCycle 분포\n")
    lines.append("| RunCycle | 이름 | 건수 | 비율 |")
    lines.append("|----------|------|------|------|")
    for rc, cnt in run_cycle_cnt.most_common():
        name = RUN_CYCLE_MAP.get(rc, f"UNKNOWN({rc})")
        lines.append(f"| {rc} | {name} | {cnt:,} | {cnt/total*100:.1f}% |")

    lines.append("\n### VhlCycle 분포\n")
    lines.append("| VhlCycle | 이름 | 건수 | 비율 |")
    lines.append("|----------|------|------|------|")
    for vc, cnt in vhl_cycle_cnt.most_common():
        name = VHL_CYCLE_MAP.get(vc, f"UNKNOWN({vc})")
        lines.append(f"| {vc} | {name} | {cnt:,} | {cnt/total*100:.1f}% |")

    lines.append("\n### DetailState 분포\n")
    lines.append("| DetailState | 이름 | 건수 | 비율 |")
    lines.append("|-------------|------|------|------|")
    for ds, cnt in det_state_cnt.most_common():
        name = VHL_DET_STATE_MAP.get(ds, f"UNKNOWN({ds})")
        lines.append(f"| {ds} | {name} | {cnt:,} | {cnt/total*100:.1f}% |")

    lines.append("\n### 주요 상태 조합 (Top 15)\n")
    lines.append("| 조합 | 건수 | 비율 |")
    lines.append("|------|------|------|")
    for combo, cnt in combo_cnt.most_common(15):
        lines.append(f"| {combo} | {cnt:,} | {cnt/total*100:.1f}% |")
    lines.append("")
    return '\n'.join(lines)


def analyze_utilization(type2):
    idle_cnt = Counter()
    active_cnt = Counter()
    loaded_cnt = Counter()
    total_cnt = Counter()

    for rec in type2:
        f = rec['fields']
        vid = f[2]
        if not vid.startswith('V'):
            continue
        total_cnt[vid] += 1
        if f[19] in ('101', '102', '103', '104', '105', '106'):
            idle_cnt[vid] += 1
        else:
            active_cnt[vid] += 1
        if f[4] == '1':
            loaded_cnt[vid] += 1

    utils = []
    loaded_ratios = []
    for vid in total_cnt:
        t = total_cnt[vid]
        a = active_cnt.get(vid, 0)
        l = loaded_cnt.get(vid, 0)
        utils.append(a / t * 100 if t > 0 else 0)
        loaded_ratios.append(l / t * 100 if t > 0 else 0)

    lines = []
    lines.append("## 4. 가동률 / 적재율 분석\n")
    lines.append("### 가동률 (Active 비율)\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    if utils:
        lines.append(f"| 평균 | {statistics.mean(utils):.1f}% |")
        lines.append(f"| 중앙값 | {statistics.median(utils):.1f}% |")
        lines.append(f"| 최소 | {min(utils):.1f}% |")
        lines.append(f"| 최대 | {max(utils):.1f}% |")
        lines.append(f"| 표준편차 | {statistics.stdev(utils):.1f}% |")

    lines.append("\n### 가동률 분포\n")
    lines.append("| 구간 | 차량 수 |")
    lines.append("|------|--------|")
    brackets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90), (90, 100.1)]
    for lo, hi in brackets:
        cnt = sum(1 for u in utils if lo <= u < hi)
        lines.append(f"| {lo:.0f}~{hi:.0f}% | {cnt} |")

    lines.append("\n### 적재율 (Loaded 비율)\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    if loaded_ratios:
        lines.append(f"| 평균 | {statistics.mean(loaded_ratios):.1f}% |")
        lines.append(f"| 중앙값 | {statistics.median(loaded_ratios):.1f}% |")
        lines.append(f"| 최소 | {min(loaded_ratios):.1f}% |")
        lines.append(f"| 최대 | {max(loaded_ratios):.1f}% |")
    lines.append("")
    return '\n'.join(lines)


def analyze_speed(type2):
    speed_cnt = Counter()
    speed_by_state = defaultdict(Counter)

    for rec in type2:
        f = rec['fields']
        if not f[2].startswith('V'):
            continue
        spd = f[18]
        st = f[3]
        speed_cnt[spd] += 1
        speed_by_state[st][spd] += 1

    total = sum(speed_cnt.values())
    lines = []
    lines.append("## 5. 속도 분석\n")
    lines.append("### 전체 속도 분포\n")
    lines.append("| 속도 | 건수 | 비율 |")
    lines.append("|------|------|------|")
    for spd, cnt in sorted(speed_cnt.items(), key=lambda x: -x[1]):
        lines.append(f"| {spd} | {cnt:,} | {cnt/total*100:.1f}% |")

    lines.append("\n### State별 속도 분포\n")
    lines.append("| State | 이름 | 속도=0 | 속도=50 | 속도=70 | 속도=80 | 속도=90 | 속도=99 |")
    lines.append("|-------|------|--------|---------|---------|---------|---------|---------|")
    for st in sorted(speed_by_state.keys()):
        name = VHL_STATE_MAP.get(st, "?")
        s = speed_by_state[st]
        t = sum(s.values())
        lines.append(f"| {st} | {name} | {s.get('0',0)/t*100:.0f}% | {s.get('50',0)/t*100:.0f}% | {s.get('70',0)/t*100:.0f}% | {s.get('80',0)/t*100:.0f}% | {s.get('90',0)/t*100:.0f}% | {s.get('99',0)/t*100:.0f}% |")
    lines.append("")
    return '\n'.join(lines)


def analyze_cycles(type2):
    """운반 사이클 분석 - 전체 V-vehicle 대상"""
    vehicle_timeline = defaultdict(list)
    for rec in type2:
        f = rec['fields']
        vid = f[2]
        if vid.startswith('V'):
            vehicle_timeline[vid].append(rec)

    # Reverse (파일이 역순)
    for vid in vehicle_timeline:
        vehicle_timeline[vid].reverse()

    cycle_times = []
    pickup_times = []  # dispatch -> pickup
    transport_times = []  # pickup -> deliver
    load_unload_events = []

    for vid in vehicle_timeline:
        events = vehicle_timeline[vid]
        cycle_start = None
        pickup_time = None
        prev_rc = None
        prev_time = None
        prev_loaded = None

        for ev in events:
            f = ev['fields']
            cat = f[10]
            sub = f[11]
            loaded = f[4]
            rc = f[10]
            t = parse_time(ev['time'])
            if t is None:
                continue

            # Load 이벤트 감지: loaded가 0->1로 변할 때
            if prev_loaded == '0' and loaded == '1' and prev_time:
                # ACQUIRE(3) -> 다음 상태 전환까지의 시간
                pass

            # RunCycle 3(ACQUIRE) 또는 4(DEPOSIT) 구간 측정
            if prev_rc in ('3', '4') and rc != prev_rc and prev_time:
                dt = (t - prev_time).total_seconds()
                if 1 < dt < 120:
                    load_unload_events.append({'type': 'ACQUIRE' if prev_rc == '3' else 'DEPOSIT', 'sec': dt})

            # 사이클 분석
            if cat == '3' and sub == '2' and loaded == '0' and cycle_start is None:
                cycle_start = t

            if cat == '4' and sub == '4' and loaded == '1' and cycle_start and pickup_time is None:
                pickup_time = t
                pickup_times.append((pickup_time - cycle_start).total_seconds())

            if cat in ('2', '3') and loaded == '0' and cycle_start and pickup_time:
                end = t
                total_sec = (end - cycle_start).total_seconds()
                if 10 < total_sec < 1800:
                    cycle_times.append(total_sec)
                    transport_times.append((end - pickup_time).total_seconds())
                cycle_start = None
                pickup_time = None

            prev_rc = rc
            prev_time = t
            prev_loaded = loaded

    lines = []
    lines.append("## 6. 운반 사이클 분석\n")

    lines.append("### 전체 사이클 타임\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    if cycle_times:
        lines.append(f"| 분석된 사이클 수 | {len(cycle_times):,} |")
        lines.append(f"| 평균 | {statistics.mean(cycle_times):.1f}초 ({statistics.mean(cycle_times)/60:.1f}분) |")
        lines.append(f"| 중앙값 | {statistics.median(cycle_times):.1f}초 ({statistics.median(cycle_times)/60:.1f}분) |")
        lines.append(f"| 표준편차 | {statistics.stdev(cycle_times):.1f}초 |")
        lines.append(f"| 최소 | {min(cycle_times):.1f}초 |")
        lines.append(f"| 최대 | {max(cycle_times):.1f}초 |")

    lines.append("\n### 사이클 타임 분포\n")
    lines.append("| 구간 | 건수 | 비율 |")
    lines.append("|------|------|------|")
    brackets = [(0, 60), (60, 120), (120, 180), (180, 300), (300, 600), (600, 900), (900, 1800)]
    for lo, hi in brackets:
        cnt = sum(1 for t in cycle_times if lo <= t < hi)
        pct = cnt / len(cycle_times) * 100 if cycle_times else 0
        lines.append(f"| {lo//60}~{hi//60}분 | {cnt:,} | {pct:.1f}% |")

    lines.append("\n### 구간별 소요시간\n")
    lines.append("| 구간 | 평균 | 중앙값 |")
    lines.append("|------|------|--------|")
    if pickup_times:
        valid_p = [t for t in pickup_times if 5 < t < 1200]
        if valid_p:
            lines.append(f"| Dispatch→Pickup (빈차 이동) | {statistics.mean(valid_p):.1f}초 | {statistics.median(valid_p):.1f}초 |")
    if transport_times:
        valid_t = [t for t in transport_times if 5 < t < 1200]
        if valid_t:
            lines.append(f"| Pickup→Deliver (적재 운반) | {statistics.mean(valid_t):.1f}초 | {statistics.median(valid_t):.1f}초 |")

    lines.append("\n### Load/Unload 소요시간 (RunCycle 기준)\n")
    lines.append("| 유형 | 건수 | 평균 | 중앙값 | 최소 | 최대 |")
    lines.append("|------|------|------|--------|------|------|")
    for lu_type in ('ACQUIRE', 'DEPOSIT'):
        vals = [e['sec'] for e in load_unload_events if e['type'] == lu_type]
        if vals:
            lines.append(f"| {lu_type} | {len(vals):,} | {statistics.mean(vals):.1f}초 | {statistics.median(vals):.1f}초 | {min(vals):.1f}초 | {max(vals):.1f}초 |")
    lines.append("")
    return '\n'.join(lines)


def analyze_stations(type2):
    from_cnt = Counter()
    to_cnt = Counter()
    od_cnt = Counter()

    for rec in type2:
        f = rec['fields']
        if not f[2].startswith('V'):
            continue
        fr = f[16]
        to = f[17]
        if fr:
            from_cnt[fr] += 1
        if to:
            to_cnt[to] += 1
        if fr and to:
            od_cnt[(fr, to)] += 1

    # Zone 분류
    zone_cnt = Counter()
    all_stations = set(from_cnt.keys()) | set(to_cnt.keys())
    for s in all_stations:
        if s.startswith('4'):
            i = 1
            while i < len(s) and s[i].isalpha():
                i += 1
            zone_cnt[s[1:i]] += 1

    lines = []
    lines.append("## 7. 스테이션 분석\n")
    lines.append(f"- 고유 출발 스테이션: {len(from_cnt):,}개")
    lines.append(f"- 고유 도착 스테이션: {len(to_cnt):,}개")
    lines.append(f"- 고유 OD 쌍: {len(od_cnt):,}개\n")

    lines.append("### Top 20 출발(From) 스테이션\n")
    lines.append("| 스테이션 | 건수 |")
    lines.append("|----------|------|")
    for st, cnt in from_cnt.most_common(20):
        lines.append(f"| {st} | {cnt:,} |")

    lines.append("\n### Top 20 도착(To) 스테이션\n")
    lines.append("| 스테이션 | 건수 |")
    lines.append("|----------|------|")
    for st, cnt in to_cnt.most_common(20):
        lines.append(f"| {st} | {cnt:,} |")

    lines.append("\n### Top 20 OD 쌍 (출발→도착)\n")
    lines.append("| 출발 | 도착 | 건수 |")
    lines.append("|------|------|------|")
    for (fr, to), cnt in od_cnt.most_common(20):
        lines.append(f"| {fr} | {to} | {cnt:,} |")

    lines.append("\n### 영역(Zone) 분류 (Top 20)\n")
    lines.append("| Zone 접두어 | 스테이션 수 |")
    lines.append("|------------|-----------|")
    for z, cnt in zone_cnt.most_common(20):
        lines.append(f"| {z} | {cnt:,} |")
    lines.append("")
    return '\n'.join(lines)


def analyze_temporal(data):
    hourly = Counter()
    ten_min = Counter()

    all_records = data['type2'] + data['type3'] + data['type1'] + data['type4']
    for rec in all_records:
        try:
            parts = rec['time'].split(' ')[1].split(':')
            h = parts[0]
            m = int(parts[1])
            hourly[h] += 1
            ten_min[f"{h}:{m//10}0"] += 1
        except Exception:
            pass

    lines = []
    lines.append("## 8. 시간대별 패턴\n")
    lines.append("### 시간별 이벤트 수\n")
    lines.append("| 시간 | 이벤트 수 | 그래프 |")
    lines.append("|------|----------|--------|")
    max_val = max(hourly.values()) if hourly else 1
    for h in sorted(hourly.keys()):
        cnt = hourly[h]
        bar_len = int(cnt / max_val * 30)
        bar = '█' * bar_len
        lines.append(f"| {h}:00 | {cnt:,} | {bar} |")

    lines.append("\n### 10분 단위 이벤트 수 (Top 20 피크)\n")
    lines.append("| 시간대 | 이벤트 수 |")
    lines.append("|--------|----------|")
    for tm, cnt in ten_min.most_common(20):
        lines.append(f"| {tm} | {cnt:,} |")
    lines.append("")
    return '\n'.join(lines)


def analyze_anomalies(type2):
    anomaly_cnt = Counter()
    anomaly_vehicles = defaultdict(set)
    speed_zero_cnt = 0
    speed_zero_by_hour = Counter()

    for rec in type2:
        f = rec['fields']
        vid = f[2]
        if not vid.startswith('V'):
            continue
        st = f[3]
        if st in ('6', '7', '8', '9'):
            name = VHL_STATE_MAP.get(st, st)
            anomaly_cnt[name] += 1
            anomaly_vehicles[name].add(vid)
        if f[18] == '0' and f[3] == '1':
            speed_zero_cnt += 1
            try:
                h = rec['time'].split(' ')[1].split(':')[0]
                speed_zero_by_hour[h] += 1
            except Exception:
                pass

    lines = []
    lines.append("## 9. 이상 상태 분석\n")
    lines.append("### 이상 상태 발생 현황\n")
    lines.append("| 상태 | 발생 건수 | 해당 차량 수 |")
    lines.append("|------|----------|------------|")
    for name, cnt in anomaly_cnt.most_common():
        vcount = len(anomaly_vehicles[name])
        lines.append(f"| {name} | {cnt:,} | {vcount} |")

    lines.append(f"\n### 속도=0 이벤트 (RUN 상태에서)\n")
    lines.append(f"- 총 건수: {speed_zero_cnt:,}\n")
    if speed_zero_by_hour:
        lines.append("| 시간 | 건수 |")
        lines.append("|------|------|")
        for h in sorted(speed_zero_by_hour.keys()):
            lines.append(f"| {h}:00 | {speed_zero_by_hour[h]:,} |")
    lines.append("")
    return '\n'.join(lines)


def analyze_type3(type3):
    station_cnt = Counter()
    vehicle_cnt = Counter()
    carrier_cnt = Counter()
    has_carrier = 0
    has_vehicle = 0

    for rec in type3:
        f = rec['fields']
        station_cnt[f[3]] += 1
        if f[4]:
            carrier_cnt[f[4]] += 1
            has_carrier += 1
        if f[7]:
            vehicle_cnt[f[7]] += 1
            has_vehicle += 1

    total = len(type3)
    lines = []
    lines.append("## 10. Type 3 (IN SERVICE) 분석\n")
    lines.append(f"- 총 건수: {total:,}")
    lines.append(f"- 캐리어 정보 포함: {has_carrier:,} ({has_carrier/total*100:.1f}%)")
    lines.append(f"- 차량 정보 포함: {has_vehicle:,} ({has_vehicle/total*100:.1f}%)")
    lines.append(f"- 고유 스테이션: {len(station_cnt):,}개")
    lines.append(f"- 고유 차량: {len(vehicle_cnt):,}대\n")

    lines.append("### Top 20 IN SERVICE 스테이션\n")
    lines.append("| 스테이션 | 건수 |")
    lines.append("|----------|------|")
    for st, cnt in station_cnt.most_common(20):
        lines.append(f"| {st} | {cnt:,} |")
    lines.append("")
    return '\n'.join(lines)


def analyze_world_model_params(type2):
    """월드 모델 구축에 참고할 수 있는 파라미터 추출"""
    # Speed profile
    speed_vals = Counter()
    # Priority distribution
    priority_cnt = Counter()
    # Idle pattern (DetailState=101,105)
    idle_moving = 0
    balance_moving = 0
    # OBS_BZ_STOP duration estimate
    obs_events = []
    prev_obs = {}

    for rec in type2:
        f = rec['fields']
        vid = f[2]
        if not vid.startswith('V'):
            continue
        speed_vals[f[18]] += 1
        priority_cnt[f[18]] += 1
        ds = f[19]
        if ds == '101':
            idle_moving += 1
        elif ds == '105':
            balance_moving += 1

    lines = []
    lines.append("## 11. 월드 모델 참고 파라미터\n")
    lines.append("### 속도 프로파일\n")
    lines.append("| 속도값 | 비율 | 용도 추정 |")
    lines.append("|--------|------|----------|")
    total = sum(speed_vals.values())
    speed_desc = {
        '0': '정지/대기', '50': '일반 주행', '70': '중간 속도',
        '80': '고속 주행', '90': '긴급/우선 운반', '99': '최고 속도'
    }
    for spd, cnt in sorted(speed_vals.items(), key=lambda x: -x[1]):
        desc = speed_desc.get(spd, '미상')
        lines.append(f"| {spd} | {cnt/total*100:.1f}% | {desc} |")

    lines.append("\n### Idle 이동 패턴\n")
    lines.append("| DetailState | 이름 | 건수 | 설명 |")
    lines.append("|-------------|------|------|------|")
    lines.append(f"| 101 | MOVING | {idle_moving:,} | 일반 idle 이동 |")
    lines.append(f"| 105 | BALANCE_MOVING | {balance_moving:,} | 밸런싱 이동 (idle 재배치) |")

    lines.append("\n### Priority 분포 (속도 기반)\n")
    lines.append("| Priority(속도) | 건수 | 비율 |")
    lines.append("|---------------|------|------|")
    for p, cnt in priority_cnt.most_common():
        lines.append(f"| {p} | {cnt:,} | {cnt/total*100:.1f}% |")

    lines.append("\n### 월드 모델 8가지 항목 데이터 검증 가능 여부\n")
    lines.append("| # | 항목 | 가능 여부 | 비고 |")
    lines.append("|---|------|----------|------|")
    lines.append("| 1 | 앞차 감속 | 부분 가능 | State=6(OBS_BZ_STOP) 이벤트로 간접 관측, layout 필요 |")
    lines.append("| 2 | 분기점 경로 | 어려움 | layout.xml 노드 그래프 필요 |")
    lines.append("| 3 | ZCU 합류점 FIFO | 어려움 | layout.xml 필요, 같은 주소 대기 패턴 관측 가능 |")
    lines.append("| 4 | Load/Unload 시간 | 가능 | RunCycle 3/4 전환으로 측정 |")
    lines.append("| 5 | Idle OHT 로직 | 부분 가능 | DetailState=101/105 패턴 분석 |")
    lines.append("| 6 | Bumping Station | 부분 가능 | idle 차량 집중 주소 추정 가능 |")
    lines.append("| 7 | 예약기능 | 어려움 | 경로 변경은 관측 가능, 예약 메커니즘 불가 |")
    lines.append("| 8 | 경로선택 알고리즘 | 불가 | LineCost/Dijkstra는 내부 로직 |")
    lines.append("")
    return '\n'.join(lines)


# ============================================================
# 리포트 생성
# ============================================================
def generate_report(csv_path, output_path):
    print(f"[1/12] CSV 로드 중: {csv_path}")
    data = load_and_parse(csv_path)
    print(f"  총 {data['total']:,}행 로드 완료")

    report = []
    report.append("# OHT XSOHS 데이터 분석 리포트\n")
    report.append(f"> 분석 대상: `{csv_path}`")
    report.append(f"> 분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"> 데이터 수집 기간: 2026-04-14 08:30 ~ 17:00\n")
    report.append("---\n")

    sections = [
        ("2/12", "데이터 개요", lambda: analyze_overview(data)),
        ("3/12", "Fleet 현황", lambda: analyze_fleet(data['type2'])),
        ("4/12", "차량 상태", lambda: analyze_states(data['type2'])),
        ("5/12", "가동률/적재율", lambda: analyze_utilization(data['type2'])),
        ("6/12", "속도 분석", lambda: analyze_speed(data['type2'])),
        ("7/12", "운반 사이클", lambda: analyze_cycles(data['type2'])),
        ("8/12", "스테이션", lambda: analyze_stations(data['type2'])),
        ("9/12", "시간대별 패턴", lambda: analyze_temporal(data)),
        ("10/12", "이상 상태", lambda: analyze_anomalies(data['type2'])),
        ("11/12", "IN SERVICE", lambda: analyze_type3(data['type3'])),
        ("12/12", "월드 모델 파라미터", lambda: analyze_world_model_params(data['type2'])),
    ]

    for step, name, fn in sections:
        print(f"[{step}] {name} 분석 중...")
        report.append(fn())

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n리포트 생성 완료: {output_path}")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'OHS/XSOHS_extracted/raw.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'OHS/XSOHS_ANALYSIS_REPORT.md'
    generate_report(csv_path, output_path)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py - OHT 7종 데이터 통합 로더

CSV/JSON 파일을 읽어 시뮬레이션에 필요한 형태로 변환한다.
- layout_cache.json → 그래프 (노드, 엣지, 거리)
- HID_Zone_Master → Zone 설정
- OHT raw → 차량 상태 타임라인
- 스타 컬럼 → 시스템 지표 타임라인
- HID_INOUT → 구간 통과 이벤트
- RAIL_CUT → 레일 차단 이벤트
- ts_resource → 작업 명령 이벤트
- oht_time_avg → TAT 평균
"""

import csv
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import (
    LAYOUT_CACHE_JSON, LAYOUT_CACHE_DIR, FAB_CONFIG_JSON, HID_ZONE_MASTER_CSV,
    DATA_DATES, STATE_NAMES, MAP_DIR, get_fab_entry,
)


# ============================================================
# FAB별 layout_cache.json 생성/확인
# ============================================================
def ensure_layout_cache(fab: str, prefix: str) -> str:
    """
    FAB/prefix에 해당하는 layout_cache.json이 존재하고 최신인지 확인.
    없거나 .layout.zip보다 오래됐으면 재생성.

    Returns: 캐시 파일 경로 (str)
    """
    entry = get_fab_entry(fab, prefix)
    if entry is None:
        raise ValueError(f"FAB '{fab}/{prefix}' not found in catalog")

    cache_path = Path(entry["layout_cache"])
    zip_path = Path(entry["layout_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"layout.zip not found: {zip_path}")

    need_build = True
    if cache_path.exists() and cache_path.stat().st_size > 100:
        # 빈 파일(파싱 실패 흔적)이면 재빌드
        if cache_path.stat().st_mtime >= zip_path.stat().st_mtime:
            need_build = False

    if not need_build:
        return str(cache_path)

    print(f"[레이아웃] {fab}/{prefix} 캐시 생성 중... ({zip_path.name})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # zip에서 layout.xml 추출 (layout_map_cre.ensure_layout_html의 추출 로직 차용)
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        names = zf.namelist()
        xml_name = None
        for n in names:
            ln = n.lower()
            if ln == 'layout/layout.xml':
                xml_name = n
                break
            elif ln == 'layout.xml' and xml_name is None:
                xml_name = n
            elif ln.endswith('.xml') and 'layout' in ln and xml_name is None:
                xml_name = n
        if not xml_name:
            raise FileNotFoundError(f"layout.xml not found in {zip_path}")
        with zf.open(xml_name) as f:
            xml_content = f.read().decode('utf-8', errors='replace')

    # XML 파싱 — 속성 순서 무관하게 유연하게
    n_nodes, n_edges = _parse_layout_xml_to_json(xml_content, str(cache_path))
    print(f"[레이아웃] {fab}/{prefix} 캐시 완료: 노드 {n_nodes}개, 엣지 {n_edges}개")
    return str(cache_path)


def _parse_layout_xml_to_json(xml_content: str, output_path: str):
    """
    layout.xml 파싱 → {nodes, adj, edges} JSON 저장.
    속성 순서 무관 (M14A/M14B/M16A 등 다양한 변형 지원).
    """
    import math

    # class="...address.Addr" 매칭 (끝이 .Addr로 끝나는 class만)
    addr_class_re = re.compile(r'class="[^"]*\.address\.Addr"')
    next_addr_class_re = re.compile(r'class="[^"]*\.NextAddr"')
    group_open_re = re.compile(r'<group\b')
    group_close_re = re.compile(r'</group>')
    # param은 key/value 순서 무관하게 개별 추출
    param_tag_re = re.compile(r'<param\b')
    key_re = re.compile(r'\bkey="([^"]+)"')
    value_re = re.compile(r'\bvalue="([^"]*)"')

    node_xy = {}
    connections = []

    current_addr_params = None
    in_next_addr = False
    next_addr_params = None

    for line in xml_content.split('\n'):
        line = line.strip()

        # Addr 그룹 시작 (속성 순서 무관)
        if group_open_re.search(line) and addr_class_re.search(line):
            # 이전 Addr 커밋
            if current_addr_params is not None and 'address' in current_addr_params:
                try:
                    addr_no = int(current_addr_params['address'])
                    if addr_no > 0:
                        x = float(current_addr_params.get('draw-x', 0))
                        y = float(current_addr_params.get('draw-y', 0))
                        node_xy[addr_no] = (x, y)
                except (ValueError, TypeError):
                    pass
            current_addr_params = {}
            in_next_addr = False
            continue

        # NextAddr 그룹 시작
        if group_open_re.search(line) and next_addr_class_re.search(line):
            in_next_addr = True
            next_addr_params = {}
            continue

        # NextAddr 그룹 종료 → 연결 커밋
        if in_next_addr and group_close_re.search(line):
            if current_addr_params and 'address' in current_addr_params and next_addr_params and 'next-address' in next_addr_params:
                try:
                    fr = int(current_addr_params['address'])
                    to = int(next_addr_params['next-address'])
                    if fr > 0 and to > 0:
                        connections.append((fr, to))
                except (ValueError, TypeError):
                    pass
            in_next_addr = False
            continue

        # param 값 추출 (key/value 순서 무관)
        if param_tag_re.search(line):
            km = key_re.search(line)
            vm = value_re.search(line)
            if km and vm:
                key, value = km.group(1), vm.group(1)
                if in_next_addr and next_addr_params is not None:
                    next_addr_params[key] = value
                elif current_addr_params is not None:
                    current_addr_params[key] = value

    # 마지막 Addr 커밋
    if current_addr_params is not None and 'address' in current_addr_params:
        try:
            addr_no = int(current_addr_params['address'])
            if addr_no > 0:
                x = float(current_addr_params.get('draw-x', 0))
                y = float(current_addr_params.get('draw-y', 0))
                node_xy[addr_no] = (x, y)
        except (ValueError, TypeError):
            pass

    # 인접 + 거리
    adj = {}
    edge_dist = {}
    for fr, to in connections:
        adj.setdefault(fr, []).append(to)
        if fr in node_xy and to in node_xy:
            x1, y1 = node_xy[fr]
            x2, y2 = node_xy[to]
            d = int(math.sqrt((x2-x1)**2 + (y2-y1)**2) * 10)
            edge_dist[(fr, to)] = max(d, 10)

    data = {
        "nodes": {str(k): list(v) for k, v in node_xy.items()},
        "adj": {str(k): v for k, v in adj.items()},
        "edges": {f"{k[0]},{k[1]}": v for k, v in edge_dist.items()},
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    return len(node_xy), len(edge_dist)


# ============================================================
# 그래프 / 레이아웃 로더
# ============================================================

class LayoutData:
    """layout_cache.json 기반 그래프 데이터"""

    def __init__(self):
        self.nodes: Dict[int, Tuple[float, float]] = {}  # nodeId → (x, y)
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.edge_dist: Dict[Tuple[int, int], float] = {}
        self.node_ids: List[int] = []
        self.bounds = {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    def load(self, path: str = None):
        path = path or str(LAYOUT_CACHE_JSON)
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)

        # 노드 좌표
        for nid_str, coords in cache.get('nodes', {}).items():
            nid = int(nid_str)
            self.nodes[nid] = (float(coords[0]), float(coords[1]))
        self.node_ids = list(self.nodes.keys())

        # 바운드 계산
        if self.nodes:
            xs = [c[0] for c in self.nodes.values()]
            ys = [c[1] for c in self.nodes.values()]
            self.bounds = {
                "min_x": min(xs), "max_x": max(xs),
                "min_y": min(ys), "max_y": max(ys),
            }

        # 인접 리스트
        for nid_str, neighbors in cache.get('adj', {}).items():
            nid = int(nid_str)
            self.adj[nid] = [int(n) for n in neighbors]

        # 엣지 거리
        for edge_key, dist in cache.get('edges', {}).items():
            parts = edge_key.split(",")
            self.edge_dist[(int(parts[0]), int(parts[1]))] = float(dist)

        return self

    def get_position(self, current_node: int, next_node: int, distance: float) -> Optional[Tuple[float, float]]:
        """주소 + 거리 → (x, y) 좌표 계산 (선형 보간)"""
        if current_node not in self.nodes or next_node not in self.nodes:
            return None

        x1, y1 = self.nodes[current_node]
        x2, y2 = self.nodes[next_node]

        edge_key = (current_node, next_node)
        edge_len = self.edge_dist.get(edge_key, 1.0)

        if edge_len <= 0:
            return (x1, y1)

        ratio = min(1.0, max(0.0, distance / edge_len))
        x = x1 + ratio * (x2 - x1)
        y = y1 + ratio * (y2 - y1)
        return (x, y)


# ============================================================
# HID Zone 로더
# ============================================================

class HIDZoneData:
    """HID_Zone_Master CSV 로더"""

    def __init__(self):
        self.zones: Dict[int, dict] = {}  # zoneId → zone info
        self.in_lane_to_zone: Dict[Tuple[int, int], int] = {}
        self.out_lane_to_zone: Dict[Tuple[int, int], int] = {}

    def load(self, path: str = None):
        path = path or str(HID_ZONE_MASTER_CSV)
        if not os.path.exists(path):
            return self

        seen_zones = set()
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                zid_str = row.get('Zone_ID') or row.get('\ufeffZone_ID') or '0'
                zid = _safe_int(zid_str)
                if zid <= 0:
                    continue

                vmax = _safe_int(row.get('Vehicle_Max', '0'))
                if vmax <= 0:
                    continue

                if zid not in seen_zones:
                    seen_zones.add(zid)
                    self.zones[zid] = {
                        'zoneId': zid,
                        'hidNo': row.get('HID_No', ''),
                        'bayZone': row.get('Bay_Zone', ''),
                        'fullName': row.get('Full_Name', f'Zone-{zid}'),
                        'vehicleMax': int(row.get('Vehicle_Max', 37)),
                        'vehiclePrecaution': int(row.get('Vehicle_Precaution', 35)),
                        'currentCount': 0,
                    }

                # IN/OUT 레인 파싱 (예: "3048→3023; 3048→3049")
                in_lanes = row.get('IN_Lanes', '')
                out_lanes = row.get('OUT_Lanes', '')

                for lane_str in in_lanes.split(';'):
                    lane_pair = _parse_lane(lane_str)
                    if lane_pair:
                        self.in_lane_to_zone[lane_pair] = zid

                for lane_str in out_lanes.split(';'):
                    lane_pair = _parse_lane(lane_str)
                    if lane_pair:
                        self.out_lane_to_zone[lane_pair] = zid

        return self


# ============================================================
# OHT Raw 데이터 파서
# ============================================================

def parse_oht_message(line: str, t=None) -> Optional[dict]:
    """
    '2,OHT,V00066,6,1,0000,1,13307,9,13308,...' → dict
    21필드 VHL_STATE_REPORT 파싱
    t: UDP 수신 시각(datetime). 결과 dict의 '_time'에 보존되어 속도 계산에 사용.
    """
    # "line" 컬럼 내부에 들어있을 수 있음
    if "2,OHT," in line:
        idx = line.index("2,OHT,")
        line = line[idx:]

    fields = line.split(",")
    if len(fields) < 11 or fields[1] != "OHT":
        return None

    try:
        vid = fields[2]
        state = int(fields[3])
        is_full = int(fields[4])
        current_addr = int(fields[7])
        distance = int(fields[8])
        next_addr = int(fields[9])
        run_cycle = int(fields[10]) if len(fields) > 10 else 0
        vhl_cycle = int(fields[11]) if len(fields) > 11 else 0
        carrier_id = fields[12] if len(fields) > 12 else ""
        destination = int(fields[13]) if len(fields) > 13 else 0
        source_port = fields[16] if len(fields) > 16 else ""
        dest_port = fields[17] if len(fields) > 17 else ""
        speed = int(fields[18]) if len(fields) > 18 else 0

        return {
            'vid': vid,
            'state': state,
            'stateName': STATE_NAMES.get(state, f"UNKNOWN({state})"),
            'isFull': is_full,
            'currentNode': current_addr,
            'distance': distance,
            'nextNode': next_addr,
            'runCycle': run_cycle,
            'vhlCycle': vhl_cycle,
            'carrierId': carrier_id,
            'destination': destination,
            'sourcePort': source_port,
            'destPort': dest_port,
            'speed': speed,
            '_time': t,
        }
    except (ValueError, IndexError):
        return None


def parse_oht_data_m14a_row(row: dict, t=None) -> Optional[dict]:
    """oht_data_m14a (파싱된 버전) 행 → dict
    t: UDP 수신 시각(datetime). 결과 dict의 '_time'에 보존되어 속도 계산에 사용.
    """
    try:
        return {
            'vid': row.get('VEHICLE', ''),
            'state': int(row.get('STATUS', 1)),
            'stateName': STATE_NAMES.get(int(row.get('STATUS', 1)), 'UNKNOWN'),
            'isFull': int(row.get('STOCK_INFO', 0)),
            'currentNode': int(row.get('ADDRESS', 0)),
            'distance': int(row.get('DISTANCE', 0)),
            'nextNode': int(row.get('NEXT_ADDRESS', 0)),
            'edge': row.get('EDGE', ''),
            'carrierId': row.get('CARRIER', ''),
            'destination': int(row.get('DESTINATION', 0)),
            'sourcePort': row.get('FROM_RETURN_PORT', ''),
            'destPort': row.get('DEST_RETURN_PORT', ''),
            'speed': int(row.get('RETURN_PRIORITY', 0)),
            '_time': t,
        }
    except (ValueError, KeyError):
        return None


# ============================================================
# 날짜별 데이터 로더
# ============================================================

class DateDataLoader:
    """특정 날짜의 CSV 데이터를 로드"""

    def __init__(self, date_key: str, date_config: Optional[dict] = None):
        self.date_key = date_key
        self.date_config = date_config if date_config is not None else DATA_DATES.get(date_key, {})
        self.data_dir = self.date_config.get('dir', '')

        # 로드된 데이터
        self.oht_timeline: List[Tuple[datetime, List[dict]]] = []  # (시각, [차량상태들])
        self.star_timeline: List[Tuple[datetime, dict]] = []  # (시각, 지표dict)
        self.hid_events: List[Tuple[datetime, dict]] = []  # (시각, 이벤트dict)
        self.rail_cut_events: List[Tuple[datetime, dict]] = []
        self.ts_events: List[Tuple[datetime, dict]] = []
        self.oht_time_avg: List[Tuple[datetime, dict]] = []

        # 시간 범위
        self.time_start: Optional[datetime] = None
        self.time_end: Optional[datetime] = None

        # 통계
        self.stats = {}

    def load_all(self) -> dict:
        """모든 데이터 로드, 통계 반환"""
        stats = {}
        stats['oht'] = self._load_oht_raw()
        stats['star'] = self._load_star()
        stats['hid'] = self._load_hid_inout()
        stats['rail_cut'] = self._load_rail_cut()
        stats['ts_resource'] = self._load_ts_resource()
        stats['oht_time_avg'] = self._load_oht_time_avg()
        stats['star_derived'] = self._derive_statecnt_from_oht()
        self.stats = stats
        return stats

    def _derive_statecnt_from_oht(self) -> dict:
        """star CSV 의 STATECNT 가 비었으면 OHT raw 의 state 로 복원.

        M16A_BR 처럼 수집측이 OHT.STATECNT.* 를 안 내려주는 FAB 에서
        oht_timeline 의 state 값을 분 단위 누적 카운트해 star_timeline 에 주입.

        주입 키:
          driving (state=1), pause (state=2), obs_bz_stop (state=6),
          congested (state=7), timeout (state=9)
        """
        if not self.oht_timeline or not self.star_timeline:
            return {'status': 'skip_no_data'}

        # star 가 이미 수치 있으면 스킵 (정상 수집 FAB)
        has_data = any(
            s.get('driving') or s.get('obs_bz_stop') or s.get('congested')
            for _, s in self.star_timeline
        )
        if has_data:
            return {'status': 'skip_existing'}

        # oht_timeline 을 분 단위로 state 카운트 (누적 상태 기반)
        vstate: Dict[str, int] = {}
        minute_counts: Dict[datetime, dict] = {}
        for t, updates in self.oht_timeline:
            for v in updates:
                s = v.get('state')
                if isinstance(s, int):
                    vstate[v['vid']] = s
            t_key = t.replace(second=0, microsecond=0)
            minute_counts[t_key] = {
                'driving':     sum(1 for s in vstate.values() if s == 1),
                'pause':       sum(1 for s in vstate.values() if s == 2),
                'obs_bz_stop': sum(1 for s in vstate.values() if s == 6),
                'congested':   sum(1 for s in vstate.values() if s == 7),
                'timeout':     sum(1 for s in vstate.values() if s == 9),
            }

        # star_timeline 에 주입: 기존 값이 falsy(0/None) 일 때만
        filled = 0
        for t, star in self.star_timeline:
            key = t.replace(second=0, microsecond=0)
            derived = minute_counts.get(key)
            if not derived:
                continue
            for k, v in derived.items():
                if not star.get(k):
                    star[k] = v
                    filled += 1

        return {
            'status': 'derived',
            'minutes_computed': len(minute_counts),
            'star_cells_filled': filled,
        }

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """다양한 시간 형식 파싱"""
        if not time_str:
            return None
        time_str = time_str.strip().strip('"')
        for fmt in [
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ]:
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=None)
            except ValueError:
                continue
        return None

    def _load_oht_raw(self) -> dict:
        """
        OHT raw CSV 로드 → 누적 상태 기반 타임라인 구축.
        각 차량의 최신 상태를 유지하면서 1초 간격 스냅샷 생성.
        """
        filename = self.date_config.get('oht_raw')
        if not filename:
            return {'count': 0, 'status': 'no_file'}

        filepath = os.path.join(str(self.data_dir), filename)
        if not os.path.exists(filepath):
            return {'count': 0, 'status': 'file_not_found', 'path': filepath}

        # oht_data_m14a가 있으면 우선 사용 (이미 파싱됨)
        m14a_file = self.date_config.get('oht_data_m14a')
        if m14a_file:
            m14a_path = os.path.join(str(self.data_dir), m14a_file)
            if os.path.exists(m14a_path):
                return self._load_oht_data_m14a(m14a_path)

        # 누적 상태: vid → 최신 상태
        vehicle_state: Dict[str, dict] = {}
        time_updates = defaultdict(list)  # 1초 단위 업데이트
        count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get('_time', '')
                line = row.get('line', '')

                if '2,OHT,' not in line:
                    continue

                t = self._parse_time(time_str)
                if not t:
                    continue

                parsed = parse_oht_message(line, t)
                if parsed and parsed['vid']:
                    # 1초 단위로 그룹화
                    t_key = t.replace(microsecond=0)
                    time_updates[t_key].append(parsed)
                    count += 1

        # 누적 상태 기반 타임라인: 모든 업데이트를 적용 후 스냅샷
        sorted_times = sorted(time_updates.keys())

        self._vehicle_state = vehicle_state
        self._oht_updates = []

        # 데이터 크기에 따라 프레임 간격 자동 조절
        snapshot_interval = 2 if count < 800000 else 5 if count < 2000000 else 10
        last_t = None

        for t_key in sorted_times:
            for v in time_updates[t_key]:
                vehicle_state[v['vid']] = v

            if last_t and (t_key - last_t).total_seconds() < snapshot_interval:
                continue

            self._oht_updates.append((t_key, time_updates[t_key]))
            last_t = t_key

        self.oht_timeline = self._oht_updates

        if self.oht_timeline:
            self.time_start = self.oht_timeline[0][0]
            self.time_end = self.oht_timeline[-1][0]

        return {
            'count': count,
            'frames': len(self.oht_timeline),
            'unique_vehicles': len(vehicle_state),
            'time_start': str(self.time_start) if self.time_start else None,
            'time_end': str(self.time_end) if self.time_end else None,
            'status': 'loaded',
        }

    def _load_oht_data_m14a(self, filepath: str) -> dict:
        """파싱된 oht_data_m14a CSV 로드 (델타 방식)"""
        vehicle_state: Dict[str, dict] = {}
        time_updates = defaultdict(list)
        count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get('_time', '')
                t = self._parse_time(time_str)
                if not t:
                    continue

                parsed = parse_oht_data_m14a_row(row, t)
                if parsed and parsed['vid']:
                    t_key = t.replace(microsecond=0)
                    time_updates[t_key].append(parsed)
                    vehicle_state[parsed['vid']] = parsed
                    count += 1

        sorted_times = sorted(time_updates.keys())
        snapshot_interval = 2
        last_t = None

        self._vehicle_state = vehicle_state
        self._oht_updates = []

        for t_key in sorted_times:
            if last_t and (t_key - last_t).total_seconds() < snapshot_interval:
                continue
            self._oht_updates.append((t_key, time_updates[t_key]))
            last_t = t_key

        self.oht_timeline = self._oht_updates

        if self.oht_timeline:
            self.time_start = self.oht_timeline[0][0]
            self.time_end = self.oht_timeline[-1][0]

        return {
            'count': count,
            'frames': len(self.oht_timeline),
            'unique_vehicles': len(vehicle_state),
            'source': 'oht_data_m14a',
            'time_start': str(self.time_start) if self.time_start else None,
            'time_end': str(self.time_end) if self.time_end else None,
            'status': 'loaded',
        }

    def _load_star(self) -> dict:
        """스타 컬럼 CSV 로드 (프리픽스 자동 인식: M14 / M16HUB / 기타)"""
        filename = self.date_config.get('star')
        if not filename:
            return {'count': 0, 'status': 'no_file'}

        filepath = os.path.join(str(self.data_dir), filename)
        if not os.path.exists(filepath):
            return {'count': 0, 'status': 'file_not_found'}

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            anchor_suffix = '.QUE.ALL.CURRENTQCNT'
            prefix = None
            for col in (reader.fieldnames or []):
                if col and col.endswith(anchor_suffix):
                    prefix = col[: -len(anchor_suffix)]
                    break
            if not prefix:
                return {'count': 0, 'status': 'prefix_not_found'}

            def C(suffix: str) -> str:
                return f"{prefix}{suffix}"

            for row in reader:
                time_str = row.get('CRT_TM', '')
                t = self._parse_time(time_str)
                if not t:
                    continue

                star = {
                    'queue_total': _safe_int(row.get(C('.QUE.ALL.CURRENTQCNT'))),
                    'queue_completed': _safe_int(row.get(C('.QUE.ALL.CURRENTQCOMPLETED'))),
                    'queue_oht': _safe_int(row.get(C('.QUE.OHT.CURRENTOHTQCNT'))),
                    'oht_util': _safe_float(row.get(C('.QUE.OHT.OHTUTIL'))),
                    'avg_load_time': _safe_float(row.get(C('.QUE.LOAD.AVGLOADTIME'))),
                    'transport_4min_over': _safe_int(row.get(C('.QUE.ALL.TRANSPORT4MINOVERCNT'))),
                    'driving': _safe_int(row.get(C('.OHT.STATECNT.DRIVING'))),
                    'obs_bz_stop': _safe_int(row.get(C('.OHT.STATECNT.OBSANDBZSTOP'))),
                    'congested': _safe_int(row.get(C('.OHT.STATECNT.CONGESTED'))),
                    'pause': _safe_int(row.get(C('.OHT.STATECNT.PAUSE'))),
                    'timeout': _safe_int(row.get(C('.OHT.STATECNT.TIMEOUT'))),
                }
                self.star_timeline.append((t, star))
                count += 1

        self.star_timeline.sort(key=lambda x: x[0])
        return {'count': count, 'status': 'loaded', 'prefix': prefix}

    def _load_hid_inout(self) -> dict:
        """HID_INOUT CSV 로드"""
        filename = self.date_config.get('hid_inout')
        if not filename:
            return {'count': 0, 'status': 'no_file'}

        filepath = os.path.join(str(self.data_dir), filename)
        if not os.path.exists(filepath):
            return {'count': 0, 'status': 'file_not_found'}

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get('_time', '')
                t = self._parse_time(time_str)
                if not t:
                    continue

                event = {
                    'vhl_id': row.get('VHL_ID', ''),
                    'from_hid': _safe_int(row.get('FROM_HIDID')),
                    'to_hid': _safe_int(row.get('TO_HIDID')),
                    'speed': _safe_float(row.get('FREE_FLOW_SPEED')),
                    'vhl_count_limit': _safe_int(row.get('VHL_COUNT_LIMIT')),
                    'vhl_precaution': _safe_int(row.get('VHL_PRECAUTION')),
                    'trans_cnt': _safe_int(row.get('TRANS_CNT')),
                }
                self.hid_events.append((t, event))
                count += 1

        self.hid_events.sort(key=lambda x: x[0])
        return {'count': count, 'status': 'loaded'}

    def _load_rail_cut(self) -> dict:
        """RAIL_CUT CSV 로드"""
        filename = self.date_config.get('rail_cut')
        if not filename:
            return {'count': 0, 'status': 'no_file'}

        filepath = os.path.join(str(self.data_dir), filename)
        if not os.path.exists(filepath):
            return {'count': 0, 'status': 'file_not_found'}

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get('EVENT_DT', '') or row.get('_time', '')
                t = self._parse_time(time_str)
                if not t:
                    continue

                # 영향 주소 파싱
                affect_str = row.get('AFFECT_ADDR_LST', '')
                affect_addrs = []
                for seg in affect_str.split(','):
                    if ':' in seg:
                        parts = seg.split(':')
                        try:
                            affect_addrs.append((int(parts[0]), int(parts[1])))
                        except ValueError:
                            pass

                event = {
                    'from_addr': _safe_int(row.get('FROM_ADDR')),
                    'to_addr': _safe_int(row.get('TO_ADDR')),
                    'state': row.get('STATE', ''),
                    'affect_addrs': affect_addrs,
                    'affect_count': len(affect_addrs),
                }
                self.rail_cut_events.append((t, event))
                count += 1

        self.rail_cut_events.sort(key=lambda x: x[0])
        return {'count': count, 'status': 'loaded'}

    def _load_ts_resource(self) -> dict:
        """ts_resource CSV 로드 (이벤트 단위)"""
        filename = self.date_config.get('ts_resource')
        if not filename:
            return {'count': 0, 'status': 'no_file'}

        filepath = os.path.join(str(self.data_dir), filename)
        if not os.path.exists(filepath):
            return {'count': 0, 'status': 'file_not_found'}

        # ts_resource는 건수가 매우 많아서 (3.3M) 분 단위 집계만
        minute_counts = defaultdict(lambda: defaultdict(int))
        count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get('_time', '')
                t = self._parse_time(time_str)
                if not t:
                    continue

                msg_name = row.get('MESSAGENAME', '')
                t_min = t.replace(second=0, microsecond=0)
                minute_counts[t_min][msg_name] += 1
                count += 1

        # 분 단위 집계를 타임라인으로
        for t_min, msg_counts in sorted(minute_counts.items()):
            event = {
                'total': sum(msg_counts.values()),
                'arrived': msg_counts.get('RAIL-VEHICLEARRIVED', 0),
                'departed': msg_counts.get('RAIL-VEHICLEDEPARTED', 0),
                'acquire_started': msg_counts.get('RAIL-VEHICLEACQUIRESTARTED', 0),
                'deposit_started': msg_counts.get('RAIL-VEHICLEDEPOSITSTARTED', 0),
                'unassigned': msg_counts.get('RAIL-VEHICLEUNASSIGNED', 0),
                'assigned': msg_counts.get('RAIL-VEHICLEASSIGNED', 0),
            }
            self.ts_events.append((t_min, event))

        return {'count': count, 'minutes': len(self.ts_events), 'status': 'loaded'}

    def _load_oht_time_avg(self) -> dict:
        """oht_time_avg CSV 로드"""
        filename = self.date_config.get('oht_time_avg')
        if not filename:
            return {'count': 0, 'status': 'no_file'}

        filepath = os.path.join(str(self.data_dir), filename)
        if not os.path.exists(filepath):
            return {'count': 0, 'status': 'file_not_found'}

        count = 0
        time_groups = defaultdict(dict)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get('_time', '')
                t = self._parse_time(time_str)
                if not t:
                    continue

                msg_type = row.get('MESSAGE_TYP', '')
                time_val = _safe_float(row.get('MESSAGE_TIME', row.get('TIME', '')))

                t_min = t.replace(second=0, microsecond=0)
                time_groups[t_min][msg_type] = time_val
                count += 1

        for t_min, msg_data in sorted(time_groups.items()):
            self.oht_time_avg.append((t_min, msg_data))

        return {'count': count, 'status': 'loaded'}

    def get_frame_at(self, target_time: datetime) -> Tuple[List[dict], Optional[dict], List[dict], List[dict]]:
        """
        특정 시각의 데이터 프레임 반환.
        Returns: (차량상태들, 스타지표, HID이벤트들, RAIL_CUT이벤트들)
        """
        # OHT: 가장 가까운 프레임 찾기
        vehicles = []
        if self.oht_timeline:
            idx = self._bisect_time(self.oht_timeline, target_time)
            if 0 <= idx < len(self.oht_timeline):
                vehicles = self.oht_timeline[idx][1]

        # 스타: 가장 가까운 지표
        star = None
        if self.star_timeline:
            idx = self._bisect_time(self.star_timeline, target_time)
            if 0 <= idx < len(self.star_timeline):
                star = self.star_timeline[idx][1]

        # HID: 현재 시각 ± 30초 범위 이벤트
        hid = []
        for t, ev in self.hid_events:
            diff = abs((t - target_time).total_seconds())
            if diff <= 30:
                hid.append(ev)

        # RAIL_CUT: 현재까지 발생한 이벤트
        rail_cuts = []
        for t, ev in self.rail_cut_events:
            if t <= target_time:
                rail_cuts.append(ev)

        return vehicles, star, hid, rail_cuts

    def _bisect_time(self, timeline: list, target: datetime) -> int:
        """이진 탐색으로 가장 가까운 시각 인덱스 찾기"""
        lo, hi = 0, len(timeline) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if timeline[mid][0] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def get_star_history(self) -> List[dict]:
        """전체 스타 타임라인 반환 (차트용)"""
        result = []
        for t, star in self.star_timeline:
            entry = {'time': t.strftime("%H:%M")}
            entry.update(star)
            result.append(entry)
        return result

    def get_hid_speed_summary(self) -> Dict[int, dict]:
        """HID 구간별 속도 통계"""
        hid_speeds = defaultdict(list)
        for t, ev in self.hid_events:
            from_hid = ev['from_hid']
            if from_hid > 0 and ev['speed'] > 0:
                hid_speeds[from_hid].append(ev['speed'])

        summary = {}
        for hid_id, speeds in hid_speeds.items():
            summary[hid_id] = {
                'hid_id': hid_id,
                'avg_speed': round(sum(speeds) / len(speeds), 1),
                'min_speed': round(min(speeds), 1),
                'max_speed': round(max(speeds), 1),
                'count': len(speeds),
            }
        return summary


# ============================================================
# 유틸리티
# ============================================================

def _parse_lane(lane_str: str) -> Optional[Tuple[int, int]]:
    """레인 문자열 파싱: '3048→3023' 또는 '3048→3023' 등"""
    lane_str = lane_str.strip()
    # 다양한 화살표 문자 대응
    for sep in ['→', '→', '->', '=>', '\u2192', '\uffeb']:
        if sep in lane_str:
            parts = lane_str.split(sep)
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except (ValueError, IndexError):
                return None
    # 숫자 2개를 추출하는 폴백
    import re as _re
    nums = _re.findall(r'\d+', lane_str)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None


def _safe_int(val, default=0) -> int:
    if val is None or val == '':
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0) -> float:
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_available_dates(date_map: Optional[dict] = None) -> List[dict]:
    """사용 가능한 날짜 목록 반환. date_map 생략 시 기본(M14A/A) 사용."""
    source = date_map if date_map is not None else DATA_DATES
    dates = []
    for key, cfg in source.items():
        data_dir = cfg.get('dir', '')
        exists = os.path.isdir(str(data_dir))
        dates.append({
            'date': key,
            'description': cfg.get('description', ''),
            'exists': exists,
        })
    return dates

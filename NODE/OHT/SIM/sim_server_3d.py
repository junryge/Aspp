#!/usr/bin/env python3
"""
OHT 실시간 시뮬레이터 - FastAPI 웹서버
- Java OHT 시스템 기반 정확한 UDP 메시지 시뮬레이션
- VHL_STATE, RUN_CYCLE, VHL_CYCLE 정확히 구현
- 속도 계산 5가지 조건 적용
- WebSocket으로 프론트엔드에 전송
- CSV 자동 저장 (ATLAS 테이블 형식)
- HID Zone 기반 차량 관리 (HID_Zone_Master.csv 연동)
"""

import os
import re
import json
import random
import math
import asyncio
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from enum import Enum
import heapq
import csv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# layout.xml -> layout.html 자동 생성 모듈
from layout_map_cre import ensure_layout_html

# ============================================================
# 설정
# ============================================================
# 스크립트 디렉토리 기준 상대 경로 사용
import pathlib
_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

LAYOUT_PATH = str(_SCRIPT_DIR / "layout" / "layout" / "layout.html")
LAYOUT_XML_PATH = str(_SCRIPT_DIR / "layout" / "layout.xml")  # 실제 환경의 layout.xml
LAYOUT_ZIP_PATH = str(_SCRIPT_DIR / "layout" / "layout" / "layout.zip")  # 백업용 ZIP (xml 없을 때)
OUTPUT_DIR = str(_SCRIPT_DIR / "output")
HID_ZONE_CSV_PATH = str(_SCRIPT_DIR / "HID_Zone_Master.csv")  # HID Zone 마스터 파일
STATION_DAT_PATH = str(_SCRIPT_DIR / "station.dat")  # Station 데이터 파일
CSV_SAVE_INTERVAL = 10  # 10초마다 CSV 저장
VEHICLE_COUNT = 450  # OHT 대수
SIMULATION_INTERVAL = 0.5  # 0.5초마다 업데이트
FAB_ID = "M14Q"
MCP_NAME = "OHT"

# ============================================================
# Enums - Java Vhl.java 기반
# ============================================================
class VHL_STATE(Enum):
    """Vehicle 상태 - Vhl.java VHL_STATE enum"""
    RUN = "1"           # 운전 중
    STOP = "2"          # 정지 중
    ABNORMAL = "3"      # 상태 이상
    MANUAL = "4"        # 수동 조치
    REMOVING = "5"      # 분리 및 제거 중
    OBS_BZ_STOP = "6"   # OBS-STOP/BZ-STOP
    JAM = "7"           # 정체
    HT_STOP = "8"       # HT-STOP
    E84_TIMEOUT = "9"   # E84 time out

class RUN_CYCLE(Enum):
    """실행 주기 - Vhl.java RUN_CYCLE enum"""
    NONE = "0"              # 사이클 없음
    POSITION_DETECT = "1"   # 위치 확인
    MOVING = "2"            # 이동 중
    ACQUIRE = "3"           # 물품 집어올리기 (구원)
    DEPOSIT = "4"           # 물품 내려놓기 (도매)
    SAMPLING = "5"          # 샘플링
    FLOOR_TRANS = "9"       # 층간 이동
    WHEELDRIVE = "21"       # 바퀴 주행
    MANUAL_CONTROL = "22"   # 수동 조작
    DRIVE_TEACHING = "23"   # 주행 학습
    TRANS_TEACHING = "24"   # 이재부 학습
    BUILDING_TRANS = "2E"   # 동 간 이동
    EVACUATION = "2F"       # 대피 이동

class VHL_CYCLE(Enum):
    """Vehicle 실행 주기 - Vhl.java VHL_CYCLE enum"""
    NONE = "0"           # 실행 사이클 없음
    MOVING = "1"         # 이동 중
    ACQUIRE_MOVING = "2" # 구원 이동
    ACQUIRING = "3"      # 구원 이송 중
    DEPOSIT_MOVING = "4" # 도매 이동
    DEPOSITING = "5"     # 도매 이송 중
    MAINT_MOVING = "6"   # 유지 이동 중
    WAITING = "7"        # 대기
    INPUT = "8"          # 투입 중

class VHL_DET_STATE(Enum):
    """작업 상태 상세"""
    NONE = "0"
    WAIT = "1"
    STAGE_WAIT = "2"
    STANDBY_WAIT = "3"
    DEPOSIT_SIG_WAIT = "4"
    ACQ_WAIT = "5"
    MAP_WAIT = "6"
    MOVING = "101"
    PARKING_UTS_MOVING = "102"
    STAGE_MOVING = "103"
    STANDBY_MOVING = "104"
    BALANCE_MOVING = "105"
    PARKING_MOVING = "106"

# ============================================================
# MCP 속도 테이블 - vhl_speed.cfg CLW07-2 기반 (최대 300 m/min)
# ============================================================
MCP_SPEED_TABLE = {
    1: 1.5, 2: 3, 3: 5, 4: 10, 5: 15, 6: 20, 7: 25, 8: 30,
    9: 35, 10: 40, 11: 45, 12: 50, 13: 55, 14: 60, 15: 65,
    16: 70, 17: 75, 18: 80, 19: 90, 20: 100, 21: 110, 22: 120,
    23: 130, 24: 140, 25: 150, 26: 160, 27: 170, 28: 180,
    29: 190, 30: 200, 31: 200, 32: 200, 33: 210, 34: 220,
    35: 230, 36: 240, 37: 250, 38: 260, 39: 270, 40: 280,
    41: 290, 42: 300
}
MAX_VELOCITY = 300.0  # m/min (CLW07-2 최대 속도)
MIN_VELOCITY = 1.5    # m/min (최소 속도)
LAST_HIS_WEIGHT = 0.7  # 이전 히스토리 가중치 (PredictionPara 기반)

# ============================================================
# 차량 길이 상수 - Java RailEdge.getDensity() 기반
# ============================================================
# FAB별 차량 길이 (mm) = 차체 길이 + 간격(300mm)
VHL_LENGTH_M14 = 784 + 300   # M14* FAB: 1084mm
VHL_LENGTH_M16 = 943 + 300   # M16* FAB: 1243mm

# UDP 거리 단위 변환 (UDP는 100mm 단위로 전송)
UDP_DISTANCE_UNIT = 100  # mm

# ============================================================
# 데이터 모델
# ============================================================
@dataclass
class Node:
    no: int
    x: float
    y: float
    stations: List[int] = field(default_factory=list)

@dataclass
class VhlUdpState:
    """Vehicle UDP 상태 - Java VhlUdpState 클래스 기반"""
    vehicleId: str
    udpCarrierId: str = ""
    state: VHL_STATE = VHL_STATE.REMOVING
    isFull: bool = False
    errorCode: str = ""
    isOnline: bool = False
    railNodeId: str = ""
    distance: float = -1.0  # 현재 번지로부터의 거리 (mm)
    nextRailNodeId: str = ""
    railEdgeId: str = ""
    runCycle: RUN_CYCLE = RUN_CYCLE.NONE
    vhlCycle: VHL_CYCLE = VHL_CYCLE.NONE
    destStationId: str = ""
    receivedTime: int = -1  # 수신 시간 (ms)
    detailState: VHL_DET_STATE = VHL_DET_STATE.NONE
    runDistance: int = 0  # 대차 주행거리
    hidId: int = -1
    currentAddress: int = -1
    nextAddress: int = -1

    def clone(self):
        """상태 복제"""
        return VhlUdpState(
            vehicleId=self.vehicleId,
            udpCarrierId=self.udpCarrierId,
            state=self.state,
            isFull=self.isFull,
            errorCode=self.errorCode,
            isOnline=self.isOnline,
            railNodeId=self.railNodeId,
            distance=self.distance,
            nextRailNodeId=self.nextRailNodeId,
            railEdgeId=self.railEdgeId,
            runCycle=self.runCycle,
            vhlCycle=self.vhlCycle,
            destStationId=self.destStationId,
            receivedTime=self.receivedTime,
            detailState=self.detailState,
            runDistance=self.runDistance,
            hidId=self.hidId,
            currentAddress=self.currentAddress,
            nextAddress=self.nextAddress
        )

@dataclass
class Vehicle:
    """Vehicle 객체 - Java Vhl.java 기반"""
    vehicleId: str
    currentNode: int
    nextNode: int = 0
    positionRatio: float = 0.0
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    smoothedVelocity: float = 0.0
    carrierId: str = ""
    destination: int = 0
    path: List[int] = field(default_factory=list)
    pathIndex: int = 0

    # Java 기반 UDP 상태
    udpState: VhlUdpState = None
    lastUdpState: VhlUdpState = None

    def __post_init__(self):
        if self.udpState is None:
            self.udpState = VhlUdpState(vehicleId=self.vehicleId)
        if self.lastUdpState is None:
            self.lastUdpState = self.udpState.clone()

    def copyCurrentVhlUdpStateToLast(self):
        """현재 상태를 이전 상태로 복사"""
        self.lastUdpState = self.udpState.clone()

    @property
    def state(self) -> int:
        """상태 코드 반환 (1=RUN, 2=STOP 등)"""
        return int(self.udpState.state.value)

    @property
    def isLoaded(self) -> int:
        """적재 상태 (0 or 1)"""
        return 1 if self.udpState.isFull else 0

    @property
    def runCycle(self) -> int:
        """실행 사이클 코드"""
        try:
            return int(self.udpState.runCycle.value)
        except ValueError:
            return 0

# ============================================================
# RailEdge 클래스 - Java RailEdge.java 기반
# ============================================================
@dataclass
class RailEdge:
    """Rail Edge - Java RailEdge.java 기반"""
    edgeId: str
    fromNodeId: int
    toNodeId: int
    length: float  # mm 단위
    fabId: str = FAB_ID
    maxVelocity: float = MAX_VELOCITY
    velocity: float = -1.0
    lastVelocity: float = -1.0
    hisCnt: int = 0
    hidId: int = -1
    changedVelocity: bool = False
    vhlIdMap: Dict[str, int] = field(default_factory=dict)

    # In/Out 카운터
    inCount: int = 0   # 진입 차량 수
    outCount: int = 0  # 진출 차량 수

    def recordIn(self):
        """차량 진입 기록"""
        self.inCount += 1

    def recordOut(self):
        """차량 진출 기록"""
        self.outCount += 1

    def getInOutRatio(self) -> float:
        """In/Out 비율 계산 (Out이 0이면 In 반환)"""
        if self.outCount == 0:
            return float(self.inCount)
        return self.inCount / self.outCount

    def resetInOut(self):
        """In/Out 카운터 초기화"""
        self.inCount = 0
        self.outCount = 0

    def addVelocity(self, velocity: float):
        """속도 추가 및 가중 평균 계산 - Java RailEdge.addVelocity 기반"""
        if math.isnan(velocity) or math.isinf(velocity):
            return

        # 속도 범위 제한
        if velocity < MIN_VELOCITY:
            velocity = MIN_VELOCITY
        elif velocity > self.maxVelocity:
            velocity = self.maxVelocity

        self.lastVelocity = self.velocity

        # 가중 평균 적용
        if self.hisCnt > 0:
            self.velocity = (self.velocity * LAST_HIS_WEIGHT) + (velocity * (1.0 - LAST_HIS_WEIGHT))
        else:
            self.velocity = velocity

        self.changedVelocity = True

    def addHistory(self):
        """히스토리 카운트 증가"""
        self.hisCnt += 1

    def addVhlId(self, vhlId: str):
        """Vehicle ID 추가"""
        self.vhlIdMap[vhlId] = 0

    def removeVhlId(self, vhlId: str):
        """Vehicle ID 제거"""
        self.vhlIdMap.pop(vhlId, None)

    def getCost(self) -> float:
        """비용 계산 - 거리/속도"""
        vel = max(self.velocity, 1.0)
        # 거리(mm) / 속도(m/min) -> ms 변환
        return self.length / (vel * 1000 / 60 / 1000)

    def getDensity(self) -> float:
        """
        밀도 계산 - Java RailEdge.getDensity() 기반

        공식: density = (vhl_length × vhl_count) / effective_rail_length × 100
        effective_rail_length = rail_length - (rail_length % vhl_length)
        """
        # FAB별 차량 길이 결정
        if self.fabId.startswith("M14"):
            vhl_length = VHL_LENGTH_M14
        elif self.fabId.startswith("M16"):
            vhl_length = VHL_LENGTH_M16
        else:
            vhl_length = VHL_LENGTH_M14  # 기본값

        # 유효 레일 길이 계산
        rail_length = self.length - (self.length % vhl_length)
        if rail_length <= vhl_length:
            rail_length = vhl_length

        # 총 차량 점유 길이
        vhl_length_sum = vhl_length * len(self.vhlIdMap)

        # 밀도 계산 (%)
        density = (vhl_length_sum / rail_length) * 100.0

        return min(density, 100.0)  # 최대 100%

    def getAbsoluteVelocity(self) -> float:
        """
        절대 속도 계산

        공식: absoluteVelocity = current_velocity / max_velocity
        """
        if self.velocity <= 0 or self.maxVelocity <= 0:
            return 0.0
        return min(self.velocity / self.maxVelocity, 1.0)


# ============================================================
# HID Zone 클래스 - HID_Zone_Master.csv 기반
# ============================================================
@dataclass
class LanePair:
    """Lane 쌍 (시작노드→종료노드)"""
    fromNode: int
    toNode: int

    @staticmethod
    def parse(lane_str: str) -> 'LanePair':
        """'3048→3023' 형식 파싱"""
        parts = lane_str.strip().split('→')
        if len(parts) == 2:
            return LanePair(int(parts[0]), int(parts[1]))
        return None

@dataclass
class HIDZone:
    """
    HID Zone 데이터 - HID_Zone_Master.csv 기반

    컬럼:
    - Zone_ID: Zone 고유 식별 번호
    - HID_No: HID 장비 번호 (HID-OHT-001)
    - Bay_Zone: Bay/구역 코드 (B01, M02, OS1 등)
    - Sub_Region: 하위 구역 번호 (1, 2, 3...)
    - Full_Name: 전체 명칭 (HID-B01-1(001))
    - Territory: 소속 영역 번호 (전체 1)
    - Type: Zone 유형 (HID = Hoist ID 구간)
    - IN_Count: 진입 Lane 개수
    - OUT_Count: 진출 Lane 개수
    - IN_Lanes: 진입 Lane 노드 쌍 리스트
    - OUT_Lanes: 진출 Lane 노드 쌍 리스트
    - Vehicle_Max: Zone 내 최대 허용 OHT 대수
    - Vehicle_Precaution: 주의 알람 발생 기준 OHT 대수
    - Project: 프로젝트명
    - ZCU: ZCU 코드
    - HID_Type: HID 타입 (HID4, MiniHID)
    """
    zoneId: int
    territory: int
    zoneType: str
    inCount: int
    outCount: int
    inLanes: List[LanePair] = field(default_factory=list)
    outLanes: List[LanePair] = field(default_factory=list)
    vehicleMax: int = 37
    vehiclePrecaution: int = 35

    # 새로운 필드 (HID_Zone_Master.csv)
    hidNo: str = ""           # HID-OHT-001
    bayZone: str = ""         # B01, M02, OS1
    subRegion: int = 0        # 하위 구역 번호
    fullName: str = ""        # HID-B01-1(001)
    project: str = ""         # M14 Project Ph-1
    zcu: str = ""             # ZC504A
    hidType: str = ""         # HID4, MiniHID

    # 실시간 상태
    currentVehicles: Set[str] = field(default_factory=set)

    # 통계
    totalInCount: int = 0   # 총 진입 횟수
    totalOutCount: int = 0  # 총 진출 횟수

    def __post_init__(self):
        if self.currentVehicles is None:
            self.currentVehicles = set()

    @property
    def vehicleCount(self) -> int:
        """현재 Zone 내 차량 수"""
        return len(self.currentVehicles)

    @property
    def isFull(self) -> bool:
        """Zone이 꽉 찼는지 여부"""
        return self.vehicleCount >= self.vehicleMax

    @property
    def isPrecautionLevel(self) -> bool:
        """주의 레벨인지 여부"""
        return self.vehicleCount >= self.vehiclePrecaution

    @property
    def occupancyRate(self) -> float:
        """점유율 (%)"""
        if self.vehicleMax <= 0:
            return 0.0
        return (self.vehicleCount / self.vehicleMax) * 100.0

    @property
    def status(self) -> str:
        """Zone 상태: NORMAL, PRECAUTION, FULL"""
        if self.isFull:
            return "FULL"
        elif self.isPrecautionLevel:
            return "PRECAUTION"
        return "NORMAL"

    def addVehicle(self, vehicleId: str, viaInLane: bool = True) -> bool:
        """차량 추가 (성공 여부 반환)

        Args:
            vehicleId: 차량 ID
            viaInLane: True면 IN Lane으로 진입, False면 OUT Lane으로 진입
        """
        if self.isFull:
            return False
        self.currentVehicles.add(vehicleId)
        if viaInLane:
            self.totalInCount += 1  # IN Lane 통한 진입
        else:
            self.totalOutCount += 1  # OUT Lane 통한 진입
        return True

    def removeVehicle(self, vehicleId: str):
        """차량 제거 (다른 Zone으로 이동 시)"""
        if vehicleId in self.currentVehicles:
            self.currentVehicles.discard(vehicleId)

    def hasVehicle(self, vehicleId: str) -> bool:
        """차량 존재 여부"""
        return vehicleId in self.currentVehicles

    def getInOutRatio(self) -> float:
        """IN Lane 진입 / OUT Lane 진입 비율"""
        if self.totalOutCount == 0:
            return float(self.totalInCount) if self.totalInCount > 0 else 1.0
        return self.totalInCount / self.totalOutCount

    def getTotalEntries(self) -> int:
        """총 진입 횟수 (IN + OUT)"""
        return self.totalInCount + self.totalOutCount

    def resetStats(self):
        """통계 초기화"""
        self.totalInCount = 0
        self.totalOutCount = 0

    def containsLane(self, fromNode: int, toNode: int) -> Tuple[bool, bool]:
        """
        특정 Lane이 이 Zone의 IN 또는 OUT Lane인지 확인
        Returns: (isInLane, isOutLane)
        """
        isIn = any(lane.fromNode == fromNode and lane.toNode == toNode for lane in self.inLanes)
        isOut = any(lane.fromNode == fromNode and lane.toNode == toNode for lane in self.outLanes)
        return (isIn, isOut)


@dataclass
class Station:
    """
    Station 데이터 - station.dat 파일 기반

    포맷: STATION = ID, "Type", ?, "Name", HexFlags, ?, NodeAddress, 0,0,0,0, ?, "EquipType", ...

    타입:
    - MAINTENANCE: 유지보수 스테이션
    - DUAL_ACCESS: 양방향 접근 스테이션 (대부분)
    - ACQUIRE: 캐리어 획득 스테이션
    - DEPOSIT: 캐리어 적재 스테이션
    - DUMMY: 더미 스테이션

    장비타입:
    - ZFS_RIGHT/ZFS_LEFT: ZFS 장비
    - UNIVERSAL: 범용 장비
    - MANUAL_ONLY: 수동 전용
    - MTL_SWITCHBACK/MTL_ELEVATOR: MTL 관련
    """
    stationId: int
    stationType: str          # MAINTENANCE, DUAL_ACCESS, etc.
    stationName: str          # 4EBE0102_2, ST-00001, etc.
    nodeAddress: int          # 노드 주소 (좌표 매핑용)
    equipType: str            # UNIVERSAL, ZFS_RIGHT, etc.
    hexFlags: int = 0
    distance1: int = 0
    distance2: int = 0

    # 좌표 (노드에서 가져옴)
    x: float = 0.0
    y: float = 0.0
    hasCoords: bool = False


def parse_stations(filepath: str) -> Dict[int, Station]:
    """
    station.dat 파일 파싱

    포맷: STATION = ID, "Type", ?, "Name", HexFlags, ?, NodeAddress, 0,0,0,0, ?, "EquipType", 0,0,0, Dist1, Dist2, ...
    """
    stations: Dict[int, Station] = {}

    if not os.path.exists(filepath):
        print(f"[경고] Station 파일 없음: {filepath}")
        return stations

    print(f"Station 파싱: {filepath}")

    import re
    pattern = re.compile(
        r'STATION\s*=\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*\d+\s*,\s*"([^"]+)"\s*,\s*(0x[0-9A-Fa-f]+|\d+)\s*,\s*\d+\s*,\s*(\d+)\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*"([^"]+)"'
    )

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue

                match = pattern.search(line)
                if match:
                    station_id = int(match.group(1))
                    station_type = match.group(2)
                    station_name = match.group(3)
                    hex_flags_str = match.group(4)
                    node_address = int(match.group(5))
                    equip_type = match.group(6)

                    # hex 플래그 파싱
                    if hex_flags_str.startswith('0x'):
                        hex_flags = int(hex_flags_str, 16)
                    else:
                        hex_flags = int(hex_flags_str)

                    stations[station_id] = Station(
                        stationId=station_id,
                        stationType=station_type,
                        stationName=station_name,
                        nodeAddress=node_address,
                        equipType=equip_type,
                        hexFlags=hex_flags
                    )

        print(f"  -> {len(stations)}개 Station 로드됨")

        # 타입별 통계
        type_counts = {}
        for s in stations.values():
            type_counts[s.stationType] = type_counts.get(s.stationType, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"     - {t}: {c}개")

    except Exception as e:
        print(f"[오류] Station 파싱 실패: {e}")

    return stations


def parse_hid_zones(filepath: str) -> Dict[int, HIDZone]:
    """
    HID_Zone_Master.csv 파일 파싱

    CSV 컬럼:
    Zone_ID,HID_No,Bay_Zone,Sub_Region,Full_Name,Territory,Type,IN_Count,OUT_Count,
    IN_Lanes,OUT_Lanes,Vehicle_Max,Vehicle_Precaution,Project,ZCU,HID_Type
    """
    zones: Dict[int, HIDZone] = {}

    # 파일 경로 처리 (상대경로면 스크립트 위치 기준)
    if not os.path.isabs(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filepath)

    if not os.path.exists(filepath):
        print(f"[경고] HID Zone 파일 없음: {filepath}")
        return zones

    print(f"HID Zone 파싱: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    zone_id = int(row['Zone_ID'])

                    # IN Lanes 파싱 (세미콜론으로 구분)
                    in_lanes = []
                    in_lanes_str = row.get('IN_Lanes', '')
                    if in_lanes_str:
                        for lane_str in in_lanes_str.split(';'):
                            lane = LanePair.parse(lane_str.strip())
                            if lane:
                                in_lanes.append(lane)

                    # OUT Lanes 파싱
                    out_lanes = []
                    out_lanes_str = row.get('OUT_Lanes', '')
                    if out_lanes_str:
                        for lane_str in out_lanes_str.split(';'):
                            lane = LanePair.parse(lane_str.strip())
                            if lane:
                                out_lanes.append(lane)

                    # 시뮬레이션용 임계값 조정
                    # OHT 대수에 따라 적절한 Zone 용량 설정
                    # 원본: 37/35, 시뮬레이션: 더 여유있게 설정
                    csv_max = int(row.get('Vehicle_Max', 37))
                    csv_precaution = int(row.get('Vehicle_Precaution', 35))

                    # 시뮬레이션용: 원본 값의 약 1/3로 조정 (OHT 500대 기준)
                    sim_max = max(10, csv_max // 3)  # 최소 10대
                    sim_precaution = max(7, csv_precaution // 3)  # 최소 7대

                    # Sub_Region 파싱 (빈 값이면 0)
                    sub_region_str = row.get('Sub_Region', '')
                    sub_region = int(sub_region_str) if sub_region_str else 0

                    zone = HIDZone(
                        zoneId=zone_id,
                        territory=int(row.get('Territory', 1)),
                        zoneType=row.get('Type', 'HID'),
                        inCount=int(row.get('IN_Count', 0)),
                        outCount=int(row.get('OUT_Count', 0)),
                        inLanes=in_lanes,
                        outLanes=out_lanes,
                        vehicleMax=sim_max,
                        vehiclePrecaution=sim_precaution,
                        # 새로운 필드 (HID_Zone_Master.csv)
                        hidNo=row.get('HID_No', ''),
                        bayZone=row.get('Bay_Zone', ''),
                        subRegion=sub_region,
                        fullName=row.get('Full_Name', ''),
                        project=row.get('Project', ''),
                        zcu=row.get('ZCU', ''),
                        hidType=row.get('HID_Type', '')
                    )
                    zones[zone_id] = zone

                except (ValueError, KeyError) as e:
                    print(f"  [경고] Zone 파싱 오류 (행 무시): {e}")
                    continue

        print(f"  로드된 Zone: {len(zones)}개")

        # 통계 출력
        total_in_lanes = sum(len(z.inLanes) for z in zones.values())
        total_out_lanes = sum(len(z.outLanes) for z in zones.values())
        total_capacity = sum(z.vehicleMax for z in zones.values())
        print(f"  총 IN Lane: {total_in_lanes}, OUT Lane: {total_out_lanes}")
        print(f"  총 허용 차량: {total_capacity}대")

    except Exception as e:
        print(f"[오류] HID Zone 파일 읽기 실패: {e}")

    return zones


def build_lane_to_zone_map(zones: Dict[int, HIDZone]) -> Tuple[Dict[Tuple[int,int], int], Dict[Tuple[int,int], int]]:
    """
    Lane -> Zone 매핑 테이블 생성

    IN Lane과 OUT Lane 모두 해당 Zone으로의 진입 Lane임
    (OUT은 퇴출이 아니라 다른 방향에서의 진입)

    Returns:
        (in_lane_map, out_lane_map):
        - in_lane_map: {(fromNode, toNode): zoneId} - IN 방향 진입 Lane
        - out_lane_map: {(fromNode, toNode): zoneId} - OUT 방향 진입 Lane
    """
    in_lane_map = {}
    out_lane_map = {}

    for zone_id, zone in zones.items():
        for lane in zone.inLanes:
            in_lane_map[(lane.fromNode, lane.toNode)] = zone_id
        for lane in zone.outLanes:
            out_lane_map[(lane.fromNode, lane.toNode)] = zone_id

    return in_lane_map, out_lane_map


# ============================================================
# 레이아웃 파서
# ============================================================
def parse_layout(filepath: str) -> Tuple[Dict[int, Node], List[Tuple[int, int, float]]]:
    print(f"레이아웃 파싱: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 노드 데이터
    a_match = re.search(r'const A=(\[.*?\]);', content, re.DOTALL)
    nodes_data = json.loads(a_match.group(1))
    nodes = {}
    for n in nodes_data:
        nodes[n['no']] = Node(no=n['no'], x=n['x'], y=n['y'], stations=n.get('stations', []))

    # 연결 데이터
    c_match = re.search(r'const C=(\[.*?\]);', content, re.DOTALL)
    connections = json.loads(c_match.group(1))

    edges = []
    for conn in connections:
        from_n, to_n = conn[0], conn[1]
        if from_n in nodes and to_n in nodes:
            n1, n2 = nodes[from_n], nodes[to_n]
            dist = math.sqrt((n2.x - n1.x)**2 + (n2.y - n1.y)**2)
            edges.append((from_n, to_n, dist))

    print(f"  노드: {len(nodes)}, 엣지: {len(edges)}")
    return nodes, edges

# ============================================================
# 시뮬레이션 엔진
# ============================================================
class SimulationEngine:
    def __init__(self, nodes: Dict[int, Node], edges: List[Tuple[int, int, float]]):
        self.nodes = nodes
        self.edges_raw = edges
        self.vehicles: Dict[str, Vehicle] = {}

        # 그래프
        self.graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for from_n, to_n, dist in edges:
            self.graph[from_n].append((to_n, dist))

        # 엣지 맵 (거리)
        self.edge_dist_map = {(e[0], e[1]): e[2] for e in edges}

        # RailEdge 맵 - Java RailEdge 기반
        self.rail_edge_map: Dict[str, RailEdge] = {}
        for from_n, to_n, dist in edges:
            edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{from_n}-{to_n}"
            self.rail_edge_map[edge_id] = RailEdge(
                edgeId=edge_id,
                fromNodeId=from_n,
                toNodeId=to_n,
                length=dist,
                fabId=FAB_ID,
                maxVelocity=MAX_VELOCITY,
                velocity=MAX_VELOCITY * 0.8  # 초기 속도
            )

        # ============================================================
        # HID Zone 관리 - HID_Zone_Master.csv 기반
        # ============================================================
        self.hid_zones: Dict[int, HIDZone] = parse_hid_zones(HID_ZONE_CSV_PATH)
        self.in_lane_to_zone, self.out_lane_to_zone = build_lane_to_zone_map(self.hid_zones)

        # 차량 -> Zone 매핑 (현재 어느 Zone에 있는지)
        self.vehicle_zone_map: Dict[str, int] = {}

        # Station 로드 및 좌표 매핑
        self.stations: Dict[int, Station] = parse_stations(STATION_DAT_PATH)
        self._map_station_coordinates()

        # 노드 -> Station 매핑 (하나의 노드에 여러 Station이 있을 수 있음)
        self.node_to_stations: Dict[int, List[Station]] = {}
        for station in self.stations.values():
            if station.nodeAddress not in self.node_to_stations:
                self.node_to_stations[station.nodeAddress] = []
            self.node_to_stations[station.nodeAddress].append(station)
        print(f"Node-Station 매핑: {len(self.node_to_stations)}개 노드에 Station 존재")

        # 속도 설정 - MCP 속도 테이블 기반
        self.base_velocity = 180.0  # m/min 기본 속도
        self.alpha = 0.3  # EMA 평활화 계수

        # CSV 데이터 버퍼
        self.vehicle_buffer: List[dict] = []
        self.rail_buffer: Dict[Tuple[int,int], dict] = defaultdict(lambda: {'pass_cnt': 0, 'velocities': []})
        self.zone_buffer: List[dict] = []  # Zone 상태 버퍼

        # 시작 시간
        self.start_time = datetime.now()
        self.simulation_tick = 0

    def init_vehicles(self, count: int):
        """Vehicle 초기화 - Java Vhl 기반 + HID Zone 연동"""
        node_list = list(self.nodes.keys())
        for i in range(count):
            vid = f"V{i+1:05d}"  # Java 형식: V00001
            start = random.choice(node_list)
            node = self.nodes[start]

            v = Vehicle(
                vehicleId=vid,
                currentNode=start,
                x=node.x,
                y=node.y
            )

            # UDP 상태 초기화
            v.udpState.state = VHL_STATE.RUN
            v.udpState.currentAddress = start
            v.udpState.isOnline = True
            v.udpState.receivedTime = int(datetime.now().timestamp() * 1000)

            self.vehicles[vid] = v

            # Zone에 차량 할당 (시작 노드 기반)
            self._assign_vehicle_to_zone(v)

            self._assign_task(v)

    def _assign_vehicle_to_zone(self, v: Vehicle):
        """차량을 적절한 Zone에 할당"""
        # 현재 Lane으로 Zone 찾기
        lane_key = (v.currentNode, v.nextNode if v.nextNode else v.currentNode)

        # IN Lane에서 Zone 찾기
        if lane_key in self.in_lane_to_zone:
            zone_id = self.in_lane_to_zone[lane_key]
            if zone_id in self.hid_zones:
                zone = self.hid_zones[zone_id]
                if zone.addVehicle(v.vehicleId):
                    self.vehicle_zone_map[v.vehicleId] = zone_id
                    v.udpState.hidId = zone_id
                    return

        # OUT Lane에서 Zone 찾기
        if lane_key in self.out_lane_to_zone:
            zone_id = self.out_lane_to_zone[lane_key]
            if zone_id in self.hid_zones:
                zone = self.hid_zones[zone_id]
                if zone.addVehicle(v.vehicleId):
                    self.vehicle_zone_map[v.vehicleId] = zone_id
                    v.udpState.hidId = zone_id
                    return

        # 어떤 Lane에도 매칭되지 않으면 랜덤 Zone 할당 (시뮬레이션 용)
        if self.hid_zones:
            available_zones = [z for z in self.hid_zones.values() if not z.isFull]
            if available_zones:
                zone = random.choice(available_zones)
                zone.addVehicle(v.vehicleId)
                self.vehicle_zone_map[v.vehicleId] = zone.zoneId
                v.udpState.hidId = zone.zoneId

    def _map_station_coordinates(self):
        """Station의 nodeAddress를 기반으로 좌표 매핑"""
        mapped_count = 0
        for station in self.stations.values():
            node = self.nodes.get(station.nodeAddress)
            if node:
                station.x = node.x
                station.y = node.y
                station.hasCoords = True
                mapped_count += 1
        print(f"Station 좌표 매핑: {mapped_count}/{len(self.stations)}개 성공")

    def _update_vehicle_zone(self, v: Vehicle, old_lane: Tuple[int, int], new_lane: Tuple[int, int]):
        """차량이 이동할 때 Zone 업데이트

        IN Lane과 OUT Lane 모두 해당 Zone으로의 진입 Lane임
        - IN Lane 진입 시: Zone에 추가 (IN 카운터 증가)
        - OUT Lane 진입 시: Zone에 추가 (OUT 카운터 증가)
        - 다른 Zone 진입 시: 기존 Zone에서 제거
        """
        current_zone_id = self.vehicle_zone_map.get(v.vehicleId)

        # new_lane이 IN 또는 OUT Lane인지 확인 -> 해당 Zone에 진입
        in_zone_id = self.in_lane_to_zone.get(new_lane)
        out_zone_id = self.out_lane_to_zone.get(new_lane)

        # IN 또는 OUT Lane을 통해 진입하는 Zone ID
        new_zone_id = in_zone_id if in_zone_id is not None else out_zone_id
        via_in_lane = in_zone_id is not None  # IN Lane 통한 진입 여부

        if new_zone_id is not None:
            # 이미 같은 Zone에 있으면 무시
            if new_zone_id == current_zone_id:
                return

            # 다른 Zone에 있었으면 먼저 제거
            if current_zone_id is not None and current_zone_id in self.hid_zones:
                self.hid_zones[current_zone_id].removeVehicle(v.vehicleId)

            # 새 Zone에 추가 (IN/OUT 구분)
            if new_zone_id in self.hid_zones:
                new_zone = self.hid_zones[new_zone_id]
                if new_zone.addVehicle(v.vehicleId, viaInLane=via_in_lane):
                    self.vehicle_zone_map[v.vehicleId] = new_zone_id
                    v.udpState.hidId = new_zone_id
                else:
                    # Zone이 꽉 찬 경우 - 정체 발생
                    v.udpState.state = VHL_STATE.JAM
                    v.velocity = 0.0

    def get_vehicle_zone(self, v: Vehicle) -> Optional[HIDZone]:
        """차량이 속한 Zone 반환"""
        zone_id = self.vehicle_zone_map.get(v.vehicleId)
        if zone_id is not None:
            return self.hid_zones.get(zone_id)
        return None

    def _get_blocking_vehicle_ahead(self, v: Vehicle) -> Tuple[Optional[Vehicle], float]:
        """같은 엣지에서 앞에 정지/정체 중인 차량 확인 (체인 효과 포함)
        Returns: (앞에 있는 정지 차량, 거리) 또는 (None, 0)
        """
        closest_blocking = None
        min_distance = float('inf')

        # 같은 엣지 (currentNode -> nextNode) 에 있는 다른 차량들 확인
        for other_v in self.vehicles.values():
            if other_v.vehicleId == v.vehicleId:
                continue

            # 같은 엣지에 있는지 확인
            if other_v.currentNode == v.currentNode and other_v.nextNode == v.nextNode:
                # 앞에 있는지 확인 (positionRatio가 더 큰 차량)
                if other_v.positionRatio > v.positionRatio:
                    distance = other_v.positionRatio - v.positionRatio

                    # 정지 조건: JAM, STOP 상태이거나 속도가 매우 낮은 경우
                    is_blocked = (
                        other_v.udpState.state == VHL_STATE.JAM or
                        other_v.udpState.state == VHL_STATE.STOP or
                        other_v.velocity < 3.0  # 속도 3 미만이면 정지로 간주
                    )

                    # 앞에 정지 차량이 있으면 무조건 blocking (거리 무관)
                    if is_blocked and distance < min_distance:
                        min_distance = distance
                        closest_blocking = other_v

            # 다음 엣지에 정지 차량이 있는지 확인 (내가 곧 그 엣지에 진입할 때)
            if v.positionRatio > 0.7:  # 현재 엣지 70% 이상 진행
                if other_v.currentNode == v.nextNode:
                    # 다음 엣지의 시작 부분 (30% 이내)에 정지 차량
                    if other_v.positionRatio < 0.3:
                        is_blocked = (
                            other_v.udpState.state == VHL_STATE.JAM or
                            other_v.udpState.state == VHL_STATE.STOP or
                            other_v.velocity < 3.0
                        )
                        if is_blocked:
                            # 가상 거리 계산
                            virtual_distance = (1.0 - v.positionRatio) + other_v.positionRatio
                            if virtual_distance < min_distance:
                                min_distance = virtual_distance
                                closest_blocking = other_v

        return closest_blocking, min_distance if closest_blocking else 0

    def _find_path(self, start: int, end: int) -> List[int]:
        """경로 탐색 - Dijkstra 알고리즘

        Args:
            start: 출발 노드
            end: 도착 노드

        Returns:
            경로 노드 리스트
        """
        if start == end or start not in self.nodes or end not in self.nodes:
            return []

        # Dijkstra 알고리즘
        dist = {start: 0}
        prev = {}
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == end:
                break
            if d > dist.get(u, float('inf')):
                continue
            for v, w in self.graph[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if end not in prev:
            return []

        path = []
        curr = end
        while curr in prev:
            path.append(curr)
            curr = prev[curr]
        path.append(start)
        path.reverse()
        return path

    def _assign_task(self, v: Vehicle):
        """작업 할당 - Java 기반 상태 설정"""
        dest = random.choice(list(self.nodes.keys()))
        attempts = 0
        while dest == v.currentNode and attempts < 10:
            dest = random.choice(list(self.nodes.keys()))
            attempts += 1

        path = self._find_path(v.currentNode, dest)
        if len(path) > 1:
            v.path = path
            v.pathIndex = 0
            v.destination = dest
            v.nextNode = path[1]

            # Java 기반 상태 설정
            v.udpState.state = VHL_STATE.RUN

            # 속도 계산 조건에 맞는 runCycle, vhlCycle 설정
            # ACQUIRE(3) 또는 DEPOSIT(4) + ACQUIRE_MOVING(2) 또는 DEPOSIT_MOVING(4)
            is_acquire = random.random() < 0.5
            if is_acquire:
                v.udpState.runCycle = RUN_CYCLE.ACQUIRE
                v.udpState.vhlCycle = VHL_CYCLE.ACQUIRE_MOVING
            else:
                v.udpState.runCycle = RUN_CYCLE.DEPOSIT
                v.udpState.vhlCycle = VHL_CYCLE.DEPOSIT_MOVING

            # 적재 상태
            v.udpState.isFull = random.random() < 0.5
            if v.udpState.isFull:
                v.carrierId = f"FOUP{random.randint(1000, 9999)}"
                v.udpState.udpCarrierId = v.carrierId
            else:
                v.carrierId = ""
                v.udpState.udpCarrierId = ""

            # 목적지 설정
            v.udpState.destStationId = str(dest)
            v.udpState.nextAddress = v.nextNode
            v.udpState.currentAddress = v.currentNode

    def update(self, dt: float):
        """모든 Vehicle 업데이트"""
        self.simulation_tick += 1
        current_time_ms = int(datetime.now().timestamp() * 1000)

        for v in self.vehicles.values():
            self._update_vehicle(v, dt, current_time_ms)

    def _check_velocity_conditions(self, v: Vehicle) -> bool:
        """
        속도 계산 5가지 조건 확인 - Java OhtMsgWorkerRunnable._checkVehicleMovement() 기반

        조건:
        1. 메시지 수신 시간 차이 < 1분 (60초)
        2. Vehicle 상태: RUN, OBS_BZ_STOP, JAM, E84_TIMEOUT 중 하나
        3. runCycle, vhlCycle이 이전과 동일
        4. runCycle: ACQUIRE 또는 DEPOSIT
        5. vhlCycle: ACQUIRE_MOVING 또는 DEPOSIT_MOVING
        """
        # 조건 1: 시간 차이 < 60초
        time_diff = v.udpState.receivedTime - v.lastUdpState.receivedTime
        if time_diff >= 60 * 1000:
            return False

        # 조건 2: 상태 확인
        valid_states = [VHL_STATE.RUN, VHL_STATE.OBS_BZ_STOP, VHL_STATE.JAM, VHL_STATE.E84_TIMEOUT]
        if v.udpState.state not in valid_states:
            return False

        # 조건 3: Cycle 일치
        if v.udpState.runCycle != v.lastUdpState.runCycle:
            return False
        if v.udpState.vhlCycle != v.lastUdpState.vhlCycle:
            return False

        # 조건 4: runCycle 확인
        if v.udpState.runCycle not in [RUN_CYCLE.ACQUIRE, RUN_CYCLE.DEPOSIT]:
            return False

        # 조건 5: vhlCycle 확인
        if v.udpState.vhlCycle not in [VHL_CYCLE.ACQUIRE_MOVING, VHL_CYCLE.DEPOSIT_MOVING]:
            return False

        return True

    def _calculate_velocity(self, v: Vehicle, rail_edge: RailEdge, last_rail_edge: Optional[RailEdge]) -> Optional[float]:
        """
        속도 계산 - Java OhtMsgWorkerRunnable._setRailEdgeVelocity() 기반

        공식: 속도(ν) = Δx / Δt × 60.0 (m/min)
               Δx: 이동 거리 (mm)
               Δt: 경과 시간 (ms)

        Case 1: 동일 RailEdge 내 이동 또는 연속 RailEdge
        Case 2: 비연속 RailEdge (경로 탐색 필요)
        """
        if not self._check_velocity_conditions(v):
            return None

        if last_rail_edge is None:
            return None

        # 시간 차이 (ms)
        elapsed_ms = v.udpState.receivedTime - v.lastUdpState.receivedTime
        if elapsed_ms <= 0:
            return None

        # UDP 거리는 100mm 단위이므로 mm로 변환
        current_distance_mm = v.udpState.distance * UDP_DISTANCE_UNIT
        last_distance_mm = v.lastUdpState.distance * UDP_DISTANCE_UNIT

        # 동일 RailEdge인 경우
        if rail_edge.edgeId == last_rail_edge.edgeId:
            # 같은 엣지 내에서 이동
            ran_distance = abs(current_distance_mm - last_distance_mm)
        # Case 1: α의 시작점과 β의 도착지점이 동일 (연속된 RailEdge)
        elif last_rail_edge.toNodeId == rail_edge.fromNodeId:
            # 거리 = (이전 엣지 길이 - 이전 거리) + 현재 거리
            ran_distance = (last_rail_edge.length - last_distance_mm) + current_distance_mm
        else:
            # Case 2: 서로 다른 RailEdge - 단순히 현재 거리만 사용
            ran_distance = current_distance_mm

        if ran_distance <= 0:
            return None

        # 속도 계산: mm / ms * 60.0 = m/min
        # (mm / ms) * (1m/1000mm) * (60000ms/min) = m/min
        speed = (ran_distance / elapsed_ms) * 60.0

        return speed

    def _update_vehicle(self, v: Vehicle, dt: float, current_time_ms: int):
        """Vehicle 업데이트 - Java processOhtReport() 기반"""

        # JAM 상태 (데드락)인 경우 - 밀도가 낮아지면 복구
        if v.udpState.state == VHL_STATE.JAM:
            edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{v.currentNode}-{v.nextNode}"
            rail_edge = self.rail_edge_map.get(edge_id)

            # JAM 복구 조건 완화 - 더 빨리 복구되도록
            recovery_chance = 0.15  # 기본 15% 복구 확률
            if rail_edge:
                density = rail_edge.getDensity()
                # 밀도가 낮을수록 복구 확률 증가
                if density < 50:
                    recovery_chance = 0.3  # 밀도 50% 미만이면 30%
                if density < 30:
                    recovery_chance = 0.5  # 밀도 30% 미만이면 50%

            if random.random() < recovery_chance:
                v.udpState.state = VHL_STATE.RUN
                v.velocity = 50.0  # 저속으로 재출발
                if rail_edge:
                    rail_edge.recordOut()  # 정체 해소 시 Out 기록
            else:
                return  # JAM 상태 유지, 이동 안함

        # 정지 상태이면 일정 확률로 작업 할당
        if v.udpState.state != VHL_STATE.RUN or not v.path:
            if random.random() < 0.05:
                self._assign_task(v)
            return

        # ============================================================
        # 앞 차량 정지 시 뒤 차량도 정지 (충돌 방지 - 체인 효과)
        # ============================================================
        blocking_vehicle, distance_ratio = self._get_blocking_vehicle_ahead(v)
        if blocking_vehicle:
            # 거리에 따른 속도 조절 (체인 효과로 뒤 차량도 정지)
            # 앞 차량이 정지/JAM이면 뒤 차량도 반드시 멈춰야 함
            if distance_ratio < 0.1:  # 매우 가까우면 완전 정지
                v.velocity = 0.0
                return  # 이동 완전 중단
            elif distance_ratio < 0.2:  # 가까우면 정지 (체인 효과 핵심)
                v.velocity = 0.0
                return  # 정지 - 앞 차가 움직일 때까지 대기
            elif distance_ratio < 0.3:  # 근접하면 거의 정지
                v.velocity = 1.0  # 아주 천천히
                return
            elif distance_ratio < 0.5:  # 어느 정도 거리면 서행
                v.velocity = min(v.velocity, 10.0)
            elif distance_ratio < 0.7:
                v.velocity = min(v.velocity, 30.0)  # 속도 제한

        # 이전 엣지 (상태 복사 전에 먼저 계산해야 함!)
        last_edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{v.lastUdpState.currentAddress}-{v.lastUdpState.nextAddress}"
        last_rail_edge = self.rail_edge_map.get(last_edge_id)

        # 이전 상태 저장 (last_edge_id 계산 후에 복사)
        v.copyCurrentVhlUdpStateToLast()

        # 현재 시간 기록
        v.udpState.receivedTime = current_time_ms

        # 속도 계산 - MCP 테이블 기반 + 노이즈
        speed_index = random.randint(20, 35)  # 100~230 m/min 범위
        target_vel = MCP_SPEED_TABLE.get(speed_index, self.base_velocity)
        target_vel *= (0.85 + random.random() * 0.3)  # ±15% 변동

        # EMA 평활화
        v.smoothedVelocity = self.alpha * target_vel + (1 - self.alpha) * v.smoothedVelocity
        v.velocity = round(v.smoothedVelocity, 2)

        # 현재 엣지
        edge_dist = self.edge_dist_map.get((v.currentNode, v.nextNode), 100)
        edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{v.currentNode}-{v.nextNode}"
        rail_edge = self.rail_edge_map.get(edge_id)

        # 이동 (m/min -> mm/update)
        # v.velocity (m/min) * dt (sec) / 60 (sec/min) * 1000 (mm/m) = mm 이동
        move_mm = v.velocity * (dt / 60.0) * 1000
        v.positionRatio += move_mm / max(edge_dist, 1)

        # UDP 상태 업데이트
        v.udpState.distance = v.positionRatio * edge_dist  # mm 단위

        # 속도 계산 및 RailEdge에 적용
        if rail_edge:
            calculated_speed = self._calculate_velocity(v, rail_edge, last_rail_edge)
            if calculated_speed is not None:
                rail_edge.addVelocity(calculated_speed)
            rail_edge.addHistory()
            rail_edge.addVhlId(v.vehicleId)

            # 이전 엣지에서 제거 + In/Out 기록
            if last_rail_edge and last_rail_edge.edgeId != rail_edge.edgeId:
                # 엣지가 변경됨 - Out/In 기록
                last_rail_edge.removeVhlId(v.vehicleId)
                last_rail_edge.recordOut()  # 이전 엣지에서 진출
                rail_edge.recordIn()        # 현재 엣지로 진입
            elif last_rail_edge is None:
                # 첫 이동 또는 이전 엣지 정보 없음 - In만 기록
                rail_edge.recordIn()

        # Rail traffic 기록
        key = (v.currentNode, v.nextNode)
        self.rail_buffer[key]['pass_cnt'] += 1
        self.rail_buffer[key]['velocities'].append(v.velocity)

        # ============================================================
        # HID Zone 기반 정체 체크 (Vehicle_Max, Vehicle_Precaution)
        # ============================================================
        current_zone = self.get_vehicle_zone(v)
        if current_zone:
            # Zone이 주의 레벨 이상이면 속도 감소 (완만하게)
            if current_zone.isPrecautionLevel:
                occupancy = current_zone.occupancyRate
                # 점유율에 따른 속도 조절: 70%에서 시작, 100%이면 30%로
                speed_factor = max(0.3, 1.0 - (occupancy - 70) / 100.0)
                v.velocity *= speed_factor

            # Zone이 꽉 찬 경우 정체 가능성 (확률 낮춤)
            if current_zone.isFull:
                if random.random() < 0.05:  # 5% 확률로 정체 (기존 30%)
                    v.udpState.state = VHL_STATE.JAM
                    v.velocity = 0.0
                    return

        # 자연스러운 데드락 발생 - 밀도 높고 In >> Out이면 JAM 상태로 전환
        # (확률 대폭 낮춤 - 500대에서도 원활하게 동작하도록)
        if rail_edge:
            density = rail_edge.getDensity()
            in_out_ratio = rail_edge.getInOutRatio()
            # 조건: 밀도 > 80% AND In/Out 비율 > 2.0 이면 정체 발생 (확률적)
            if density > 80 and in_out_ratio > 2.0:
                jam_probability = min(0.2, (density - 80) / 200 + (in_out_ratio - 2) * 0.05)
                if random.random() < jam_probability:
                    v.udpState.state = VHL_STATE.JAM
                    v.velocity = 0.0
                    return  # 정체 상태로 전환, 이동 중단

        # 위치 계산 (보간)
        n1 = self.nodes.get(v.currentNode)
        n2 = self.nodes.get(v.nextNode)
        if n1 and n2:
            ratio = min(1.0, v.positionRatio)
            v.x = n1.x + (n2.x - n1.x) * ratio
            v.y = n1.y + (n2.y - n1.y) * ratio

        # 다음 노드 도달
        if v.positionRatio >= 1.0:
            old_lane = (v.currentNode, v.nextNode)
            v.positionRatio = 0.0
            v.udpState.distance = 0.0
            v.pathIndex += 1

            if v.pathIndex < len(v.path) - 1:
                v.currentNode = v.path[v.pathIndex]
                v.nextNode = v.path[v.pathIndex + 1]
                v.udpState.currentAddress = v.currentNode
                v.udpState.nextAddress = v.nextNode

                # Zone 업데이트 (Lane 변경 시)
                new_lane = (v.currentNode, v.nextNode)
                self._update_vehicle_zone(v, old_lane, new_lane)
            else:
                # 목적지 도착
                v.currentNode = v.destination
                v.udpState.state = VHL_STATE.STOP
                v.udpState.runCycle = RUN_CYCLE.NONE
                v.udpState.vhlCycle = VHL_CYCLE.NONE
                v.path = []

    def get_state(self) -> dict:
        """현재 상태 반환 (WebSocket용) - Java 기반 확장 + HID Zone 정보"""
        vehicles = []
        for v in self.vehicles.values():
            # 남은 경로 (현재 위치부터 목적지까지)
            remaining_path = []
            if v.path and v.pathIndex < len(v.path):
                remaining_path = v.path[v.pathIndex:]

            # 차량이 속한 Zone 정보
            zone_id = self.vehicle_zone_map.get(v.vehicleId, -1)

            # 현재 노드에 있는 Station 정보
            current_stations = []
            if v.currentNode in self.node_to_stations:
                for station in self.node_to_stations[v.currentNode]:
                    current_stations.append({
                        'stationId': station.stationId,
                        'stationName': station.stationName,
                        'stationType': station.stationType,
                        'equipType': station.equipType
                    })

            vehicles.append({
                'vehicleId': v.vehicleId,
                'x': round(v.x, 2),
                'y': round(v.y, 2),
                'state': v.state,
                'stateName': v.udpState.state.name,
                'isLoaded': v.isLoaded,
                'velocity': v.velocity,
                'currentNode': v.currentNode,
                'nextNode': v.nextNode,
                'destination': v.destination,
                'carrierId': v.carrierId,
                'path': remaining_path,
                # Java 기반 확장 필드
                'runCycle': v.udpState.runCycle.value,
                'runCycleName': v.udpState.runCycle.name,
                'vhlCycle': v.udpState.vhlCycle.value,
                'vhlCycleName': v.udpState.vhlCycle.name,
                'distance': round(v.udpState.distance, 2),
                'detailState': v.udpState.detailState.name,
                # HID Zone 정보
                'hidZoneId': zone_id,
                # 현재 Station 정보
                'currentStations': current_stations
            })

        # RailEdge In/Out 통계 계산
        total_in = sum(re.inCount for re in self.rail_edge_map.values())
        total_out = sum(re.outCount for re in self.rail_edge_map.values())

        # 데드락 위험 구간 찾기 (In > Out * 1.5 이고 밀도 > 50%)
        deadlock_edges = []
        for re in self.rail_edge_map.values():
            density = re.getDensity()
            if re.inCount > re.outCount * 1.5 and density > 50:
                deadlock_edges.append({
                    'edgeId': re.edgeId,
                    'fromNode': re.fromNodeId,
                    'toNode': re.toNodeId,
                    'inCount': re.inCount,
                    'outCount': re.outCount,
                    'density': round(density, 2),
                    'vhlCount': len(re.vhlIdMap)
                })

        # ============================================================
        # HID Zone 통계
        # ============================================================
        zone_stats = []
        precaution_zones = []
        full_zones = []

        for zone in self.hid_zones.values():
            zone_data = {
                'zoneId': zone.zoneId,
                'vehicleCount': zone.vehicleCount,
                'vehicleMax': zone.vehicleMax,
                'vehiclePrecaution': zone.vehiclePrecaution,
                'occupancyRate': round(zone.occupancyRate, 1),
                'status': zone.status,
                'totalIn': zone.totalInCount,
                'totalOut': zone.totalOutCount,
                'inOutRatio': round(zone.getInOutRatio(), 2)
            }
            zone_stats.append(zone_data)

            if zone.status == 'PRECAUTION':
                precaution_zones.append(zone_data)
            elif zone.status == 'FULL':
                full_zones.append(zone_data)

        # Zone 통계 요약
        total_zones = len(self.hid_zones)
        normal_zones = total_zones - len(precaution_zones) - len(full_zones)
        total_zone_capacity = sum(z.vehicleMax for z in self.hid_zones.values())
        total_zone_vehicles = sum(z.vehicleCount for z in self.hid_zones.values())

        return {
            'timestamp': datetime.now().isoformat(),
            'fabId': FAB_ID,
            'mcpName': MCP_NAME,
            'vehicles': vehicles,
            'stats': {
                'total': len(vehicles),
                'running': sum(1 for v in vehicles if v['state'] == 1),
                'stopped': sum(1 for v in vehicles if v['state'] == 2),
                'jammed': sum(1 for v in vehicles if v['state'] == 7),
                'loaded': sum(1 for v in vehicles if v['isLoaded'] == 1),
                'empty': sum(1 for v in vehicles if v['isLoaded'] == 0)
            },
            'inOutStats': {
                'totalIn': total_in,
                'totalOut': total_out,
                'ratio': round(total_in / max(total_out, 1), 2),
                'deadlockRiskCount': len(deadlock_edges)
            },
            'deadlockEdges': deadlock_edges[:10],  # 상위 10개만
            # HID Zone 통계
            'zoneStats': {
                'totalZones': total_zones,
                'normalZones': normal_zones,
                'precautionZones': len(precaution_zones),
                'fullZones': len(full_zones),
                'totalCapacity': total_zone_capacity,
                'totalVehiclesInZones': total_zone_vehicles,
                'overallOccupancy': round((total_zone_vehicles / max(total_zone_capacity, 1)) * 100, 1)
            },
            'precautionZoneList': precaution_zones[:5],  # 상위 5개
            'fullZoneList': full_zones[:5],  # 상위 5개
            # Zone별 현재 상태 (렌더링용)
            'zoneStatusMap': {
                z.zoneId: {
                    'status': z.status,
                    'vehicleCount': z.vehicleCount,
                    'occupancyRate': round(z.occupancyRate, 1)
                }
                for z in self.hid_zones.values()
            }
        }

    def generate_udp_message(self, v: Vehicle) -> str:
        """
        UDP 메시지 생성 - Java VHL_STATE_REPORT 형식 (23개 필드)

        Index | 필드명               | 설명
        ------|----------------------|---------------------------
        0     | MessageId            | 메시지 ID (2=VHL_STATE_REPORT)
        1     | McpName              | MCP 이름
        2     | VehicleId            | 차량 ID
        3     | State                | 상태 (1=RUN, 2=STOP, 등)
        4     | IsFull               | 재하정보 (0/1)
        5     | ErrorCode            | 에러 코드
        6     | IsOnline             | 통신 상태 (0/1)
        7     | CurrentAddress       | 현재 번지
        8     | Distance             | 거리 (100mm 단위)
        9     | NextAddress          | 다음 번지
        10    | RunCycle             | 실행 Cycle
        11    | VhlCycle             | Vehicle Cycle
        12    | CarrierId            | Carrier ID
        13    | Destination          | 목적지
        14    | EMState              | E/M 상태
        15    | GroupId              | Group ID
        16    | SourcePort           | 반송원 Port
        17    | DestPort             | 반송처 Port
        18    | Priority             | 우선도
        19    | DetailState          | 작업 상태 상세
        20    | RunDistance          | 주행 거리
        21    | CommandId            | Command ID
        22    | BayName              | Bay 명
        """
        # 거리는 100mm 단위로 전송 (내부는 mm 단위)
        distance_100mm = int(v.udpState.distance / UDP_DISTANCE_UNIT) if v.udpState.distance > 0 else 0

        return ','.join([
            '2',                                  # [0] MessageId (VHL_STATE_REPORT)
            MCP_NAME,                             # [1] McpName
            v.vehicleId,                          # [2] VehicleId
            v.udpState.state.value,               # [3] State
            '1' if v.udpState.isFull else '0',    # [4] IsFull
            v.udpState.errorCode or '0000',       # [5] ErrorCode
            '1' if v.udpState.isOnline else '0',  # [6] IsOnline
            str(v.udpState.currentAddress),       # [7] CurrentAddress
            str(distance_100mm),                  # [8] Distance (100mm 단위)
            str(v.udpState.nextAddress),          # [9] NextAddress
            v.udpState.runCycle.value,            # [10] RunCycle
            v.udpState.vhlCycle.value,            # [11] VhlCycle
            v.udpState.udpCarrierId or '',        # [12] CarrierId
            str(v.destination),                   # [13] Destination
            '00000000',                           # [14] EMState
            '0000',                               # [15] GroupId
            '',                                   # [16] SourcePort
            '',                                   # [17] DestPort
            '50',                                 # [18] Priority
            v.udpState.detailState.value,         # [19] DetailState
            str(v.udpState.runDistance),          # [20] RunDistance
            '',                                   # [21] CommandId
            '',                                   # [22] BayName
        ])

    def record_to_buffer(self):
        """CSV 버퍼에 기록 - ATLAS 테이블 형식"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        for v in self.vehicles.values():
            # ATLAS_VEHICLE 테이블 형식
            self.vehicle_buffer.append({
                'createTime': now,
                'fabId': FAB_ID,
                'mcpName': MCP_NAME,
                'vehicleId': v.vehicleId,
                'state': v.udpState.state.value,
                'stateName': v.udpState.state.name,
                'runCycle': v.udpState.runCycle.value,
                'runCycleName': v.udpState.runCycle.name,
                'vhlCycle': v.udpState.vhlCycle.value,
                'vhlCycleName': v.udpState.vhlCycle.name,
                'currentAddress': v.currentNode,
                'nextAddress': v.nextNode,
                'distance': round(v.udpState.distance, 2),
                'destination': v.destination,
                'carrierId': v.carrierId,
                'isLoaded': 1 if v.udpState.isFull else 0,
                'velocity': v.velocity,
                'x': round(v.x, 2),
                'y': round(v.y, 2),
                'detailState': v.udpState.detailState.value,
                'runDistance': v.udpState.runDistance,
                'udpMessage': self.generate_udp_message(v)
            })

    def save_csv(self):
        """CSV 파일 저장 - ATLAS 테이블 형식"""
        if not self.vehicle_buffer:
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Vehicle CSV - ATLAS_VEHICLE 테이블
        veh_file = os.path.join(OUTPUT_DIR, f'ATLAS_VEHICLE_{timestamp}.csv')
        with open(veh_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.vehicle_buffer[0].keys())
            writer.writeheader()
            writer.writerows(self.vehicle_buffer)
        print(f"저장: {veh_file} ({len(self.vehicle_buffer)} rows)")

        # Rail Traffic CSV - ATLAS_RAIL_TRAFFIC 테이블
        rail_file = os.path.join(OUTPUT_DIR, f'ATLAS_RAIL_TRAFFIC_{timestamp}.csv')
        rail_rows = []
        for (from_n, to_n), data in self.rail_buffer.items():
            if data['velocities']:
                edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{from_n}-{to_n}"
                rail_edge = self.rail_edge_map.get(edge_id)
                # In/Out 비율로 데드락 위험도 계산 (In > Out이면 정체 위험)
                in_cnt = rail_edge.inCount if rail_edge else 0
                out_cnt = rail_edge.outCount if rail_edge else 0
                in_out_ratio = rail_edge.getInOutRatio() if rail_edge else 0
                # 데드락 위험: In이 Out보다 많고, 밀도가 높으면 위험
                density = rail_edge.getDensity() if rail_edge else 0
                deadlock_risk = 'HIGH' if (in_cnt > out_cnt * 1.5 and density > 50) else \
                               ('MEDIUM' if (in_cnt > out_cnt and density > 30) else 'LOW')

                rail_rows.append({
                    'createTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'fabId': FAB_ID,
                    'mcpName': MCP_NAME,
                    'railEdgeId': edge_id,
                    'fromNode': from_n,
                    'toNode': to_n,
                    'avgVelocity': round(sum(data['velocities']) / len(data['velocities']), 2),
                    'maxVelocity': round(max(data['velocities']), 2),
                    'minVelocity': round(min(data['velocities']), 2),
                    'calculatedVelocity': round(rail_edge.velocity, 2) if rail_edge else 0,
                    'absoluteVelocity': round(rail_edge.getAbsoluteVelocity(), 4) if rail_edge else 0,
                    'density': round(density, 2),
                    'inCount': in_cnt,
                    'outCount': out_cnt,
                    'inOutRatio': round(in_out_ratio, 2),
                    'deadlockRisk': deadlock_risk,
                    'passCnt': data['pass_cnt'],
                    'vhlCount': len(rail_edge.vhlIdMap) if rail_edge else 0,
                    'length': round(rail_edge.length, 2) if rail_edge else 0
                })

        if rail_rows:
            with open(rail_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rail_rows[0].keys())
                writer.writeheader()
                writer.writerows(rail_rows)
            print(f"저장: {rail_file} ({len(rail_rows)} rows)")

        # 버퍼 초기화
        self.vehicle_buffer = []
        self.rail_buffer = defaultdict(lambda: {'pass_cnt': 0, 'velocities': []})

# ============================================================
# FastAPI 앱
# ============================================================
app = FastAPI(title="OHT Simulator")

# 전역 변수
engine: Optional[SimulationEngine] = None
layout_data: dict = None
is_running = False

@app.on_event("startup")
async def startup():
    global engine, layout_data, is_running

    # layout.html 자동 생성 (없거나 오래된 경우)
    ensure_layout_html(LAYOUT_PATH, LAYOUT_XML_PATH, LAYOUT_ZIP_PATH)

    # 레이아웃 로드
    nodes, edges = parse_layout(LAYOUT_PATH)

    # 엔진 초기화
    engine = SimulationEngine(nodes, edges)
    engine.init_vehicles(VEHICLE_COUNT)

    # 프론트엔드용 레이아웃 데이터 (Zone Lane 정보 포함)
    layout_data = {
        'nodes': [{'no': n.no, 'x': n.x, 'y': n.y} for n in nodes.values()],
        'edges': [{'from': e[0], 'to': e[1]} for e in edges],
        # HID Zone Lane 정보 (HID_Zone_Master.csv 기반)
        'hidZones': [
            {
                'zoneId': z.zoneId,
                'hidNo': z.hidNo,
                'bayZone': z.bayZone,
                'subRegion': z.subRegion,
                'fullName': z.fullName,
                'hidType': z.hidType,
                'inLanes': [{'from': lane.fromNode, 'to': lane.toNode} for lane in z.inLanes],
                'outLanes': [{'from': lane.fromNode, 'to': lane.toNode} for lane in z.outLanes],
                'vehicleMax': z.vehicleMax,
                'vehiclePrecaution': z.vehiclePrecaution
            }
            for z in engine.hid_zones.values()
        ],
        'stations': [
            {
                'stationId': s.stationId,
                'stationType': s.stationType,
                'stationName': s.stationName,
                'nodeAddress': s.nodeAddress,
                'equipType': s.equipType,
                'x': s.x,
                'y': s.y,
                'hasCoords': s.hasCoords
            }
            for s in engine.stations.values() if s.hasCoords
        ]
    }

    is_running = True

    # 백그라운드 태스크 시작
    asyncio.create_task(simulation_loop())
    asyncio.create_task(csv_save_loop())
    asyncio.create_task(output_cleanup_loop())  # 10분마다 OUTPUT 파일 삭제

    print(f"\n서버 시작: http://localhost:8000")
    print(f"OHT {VEHICLE_COUNT}대 시뮬레이션 시작")
    print(f"HID Zone {len(engine.hid_zones)}개 로드됨")
    print(f"Station {len([s for s in engine.stations.values() if s.hasCoords])}개 로드됨 (좌표 있음)\n")

async def simulation_loop():
    """시뮬레이션 메인 루프"""
    global engine, is_running
    while is_running:
        engine.update(SIMULATION_INTERVAL)
        engine.record_to_buffer()
        await asyncio.sleep(SIMULATION_INTERVAL)

async def csv_save_loop():
    """CSV 저장 루프"""
    global engine, is_running
    while is_running:
        await asyncio.sleep(CSV_SAVE_INTERVAL)
        engine.save_csv()

async def output_cleanup_loop():
    """OUTPUT 디렉토리 정리 루프 - 10분마다 파일 완전 삭제"""
    global is_running
    cleanup_interval = 600  # 10분 = 600초

    while is_running:
        await asyncio.sleep(cleanup_interval)
        try:
            deleted_count = 0
            deleted_size = 0

            if os.path.exists(OUTPUT_DIR):
                for filename in os.listdir(OUTPUT_DIR):
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    if os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)  # 완전 삭제 (휴지통 X)
                        deleted_count += 1
                        deleted_size += file_size

            if deleted_count > 0:
                size_mb = deleted_size / (1024 * 1024)
                print(f"[OUTPUT 정리] {deleted_count}개 파일 삭제됨 ({size_mb:.2f} MB 확보)")
        except Exception as e:
            print(f"[OUTPUT 정리 오류] {e}")

# WebSocket 연결 관리
connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    print(f"WebSocket 연결: {len(connections)}개")

    try:
        # 레이아웃 전송
        await websocket.send_json({'type': 'layout', 'data': layout_data})

        # 실시간 데이터 전송
        while True:
            state = engine.get_state()
            await websocket.send_json({'type': 'update', 'data': state})
            await asyncio.sleep(SIMULATION_INTERVAL)
    except WebSocketDisconnect:
        connections.remove(websocket)
        print(f"WebSocket 해제: {len(connections)}개")

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.get("/api/layout")
async def get_layout():
    return layout_data

@app.get("/api/state")
async def get_state():
    return engine.get_state()

@app.post("/api/create-deadlock")
async def create_deadlock():
    """데드락 상황 시뮬레이션 - 특정 구간에 차량 집중시키고 정체(JAM) 상태로 만듦"""
    global engine

    # 랜덤 노드 선택하여 차량 집중
    import random
    node_list = list(engine.nodes.keys())
    target_node = random.choice(node_list)

    # 주변 노드 찾기
    neighbors = [n for n, _ in engine.graph.get(target_node, [])]
    if not neighbors:
        return {"status": "error", "message": "No neighbors found"}

    # 가장 가까운 이웃 노드 하나 선택 (집중을 위해)
    target_next = neighbors[0]

    # 15대의 차량을 해당 구간에 집중 + JAM 상태로 만듦
    moved_count = 0
    jam_count = 0
    edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{target_node}-{target_next}"
    rail_edge = engine.rail_edge_map.get(edge_id)

    for v in list(engine.vehicles.values())[:15]:
        v.currentNode = target_node
        v.nextNode = target_next
        v.path = [target_node, target_next]
        v.pathIndex = 0
        v.positionRatio = random.random() * 0.5  # 구간 전반에 분포

        # 위치 업데이트
        n1 = engine.nodes.get(target_node)
        n2 = engine.nodes.get(target_next)
        if n1 and n2:
            v.x = n1.x + (n2.x - n1.x) * v.positionRatio
            v.y = n1.y + (n2.y - n1.y) * v.positionRatio

        # 핵심: JAM 상태로 설정하여 멈추게 함!
        v.udpState.state = VHL_STATE.JAM
        v.udpState.currentAddress = target_node
        v.udpState.nextAddress = target_next
        v.velocity = 0.0  # 완전 정지
        v.smoothedVelocity = 0.0

        # RailEdge에 기록
        if rail_edge:
            rail_edge.recordIn()  # In 카운트 증가
            rail_edge.addVhlId(v.vehicleId)

        moved_count += 1
        jam_count += 1

    return {
        "status": "ok",
        "message": f"데드락 생성! 노드 {target_node}->{target_next} 구간에 {jam_count}대 정체",
        "movedVehicles": moved_count,
        "jammedVehicles": jam_count,
        "targetNode": target_node,
        "targetEdge": edge_id
    }

@app.post("/api/reset-inout")
async def reset_inout():
    """In/Out 카운터 초기화"""
    global engine
    for rail_edge in engine.rail_edge_map.values():
        rail_edge.resetInOut()
    return {"status": "ok", "message": "In/Out counters reset"}

@app.post("/api/reroute-all")
async def reroute_all():
    """모든 차량 경로 재탐색 (수동)"""
    global engine
    rerouted = 0

    for v in engine.vehicles.values():
        if not v.path or v.udpState.state != VHL_STATE.RUN:
            continue

        current_node = v.currentNode
        destination = v.destination

        new_path = engine._find_path(current_node, destination)

        if new_path and len(new_path) > 1:
            v.path = new_path
            v.pathIndex = 0
            v.nextNode = new_path[1] if len(new_path) > 1 else current_node
            v.udpState.nextAddress = v.nextNode
            rerouted += 1

    return {
        "status": "ok",
        "message": f"{rerouted}대 경로 재탐색 완료",
        "reroutedCount": rerouted
    }

@app.get("/api/path-info")
async def get_path_info(start: int, end: int):
    """두 노드 간 경로 정보 조회"""
    global engine

    # 경로 탐색
    path = engine._find_path(start, end)

    return {
        "status": "ok",
        "path": path,
        "length": len(path)
    }

@app.get("/api/jam-stats")
async def get_jam_stats():
    """JAM 통계 데이터"""
    global engine

    # 현재 JAM 상태 차량
    jam_vehicles = [v for v in engine.vehicles.values() if v.udpState.state == VHL_STATE.JAM]

    # 위험 구간 수
    high_risk_edges = []
    medium_risk_edges = []

    for edge_id, rail_edge in engine.rail_edge_map.items():
        density = rail_edge.getDensity()
        in_cnt = rail_edge.inCount
        out_cnt = rail_edge.outCount

        if in_cnt > out_cnt * 1.5 and density > 50:
            high_risk_edges.append({
                'edgeId': edge_id,
                'density': round(density, 2),
                'inOut': f"{in_cnt}/{out_cnt}"
            })
        elif in_cnt > out_cnt and density > 30:
            medium_risk_edges.append({
                'edgeId': edge_id,
                'density': round(density, 2),
                'inOut': f"{in_cnt}/{out_cnt}"
            })

    return {
        "status": "ok",
        "jamVehicleCount": len(jam_vehicles),
        "jamVehicles": [v.vehicleId for v in jam_vehicles[:20]],
        "highRiskCount": len(high_risk_edges),
        "mediumRiskCount": len(medium_risk_edges),
        "highRiskEdges": high_risk_edges[:10],
        "mediumRiskEdges": medium_risk_edges[:10]
    }

@app.post("/api/set-vehicle-count")
async def set_vehicle_count(count: int):
    """OHT 대수 변경"""
    global engine
    current_count = len(engine.vehicles)

    if count < 1:
        return {"status": "error", "message": "최소 1대 이상이어야 합니다"}
    if count > 2000:
        return {"status": "error", "message": "최대 2000대까지 가능합니다"}

    if count > current_count:
        # OHT 추가
        added = 0
        node_list = list(engine.nodes.keys())
        for i in range(count - current_count):
            new_id = f"V{current_count + i + 1:04d}"
            if new_id not in engine.vehicles:
                start_node = random.choice(node_list)
                v = Vehicle(vehicleId=new_id, currentNode=start_node, x=0, y=0)
                n = engine.nodes.get(start_node)
                if n:
                    v.x, v.y = n.x, n.y
                # 인접 노드 찾기
                if engine.graph[start_node]:
                    v.nextNode = engine.graph[start_node][0][0]
                else:
                    v.nextNode = start_node
                v.udpState.currentAddress = v.currentNode
                v.udpState.nextAddress = v.nextNode
                engine.vehicles[new_id] = v
                engine._assign_vehicle_to_zone(v)
                added += 1
        return {"status": "ok", "message": f"{added}대 추가됨", "total": len(engine.vehicles)}

    elif count < current_count:
        # OHT 제거
        removed = 0
        vehicles_to_remove = list(engine.vehicles.keys())[count:]
        for vid in vehicles_to_remove:
            # Zone에서 제거
            if vid in engine.vehicle_zone_map:
                zone_id = engine.vehicle_zone_map[vid]
                if zone_id in engine.hid_zones:
                    engine.hid_zones[zone_id].removeVehicle(vid)
                del engine.vehicle_zone_map[vid]
            del engine.vehicles[vid]
            removed += 1
        return {"status": "ok", "message": f"{removed}대 제거됨", "total": len(engine.vehicles)}

    return {"status": "ok", "message": "변경 없음", "total": current_count}

# ============================================================
# HTML 프론트엔드
# ============================================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>OHT 실시간 시뮬레이터</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #eee; overflow: hidden; }

#header {
    position: fixed; top: 0; left: 0; right: 0; height: 50px;
    background: linear-gradient(90deg, #1a1a3e, #2a2a5e);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 20px; z-index: 1000;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
#header h1 { font-size: 18px; color: #00d4ff; }
#header .status { display: flex; gap: 20px; font-size: 13px; }
#header .status span { color: #00d4ff; font-weight: bold; }
.live-dot { width: 10px; height: 10px; background: #00ff88; border-radius: 50%; animation: pulse 1s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

#sidebar {
    position: fixed; top: 50px; left: 0; bottom: 0; width: 240px;
    background: rgba(20, 20, 40, 0.95); padding: 15px;
    border-right: 1px solid #333; z-index: 900;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #444 #1a1a2e;
    transition: transform 0.3s ease;
}
#sidebar.collapsed { transform: translateX(-240px); }
#sidebar-toggle {
    position: fixed; top: 55px; left: 240px; z-index: 950;
    width: 24px; height: 50px; background: rgba(20, 20, 40, 0.95);
    border: 1px solid #333; border-left: none;
    border-radius: 0 8px 8px 0; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #00d4ff; font-size: 14px; transition: left 0.3s ease;
}
#sidebar-toggle:hover { background: rgba(0, 212, 255, 0.2); }
#sidebar-toggle.collapsed { left: 0; }
#sidebar::-webkit-scrollbar { width: 6px; }
#sidebar::-webkit-scrollbar-track { background: #1a1a2e; }
#sidebar::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
#sidebar::-webkit-scrollbar-thumb:hover { background: #555; }
#sidebar h3 { color: #00d4ff; font-size: 13px; margin: 15px 0 10px; }
#sidebar .section { background: #1a1a2e; padding: 12px; border-radius: 8px; margin-bottom: 12px; }
.stat-row { display: flex; justify-content: space-between; padding: 5px 0; font-size: 12px; }
.stat-row .val { color: #00d4ff; font-weight: bold; font-size: 14px; }

.legend { margin-top: 10px; }
.legend-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 11px; }
.legend-dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid #fff; }

#canvas-container { position: fixed; top: 50px; left: 240px; right: 280px; bottom: 0; transition: left 0.3s ease, right 0.3s ease; }
#canvas-container.expanded { left: 0; }
#canvas-container.right-expanded { right: 0; }
canvas { display: block; }

/* 오른쪽 사이드바 - OHT 상태 */
#right-sidebar {
    position: fixed; top: 50px; right: 0; bottom: 0; width: 280px;
    background: rgba(20, 20, 40, 0.95); padding: 10px;
    border-left: 1px solid #333; z-index: 900;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #444 #1a1a2e;
    transition: transform 0.3s ease;
}
#right-sidebar.collapsed { transform: translateX(280px); }
#right-sidebar-toggle {
    position: fixed; top: 55px; right: 280px; z-index: 950;
    width: 24px; height: 50px; background: rgba(20, 20, 40, 0.95);
    border: 1px solid #333; border-right: none;
    border-radius: 8px 0 0 8px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #00d4ff; font-size: 14px; transition: right 0.3s ease;
}
#right-sidebar-toggle:hover { background: rgba(0, 212, 255, 0.2); }
#right-sidebar-toggle.collapsed { right: 0; }
#right-sidebar h3 { color: #00d4ff; font-size: 13px; margin: 10px 0 8px; }
#right-sidebar::-webkit-scrollbar { width: 6px; }
#right-sidebar::-webkit-scrollbar-track { background: #1a1a2e; }
#right-sidebar::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

/* 사이드바 탭 스타일 */
.sidebar-tabs { display: flex; gap: 4px; margin-bottom: 12px; }
.sidebar-tab {
    flex: 1; padding: 8px 4px; font-size: 12px; font-weight: bold;
    background: #1a1a2e; color: #888; border: 1px solid #333;
    border-radius: 6px; cursor: pointer; transition: all 0.2s ease;
}
.sidebar-tab:hover { background: #252550; color: #aaa; }
.sidebar-tab.active { background: #00d4ff; color: #000; border-color: #00d4ff; }
.tab-content { display: block; }

/* OHT 리스트 스타일 */
.oht-list { max-height: calc(100vh - 220px); overflow-y: auto; }
.oht-item {
    background: #1a1a2e; padding: 8px 10px; border-radius: 6px;
    margin-bottom: 6px; cursor: pointer; transition: all 0.2s ease;
    border-left: 3px solid #444;
}
.oht-item:hover { background: #252550; }
.oht-item.selected { background: #1a3a5e; border-left-color: #00d4ff; }
.oht-item.running { border-left-color: #00ff88; }
.oht-item.loaded { border-left-color: #ff9900; }
.oht-item.stopped { border-left-color: #ff3366; }
.oht-item.jam { border-left-color: #ff0000; animation: blink 0.5s infinite; }
.oht-item .oht-header { display: flex; justify-content: space-between; align-items: center; }
.oht-item .oht-id { font-weight: bold; color: #00d4ff; font-size: 12px; }
.oht-item .oht-state { font-size: 10px; padding: 2px 6px; border-radius: 3px; background: #333; }
.oht-item .oht-detail { display: none; margin-top: 8px; padding-top: 8px; border-top: 1px solid #333; font-size: 11px; }
.oht-item.expanded .oht-detail { display: block; }
.oht-item .detail-row { display: flex; justify-content: space-between; padding: 2px 0; }
.oht-item .detail-row .label { color: #888; }
.oht-item .detail-row .value { color: #fff; }

/* OHT 필터 */
.oht-filter { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 10px; }
.oht-filter button { padding: 4px 8px; font-size: 10px; background: #333; color: #fff; border: none; border-radius: 3px; cursor: pointer; }
.oht-filter button.active { background: #00d4ff; color: #000; }
.oht-filter button:hover { background: #444; }
.oht-filter button.active:hover { background: #00b8e6; }

/* Zone 필터 */
.zone-filter { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 10px; }
.zone-filter button { padding: 4px 8px; font-size: 10px; background: #333; color: #fff; border: none; border-radius: 3px; cursor: pointer; }
.zone-filter button.active { background: #00d4ff; color: #000; }
.zone-filter button:hover { background: #444; }
.zone-filter button.active:hover { background: #00b8e6; }

/* Zone 리스트 */
.zone-list { max-height: calc(100vh - 220px); overflow-y: auto; }
.zone-item {
    background: #1a1a2e; padding: 8px 10px; border-radius: 6px;
    margin-bottom: 6px; cursor: pointer; transition: all 0.2s ease;
    border-left: 3px solid #444;
}
.zone-item:hover { background: #252550; }
.zone-item.selected { background: #1a3a5e; border-left-color: #00d4ff; }
.zone-item.normal { border-left-color: #00ff88; }
.zone-item.precaution { border-left-color: #ff9900; }
.zone-item.full { border-left-color: #ff3366; animation: blink 0.5s infinite; }
.zone-item .zone-header { display: flex; justify-content: space-between; align-items: center; }
.zone-item .zone-id { font-weight: bold; color: #00d4ff; font-size: 12px; }
.zone-item .zone-state { font-size: 10px; padding: 2px 6px; border-radius: 3px; background: #333; }
.zone-item .zone-detail { display: none; margin-top: 8px; padding-top: 8px; border-top: 1px solid #333; font-size: 11px; }
.zone-item .zone-detail.show { display: block; }
.zone-item .zone-info { display: flex; justify-content: space-between; margin-bottom: 4px; }
.zone-item .zone-info .label { color: #888; }
.zone-item .zone-info .value { color: #fff; }
.zone-item .zone-progress { height: 4px; background: #333; border-radius: 2px; margin-top: 6px; }
.zone-item .zone-progress-bar { height: 100%; border-radius: 2px; transition: width 0.3s ease; }
.zone-item.normal .zone-progress-bar { background: #00ff88; }
.zone-item.precaution .zone-progress-bar { background: #ff9900; }
.zone-item.full .zone-progress-bar { background: #ff3366; }

#tooltip {
    position: fixed; background: rgba(0,20,40,0.95); border: 2px solid #00d4ff;
    padding: 12px; border-radius: 8px; font-size: 11px; display: none; z-index: 2000;
    min-width: 160px; pointer-events: none;
}
#tooltip .title { color: #00d4ff; font-weight: bold; font-size: 14px; margin-bottom: 8px; }
#tooltip .row { display: flex; justify-content: space-between; padding: 3px 0; }
#tooltip .label { color: #888; }
#tooltip .value { color: #fff; font-weight: bold; }

#info-panel {
    position: fixed; bottom: 20px; right: 20px;
    background: rgba(0,0,0,0.8); padding: 15px 20px;
    border-radius: 8px; border: 1px solid #00d4ff; z-index: 1000;
}
#info-panel .time { font-size: 24px; color: #00d4ff; font-weight: bold; }
#info-panel .label { font-size: 11px; color: #888; }
</style>
</head>
<body>

<div id="header">
    <h1>SK Hynix M14 OHT Simulator <span style="font-size:14px;color:#00d4ff;">(Pseudo-3D)</span></h1>
    <div class="status">
        <div style="display:flex;align-items:center;gap:8px;">
            <button id="btnToggle3D" style="padding:6px 14px;background:#ff9900;color:#000;border:none;border-radius:4px;cursor:pointer;font-size:12px;font-weight:bold;">3D 효과 OFF</button>
        </div>
        <div style="display:flex;align-items:center;gap:8px;"><div class="live-dot"></div> LIVE</div>
        <div>노드: <span id="nodeCount">-</span></div>
        <div>OHT: <span id="vehCount">-</span></div>
        <div>운행: <span id="runCount">-</span></div>
        <div>적재: <span id="loadCount">-</span></div>
    </div>
</div>

<div id="sidebar-toggle" class="collapsed" onclick="toggleSidebar()" title="사이드바 접기/펼치기">▶</div>
<div id="sidebar" class="collapsed">
    <div class="section">
        <h3>OHT 대수 설정</h3>
        <div style="display:flex;gap:5px;align-items:center;">
            <input type="number" id="inputVehicleCount" min="1" max="2000" value="450"
                   style="flex:1;padding:6px 8px;background:#1a1a3e;color:#fff;border:1px solid #444;border-radius:4px;font-size:12px;width:80px;">
            <span style="font-size:11px;color:#888;">대</span>
        </div>
        <div style="margin-top:8px;display:flex;gap:5px;">
            <button id="btnSetVehicles" style="flex:1;padding:6px 8px;background:#00d4ff;color:#000;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:bold;">적용</button>
            <button id="btnQuick50" style="padding:6px 8px;background:#333;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:10px;">50</button>
            <button id="btnQuick100" style="padding:6px 8px;background:#333;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:10px;">100</button>
            <button id="btnQuick500" style="padding:6px 8px;background:#333;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:10px;">500</button>
        </div>
        <div id="vehicleMsg" style="margin-top:6px;font-size:10px;color:#888;"></div>
    </div>

    <div class="section">
        <h3>실시간 통계</h3>
        <div class="stat-row"><span>총 OHT</span><span class="val" id="statTotal">-</span></div>
        <div class="stat-row"><span>운행중</span><span class="val" id="statRunning">-</span></div>
        <div class="stat-row"><span>적재중</span><span class="val" id="statLoaded">-</span></div>
        <div class="stat-row"><span>정지</span><span class="val" id="statStopped">-</span></div>
        <div class="stat-row"><span>정체(JAM)</span><span class="val" id="statJammed" style="color:#ff0000">0</span></div>
    </div>

    <div class="section">
        <h3>속도 정보</h3>
        <div class="stat-row"><span>평균 속도</span><span class="val" id="statAvgVel">- m/min</span></div>
        <div class="stat-row"><span>최대 속도</span><span class="val" id="statMaxVel">- m/min</span></div>
        <div class="stat-row"><span>최소 속도</span><span class="val" id="statMinVel">- m/min</span></div>
        <div class="stat-row"><span>절대속도</span><span class="val" id="statAbsVel">- %</span></div>
    </div>

    <div class="section">
        <h3>In/Out (데드락)</h3>
        <div class="stat-row"><span>Total In</span><span class="val" id="statTotalIn">0</span></div>
        <div class="stat-row"><span>Total Out</span><span class="val" id="statTotalOut">0</span></div>
        <div class="stat-row"><span>In/Out 비율</span><span class="val" id="statInOutRatio">0</span></div>
        <div class="stat-row"><span>위험 구간</span><span class="val" id="statDeadlockCnt" style="color:#ff3366">0</span></div>
        <div style="margin-top:10px;display:flex;gap:5px;">
            <button id="btnCreateDeadlock" style="flex:1;padding:6px 8px;background:#ff3366;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;">상황만들기</button>
            <button id="btnResetInOut" style="flex:1;padding:6px 8px;background:#555;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;">초기화</button>
        </div>
        <div id="deadlockMsg" style="margin-top:8px;font-size:10px;color:#888;"></div>
    </div>

    <div class="section">
        <h3>HID Zone 현황</h3>
        <div class="stat-row"><span>총 Zone</span><span class="val" id="statTotalZones">187</span></div>
        <div class="stat-row"><span>정상</span><span class="val" id="statNormalZones" style="color:#00ff88">-</span></div>
        <div class="stat-row"><span>주의</span><span class="val" id="statPrecautionZones" style="color:#ffaa00">-</span></div>
        <div class="stat-row"><span>포화</span><span class="val" id="statFullZones" style="color:#ff3366">-</span></div>
        <div class="stat-row"><span>전체 점유율</span><span class="val" id="statZoneOccupancy">- %</span></div>
        <div style="margin-top:10px;">
            <button id="btnToggleZones" style="width:100%;padding:6px 8px;background:#00d4ff;color:#000;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:bold;">Zone 표시 ON</button>
        </div>
        <div style="margin-top:6px;display:flex;gap:4px;">
            <button id="btnToggleStations" style="flex:1;padding:5px 2px;background:#555;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:9px;">Station OFF</button>
            <button id="btnToggleStationIds" style="flex:1;padding:5px 2px;background:#555;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:9px;">ID OFF</button>
            <button id="btnToggleStationNames" style="flex:1;padding:5px 2px;background:#555;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:9px;">이름 OFF</button>
        </div>
        <div class="legend" style="margin-top:8px;">
            <div class="legend-item"><div style="width:20px;height:3px;background:#00ff88;margin-right:8px;"></div>정상 Zone</div>
            <div class="legend-item"><div style="width:20px;height:3px;background:#ffaa00;margin-right:8px;"></div>주의 Zone</div>
            <div style="font-size:10px;color:#888;margin-top:4px;">실선: IN Lane, 점선: OUT Lane</div>
        </div>
        <!-- Station 범례 -->
        <div class="legend" style="margin-top:8px;border-top:1px solid #333;padding-top:8px;">
            <div style="font-size:10px;color:#00aaff;font-weight:bold;margin-bottom:4px;">Station 타입 (색상)</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#00aaff;margin-right:6px;border-radius:50%;"></div>DUAL_ACCESS</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#ff6600;margin-right:6px;border-radius:50%;"></div>MAINTENANCE</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#00ff00;margin-right:6px;border-radius:50%;"></div>ACQUIRE</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#ff00ff;margin-right:6px;border-radius:50%;"></div>DEPOSIT</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#666666;margin-right:6px;border-radius:50%;"></div>DUMMY</div>
            <div style="font-size:10px;color:#00aaff;font-weight:bold;margin-top:6px;margin-bottom:4px;">장비 타입 (모양)</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#00aaff;margin-right:6px;"></div>ZFS (■)</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#00aaff;margin-right:6px;border-radius:50%;"></div>UNIVERSAL (●)</div>
            <div class="legend-item"><div style="width:8px;height:8px;background:#00aaff;margin-right:6px;transform:rotate(45deg);"></div>기타 (◆)</div>
        </div>
        <div id="zoneAlertList" style="margin-top:8px;font-size:10px;max-height:60px;overflow-y:auto;"></div>
    </div>

    <div class="section">
        <h3>정체(JAM) 현황</h3>
        <div class="stat-row"><span>현재 JAM 차량</span><span class="val" id="jamStatVehicles" style="color:#ff0000">0</span></div>
        <div class="stat-row"><span>HIGH 위험</span><span class="val" id="jamStatHigh" style="color:#ff3366">0</span></div>
        <div class="stat-row"><span>MEDIUM 위험</span><span class="val" id="jamStatMedium" style="color:#ffaa00">0</span></div>
    </div>

    <div class="section">
        <h3>시스템 정보</h3>
        <div class="stat-row"><span>FAB ID</span><span class="val">M14Q</span></div>
        <div class="stat-row"><span>차량 길이</span><span class="val">1084mm</span></div>
        <div class="stat-row"><span>최대속도</span><span class="val">300 m/min</span></div>
        <div class="stat-row"><span>거리단위</span><span class="val">100mm</span></div>
    </div>

    <div class="section">
        <h3>범례</h3>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div>운행중 (공차)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ff9900"></div>운행중 (적재)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ff3366"></div>정지</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ff0000;animation:blink 0.5s infinite"></div>정체 (JAM)</div>
        </div>
    </div>

    <div class="section">
        <h3>컨트롤</h3>
        <div class="stat-row"><span>줌</span><span class="val" id="zoomLevel">100%</span></div>
        <div style="font-size:11px;color:#888;margin-top:8px;">
            마우스 휠: 줌<br>
            드래그: 이동<br>
            더블클릭: 경로표시<br>
            호버: 상세정보
        </div>
    </div>
</div>

<div id="canvas-container" class="expanded right-expanded">
    <canvas id="canvas"></canvas>
    <div id="three-container" style="display:none;width:100%;height:100%;"></div>
</div>

<!-- 오른쪽 사이드바: OHT/Zone 상태 (기본 접힘) -->
<div id="right-sidebar-toggle" class="collapsed" onclick="toggleRightSidebar()" title="목록 접기/펼치기">◀</div>
<div id="right-sidebar" class="collapsed">
    <!-- 탭 버튼 -->
    <div class="sidebar-tabs">
        <button class="sidebar-tab active" data-tab="oht">OHT 상태</button>
        <button class="sidebar-tab" data-tab="zone">HID Zone</button>
    </div>

    <!-- OHT 탭 콘텐츠 -->
    <div class="tab-content" id="tab-oht">
        <div class="oht-filter">
            <button class="active" data-filter="all">전체</button>
            <button data-filter="running">운행</button>
            <button data-filter="loaded">적재</button>
            <button data-filter="stopped">정지</button>
            <button data-filter="jam">JAM</button>
        </div>
        <div style="margin-bottom:8px;">
            <input type="text" id="ohtSearch" placeholder="OHT ID 검색..."
                   style="width:100%;padding:6px 8px;background:#1a1a3e;color:#fff;border:1px solid #444;border-radius:4px;font-size:11px;">
        </div>
        <div class="oht-list" id="ohtList">
            <!-- OHT 목록이 여기에 동적으로 추가됨 -->
        </div>
        <div style="margin-top:10px;padding:8px;background:#1a1a2e;border-radius:6px;font-size:10px;color:#888;">
            <div>클릭: 선택/해제 | 더블클릭: 상세정보</div>
        </div>
    </div>

    <!-- HID Zone 탭 콘텐츠 -->
    <div class="tab-content" id="tab-zone" style="display:none;">
        <div class="zone-filter">
            <button class="active" data-zone-filter="all">전체</button>
            <button data-zone-filter="full">포화</button>
            <button data-zone-filter="precaution">주의</button>
            <button data-zone-filter="normal">정상</button>
        </div>
        <div style="margin-bottom:8px;">
            <input type="text" id="zoneSearch" placeholder="Zone ID 검색..."
                   style="width:100%;padding:6px 8px;background:#1a1a3e;color:#fff;border:1px solid #444;border-radius:4px;font-size:11px;">
        </div>
        <div class="zone-list" id="zoneList">
            <!-- Zone 목록이 여기에 동적으로 추가됨 -->
        </div>
        <div style="margin-top:10px;padding:8px;background:#1a1a2e;border-radius:6px;font-size:10px;color:#888;">
            <div>클릭: 선택 | 더블클릭: 상세정보</div>
        </div>
    </div>
</div>

<div id="tooltip"></div>

<div id="info-panel">
    <div class="label">시뮬레이션 시간</div>
    <div class="time" id="simTime">00:00:00</div>
</div>

<!-- Three.js 라이브러리 -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
// 왼쪽 사이드바 토글 함수
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebar-toggle');
    const container = document.getElementById('canvas-container');

    sidebar.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
    container.classList.toggle('expanded');

    // 아이콘 변경
    toggle.textContent = sidebar.classList.contains('collapsed') ? '▶' : '◀';

    // 캔버스 리사이즈
    setTimeout(resize, 350);
}

// 오른쪽 사이드바 토글 함수
function toggleRightSidebar() {
    const sidebar = document.getElementById('right-sidebar');
    const toggle = document.getElementById('right-sidebar-toggle');
    const container = document.getElementById('canvas-container');

    sidebar.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
    container.classList.toggle('right-expanded');

    // 아이콘 변경
    toggle.textContent = sidebar.classList.contains('collapsed') ? '◀' : '▶';

    // 캔버스 리사이즈
    setTimeout(resize, 350);
}

// OHT 목록 관련 변수
let ohtFilter = 'all';
let ohtSearchText = '';
let expandedOhtId = null;

// Zone 필터 및 검색 상태
let zoneFilter = 'all';
let zoneSearchText = '';
let expandedZoneId = null;
let selectedZoneId = null;

// OHT 상태에 따른 클래스 반환
// VHL_STATE enum: RUN="1", STOP="2", ABNORMAL="3", MANUAL="4", REMOVING="5", OBS_BZ_STOP="6", JAM="7"
function getOhtStateClass(v) {
    const state = String(v.state);
    if (state === '7') return 'jam';  // JAM
    if (state === '2' || state === '6' || state === '8') return 'stopped';  // STOP, OBS_BZ_STOP, HT_STOP
    if (v.loaded) return 'loaded';
    if (state === '1') return 'running';  // RUN
    return 'stopped';  // 기타 상태는 정지로 표시
}

// OHT 상태 텍스트
// VHL_STATE enum: RUN="1", STOP="2", ABNORMAL="3", MANUAL="4", REMOVING="5", OBS_BZ_STOP="6", JAM="7", HT_STOP="8", E84_TIMEOUT="9"
function getOhtStateText(v) {
    const states = {
        '1': 'RUN', '2': 'STOP', '3': 'ABNORMAL', '4': 'MANUAL',
        '5': 'REMOVING', '6': 'OBS_STOP', '7': 'JAM', '8': 'HT_STOP', '9': 'E84_TIMEOUT'
    };
    return states[String(v.state)] || 'UNKNOWN';
}

// OHT 목록 업데이트
function updateOhtList() {
    const listEl = document.getElementById('ohtList');
    if (!listEl) {
        console.log('OHT 목록 요소를 찾을 수 없음');
        return;
    }

    // vehicles 변수 접근 확인
    if (typeof vehicles === 'undefined') {
        console.log('vehicles 변수가 정의되지 않음');
        listEl.innerHTML = '<div style="text-align:center;color:#888;padding:20px;">초기화 중...</div>';
        return;
    }

    const vehCount = Object.keys(vehicles).length;
    console.log('updateOhtList 호출됨, vehicles 수:', vehCount);

    if (vehCount === 0) {
        listEl.innerHTML = '<div style="text-align:center;color:#888;padding:20px;">OHT 데이터 로딩 중...</div>';
        return;
    }

    const vehArray = Object.values(vehicles);

    // 필터링
    let filtered = vehArray.filter(v => {
        // 상태 필터
        if (ohtFilter !== 'all') {
            const stateClass = getOhtStateClass(v);
            if (ohtFilter === 'running' && stateClass !== 'running') return false;
            if (ohtFilter === 'loaded' && stateClass !== 'loaded') return false;
            if (ohtFilter === 'stopped' && stateClass !== 'stopped') return false;
            if (ohtFilter === 'jam' && stateClass !== 'jam') return false;
        }
        // 검색 필터
        if (ohtSearchText && !v.vehicleId.toLowerCase().includes(ohtSearchText.toLowerCase())) {
            return false;
        }
        return true;
    });

    // 정렬 (JAM 우선, 그 다음 ID 순)
    filtered.sort((a, b) => {
        const aJam = a.state === 5 ? 0 : 1;
        const bJam = b.state === 5 ? 0 : 1;
        if (aJam !== bJam) return aJam - bJam;
        return a.vehicleId.localeCompare(b.vehicleId);
    });

    // HTML 생성
    let html = '';
    filtered.forEach(v => {
        const stateClass = getOhtStateClass(v);
        const isSelected = selectedVehicles.has(v.vehicleId);
        const isExpanded = expandedOhtId === v.vehicleId;

        html += '<div class="oht-item ' + stateClass + (isSelected ? ' selected' : '') + (isExpanded ? ' expanded' : '') + '" data-id="' + v.vehicleId + '">';
        html += '  <div class="oht-header">';
        html += '    <span class="oht-id">' + v.vehicleId + '</span>';
        html += '    <span class="oht-state" style="background:' + (stateClass === 'jam' ? '#ff0000' : stateClass === 'stopped' ? '#ff3366' : stateClass === 'loaded' ? '#ff9900' : '#00ff88') + ';color:#000;">' + getOhtStateText(v) + '</span>';
        html += '  </div>';
        html += '  <div class="oht-detail">';
        html += '    <div class="detail-row"><span class="label">속도</span><span class="value">' + (v.velocity || 0).toFixed(1) + ' m/min</span></div>';
        html += '    <div class="detail-row"><span class="label">적재</span><span class="value">' + (v.loaded ? 'YES' : 'NO') + '</span></div>';
        html += '    <div class="detail-row"><span class="label">현재 노드</span><span class="value">' + (v.currentNode || '-') + '</span></div>';
        html += '    <div class="detail-row"><span class="label">목적지</span><span class="value">' + (v.destNode || '-') + '</span></div>';
        html += '    <div class="detail-row"><span class="label">HID Zone</span><span class="value">' + (v.hidZoneId >= 0 ? 'HID ' + v.hidZoneId : '-') + '</span></div>';
        html += '    <div class="detail-row"><span class="label">Station</span><span class="value" style="color:#ff9800;">' + (v.currentStations && v.currentStations.length > 0 ? v.currentStations.map(s => s.stationName).join(', ') : '-') + '</span></div>';
        html += '    <div class="detail-row"><span class="label">RUN CYCLE</span><span class="value">' + (v.runCycle || '-') + '</span></div>';
        html += '  </div>';
        html += '</div>';
    });

    if (filtered.length === 0) {
        html = '<div style="text-align:center;color:#888;padding:20px;">표시할 OHT가 없습니다</div>';
    }

    listEl.innerHTML = html;

    // 클릭 이벤트 바인딩
    listEl.querySelectorAll('.oht-item').forEach(item => {
        const id = item.dataset.id;

        // 클릭: 선택/해제
        item.addEventListener('click', () => {
            if (selectedVehicles.has(id)) {
                selectedVehicles.delete(id);
                item.classList.remove('selected');
            } else {
                selectedVehicles.add(id);
                item.classList.add('selected');
            }
            draw();
        });

        // 더블클릭: 상세정보 토글
        item.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            if (expandedOhtId === id) {
                expandedOhtId = null;
                item.classList.remove('expanded');
            } else {
                // 이전 확장된 항목 닫기
                listEl.querySelectorAll('.oht-item.expanded').forEach(el => el.classList.remove('expanded'));
                expandedOhtId = id;
                item.classList.add('expanded');
            }
        });
    });
}

// Zone 상태에 따른 클래스 반환
function getZoneStateClass(zone) {
    const count = zone.vehicleCount || 0;
    const max = zone.vehicleMax || 1;
    const precaution = zone.vehiclePrecaution || Math.ceil(max * 0.7);

    if (count >= max) return 'full';
    if (count >= precaution) return 'precaution';
    return 'normal';
}

// Zone 상태 텍스트 반환
function getZoneStateText(zone) {
    const stateClass = getZoneStateClass(zone);
    if (stateClass === 'full') return '포화';
    if (stateClass === 'precaution') return '주의';
    return '정상';
}

// Zone 리스트 업데이트
function updateZoneList() {
    const listEl = document.getElementById('zoneList');
    if (!listEl) {
        return;
    }

    if (!window.hidZones || window.hidZones.length === 0) {
        listEl.innerHTML = '<div style="text-align:center;color:#888;padding:20px;">Zone 데이터 로딩 중...</div>';
        return;
    }

    let filtered = window.hidZones.slice();

    // 필터링
    if (zoneFilter !== 'all') {
        filtered = filtered.filter(z => {
            const stateClass = getZoneStateClass(z);
            return stateClass === zoneFilter;
        });
    }

    // 검색 필터
    if (zoneSearchText) {
        const searchLower = zoneSearchText.toLowerCase();
        filtered = filtered.filter(z => {
            const name = (z.fullName || z.label || String(z.zoneId)).toLowerCase();
            return name.includes(searchLower);
        });
    }

    // 정렬 (포화 > 주의 > 정상, 같은 상태 내에서는 ID 순)
    filtered.sort((a, b) => {
        const aState = getZoneStateClass(a);
        const bState = getZoneStateClass(b);
        const stateOrder = { 'full': 0, 'precaution': 1, 'normal': 2 };
        if (stateOrder[aState] !== stateOrder[bState]) {
            return stateOrder[aState] - stateOrder[bState];
        }
        return (a.zoneId || 0) - (b.zoneId || 0);
    });

    // HTML 생성
    let html = '';
    filtered.forEach(z => {
        const stateClass = getZoneStateClass(z);
        const isSelected = selectedZoneId === z.zoneId;
        const isExpanded = expandedZoneId === z.zoneId;
        const displayName = z.fullName || z.label || 'Zone ' + z.zoneId;
        const count = z.vehicleCount || 0;
        const max = z.vehicleMax || 1;
        const percent = Math.min(100, Math.round((count / max) * 100));
        const precaution = z.vehiclePrecaution || Math.ceil(max * 0.7);

        html += '<div class="zone-item ' + stateClass + (isSelected ? ' selected' : '') + '" data-id="' + z.zoneId + '">';
        html += '  <div class="zone-header">';
        html += '    <span class="zone-id">' + displayName + '</span>';
        html += '    <span class="zone-state" style="background:' + (stateClass === 'full' ? '#ff3366' : stateClass === 'precaution' ? '#ff9900' : '#00ff88') + ';color:#000;">' + getZoneStateText(z) + '</span>';
        html += '  </div>';
        html += '  <div class="zone-progress"><div class="zone-progress-bar" style="width:' + percent + '%;"></div></div>';
        html += '  <div class="zone-detail' + (isExpanded ? ' show' : '') + '">';
        html += '    <div class="zone-info"><span class="label">현재 OHT:</span><span class="value">' + count + ' / ' + max + '</span></div>';
        html += '    <div class="zone-info"><span class="label">주의 기준:</span><span class="value">' + precaution + '</span></div>';
        if (z.bayZone) html += '    <div class="zone-info"><span class="label">Bay Zone:</span><span class="value">' + z.bayZone + '</span></div>';
        if (z.subRegion) html += '    <div class="zone-info"><span class="label">Sub Region:</span><span class="value">' + z.subRegion + '</span></div>';
        if (z.inCount || z.outCount) html += '    <div class="zone-info"><span class="label">IN/OUT:</span><span class="value">' + (z.inCount || 0) + ' / ' + (z.outCount || 0) + '</span></div>';
        html += '  </div>';
        html += '</div>';
    });

    if (filtered.length === 0) {
        html = '<div style="text-align:center;color:#888;padding:20px;">표시할 Zone이 없습니다</div>';
    }

    listEl.innerHTML = html;

    // 클릭 이벤트 바인딩
    listEl.querySelectorAll('.zone-item').forEach(item => {
        const id = parseInt(item.dataset.id);

        // 클릭: 선택
        item.addEventListener('click', () => {
            // 선택된 zone 토글
            if (selectedZoneId === id) {
                selectedZoneId = null;
                item.classList.remove('selected');
            } else {
                // 이전 선택 해제
                listEl.querySelectorAll('.zone-item.selected').forEach(el => el.classList.remove('selected'));
                selectedZoneId = id;
                item.classList.add('selected');
            }
            draw();
        });

        // 더블클릭: 상세정보 토글
        item.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            const detail = item.querySelector('.zone-detail');
            if (expandedZoneId === id) {
                expandedZoneId = null;
                if (detail) detail.classList.remove('show');
            } else {
                // 이전 확장된 항목 닫기
                listEl.querySelectorAll('.zone-detail.show').forEach(el => el.classList.remove('show'));
                expandedZoneId = id;
                if (detail) detail.classList.add('show');
            }
        });
    });
}

// 필터 버튼 및 검색 이벤트 (DOM 로드 후)
document.addEventListener('DOMContentLoaded', () => {
    // 필터 버튼 이벤트
    document.querySelectorAll('.oht-filter button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.oht-filter button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ohtFilter = btn.dataset.filter;
            updateOhtList();
        });
    });

    // 검색 이벤트
    const searchEl = document.getElementById('ohtSearch');
    if (searchEl) {
        searchEl.addEventListener('input', (e) => {
            ohtSearchText = e.target.value;
            updateOhtList();
        });
    }

    // Zone 필터 버튼 이벤트
    document.querySelectorAll('.zone-filter button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.zone-filter button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            zoneFilter = btn.dataset.zoneFilter;
            updateZoneList();
        });
    });

    // Zone 검색 이벤트
    const zoneSearchEl = document.getElementById('zoneSearch');
    if (zoneSearchEl) {
        zoneSearchEl.addEventListener('input', (e) => {
            zoneSearchText = e.target.value;
            updateZoneList();
        });
    }

    // 사이드바 탭 전환 이벤트
    document.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // 모든 탭 버튼 비활성화
            document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
            // 클릭한 탭 활성화
            tab.classList.add('active');

            // 모든 탭 콘텐츠 숨기기
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = 'none';
            });

            // 선택한 탭 콘텐츠 보이기
            const tabId = tab.dataset.tab;
            const targetContent = document.getElementById('tab-' + tabId);
            if (targetContent) {
                targetContent.style.display = 'block';
            }
        });
    });

    console.log('OHT 목록 및 Zone 목록 이벤트 리스너 등록 완료');
});

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let layout = null;
let vehicles = {};
let nodeMap = {};

let offsetX = 0, offsetY = 0, scale = 1;
let isDragging = false, lastMouse = {x: 0, y: 0};
let startTime = Date.now();
let selectedVehicles = new Set();  // 더블클릭으로 선택된 OHT들 (여러개 가능)

// ============================================================
// Pseudo-3D 효과 설정
// ============================================================
let isPseudo3D = false;  // 기본값: 3D 효과 꺼짐 (버튼 누르면 활성화)
const ISO_ANGLE = Math.PI / 6;  // 30도 (isometric 각도)
const ISO_SCALE_Y = 0.6;  // Y축 압축 비율
const RAIL_HEIGHT = 8;  // 레일 높이 (3D 효과용)
const OHT_HEIGHT = 12;  // OHT 높이 (3D 효과용)
const LIGHT_ANGLE = -Math.PI / 4;  // 조명 각도 (좌상단에서)

// Isometric 변환 함수 (2D 좌표를 pseudo-3D로 변환)
function toIso(x, y, z = 0) {
    if (!isPseudo3D) return { x: x, y: y };
    const isoX = x;
    const isoY = y * ISO_SCALE_Y - z * 0.7;
    return { x: isoX, y: isoY };
}

// 색상을 밝게/어둡게 조절
function adjustColor(hex, factor) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const nr = Math.min(255, Math.max(0, Math.round(r * factor)));
    const ng = Math.min(255, Math.max(0, Math.round(g * factor)));
    const nb = Math.min(255, Math.max(0, Math.round(b * factor)));
    return '#' + [nr, ng, nb].map(c => c.toString(16).padStart(2, '0')).join('');
}

// 3D 박스 그리기 (OHT용)
function draw3DBox(cx, cy, w, h, depth, color) {
    const top = toIso(cx, cy, depth);
    const halfW = w / 2;
    const halfH = h / 2;

    // 상단면 (밝은 색)
    ctx.fillStyle = adjustColor(color, 1.3);
    ctx.beginPath();
    ctx.moveTo(top.x - halfW, top.y - halfH);
    ctx.lineTo(top.x + halfW, top.y - halfH);
    ctx.lineTo(top.x + halfW, top.y + halfH);
    ctx.lineTo(top.x - halfW, top.y + halfH);
    ctx.closePath();
    ctx.fill();

    // 앞면 (원래 색)
    const front = toIso(cx, cy + halfH, 0);
    const frontTop = toIso(cx, cy + halfH, depth);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(frontTop.x - halfW, frontTop.y);
    ctx.lineTo(frontTop.x + halfW, frontTop.y);
    ctx.lineTo(front.x + halfW, front.y);
    ctx.lineTo(front.x - halfW, front.y);
    ctx.closePath();
    ctx.fill();

    // 우측면 (어두운 색)
    const right = toIso(cx + halfW, cy, 0);
    const rightTop = toIso(cx + halfW, cy, depth);
    ctx.fillStyle = adjustColor(color, 0.7);
    ctx.beginPath();
    ctx.moveTo(rightTop.x, rightTop.y - halfH);
    ctx.lineTo(rightTop.x, rightTop.y + halfH);
    ctx.lineTo(right.x, right.y + halfH);
    ctx.lineTo(right.x, right.y - halfH);
    ctx.closePath();
    ctx.fill();

    // 테두리
    ctx.strokeStyle = adjustColor(color, 0.5);
    ctx.lineWidth = 1 / scale;
    ctx.stroke();
}

// 3D 레일 그리기 (두께감 있는 파이프 형태)
function draw3DRail(fromX, fromY, toX, toY, railColor = '#2a4a6a') {
    const width = isPseudo3D ? 4 / scale : 1.5 / scale;
    const height = isPseudo3D ? RAIL_HEIGHT / scale : 0;

    // 레일 방향 벡터
    const dx = toX - fromX;
    const dy = toY - fromY;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;

    const nx = -dy / len * width;  // 수직 벡터
    const ny = dx / len * width;

    if (isPseudo3D) {
        // 상단면 (밝은 색)
        const ft = toIso(fromX, fromY, height);
        const tt = toIso(toX, toY, height);
        ctx.fillStyle = adjustColor(railColor, 1.4);
        ctx.beginPath();
        ctx.moveTo(ft.x + nx, ft.y + ny * ISO_SCALE_Y);
        ctx.lineTo(tt.x + nx, tt.y + ny * ISO_SCALE_Y);
        ctx.lineTo(tt.x - nx, tt.y - ny * ISO_SCALE_Y);
        ctx.lineTo(ft.x - nx, ft.y - ny * ISO_SCALE_Y);
        ctx.closePath();
        ctx.fill();

        // 측면 (그림자)
        const fb = toIso(fromX, fromY, 0);
        const tb = toIso(toX, toY, 0);
        ctx.fillStyle = adjustColor(railColor, 0.6);
        ctx.beginPath();
        ctx.moveTo(ft.x - nx, ft.y - ny * ISO_SCALE_Y);
        ctx.lineTo(tt.x - nx, tt.y - ny * ISO_SCALE_Y);
        ctx.lineTo(tb.x - nx, tb.y - ny * ISO_SCALE_Y);
        ctx.lineTo(fb.x - nx, fb.y - ny * ISO_SCALE_Y);
        ctx.closePath();
        ctx.fill();
    } else {
        // 기본 2D 레일
        ctx.strokeStyle = railColor;
        ctx.lineWidth = 1.5 / scale;
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
    }
}

// 캔버스 크기
function resize() {
    const container = document.getElementById('canvas-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    render();
}
window.addEventListener('resize', resize);
resize();

// WebSocket 연결
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'layout') {
        layout = msg.data;
        nodeMap = {};
        layout.nodes.forEach(n => nodeMap[n.no] = n);
        document.getElementById('nodeCount').textContent = layout.nodes.length.toLocaleString();

        // HID Zone 정보 저장
        window.hidZones = layout.hidZones || [];
        console.log('HID Zones 로드:', window.hidZones.length + '개');

        // Station 정보 저장
        window.stations = layout.stations || [];
        console.log('Stations 로드:', window.stations.length + '개');

        // 디버깅: Zone Lane 노드 존재 여부 확인
        if (window.hidZones.length > 0) {
            let foundCount = 0, notFoundCount = 0;
            const notFoundNodes = new Set();
            window.hidZones.forEach(zone => {
                zone.inLanes.forEach(lane => {
                    if (nodeMap[lane.from] && nodeMap[lane.to]) foundCount++;
                    else {
                        notFoundCount++;
                        if (!nodeMap[lane.from]) notFoundNodes.add(lane.from);
                        if (!nodeMap[lane.to]) notFoundNodes.add(lane.to);
                    }
                });
                zone.outLanes.forEach(lane => {
                    if (nodeMap[lane.from] && nodeMap[lane.to]) foundCount++;
                    else {
                        notFoundCount++;
                        if (!nodeMap[lane.from]) notFoundNodes.add(lane.from);
                        if (!nodeMap[lane.to]) notFoundNodes.add(lane.to);
                    }
                });
            });
            console.log('Zone Lane 노드 매칭: 성공=' + foundCount + ', 실패=' + notFoundCount);
            if (notFoundNodes.size > 0) {
                console.log('누락된 노드 (샘플):', Array.from(notFoundNodes).slice(0, 10));
            }
        }

        fitView();

        // Zone 목록 초기화
        updateZoneList();
    }
    else if (msg.type === 'update') {
        // 부드러운 보간을 위해 타겟 위치 설정
        const receivedCount = msg.data.vehicles ? msg.data.vehicles.length : 0;
        msg.data.vehicles.forEach(v => {
            if (!vehicles[v.vehicleId]) {
                vehicles[v.vehicleId] = {...v, dispX: v.x, dispY: v.y};
            } else {
                Object.assign(vehicles[v.vehicleId], v);
            }
        });
        console.log('WebSocket update - 수신:', receivedCount, '현재 vehicles:', Object.keys(vehicles).length);

        // 통계 업데이트
        const stats = msg.data.stats;
        document.getElementById('vehCount').textContent = stats.total;
        document.getElementById('runCount').textContent = stats.running;
        document.getElementById('loadCount').textContent = stats.loaded;
        document.getElementById('statTotal').textContent = stats.total;
        document.getElementById('statRunning').textContent = stats.running;
        document.getElementById('statLoaded').textContent = stats.loaded;
        document.getElementById('statStopped').textContent = stats.stopped || 0;
        document.getElementById('statJammed').textContent = stats.jammed || 0;

        // OHT 대수 입력란에 현재 값 반영 (첫 연결 시)
        const inputEl = document.getElementById('inputVehicleCount');
        if (inputEl && !inputEl.dataset.initialized) {
            inputEl.value = stats.total;
            inputEl.dataset.initialized = 'true';
        }

        // 속도 통계 계산
        const runningVehicles = msg.data.vehicles.filter(v => v.state === 1 && v.velocity > 0);
        if (runningVehicles.length > 0) {
            const velocities = runningVehicles.map(v => v.velocity);
            const avgVel = velocities.reduce((a,b) => a+b, 0) / velocities.length;
            const maxVel = Math.max(...velocities);
            const minVel = Math.min(...velocities);
            const absVel = (avgVel / 300) * 100;  // 최대 300 m/min 기준

            document.getElementById('statAvgVel').textContent = avgVel.toFixed(1) + ' m/min';
            document.getElementById('statMaxVel').textContent = maxVel.toFixed(1) + ' m/min';
            document.getElementById('statMinVel').textContent = minVel.toFixed(1) + ' m/min';
            document.getElementById('statAbsVel').textContent = absVel.toFixed(1) + ' %';
        }

        // In/Out 통계 업데이트
        if (msg.data.inOutStats) {
            const ios = msg.data.inOutStats;
            document.getElementById('statTotalIn').textContent = ios.totalIn;
            document.getElementById('statTotalOut').textContent = ios.totalOut;
            document.getElementById('statInOutRatio').textContent = ios.ratio;
            const deadlockEl = document.getElementById('statDeadlockCnt');
            deadlockEl.textContent = ios.deadlockRiskCount;
            deadlockEl.style.color = ios.deadlockRiskCount > 0 ? '#ff3366' : '#00ff88';
        }

        // 데드락 위험 구간 저장 (렌더링용)
        window.deadlockEdges = msg.data.deadlockEdges || [];

        // HID Zone 통계 업데이트
        if (msg.data.zoneStats) {
            const zs = msg.data.zoneStats;
            document.getElementById('statTotalZones').textContent = zs.totalZones;
            document.getElementById('statNormalZones').textContent = zs.normalZones;
            document.getElementById('statPrecautionZones').textContent = zs.precautionZones;
            document.getElementById('statFullZones').textContent = zs.fullZones;
            document.getElementById('statZoneOccupancy').textContent = zs.overallOccupancy + ' %';

            // 주의/포화 Zone 목록 표시
            const alertList = document.getElementById('zoneAlertList');
            let alertHtml = '';

            if (msg.data.fullZoneList && msg.data.fullZoneList.length > 0) {
                alertHtml += '<div style="color:#ff3366;margin-bottom:4px;"><b>포화 Zone:</b></div>';
                msg.data.fullZoneList.forEach(z => {
                    alertHtml += `<div style="color:#ff3366;">Zone ${z.zoneId}: ${z.vehicleCount}/${z.vehicleMax}</div>`;
                });
            }

            if (msg.data.precautionZoneList && msg.data.precautionZoneList.length > 0) {
                alertHtml += '<div style="color:#ffaa00;margin-bottom:4px;margin-top:4px;"><b>주의 Zone:</b></div>';
                msg.data.precautionZoneList.forEach(z => {
                    alertHtml += `<div style="color:#ffaa00;">Zone ${z.zoneId}: ${z.vehicleCount}/${z.vehicleMax} (${z.occupancyRate}%)</div>`;
                });
            }

            alertList.innerHTML = alertHtml || '<span style="color:#888;">정상 운영 중</span>';
        }

        // Zone 상태 맵 저장 (렌더링용)
        window.zoneStatusMap = msg.data.zoneStatusMap || {};

        // HID Zone 데이터 실시간 업데이트 (zoneStatusMap 사용)
        if (window.zoneStatusMap && window.hidZones) {
            window.hidZones.forEach(zone => {
                const statusInfo = window.zoneStatusMap[zone.zoneId];
                if (statusInfo) {
                    zone.vehicleCount = statusInfo.vehicleCount;
                    zone.occupancyRate = statusInfo.occupancyRate;
                    zone.status = statusInfo.status;
                }
            });
        }

        // OHT 목록 업데이트 (오른쪽 사이드바)
        updateOhtList();

        // Zone 목록 업데이트 (오른쪽 사이드바)
        updateZoneList();
    }
};

ws.onclose = () => console.log('WebSocket 연결 해제');
ws.onerror = (e) => console.error('WebSocket 오류:', e);

// 애니메이션 루프
function animate() {
    // 부드러운 보간
    Object.values(vehicles).forEach(v => {
        v.dispX = v.dispX + (v.x - v.dispX) * 0.15;
        v.dispY = v.dispY + (v.y - v.dispY) * 0.15;
    });

    // 시간 업데이트
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('simTime').textContent = `${h}:${m}:${s}`;

    render();
    requestAnimationFrame(animate);
}
animate();

// 렌더링
function render() {
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!layout) return;

    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // 엣지 (레일) - pseudo-3D 효과 적용
    layout.edges.forEach(e => {
        const from = nodeMap[e.from], to = nodeMap[e.to];
        if (from && to) {
            draw3DRail(from.x, from.y, to.x, to.y, '#2a4a6a');
        }
    });

    // 노드 점 (pseudo-3D 모드에서는 3D 효과 적용)
    const nodeSize = Math.max(1.5, 3 / scale);
    layout.nodes.forEach(n => {
        if (isPseudo3D) {
            // 3D 효과: 노드를 작은 원기둥처럼 표현
            const pos = toIso(n.x, n.y, RAIL_HEIGHT / scale);
            const posBase = toIso(n.x, n.y, 0);

            // 그림자 (기둥)
            ctx.fillStyle = '#1a2a3a';
            ctx.beginPath();
            ctx.ellipse(posBase.x, posBase.y, nodeSize, nodeSize * 0.5, 0, 0, Math.PI * 2);
            ctx.fill();

            // 상단 원
            ctx.fillStyle = '#4a6a8a';
            ctx.beginPath();
            ctx.ellipse(pos.x, pos.y, nodeSize, nodeSize * 0.5, 0, 0, Math.PI * 2);
            ctx.fill();
        } else {
            ctx.fillStyle = '#4a4a6a';
            ctx.beginPath();
            ctx.arc(n.x, n.y, nodeSize, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // ============================================================
    // HID Zone Lane 표시 (pseudo-3D 변환 적용)
    // ============================================================
    if (window.hidZones && window.showZones !== false) {
        const zoneStatusMap = window.zoneStatusMap || {};
        const zoneZ = RAIL_HEIGHT / scale + 2;  // Zone 표시 높이

        window.hidZones.forEach(zone => {
            const status = zoneStatusMap[zone.zoneId];
            let zoneColor, zoneAlpha;

            // 상태별 색상
            if (status && status.status === 'FULL') {
                zoneColor = '#ff3366';
                zoneAlpha = 0.8;
            } else if (status && status.status === 'PRECAUTION') {
                zoneColor = '#ffaa00';
                zoneAlpha = 0.6;
            } else {
                zoneColor = '#00ff88';
                zoneAlpha = 0.3;
            }

            // IN Lane (진입) - 실선
            ctx.strokeStyle = zoneColor;
            ctx.lineWidth = Math.max(3, 5 / scale);
            ctx.globalAlpha = zoneAlpha;
            ctx.setLineDash([]);

            let firstLaneMid = null;  // 첫 번째 Lane 중간점 저장

            zone.inLanes.forEach((lane, idx) => {
                const from = nodeMap[lane.from];
                const to = nodeMap[lane.to];
                if (from && to) {
                    const fromIso = toIso(from.x, from.y, zoneZ);
                    const toIso_ = toIso(to.x, to.y, zoneZ);
                    ctx.beginPath();
                    ctx.moveTo(fromIso.x, fromIso.y);
                    ctx.lineTo(toIso_.x, toIso_.y);
                    ctx.stroke();

                    // 첫 번째 Lane의 중간점 저장 (Zone ID 표시용)
                    if (idx === 0) {
                        const midIso = toIso((from.x + to.x) / 2, (from.y + to.y) / 2, zoneZ);
                        firstLaneMid = { x: midIso.x, y: midIso.y };
                    }

                    // 화살표 (진입 방향)
                    const angle = Math.atan2(toIso_.y - fromIso.y, toIso_.x - fromIso.x);
                    const arrowLen = 15 / scale;
                    const midIso = toIso((from.x + to.x) / 2, (from.y + to.y) / 2, zoneZ);
                    ctx.beginPath();
                    ctx.moveTo(midIso.x, midIso.y);
                    ctx.lineTo(midIso.x - arrowLen * Math.cos(angle - 0.4), midIso.y - arrowLen * Math.sin(angle - 0.4));
                    ctx.moveTo(midIso.x, midIso.y);
                    ctx.lineTo(midIso.x - arrowLen * Math.cos(angle + 0.4), midIso.y - arrowLen * Math.sin(angle + 0.4));
                    ctx.stroke();
                }
            });

            // OUT Lane (진출) - 점선
            ctx.setLineDash([6/scale, 3/scale]);
            zone.outLanes.forEach(lane => {
                const from = nodeMap[lane.from];
                const to = nodeMap[lane.to];
                if (from && to) {
                    const fromIso = toIso(from.x, from.y, zoneZ);
                    const toIso_ = toIso(to.x, to.y, zoneZ);
                    ctx.beginPath();
                    ctx.moveTo(fromIso.x, fromIso.y);
                    ctx.lineTo(toIso_.x, toIso_.y);
                    ctx.stroke();
                }
            });

            // Zone ID 텍스트 표시
            if (firstLaneMid) {
                ctx.globalAlpha = 1.0;
                ctx.setLineDash([]);

                // 배경 박스
                const fontSize = Math.max(8, 10 / scale);
                // Zone_ID 표시
                let label = 'HID ' + zone.zoneId;
                ctx.font = 'bold ' + fontSize + 'px sans-serif';
                const textWidth = ctx.measureText(label).width;
                const padding = 3 / scale;

                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(
                    firstLaneMid.x - textWidth/2 - padding,
                    firstLaneMid.y - fontSize/2 - padding,
                    textWidth + padding * 2,
                    fontSize + padding * 2
                );

                // 텍스트
                ctx.fillStyle = zoneColor;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, firstLaneMid.x, firstLaneMid.y);

                // 차량 수 표시 (상태가 있으면)
                if (status) {
                    const countLabel = status.vehicleCount + '/' + zone.vehicleMax;
                    const countY = firstLaneMid.y + fontSize + padding * 2;
                    const countFontSize = Math.max(8, 11 / scale);
                    ctx.font = countFontSize + 'px sans-serif';
                    const countWidth = ctx.measureText(countLabel).width;

                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(
                        firstLaneMid.x - countWidth/2 - padding,
                        countY - countFontSize/2 - padding,
                        countWidth + padding * 2,
                        countFontSize + padding * 2
                    );

                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(countLabel, firstLaneMid.x, countY);
                }
            }

            // 선택된 Zone 노란색 하이라이트 표시
            if (selectedZoneId === zone.zoneId) {
                // Zone의 모든 노드 좌표 수집
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                const allLanes = [...zone.inLanes, ...zone.outLanes];
                allLanes.forEach(lane => {
                    const from = nodeMap[lane.from];
                    const to = nodeMap[lane.to];
                    if (from) {
                        minX = Math.min(minX, from.x); maxX = Math.max(maxX, from.x);
                        minY = Math.min(minY, from.y); maxY = Math.max(maxY, from.y);
                    }
                    if (to) {
                        minX = Math.min(minX, to.x); maxX = Math.max(maxX, to.x);
                        minY = Math.min(minY, to.y); maxY = Math.max(maxY, to.y);
                    }
                });

                if (minX !== Infinity) {
                    // 여유 공간 추가
                    const padding = 50;
                    minX -= padding; maxX += padding;
                    minY -= padding; maxY += padding;

                    // 4개 코너를 isometric 변환
                    const topLeft = toIso(minX, minY, zoneZ);
                    const topRight = toIso(maxX, minY, zoneZ);
                    const bottomRight = toIso(maxX, maxY, zoneZ);
                    const bottomLeft = toIso(minX, maxY, zoneZ);

                    // 노란색 사각형 그리기
                    ctx.strokeStyle = '#ffff00';
                    ctx.lineWidth = Math.max(3, 4 / scale);
                    ctx.globalAlpha = 1.0;
                    ctx.setLineDash([10/scale, 5/scale]);

                    ctx.beginPath();
                    ctx.moveTo(topLeft.x, topLeft.y);
                    ctx.lineTo(topRight.x, topRight.y);
                    ctx.lineTo(bottomRight.x, bottomRight.y);
                    ctx.lineTo(bottomLeft.x, bottomLeft.y);
                    ctx.closePath();
                    ctx.stroke();

                    // 내부 반투명 노란색 채우기
                    ctx.fillStyle = 'rgba(255, 255, 0, 0.15)';
                    ctx.fill();

                    ctx.setLineDash([]);
                }
            }
        });

        ctx.globalAlpha = 1.0;
        ctx.setLineDash([]);
    }

    // ============================================================
    // Station 표시 (pseudo-3D 변환 적용)
    // Station, ID, 이름 중 하나라도 ON이면 마커 표시
    // ============================================================
    if (window.stations && (window.showStations || window.showStationIds || window.showStationNames)) {
        const stationZ = RAIL_HEIGHT / scale + 1;  // Station 높이

        window.stations.forEach(station => {
            if (!station.hasCoords) return;

            const pos = toIso(station.x, station.y, stationZ);
            const size = Math.max(3, 5 / scale);

            // 타입별 색상
            let stationColor;
            switch (station.stationType) {
                case 'MAINTENANCE': stationColor = '#ff6600'; break;
                case 'ACQUIRE': stationColor = '#00ff00'; break;
                case 'DEPOSIT': stationColor = '#ff00ff'; break;
                case 'DUMMY': stationColor = '#666666'; break;
                default: stationColor = '#00aaff';  // DUAL_ACCESS 등
            }

            // 장비 타입에 따른 마커 모양
            ctx.globalAlpha = 0.8;
            ctx.fillStyle = stationColor;

            if (station.equipType === 'ZFS_RIGHT' || station.equipType === 'ZFS_LEFT') {
                // ZFS: 작은 사각형
                ctx.fillRect(pos.x - size/2, pos.y - size/2, size, size);
            } else if (station.equipType === 'UNIVERSAL') {
                // UNIVERSAL: 작은 원
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, size/2, 0, Math.PI * 2);
                ctx.fill();
            } else {
                // 기타: 다이아몬드
                ctx.beginPath();
                ctx.moveTo(pos.x, pos.y - size/2);
                ctx.lineTo(pos.x + size/2, pos.y);
                ctx.lineTo(pos.x, pos.y + size/2);
                ctx.lineTo(pos.x - size/2, pos.y);
                ctx.closePath();
                ctx.fill();
            }

            // ID 또는 이름 라벨 표시
            if (window.showStationIds || window.showStationNames) {
                ctx.globalAlpha = 1.0;
                const fontSize = Math.max(8, 10 / scale);
                ctx.font = fontSize + 'px sans-serif';
                const padding = 2 / scale;
                let offsetY = size/2 + 2/scale;

                // ID 표시
                if (window.showStationIds) {
                    const idLabel = String(station.stationId);
                    const idWidth = ctx.measureText(idLabel).width;

                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(
                        pos.x - idWidth/2 - padding,
                        pos.y + offsetY,
                        idWidth + padding * 2,
                        fontSize + padding * 2
                    );

                    ctx.fillStyle = '#ffff00';  // ID는 노란색
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText(idLabel, pos.x, pos.y + offsetY + padding);

                    offsetY += fontSize + padding * 2 + 1/scale;
                }

                // 이름 표시
                if (window.showStationNames) {
                    const nameLabel = station.stationName;
                    const nameWidth = ctx.measureText(nameLabel).width;

                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(
                        pos.x - nameWidth/2 - padding,
                        pos.y + offsetY,
                        nameWidth + padding * 2,
                        fontSize + padding * 2
                    );

                    ctx.fillStyle = stationColor;  // 이름은 타입 색상
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText(nameLabel, pos.x, pos.y + offsetY + padding);
                }
            }
        });

        ctx.globalAlpha = 1.0;
    }

    // OHT 크기
    const vehSize = Math.max(4, 7 / scale);

    // 선택된 OHT들 경로 표시 (여러개) - pseudo-3D 변환 적용
    const colors = ['#00ffff', '#ff00ff', '#ffff00', '#00ff00', '#ff8800', '#8800ff'];
    let colorIdx = 0;
    selectedVehicles.forEach(vid => {
        const sv = vehicles[vid];
        if (!sv) return;

        const path = sv.path || [];
        const color = colors[colorIdx % colors.length];
        colorIdx++;
        const pathZ = RAIL_HEIGHT / scale + 5;  // 경로 표시 높이

        if (path.length > 0) {
            // 현재 위치에서 시작하는 경로선
            ctx.strokeStyle = color;
            ctx.lineWidth = 4 / scale;
            ctx.setLineDash([8/scale, 4/scale]);
            ctx.beginPath();
            const startIso = toIso(sv.dispX, sv.dispY, pathZ);
            ctx.moveTo(startIso.x, startIso.y);

            // 경로의 각 노드를 따라 선 그리기
            path.forEach(nodeNo => {
                const node = nodeMap[nodeNo];
                if (node) {
                    const nodeIso = toIso(node.x, node.y, pathZ);
                    ctx.lineTo(nodeIso.x, nodeIso.y);
                }
            });
            ctx.stroke();
            ctx.setLineDash([]);

            // 경로 상의 노드들 강조
            ctx.fillStyle = color;
            path.forEach((nodeNo, idx) => {
                const node = nodeMap[nodeNo];
                if (node) {
                    const nodeIso = toIso(node.x, node.y, pathZ);
                    ctx.beginPath();
                    ctx.arc(nodeIso.x, nodeIso.y, 4 / scale, 0, Math.PI * 2);
                    ctx.fill();
                }
            });

            // 목적지 마커 (마지막 노드)
            const destNode = nodeMap[sv.destination];
            if (destNode) {
                const destIso = toIso(destNode.x, destNode.y, pathZ);
                ctx.fillStyle = '#ff0066';
                ctx.beginPath();
                ctx.arc(destIso.x, destIso.y, 12 / scale, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2 / scale;
                ctx.stroke();

                // 목적지 텍스트
                ctx.fillStyle = '#fff';
                ctx.font = `bold ${14/scale}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.fillText(vid, destIso.x, destIso.y - 18/scale);
            }
        }

        // 선택된 OHT 강조 테두리
        const ohtZ = RAIL_HEIGHT / scale + OHT_HEIGHT / scale;
        const svIso = toIso(sv.dispX, sv.dispY, ohtZ);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3 / scale;
        ctx.beginPath();
        if (isPseudo3D) {
            ctx.ellipse(svIso.x, svIso.y, vehSize + 8/scale, (vehSize + 8/scale) * 0.6, 0, 0, Math.PI * 2);
        } else {
            ctx.arc(sv.dispX, sv.dispY, vehSize + 5/scale, 0, Math.PI * 2);
        }
        ctx.stroke();
    });

    // OHT - pseudo-3D 효과 적용
    Object.values(vehicles).forEach(v => {
        let color = '#00ff88';
        if (v.state === 7) color = '#ff0000';  // JAM (정체) - 빨간색 깜빡임
        else if (v.state === 2) color = '#ff3366';  // 정지
        else if (v.isLoaded === 1) color = '#ff9900';  // 적재

        if (isPseudo3D) {
            // ============================================================
            // Pseudo-3D OHT 렌더링 (3D 박스 형태)
            // ============================================================
            const ohtW = vehSize * 2.5;  // OHT 너비
            const ohtH = vehSize * 1.5;  // OHT 깊이
            const ohtD = OHT_HEIGHT / scale;  // OHT 높이

            // 바닥 그림자 (더 진하게)
            const shadowPos = toIso(v.dispX + 3/scale, v.dispY + 3/scale, 0);
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.beginPath();
            ctx.ellipse(shadowPos.x, shadowPos.y, ohtW * 0.7, ohtH * 0.4, 0, 0, Math.PI * 2);
            ctx.fill();

            // 연결봉 (레일에서 OHT 본체로)
            const railTop = toIso(v.dispX, v.dispY, RAIL_HEIGHT / scale);
            const ohtBottom = toIso(v.dispX, v.dispY, RAIL_HEIGHT / scale + ohtD * 0.3);
            ctx.strokeStyle = '#556677';
            ctx.lineWidth = 3 / scale;
            ctx.beginPath();
            ctx.moveTo(railTop.x, railTop.y);
            ctx.lineTo(ohtBottom.x, ohtBottom.y);
            ctx.stroke();

            // OHT 본체 (3D 박스)
            const baseZ = RAIL_HEIGHT / scale + ohtD * 0.3;

            // 뒷면 (어두운 색) - 먼저 그려야 가려짐
            const backTop = toIso(v.dispX, v.dispY - ohtH/2, baseZ + ohtD);
            const backBot = toIso(v.dispX, v.dispY - ohtH/2, baseZ);
            ctx.fillStyle = adjustColor(color, 0.5);
            ctx.beginPath();
            ctx.moveTo(backTop.x - ohtW/2, backTop.y);
            ctx.lineTo(backTop.x + ohtW/2, backTop.y);
            ctx.lineTo(backBot.x + ohtW/2, backBot.y);
            ctx.lineTo(backBot.x - ohtW/2, backBot.y);
            ctx.closePath();
            ctx.fill();

            // 왼쪽면
            const leftTopF = toIso(v.dispX - ohtW/2, v.dispY + ohtH/2, baseZ + ohtD);
            const leftTopB = toIso(v.dispX - ohtW/2, v.dispY - ohtH/2, baseZ + ohtD);
            const leftBotF = toIso(v.dispX - ohtW/2, v.dispY + ohtH/2, baseZ);
            const leftBotB = toIso(v.dispX - ohtW/2, v.dispY - ohtH/2, baseZ);
            ctx.fillStyle = adjustColor(color, 0.8);
            ctx.beginPath();
            ctx.moveTo(leftTopF.x, leftTopF.y);
            ctx.lineTo(leftTopB.x, leftTopB.y);
            ctx.lineTo(leftBotB.x, leftBotB.y);
            ctx.lineTo(leftBotF.x, leftBotF.y);
            ctx.closePath();
            ctx.fill();

            // 상단면 (가장 밝은 색)
            const topFL = toIso(v.dispX - ohtW/2, v.dispY + ohtH/2, baseZ + ohtD);
            const topFR = toIso(v.dispX + ohtW/2, v.dispY + ohtH/2, baseZ + ohtD);
            const topBR = toIso(v.dispX + ohtW/2, v.dispY - ohtH/2, baseZ + ohtD);
            const topBL = toIso(v.dispX - ohtW/2, v.dispY - ohtH/2, baseZ + ohtD);
            ctx.fillStyle = adjustColor(color, 1.3);
            ctx.beginPath();
            ctx.moveTo(topFL.x, topFL.y);
            ctx.lineTo(topFR.x, topFR.y);
            ctx.lineTo(topBR.x, topBR.y);
            ctx.lineTo(topBL.x, topBL.y);
            ctx.closePath();
            ctx.fill();

            // 앞면 (원래 색)
            const frontTopL = toIso(v.dispX - ohtW/2, v.dispY + ohtH/2, baseZ + ohtD);
            const frontTopR = toIso(v.dispX + ohtW/2, v.dispY + ohtH/2, baseZ + ohtD);
            const frontBotL = toIso(v.dispX - ohtW/2, v.dispY + ohtH/2, baseZ);
            const frontBotR = toIso(v.dispX + ohtW/2, v.dispY + ohtH/2, baseZ);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(frontTopL.x, frontTopL.y);
            ctx.lineTo(frontTopR.x, frontTopR.y);
            ctx.lineTo(frontBotR.x, frontBotR.y);
            ctx.lineTo(frontBotL.x, frontBotL.y);
            ctx.closePath();
            ctx.fill();

            // 오른쪽면 (어두운 색)
            const rightTopF = toIso(v.dispX + ohtW/2, v.dispY + ohtH/2, baseZ + ohtD);
            const rightTopB = toIso(v.dispX + ohtW/2, v.dispY - ohtH/2, baseZ + ohtD);
            const rightBotF = toIso(v.dispX + ohtW/2, v.dispY + ohtH/2, baseZ);
            const rightBotB = toIso(v.dispX + ohtW/2, v.dispY - ohtH/2, baseZ);
            ctx.fillStyle = adjustColor(color, 0.6);
            ctx.beginPath();
            ctx.moveTo(rightTopF.x, rightTopF.y);
            ctx.lineTo(rightTopB.x, rightTopB.y);
            ctx.lineTo(rightBotB.x, rightBotB.y);
            ctx.lineTo(rightBotF.x, rightBotF.y);
            ctx.closePath();
            ctx.fill();

            // 테두리
            ctx.strokeStyle = adjustColor(color, 0.4);
            ctx.lineWidth = 0.5 / scale;
            ctx.stroke();

            // JAM 상태 깜빡임 효과 (3D 글로우)
            if (v.state === 7) {
                const blinkAlpha = 0.3 + 0.5 * Math.abs(Math.sin(Date.now() / 200));
                const glowPos = toIso(v.dispX, v.dispY, baseZ + ohtD/2);
                ctx.fillStyle = `rgba(255, 50, 50, ${blinkAlpha})`;
                ctx.beginPath();
                ctx.ellipse(glowPos.x, glowPos.y, ohtW * 0.8, ohtH * 0.5, 0, 0, Math.PI * 2);
                ctx.fill();
            }

            // 적재 표시 (상단에 작은 박스)
            if (v.isLoaded === 1) {
                const cargoZ = baseZ + ohtD + 3/scale;
                const cargoPos = toIso(v.dispX, v.dispY, cargoZ);
                const cargoW = ohtW * 0.6;
                const cargoH = ohtH * 0.6;

                // 화물 상자 (FOUP)
                ctx.fillStyle = '#ffffff';
                const cargoTopL = toIso(v.dispX - cargoW/2, v.dispY + cargoH/2, cargoZ + 5/scale);
                const cargoTopR = toIso(v.dispX + cargoW/2, v.dispY + cargoH/2, cargoZ + 5/scale);
                const cargoBotL = toIso(v.dispX - cargoW/2, v.dispY + cargoH/2, cargoZ);
                const cargoBotR = toIso(v.dispX + cargoW/2, v.dispY + cargoH/2, cargoZ);
                ctx.beginPath();
                ctx.moveTo(cargoTopL.x, cargoTopL.y);
                ctx.lineTo(cargoTopR.x, cargoTopR.y);
                ctx.lineTo(cargoBotR.x, cargoBotR.y);
                ctx.lineTo(cargoBotL.x, cargoBotL.y);
                ctx.closePath();
                ctx.fill();
                ctx.fillStyle = '#dddddd';
                ctx.fill();
            }

        } else {
            // ============================================================
            // 기존 2D OHT 렌더링
            // ============================================================
            // 그림자
            ctx.fillStyle = 'rgba(0,0,0,0.4)';
            ctx.beginPath();
            ctx.arc(v.dispX + 2/scale, v.dispY + 2/scale, vehSize, 0, Math.PI * 2);
            ctx.fill();

            // 본체
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(v.dispX, v.dispY, vehSize, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5 / scale;
            ctx.stroke();

            // JAM 상태 (정체) 깜빡임 효과
            if (v.state === 7) {
                const blinkAlpha = 0.3 + 0.7 * Math.abs(Math.sin(Date.now() / 200));
                ctx.fillStyle = `rgba(255, 0, 0, ${blinkAlpha})`;
                ctx.beginPath();
                ctx.arc(v.dispX, v.dispY, vehSize + 3/scale, 0, Math.PI * 2);
                ctx.fill();
            }

            // 적재 표시
            if (v.isLoaded === 1) {
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(v.dispX, v.dispY, vehSize * 0.4, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    });

    ctx.restore();

    document.getElementById('zoomLevel').textContent = Math.round(scale * 100) + '%';
}

// 마우스 이벤트
canvas.addEventListener('mousedown', e => {
    isDragging = true;
    lastMouse = {x: e.clientX, y: e.clientY};
});
canvas.addEventListener('mousemove', e => {
    if (isDragging) {
        offsetX += e.clientX - lastMouse.x;
        offsetY += e.clientY - lastMouse.y;
        lastMouse = {x: e.clientX, y: e.clientY};
    }
    updateTooltip(e);
});
canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mouseleave', () => { isDragging = false; document.getElementById('tooltip').style.display = 'none'; });

// 더블클릭 - OHT 선택 토글 (여러개 선택 가능) - pseudo-3D 좌표 변환 적용
canvas.addEventListener('dblclick', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - offsetX) / scale;
    const my = (e.clientY - rect.top - offsetY) / scale;

    let clicked = null;
    let minDist = 25 / scale;
    Object.values(vehicles).forEach(v => {
        // pseudo-3D 모드에서는 변환된 좌표로 비교
        if (isPseudo3D) {
            const ohtZ = RAIL_HEIGHT / scale + OHT_HEIGHT / scale * 0.5;
            const vIso = toIso(v.dispX, v.dispY, ohtZ);
            const d = Math.hypot(vIso.x - mx, vIso.y - my);
            if (d < minDist) { minDist = d; clicked = v.vehicleId; }
        } else {
            const d = Math.hypot(v.dispX - mx, v.dispY - my);
            if (d < minDist) { minDist = d; clicked = v.vehicleId; }
        }
    });

    if (clicked) {
        // 이미 선택된 OHT면 해제, 아니면 추가
        if (selectedVehicles.has(clicked)) {
            selectedVehicles.delete(clicked);
            console.log('선택 해제:', clicked);
        } else {
            selectedVehicles.add(clicked);
            console.log('선택 추가:', clicked);
        }
    }
    // 빈 곳 더블클릭은 아무것도 안함 (기존 선택 유지)
});

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const zoom = e.deltaY < 0 ? 1.15 : 0.87;
    const newScale = Math.max(0.02, Math.min(15, scale * zoom));

    offsetX = mx - (mx - offsetX) * (newScale / scale);
    offsetY = my - (my - offsetY) * (newScale / scale);
    scale = newScale;

    // 확대 50% 이상이면 Station ID 자동 ON, 미만이면 OFF
    updateStationIdAutoToggle();
});

// Station ID 자동 토글 함수 (확대 50% 기준)
function updateStationIdAutoToggle() {
    const zoomPercent = scale * 100;
    const btn = document.getElementById('btnToggleStationIds');

    if (zoomPercent >= 50 && !window.showStationIds) {
        window.showStationIds = true;
        if (btn) {
            btn.textContent = 'ID ON';
            btn.style.background = '#00d4ff';
            btn.style.color = '#000';
        }
    } else if (zoomPercent < 50 && window.showStationIds) {
        window.showStationIds = false;
        if (btn) {
            btn.textContent = 'ID OFF';
            btn.style.background = '#555';
            btn.style.color = '#fff';
        }
    }
}

function updateTooltip(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - offsetX) / scale;
    const my = (e.clientY - rect.top - offsetY) / scale;

    let closest = null, minDist = 20 / scale;
    Object.values(vehicles).forEach(v => {
        // pseudo-3D 모드에서는 변환된 좌표로 비교
        if (isPseudo3D) {
            const ohtZ = RAIL_HEIGHT / scale + OHT_HEIGHT / scale * 0.5;
            const vIso = toIso(v.dispX, v.dispY, ohtZ);
            const d = Math.hypot(vIso.x - mx, vIso.y - my);
            if (d < minDist) { minDist = d; closest = v; }
        } else {
            const d = Math.hypot(v.dispX - mx, v.dispY - my);
            if (d < minDist) { minDist = d; closest = v; }
        }
    });

    const tooltip = document.getElementById('tooltip');
    if (closest) {
        // 상태명 매핑
        const stateNames = {1:'운행', 2:'정지', 3:'이상', 4:'수동', 5:'제거중', 6:'OBS정지', 7:'정체', 8:'HT정지', 9:'E84타임아웃'};
        const stateName = stateNames[closest.state] || closest.stateName || '알수없음';

        // 거리 100mm 단위 표시
        const distanceDisplay = closest.distance > 0 ? (closest.distance / 100).toFixed(1) + ' (x100mm)' : '0';

        tooltip.innerHTML = `
            <div class="title">${closest.vehicleId}</div>
            <div class="row"><span class="label">상태</span><span class="value">${stateName}</span></div>
            <div class="row"><span class="label">속도</span><span class="value">${closest.velocity} m/min</span></div>
            <div class="row"><span class="label">적재</span><span class="value">${closest.isLoaded === 1 ? 'O (' + closest.carrierId + ')' : 'X'}</span></div>
            <div class="row"><span class="label">현재번지</span><span class="value">${closest.currentNode}</span></div>
            <div class="row"><span class="label">다음번지</span><span class="value">${closest.nextNode || '-'}</span></div>
            <div class="row"><span class="label">거리</span><span class="value">${distanceDisplay}</span></div>
            <div class="row"><span class="label">목적지</span><span class="value">${closest.destination || '-'}</span></div>
            <div class="row"><span class="label">HID Zone</span><span class="value" style="color:#00d4ff;">${closest.hidZoneId >= 0 ? 'HID ' + closest.hidZoneId : '-'}</span></div>
            <hr style="border:none;border-top:1px solid #444;margin:6px 0;">
            <div class="row"><span class="label">RunCycle</span><span class="value">${closest.runCycleName || closest.runCycle}</span></div>
            <div class="row"><span class="label">VhlCycle</span><span class="value">${closest.vhlCycleName || closest.vhlCycle}</span></div>
            <div class="row"><span class="label">상세상태</span><span class="value">${closest.detailState || '-'}</span></div>
        `;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY + 15) + 'px';
    } else {
        tooltip.style.display = 'none';
    }
}

function fitView() {
    if (!layout || !layout.nodes.length) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    layout.nodes.forEach(n => {
        minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    });

    const pad = 50;
    const w = maxX - minX, h = maxY - minY;
    const sx = (canvas.width - pad * 2) / w;
    const sy = (canvas.height - pad * 2) / h;
    scale = Math.min(sx, sy);

    offsetX = pad - minX * scale + (canvas.width - pad * 2 - w * scale) / 2;
    offsetY = pad - minY * scale + (canvas.height - pad * 2 - h * scale) / 2;

    // 확대 50% 기준 Station ID 자동 토글
    updateStationIdAutoToggle();
}

// 데드락 상황 만들기 버튼
document.getElementById('btnCreateDeadlock').addEventListener('click', async () => {
    const btn = document.getElementById('btnCreateDeadlock');
    const msg = document.getElementById('deadlockMsg');
    btn.disabled = true;
    btn.textContent = '생성중...';
    msg.textContent = '';

    try {
        const res = await fetch('/api/create-deadlock', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            msg.style.color = '#00ff88';
            msg.textContent = `노드 ${data.targetNode}에 ${data.movedVehicles}대 집중!`;
        } else {
            msg.style.color = '#ff3366';
            msg.textContent = data.message;
        }
    } catch (e) {
        msg.style.color = '#ff3366';
        msg.textContent = 'Error: ' + e.message;
    }

    btn.disabled = false;
    btn.textContent = '상황만들기';
});

// In/Out 초기화 버튼
document.getElementById('btnResetInOut').addEventListener('click', async () => {
    const btn = document.getElementById('btnResetInOut');
    const msg = document.getElementById('deadlockMsg');
    btn.disabled = true;

    try {
        const res = await fetch('/api/reset-inout', { method: 'POST' });
        const data = await res.json();
        msg.style.color = '#00d4ff';
        msg.textContent = 'In/Out 카운터 초기화됨';
    } catch (e) {
        msg.style.color = '#ff3366';
        msg.textContent = 'Error: ' + e.message;
    }

    btn.disabled = false;
});

// Zone 표시 토글 버튼
window.showZones = true;  // 기본값: 표시
document.getElementById('btnToggleZones').addEventListener('click', () => {
    const btn = document.getElementById('btnToggleZones');
    window.showZones = !window.showZones;

    if (window.showZones) {
        btn.textContent = 'Zone 표시 ON';
        btn.style.background = '#00d4ff';
        btn.style.color = '#000';
    } else {
        btn.textContent = 'Zone 표시 OFF';
        btn.style.background = '#555';
        btn.style.color = '#fff';
    }
});

// Station 표시 토글 버튼 (각각 독립적으로 동작)
// - Station: 마커만 표시
// - ID: 마커 + ID 라벨 표시
// - 이름: 마커 + 이름 라벨 표시
window.showStations = false;
window.showStationIds = false;
window.showStationNames = false;

document.getElementById('btnToggleStations').addEventListener('click', () => {
    const btn = document.getElementById('btnToggleStations');
    window.showStations = !window.showStations;

    if (window.showStations) {
        btn.textContent = 'Station ON';
        btn.style.background = '#00aaff';
        btn.style.color = '#000';
    } else {
        btn.textContent = 'Station OFF';
        btn.style.background = '#555';
        btn.style.color = '#fff';
    }
});

// Station ID 표시 토글 버튼
document.getElementById('btnToggleStationIds').addEventListener('click', () => {
    const btn = document.getElementById('btnToggleStationIds');
    window.showStationIds = !window.showStationIds;

    if (window.showStationIds) {
        btn.textContent = 'ID ON';
        btn.style.background = '#ffff00';
        btn.style.color = '#000';
    } else {
        btn.textContent = 'ID OFF';
        btn.style.background = '#555';
        btn.style.color = '#fff';
    }
});

// Station 이름 표시 토글 버튼
document.getElementById('btnToggleStationNames').addEventListener('click', () => {
    const btn = document.getElementById('btnToggleStationNames');
    window.showStationNames = !window.showStationNames;

    if (window.showStationNames) {
        btn.textContent = '이름 ON';
        btn.style.background = '#00aaff';
        btn.style.color = '#000';
    } else {
        btn.textContent = '이름 OFF';
        btn.style.background = '#555';
        btn.style.color = '#fff';
    }
});

// OHT 대수 변경 기능
async function setVehicleCount(count) {
    const msgEl = document.getElementById('vehicleMsg');
    const inputEl = document.getElementById('inputVehicleCount');
    const btn = document.getElementById('btnSetVehicles');

    btn.disabled = true;
    btn.textContent = '처리중...';
    msgEl.textContent = '요청 중...';
    msgEl.style.color = '#888';

    try {
        const response = await fetch('/api/set-vehicle-count?count=' + count, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.status === 'ok') {
            msgEl.textContent = data.message + ' (총 ' + data.total + '대)';
            msgEl.style.color = '#00ff88';
            inputEl.value = data.total;
        } else {
            msgEl.textContent = data.message;
            msgEl.style.color = '#ff3366';
        }
    } catch (e) {
        msgEl.textContent = '오류: ' + e.message;
        msgEl.style.color = '#ff3366';
    }

    btn.disabled = false;
    btn.textContent = '적용';

    setTimeout(() => {
        msgEl.textContent = '';
    }, 3000);
}

document.getElementById('btnSetVehicles').addEventListener('click', () => {
    const count = parseInt(document.getElementById('inputVehicleCount').value);
    if (count && count > 0) {
        setVehicleCount(count);
    }
});

document.getElementById('btnQuick50').addEventListener('click', () => {
    document.getElementById('inputVehicleCount').value = 50;
    setVehicleCount(50);
});

document.getElementById('btnQuick100').addEventListener('click', () => {
    document.getElementById('inputVehicleCount').value = 100;
    setVehicleCount(100);
});

document.getElementById('btnQuick500').addEventListener('click', () => {
    document.getElementById('inputVehicleCount').value = 500;
    setVehicleCount(500);
});

// Enter 키로 적용
document.getElementById('inputVehicleCount').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const count = parseInt(e.target.value);
        if (count && count > 0) {
            setVehicleCount(count);
        }
    }
});

// JAM 통계 로드
async function loadJamStats() {
    try {
        const res = await fetch('/api/jam-stats');
        const data = await res.json();
        document.getElementById('jamStatVehicles').textContent = data.jam_vehicle_count || 0;
        document.getElementById('jamStatHigh').textContent = data.high_risk_edges || 0;
        document.getElementById('jamStatMedium').textContent = data.medium_risk_edges || 0;
    } catch (e) {
        console.error('JAM 통계 로드 실패:', e);
    }
}

// 초기 JAM 통계 로드 및 5초마다 갱신
loadJamStats();
setInterval(loadJamStats, 5000);

// ============================================================
// Three.js 3D 뷰 (깔끔한 버전)
// ============================================================
let is3DMode = false;
let scene, camera, renderer, controls;
let railMeshes = [];
let ohtMeshes = {};

const RAIL_H = 200;
const SC = 0.2;
const CX = 10000, CY = 10000;

function init3D() {
    const ct = document.getElementById('three-container');

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    camera = new THREE.PerspectiveCamera(45, ct.clientWidth / ct.clientHeight, 10, 30000);
    camera.position.set(0, 4000, 4000);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(ct.clientWidth, ct.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    ct.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2.2;
    controls.minDistance = 1000;
    controls.maxDistance = 15000;

    // 조명
    scene.add(new THREE.AmbientLight(0x4466aa, 0.4));
    const sun = new THREE.DirectionalLight(0xffffff, 0.6);
    sun.position.set(2000, 3000, 1500);
    scene.add(sun);
    scene.add(new THREE.HemisphereLight(0x4488ff, 0x002244, 0.3));

    // 바닥 그리드
    const grid = new THREE.GridHelper(8000, 40, 0x1a3050, 0x0d1825);
    scene.add(grid);

    createRails();
    animate3D();
}

function createRails() {
    if (!layout?.edges) return;
    railMeshes.forEach(m => scene.remove(m));
    railMeshes = [];

    const mat = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.7 });

    layout.edges.forEach(e => {
        const fn = nodeMap[e.from], tn = nodeMap[e.to];
        if (!fn || !tn) return;

        const x1 = (fn.x - CX) * SC, z1 = (fn.y - CY) * SC;
        const x2 = (tn.x - CX) * SC, z2 = (tn.y - CY) * SC;
        const len = Math.hypot(x2-x1, z2-z1);
        if (len < 2) return;

        const g = new THREE.BoxGeometry(len, 4, 6);
        const m = new THREE.Mesh(g, mat);
        m.position.set((x1+x2)/2, RAIL_H, (z1+z2)/2);
        m.rotation.y = -Math.atan2(z2-z1, x2-x1);
        scene.add(m);
        railMeshes.push(m);
    });
}

function createOHT(c) {
    const g = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: c });

    // 본체
    const body = new THREE.Mesh(new THREE.BoxGeometry(30, 20, 45), mat);
    body.position.y = -15;
    body.name = 'b';
    g.add(body);

    // 연결부
    g.add(new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 20, 6), new THREE.MeshBasicMaterial({color: 0x445566})));

    return g;
}

function updateOHTs() {
    if (!vehicles) return;

    for (const [id, v] of Object.entries(vehicles)) {
        let c = 0x00ff88;
        if (v.state === 'JAM') c = 0xff2222;
        else if (v.state === 'STOP') c = 0xff5588;
        else if (v.loaded) c = 0xffaa00;

        let o = ohtMeshes[id];
        if (!o) { o = createOHT(c); scene.add(o); ohtMeshes[id] = o; }

        o.position.set((v.x - CX) * SC, RAIL_H, (v.y - CY) * SC);
        const b = o.getObjectByName('b');
        if (b) b.material.color.setHex(c);
    }

    for (const id of Object.keys(ohtMeshes)) {
        if (!vehicles[id]) { scene.remove(ohtMeshes[id]); delete ohtMeshes[id]; }
    }
}

function animate3D() {
    if (!is3DMode) return;
    requestAnimationFrame(animate3D);
    controls.update();
    updateOHTs();
    renderer.render(scene, camera);
}

// Pseudo-3D 효과 토글 (Three.js 대신 2D 캔버스에서 3D 효과)
function toggle3DMode() {
    isPseudo3D = !isPseudo3D;
    const btn = document.getElementById('btnToggle3D');

    if (isPseudo3D) {
        btn.textContent = '3D 효과 ON';
        btn.style.background = '#00d4ff';
    } else {
        btn.textContent = '3D 효과 OFF';
        btn.style.background = '#ff9900';
    }
    render();
}

document.getElementById('btnToggle3D').addEventListener('click', toggle3DMode);
</script>
</body>
</html>
"""

# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=10004)  # 3D 버전: 포트 10004
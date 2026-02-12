#!/usr/bin/env python3
"""

pip install pyinstaller
pyinstaller build_exe.spec

3D Campus Builder v1.0
=====================
SK Hynix 스타일 3D 캠퍼스 시각화 HTML을 생성하는 Python GUI 프로그램

기능:
- 2D 캔버스에서 건물/도로/나무/주차장/호수 배치 (마우스 드래그)
- 건물 속성 편집 (이름, 크기, 색상, 층수, 설명)
- 프로젝트 저장/불러오기 (JSON)
- 인터랙티브 3D HTML 파일 생성 (Three.js)
  - 마우스 궤도/팬/줌 컨트롤
  - 건물 클릭 시 정보 표시
  - 낮/밤 모드 전환
  - 검색 기능
  - 미니맵
  - 건물 목록 사이드바
  - 그리드 표시 토글
  - 라벨 표시 토글
  - 그림자 효과
  - 부드러운 카메라 애니메이션
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import json
import os
import math
import webbrowser
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ============================================================
# 데이터 모델
# ============================================================

@dataclass
class Building:
    id: str = ""
    name: str = "건물"
    x: float = 0
    z: float = 0
    width: float = 30
    depth: float = 30
    height: float = 20
    color: str = "#4488cc"
    floors: int = 3
    building_type: str = "office"
    description: str = ""
    roof_type: str = "flat"  # flat, peaked, dome

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Road:
    id: str = ""
    x: float = 0
    z: float = 0
    length: float = 100
    width: float = 8
    rotation: float = 0  # degrees
    color: str = "#555555"
    name: str = "도로"

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Tree:
    id: str = ""
    x: float = 0
    z: float = 0
    trunk_height: float = 4
    canopy_radius: float = 3
    color: str = "#228B22"
    name: str = "나무"

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class ParkingLot:
    id: str = ""
    x: float = 0
    z: float = 0
    width: float = 40
    depth: float = 25
    color: str = "#888888"
    name: str = "주차장"

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Lake:
    id: str = ""
    x: float = 0
    z: float = 0
    radius: float = 25
    color: str = "#4499cc"
    name: str = "호수"

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


SPEECH_BUBBLES = [
    "안녕하세요!", "좋은 아침!", "회의 가야해요", "점심 뭐 먹지?",
    "커피 마시러 가요~", "오늘 날씨 좋다!", "퇴근하고 싶다...",
    "파이팅!", "수고하셨습니다", "잠깐만요!", "화이팅~",
    "배고프다", "오늘도 힘내자!", "좋은 하루!", "감사합니다",
    "먼저 갈게요~", "조심히 가세요", "다음에 봐요!", "잘 부탁드려요",
]


@dataclass
class Person:
    id: str = ""
    x: float = 0
    z: float = 0
    speed: float = 0.5  # walking speed
    direction: float = 0  # degrees, walking direction
    shirt_color: str = "#3366cc"
    pants_color: str = "#333333"
    name: str = "사람"
    walk_radius: float = 30  # how far they walk from origin point
    speech: str = ""  # 말풍선 텍스트 (비어있으면 랜덤)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.speech:
            import random
            self.speech = random.choice(SPEECH_BUBBLES)


@dataclass
class Gate:
    id: str = ""
    x: float = 0
    z: float = 0
    width: float = 20
    height: float = 8
    depth: float = 3
    color: str = "#aa8855"
    name: str = "정문"
    gate_type: str = "main"  # main, side
    has_barrier: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class WaterTank:
    id: str = ""
    x: float = 0
    z: float = 0
    radius: float = 12
    height: float = 28
    color: str = "#cccccc"
    name: str = "물탱크"

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class LPGTank:
    id: str = ""
    x: float = 0
    z: float = 0
    length: float = 20
    radius: float = 7
    color: str = "#ffffff"
    name: str = "LPG저장탱크"

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Chimney:
    id: str = ""
    x: float = 0
    z: float = 0
    height: float = 80
    radius: float = 5
    color: str = "#666666"
    name: str = "굴뚝"
    has_smoke: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Wall:
    id: str = ""
    x: float = 0
    z: float = 0
    length: float = 40
    height: float = 12
    thickness: float = 2
    rotation: float = 0
    color: str = "#888888"
    name: str = "벽"
    wall_type: str = "concrete"  # concrete, fence, brick

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Truck:
    id: str = ""
    x: float = 0
    z: float = 0
    direction: float = 0
    color: str = "#ffffff"
    name: str = "트럭"
    truck_type: str = "cargo"  # cargo, tanker, flatbed
    speed: float = 0.5
    route_radius: float = 30

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class TransportLine:
    id: str = ""
    x1: float = 0
    z1: float = 0
    x2: float = 50
    z2: float = 0
    height: float = 20
    color: str = "#44aaee"
    name: str = "연결통로"
    transport_type: str = "conveyor"  # conveyor, lifter, rail

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class CampusProject:
    name: str = "SK Hynix 3D Campus"
    version: str = "1.0"
    ground_width: float = 600
    ground_depth: float = 600
    ground_color: str = "#4a7c59"
    sky_color: str = "#87CEEB"
    buildings: List[Dict] = field(default_factory=list)
    roads: List[Dict] = field(default_factory=list)
    trees: List[Dict] = field(default_factory=list)
    parking_lots: List[Dict] = field(default_factory=list)
    lakes: List[Dict] = field(default_factory=list)
    persons: List[Dict] = field(default_factory=list)
    gates: List[Dict] = field(default_factory=list)
    water_tanks: List[Dict] = field(default_factory=list)
    lpg_tanks: List[Dict] = field(default_factory=list)
    chimneys: List[Dict] = field(default_factory=list)
    walls: List[Dict] = field(default_factory=list)
    trucks: List[Dict] = field(default_factory=list)
    transport_lines: List[Dict] = field(default_factory=list)
    label_scale: float = 1.5  # 이름표 크기 배율 (기본 1.5배)


# ============================================================
# HTML 생성 엔진
# ============================================================

def generate_html(project: CampusProject) -> str:
    """Three.js 기반 인터랙티브 3D 캠퍼스 HTML을 생성합니다."""

    buildings_json = json.dumps(project.buildings, ensure_ascii=False)
    roads_json = json.dumps(project.roads, ensure_ascii=False)
    trees_json = json.dumps(project.trees, ensure_ascii=False)
    parkings_json = json.dumps(project.parking_lots, ensure_ascii=False)
    lakes_json = json.dumps(project.lakes, ensure_ascii=False)
    persons_json = json.dumps(project.persons, ensure_ascii=False)
    gates_json = json.dumps(project.gates, ensure_ascii=False)
    water_tanks_json = json.dumps(project.water_tanks, ensure_ascii=False)
    lpg_tanks_json = json.dumps(project.lpg_tanks, ensure_ascii=False)
    chimneys_json = json.dumps(project.chimneys, ensure_ascii=False)
    walls_json = json.dumps(project.walls, ensure_ascii=False)
    trucks_json = json.dumps(project.trucks, ensure_ascii=False)
    transport_lines_json = json.dumps(project.transport_lines, ensure_ascii=False)
    label_scale = project.label_scale

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{project.name} - 3D Campus View</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700;900&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  background: #0a0a12;
  color: #e0e0e0;
  font-family: 'Noto Sans KR', sans-serif;
  overflow: hidden;
  width: 100vw;
  height: 100vh;
}}

#canvas-container {{
  width: 100%;
  height: 100%;
  position: relative;
}}

canvas {{ display: block; }}

/* Title overlay */
.title-overlay {{
  position: fixed;
  top: 20px;
  left: 24px;
  text-align: left;
  z-index: 100;
  pointer-events: none;
}}

.title-overlay h1 {{
  font-size: 22px;
  font-weight: 900;
  letter-spacing: 2px;
  color: #fff;
  text-shadow: 0 2px 20px rgba(0,0,0,0.8);
  margin-bottom: 4px;
}}

.title-overlay p {{
  font-size: 11px;
  color: #888;
  letter-spacing: 4px;
  text-transform: uppercase;
}}

/* Info panel */
.info-panel {{
  position: fixed;
  bottom: 24px;
  left: 24px;
  background: rgba(10,10,20,0.92);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 0;
  z-index: 100;
  min-width: 280px;
  max-width: 340px;
  max-height: 70vh;
  overflow-y: auto;
  transition: all 0.3s ease;
  display: none;
}}

.info-panel::-webkit-scrollbar {{ width: 4px; }}
.info-panel::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 2px; }}

.info-panel.visible {{ display: block; }}

.info-panel .panel-header {{
  padding: 14px 18px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  position: sticky;
  top: 0;
  background: rgba(10,10,20,0.95);
  border-radius: 14px 14px 0 0;
  z-index: 2;
}}

.info-panel .building-name {{
  font-size: 17px;
  font-weight: 800;
  color: #fff;
  margin-bottom: 3px;
  display: flex;
  align-items: center;
  gap: 8px;
}}

.info-panel .building-name .color-badge {{
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  flex-shrink: 0;
}}

.info-panel .building-type {{
  font-size: 10px;
  color: #888;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 0;
}}

.info-panel .panel-body {{ padding: 12px 18px 16px; }}

.info-panel .building-detail {{
  font-size: 12px;
  color: #aaa;
  line-height: 1.6;
  margin-bottom: 10px;
}}

.info-panel .spec-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin-bottom: 12px;
}}

.info-panel .spec-item {{
  background: rgba(255,255,255,0.03);
  border-radius: 6px;
  padding: 7px 10px;
}}

.info-panel .spec-label {{
  font-size: 9px;
  color: #555;
  letter-spacing: 1px;
}}

.info-panel .spec-value {{
  font-size: 13px;
  font-weight: 700;
  color: #ddd;
  margin-top: 1px;
}}

/* Legend */
.legend {{
  position: fixed;
  top: 24px;
  right: 24px;
  background: rgba(10,10,20,0.85);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 14px 18px;
  z-index: 100;
  font-size: 11px;
  max-height: 60vh;
  overflow-y: auto;
  min-width: 180px;
}}

.legend-title {{
  font-size: 10px;
  color: #666;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 10px;
}}

.legend-item {{
  display: flex;
  align-items: center;
  margin-bottom: 6px;
  color: #aaa;
}}

.legend-item .swatch {{
  width: 14px;
  height: 14px;
  border-radius: 3px;
  margin-right: 8px;
  flex-shrink: 0;
}}

/* Minimap */
#minimap {{
  position: fixed;
  bottom: 24px;
  right: 24px;
  width: 160px;
  height: 140px;
  background: rgba(8,10,18,0.92);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
  z-index: 100;
  overflow: hidden;
}}

#minimap canvas {{
  display: block;
  width: 100%;
  height: 100%;
}}

.minimap-label {{
  position: absolute;
  top: 5px;
  left: 8px;
  font-size: 7px;
  color: #445;
  letter-spacing: 2px;
  text-transform: uppercase;
  pointer-events: none;
  font-weight: 600;
}}

.minimap-compass {{
  position: absolute;
  top: 5px;
  right: 8px;
  font-size: 8px;
  color: #e74c3c;
  pointer-events: none;
  font-weight: 700;
}}

/* Tooltip */
#tooltip {{
  position: fixed;
  pointer-events: none;
  background: rgba(0,0,0,0.85);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  color: #fff;
  font-weight: 600;
  z-index: 200;
  display: none;
  white-space: nowrap;
  transform: translate(-50%, -120%);
}}

/* Time toggle panel */
#timeToggle {{
  position: fixed;
  top: 75px;
  left: 24px;
  z-index: 200;
  background: rgba(10,10,20,0.85);
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.1);
  backdrop-filter: blur(8px);
  user-select: none;
}}

#timeToggleHeader {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  cursor: move;
}}

#timeToggleHeader span {{
  font-size: 11px;
  color: #666;
  letter-spacing: 1px;
}}

#btnCollapse {{
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  font-size: 14px;
  padding: 0 4px;
}}

#timeToggleBody {{
  display: flex;
  gap: 6px;
  padding: 4px 12px 8px 12px;
}}

.time-btn {{
  padding: 6px 14px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  background: rgba(100,100,200,0.15);
  color: #aabbdd;
  transition: all 0.3s;
  white-space: nowrap;
}}

.time-btn.active {{
  box-shadow: 0 0 8px rgba(100,200,255,0.3);
  background: rgba(100,200,255,0.3);
  color: #88ddff;
}}

#btnMorning {{ background: rgba(255,180,50,0.2); color: #ffcc66; }}
#btnMorning.active {{ background: rgba(255,180,50,0.4); box-shadow: 0 0 8px rgba(255,180,50,0.3); }}

#btnNight {{ background: rgba(100,100,200,0.15); color: #aabbdd; }}
#btnNight.active {{ background: rgba(100,100,200,0.3); box-shadow: 0 0 8px rgba(100,100,200,0.3); }}

/* Brightness slider */
.brightness-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px 10px 12px;
}}
.brightness-row label {{
  font-size: 11px;
  color: #888;
  white-space: nowrap;
}}
.brightness-row input[type=range] {{
  flex: 1;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(255,255,255,0.15);
  border-radius: 2px;
  outline: none;
}}
.brightness-row input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #88ddff;
  cursor: pointer;
  box-shadow: 0 0 6px rgba(100,200,255,0.4);
}}
.brightness-row .val {{
  font-size: 11px;
  color: #aaa;
  min-width: 28px;
  text-align: right;
}}

/* Controls hint */
.controls-hint {{
  position: fixed;
  bottom: 24px;
  left: 200px;
  background: rgba(10,10,20,0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 10px 14px;
  z-index: 100;
  font-size: 10px;
  color: #666;
  line-height: 1.8;
}}

/* Loading */
.loading {{
  position: fixed;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #060810;
  z-index: 999;
  transition: opacity 1.2s ease;
}}

.loading.hidden {{ opacity: 0; pointer-events: none; }}

.load-logo {{
  font-size: 42px;
  font-weight: 900;
  color: #fff;
  letter-spacing: 4px;
  margin-bottom: 2px;
  text-shadow: 0 0 30px rgba(231,76,60,0.3), 0 4px 20px rgba(0,0,0,0.6);
  animation: ldIn 1.2s ease both;
}}

.load-campus {{
  font-size: 13px;
  color: #556;
  letter-spacing: 6px;
  text-transform: uppercase;
  margin-bottom: 6px;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5);
  animation: ldIn 1s ease 0.4s both;
}}

.load-bar-bg {{
  width: 240px;
  height: 2px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 16px;
  animation: ldIn 0.8s ease 0.9s both;
}}

.load-bar {{
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #e74c3c, #4af, #3a7);
  border-radius: 2px;
  animation: barGo 4s ease-in-out 1s forwards;
}}

.load-msg {{
  font-size: 10px;
  color: #334;
  letter-spacing: 4px;
  text-transform: uppercase;
  animation: ldPulse 1.5s ease-in-out 1s infinite;
}}

@keyframes ldIn {{ from {{ opacity: 0; transform: translateY(16px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes ldPulse {{ 0%, 100% {{ opacity: 0.2; }} 50% {{ opacity: 0.6; }} }}
@keyframes barGo {{ 0% {{ width: 0%; }} 20% {{ width: 22%; }} 45% {{ width: 50%; }} 70% {{ width: 75%; }} 90% {{ width: 92%; }} 100% {{ width: 100%; }} }}

</style>
</head>
<body>

<!-- Loading screen -->
<div class="loading" id="loading">
  <div class="load-logo">{project.name}</div>
  <div class="load-campus">3D Campus Visualization</div>
  <div class="load-bar-bg"><div class="load-bar"></div></div>
  <div class="load-msg">Initializing 3D Environment...</div>
</div>

<!-- Title overlay -->
<div class="title-overlay">
  <h1><span style="color:#fff">SK </span><span style="color:#e84040">hynix</span></h1>
  <p style="letter-spacing:3px;color:rgba(255,255,255,0.5)">ICHEON CAMPUS 3D</p>
  <p style="letter-spacing:2px;color:#4dd9a0;font-weight:600">AMOS MONITORING SYSTEM</p>
</div>

<!-- Legend -->
<div class="legend" id="legendPanel">
  <div class="legend-title">범 례</div>
  <div id="legendItems"></div>
  <div style="margin-top:8px;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px;">
    <button onclick="addLegendItem()" style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);color:#aaa;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:10px;width:100%;">+ 항목 추가</button>
  </div>
</div>

<!-- Time toggle panel -->
<div id="timeToggle">
  <div id="timeToggleHeader">
    <span>⏰ 시간 설정</span>
    <button id="btnCollapse" onclick="toggleTimePanel()">▼</button>
  </div>
  <div id="timeToggleBody">
    <button id="btnMorning" class="time-btn active" onclick="setTimeMode('morning')">🌅 아침</button>
    <button id="btnAuto" class="time-btn" onclick="setTimeMode('auto')">🕐 실시간</button>
    <button id="btnNight" class="time-btn" onclick="setTimeMode('night')">🌙 밤</button>
  </div>
  <div class="brightness-row">
    <label>☀️ 밝기</label>
    <input type="range" id="brightnessSlider" min="30" max="250" value="120" oninput="setBrightness(this.value)">
    <span class="val" id="brightnessVal">120%</span>
  </div>
  <div class="brightness-row" style="margin-top:6px;">
    <label>🚶 사람 속도</label>
    <input type="range" id="personSpeedSlider" min="0" max="200" value="100" oninput="setPersonSpeed(this.value)">
    <span class="val" id="personSpeedVal">100%</span>
  </div>
</div>

<!-- Info panel -->
<div class="info-panel" id="infoPanel">
  <div class="panel-header">
    <div class="building-name"><span class="color-badge" id="bBadge"></span><span id="bName"></span></div>
    <div class="building-type" id="bType"></div>
  </div>
  <div class="panel-body" id="panelBody"></div>
</div>

<!-- Minimap -->
<div id="minimap">
  <canvas id="minimapCanvas"></canvas>
  <div class="minimap-label">MINIMAP</div>
  <div class="minimap-compass">N</div>
</div>

<!-- Controls hint -->
<div class="controls-hint">
  좌클릭: 회전 | 우클릭: 이동 | 휠: 줌 | ESC: 선택 해제
</div>

<!-- Export: Python GUI에서 파일 > OBJ/React 내보내기 사용 -->

<!-- Tooltip -->
<div id="tooltip"></div>

<!-- Canvas container -->
<div id="canvas-container"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
// ========== Data ==========
const CAMPUS_NAME = {json.dumps(project.name, ensure_ascii=False)};
const GROUND_W = {project.ground_width};
const GROUND_D = {project.ground_depth};
const GROUND_SIZE = Math.max(GROUND_W, GROUND_D);
const SKY_COLOR = "{project.sky_color}";

const buildingsData = {buildings_json};
const roadsData = {roads_json};
const treesData = {trees_json};
const parkingsData = {parkings_json};
const lakesData = {lakes_json};
const personsData = {persons_json};
const gatesData = {gates_json};
const waterTanksData = {water_tanks_json};
const lpgTanksData = {lpg_tanks_json};
const chimneysData = {chimneys_json};
const wallsData = {walls_json};
const trucksData = {trucks_json};
const transportLinesData = {transport_lines_json};
let smokeParticles = [];
let movingTrucks = [];
let roadSegments = [];
let checkWallCollision = () => false;
let checkBuildingCollision = () => false;
let globalPersonSpeed = 1.0;  // 전역 사람 속도 배율
let globalLabelScale = {label_scale};   // 이름표 크기 배율 (Python에서 설정)
let allNameTagSprites = [];   // 이름표 스프라이트 목록

// 그라데이션 재질 생성 함수
function createGradientMaterial(baseColor, height) {{
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const c = new THREE.Color(baseColor);
  const topColor = c.clone().offsetHSL(0, 0, 0.15);
  const botColor = c.clone().offsetHSL(0, 0, -0.1);
  const grad = ctx.createLinearGradient(0, 0, 0, 128);
  grad.addColorStop(0, '#' + topColor.getHexString());
  grad.addColorStop(1, '#' + botColor.getHexString());
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 2, 128);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  return new THREE.MeshStandardMaterial({{
    map: tex,
    roughness: 0.65,
    metalness: 0.15
  }});
}}

// ========== Global variables ==========
let scene, camera, renderer;
let buildingMeshes = [];
let personMeshes = [];
let allMeshes = [];
let raycaster, mouse;
let selectedMesh = null;
let timeMode = 'morning'; // morning, auto, night
let brightnessMultiplier = 1.2; // 밝기 배율
let showLabels = true;
let showGrid = true;
let shadowsEnabled = true;
let ambientLight, dirLight, hemiLight;
let gridHelper;
let groundMesh;
let pointLight;
let sunMesh, moonMesh, moonLight, sunGlowMat, moonGlowMat, moonGlow2Mat;
const skyRadius = 350;
let labelElements = [];
let frameCount = 0;
let lastTime = performance.now();

// Camera control variables
let camTarget = new THREE.Vector3(0, 0, 0);
let camSpherical = {{ radius: 300, theta: Math.PI / 4, phi: Math.PI / 3 }};
let isDragging = false;
let isPanning = false;
let lastMouse = {{ x: 0, y: 0 }};
let targetSpherical = {{ ...camSpherical }};
let targetTarget = camTarget.clone();
let damping = 0.08;
let autoRotate = false;

// ========== Initialization ==========
function init() {{
  updateLoading(10);

  // Scene setup
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x99ccee);
  // fog는 updateSky에서 설정

  // Camera
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, GROUND_SIZE * 8);
  updateCameraPosition();

  // Renderer
  renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.8;
  document.getElementById('canvas-container').appendChild(renderer.domElement);

  updateLoading(20);

  // Raycaster
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // Lighting
  setupLighting();
  updateLoading(30);

  // Environment
  setupEnvironment();
  updateLoading(50);

  // Create objects
  createBuildings();
  updateLoading(70);
  createRoads();
  createTrees();
  createParkings();
  createLakes();
  createPersons();
  createGates();
  createInfrastructure();
  updateLoading(85);

  // Setup UI
  setupUI();
  setupControls();
  updateLoading(95);

  // Start animation
  updateStats();
  updateLoading(100);
  setTimeout(() => {{
    document.getElementById('loading').classList.add('hidden');
    setTimeout(() => document.getElementById('loading').style.display = 'none', 1200);
  }}, 500);

  // Apply initial time mode
  updateSky();
  animate();
}}

function updateLoading(pct) {{
  const bar = document.querySelector('.load-bar');
  if (bar) bar.style.width = Math.max(bar.offsetWidth === 0 ? 0 : parseFloat(bar.style.width), pct) + '%';
}}

// ========== Lighting ==========
function setupLighting() {{
  // Ambient light
  ambientLight = new THREE.AmbientLight(0x8899bb, 1.2);
  scene.add(ambientLight);

  // Hemisphere light
  hemiLight = new THREE.HemisphereLight(0xaaccee, 0x556677, 1.0);
  scene.add(hemiLight);

  // Directional light (sun)
  dirLight = new THREE.DirectionalLight(0xfff0dd, 1.6);
  dirLight.position.set(150, 200, 100);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.near = 1;
  dirLight.shadow.camera.far = GROUND_SIZE * 2;
  dirLight.shadow.camera.left = -GROUND_SIZE;
  dirLight.shadow.camera.right = GROUND_SIZE;
  dirLight.shadow.camera.top = GROUND_SIZE;
  dirLight.shadow.camera.bottom = -GROUND_SIZE;
  dirLight.shadow.bias = -0.001;
  scene.add(dirLight);

  // Point light for accents
  pointLight = new THREE.PointLight(0x4488ff, 0.3, 600);
  pointLight.position.set(-200, 150, 200);
  scene.add(pointLight);

  // ====== 천체 시스템 (해/달) ======
  // 해 (Sun)
  const sunGeo = new THREE.SphereGeometry(18, 32, 32);
  const sunMat = new THREE.MeshBasicMaterial({{ color: 0xffdd44 }});
  sunMesh = new THREE.Mesh(sunGeo, sunMat);
  scene.add(sunMesh);
  // 해 글로우
  const sunGlowGeo = new THREE.SphereGeometry(25, 32, 32);
  sunGlowMat = new THREE.MeshBasicMaterial({{ color: 0xffaa22, transparent: true, opacity: 0.2 }});
  const sunGlow = new THREE.Mesh(sunGlowGeo, sunGlowMat);
  sunMesh.add(sunGlow);

  // 달 (Moon)
  const moonGeo = new THREE.SphereGeometry(20, 32, 32);
  const moonMatObj = new THREE.MeshBasicMaterial({{ color: 0xfffff0 }});
  moonMesh = new THREE.Mesh(moonGeo, moonMatObj);
  scene.add(moonMesh);
  // 달 글로우 (내부)
  const moonGlowGeo = new THREE.SphereGeometry(30, 32, 32);
  moonGlowMat = new THREE.MeshBasicMaterial({{ color: 0xcceeff, transparent: true, opacity: 0.25 }});
  const moonGlow = new THREE.Mesh(moonGlowGeo, moonGlowMat);
  moonMesh.add(moonGlow);
  // 달 글로우 (외부 후광)
  const moonGlow2Geo = new THREE.SphereGeometry(50, 32, 32);
  moonGlow2Mat = new THREE.MeshBasicMaterial({{ color: 0x99bbdd, transparent: true, opacity: 0.1 }});
  const moonGlow2 = new THREE.Mesh(moonGlow2Geo, moonGlow2Mat);
  moonMesh.add(moonGlow2);
  // 달빛 포인트라이트 (달 조명 역할)
  moonLight = new THREE.PointLight(0x8899cc, 0, 800);
  moonMesh.add(moonLight);
}}

// ========== Environment ==========
function setupEnvironment() {{
  // Sky with gradient
  const skyGeo = new THREE.SphereGeometry(GROUND_SIZE * 6, 32, 32);
  const skyMat = new THREE.ShaderMaterial({{
    uniforms: {{
      topColor: {{ value: new THREE.Color(0x2277cc) }},
      bottomColor: {{ value: new THREE.Color(0x99ccee) }},
      offset: {{ value: 20 }},
      exponent: {{ value: 0.6 }}
    }},
    vertexShader: `
      varying vec3 vWorldPosition;
      void main() {{
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }}
    `,
    fragmentShader: `
      uniform vec3 topColor;
      uniform vec3 bottomColor;
      uniform float offset;
      uniform float exponent;
      varying vec3 vWorldPosition;
      void main() {{
        float h = normalize(vWorldPosition + offset).y;
        gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
      }}
    `,
    side: THREE.BackSide
  }});
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  // Ground
  const groundGeo = new THREE.PlaneGeometry(GROUND_W * 2, GROUND_D * 2, 50, 50);
  const groundMat = new THREE.MeshStandardMaterial({{
    color: 0x141820,
    roughness: 0.95,
    metalness: 0.0
  }});
  groundMesh = new THREE.Mesh(groundGeo, groundMat);
  groundMesh.rotation.x = -Math.PI / 2;
  groundMesh.receiveShadow = true;
  scene.add(groundMesh);

  // Grid
  const gridSize = Math.max(GROUND_W, GROUND_D) * 2;
  gridHelper = new THREE.GridHelper(gridSize, 60, 0x555555, 0x333333);
  gridHelper.position.y = 0.1;
  gridHelper.material.opacity = 0.07;
  gridHelper.material.transparent = true;
  scene.add(gridHelper);
}}

// ========== Building creation ==========
function createBuildings() {{
  buildingsData.forEach((b) => {{
    const group = new THREE.Group();
    group.userData = {{ ...b, objectType: 'building' }};

    const color = new THREE.Color(b.color);

    // === Industrial facility types ===
    if (b.building_type === 'water_tank' || b.building_type === 'cooling_tower') {{
      // Cylindrical tank
      const tankR = Math.min(b.width, b.depth) / 2;
      const tankGeo = new THREE.CylinderGeometry(tankR, tankR, b.height, 24);
      const tankMat = new THREE.MeshStandardMaterial({{
        color: color,
        roughness: b.building_type === 'cooling_tower' ? 0.7 : 0.3,
        metalness: b.building_type === 'cooling_tower' ? 0.1 : 0.6,
      }});
      const tank = new THREE.Mesh(tankGeo, tankMat);
      tank.position.y = b.height / 2;
      tank.castShadow = true;
      tank.receiveShadow = true;
      group.add(tank);
      // Top lid
      const lidGeo = new THREE.CylinderGeometry(tankR + 0.5, tankR + 0.5, 0.5, 24);
      const lidMat = new THREE.MeshStandardMaterial({{ color: 0x666666, roughness: 0.5, metalness: 0.4 }});
      const lid = new THREE.Mesh(lidGeo, lidMat);
      lid.position.y = b.height + 0.25;
      lid.castShadow = true;
      group.add(lid);
      // Support ring
      const ringGeo = new THREE.TorusGeometry(tankR + 0.3, 0.4, 8, 24);
      const ringMat = new THREE.MeshStandardMaterial({{ color: 0x555555, metalness: 0.6, roughness: 0.4 }});
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = b.height * 0.3;
      group.add(ring);
      const ring2 = ring.clone();
      ring2.position.y = b.height * 0.7;
      group.add(ring2);
    }} else if (b.building_type === 'lpg_tank') {{
      // Horizontal capsule tank
      const tankR = Math.min(b.height, b.depth) / 2;
      const tankLen = b.width;
      const bodyGeo = new THREE.CylinderGeometry(tankR, tankR, tankLen, 24);
      const tankMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.2, metalness: 0.7 }});
      const bodyMesh = new THREE.Mesh(bodyGeo, tankMat);
      bodyMesh.rotation.z = Math.PI / 2;
      bodyMesh.position.y = tankR + 2;
      bodyMesh.castShadow = true;
      group.add(bodyMesh);
      // End caps (spheres)
      const capGeo = new THREE.SphereGeometry(tankR, 16, 16);
      const cap1 = new THREE.Mesh(capGeo, tankMat);
      cap1.position.set(-tankLen/2, tankR + 2, 0);
      group.add(cap1);
      const cap2 = new THREE.Mesh(capGeo, tankMat);
      cap2.position.set(tankLen/2, tankR + 2, 0);
      group.add(cap2);
      // Support legs
      const legGeo = new THREE.BoxGeometry(1.5, tankR + 2, 1.5);
      const legMat = new THREE.MeshStandardMaterial({{ color: 0x666666, metalness: 0.5, roughness: 0.5 }});
      const leg1 = new THREE.Mesh(legGeo, legMat);
      leg1.position.set(-tankLen * 0.3, (tankR + 2) / 2, 0);
      leg1.castShadow = true;
      group.add(leg1);
      const leg2 = new THREE.Mesh(legGeo, legMat);
      leg2.position.set(tankLen * 0.3, (tankR + 2) / 2, 0);
      leg2.castShadow = true;
      group.add(leg2);
    }} else if (b.building_type === 'chimney') {{
      // Tall chimney / smokestack
      const baseR = Math.min(b.width, b.depth) / 2;
      const topR = baseR * 0.6;
      const chimneyGeo = new THREE.CylinderGeometry(topR, baseR, b.height, 16);
      const chimneyMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.8, metalness: 0.1 }});
      const chimneyMesh = new THREE.Mesh(chimneyGeo, chimneyMat);
      chimneyMesh.position.y = b.height / 2;
      chimneyMesh.castShadow = true;
      group.add(chimneyMesh);
      // Red/white warning stripes at top
      const stripeGeo = new THREE.CylinderGeometry(topR + 0.2, topR + 0.2, 3, 16);
      const stripeMat = new THREE.MeshStandardMaterial({{ color: 0xcc3333, roughness: 0.6 }});
      const stripe = new THREE.Mesh(stripeGeo, stripeMat);
      stripe.position.y = b.height - 1.5;
      group.add(stripe);
      const stripe2Geo = new THREE.CylinderGeometry(topR + 0.2, topR + 0.2, 3, 16);
      const stripe2Mat = new THREE.MeshStandardMaterial({{ color: 0xeeeeee, roughness: 0.6 }});
      const stripe2 = new THREE.Mesh(stripe2Geo, stripe2Mat);
      stripe2.position.y = b.height - 4.5;
      group.add(stripe2);
      // Top opening
      const openGeo = new THREE.CylinderGeometry(topR + 0.5, topR, 1.5, 16);
      const openMat = new THREE.MeshStandardMaterial({{ color: 0x444444, roughness: 0.5, metalness: 0.4 }});
      const openMesh = new THREE.Mesh(openGeo, openMat);
      openMesh.position.y = b.height + 0.75;
      group.add(openMesh);
    }} else {{
      // === Standard box building ===
      const geo = new THREE.BoxGeometry(b.width, b.height, b.depth);
      const mat = createGradientMaterial(b.color, b.height);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.y = b.height / 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);

      // Windows (4면 전체, 층별 유리 패널)
      if (b.floors > 0) {{
        const winSpacingX = 6;
        const winSpacingY = 6;
        const winW = 3.5;
        const winH = 2.8;
        const winColsF = Math.max(1, Math.floor((b.width - 4) / winSpacingX));
        const winColsS = Math.max(1, Math.floor((b.depth - 4) / winSpacingX));
        const winRows = Math.max(1, Math.floor((b.height - 3) / winSpacingY));

        for (let row = 0; row < winRows; row++) {{
          const wy = 3 + row * winSpacingY;
          const flicker = () => 0.15 + Math.random() * 0.6;
          const bright = () => 0.5 + Math.random() * 0.4;

          for (let col = 0; col < winColsF; col++) {{
            const wx = -b.width/2 + 3 + col * winSpacingX;
            const wGeo = new THREE.PlaneGeometry(winW, winH);
            const wMat = new THREE.MeshStandardMaterial({{
              color: 0x8ac4ed, emissive: 0x3388bb,
              emissiveIntensity: flicker(),
              transparent: true, opacity: bright(),
              side: THREE.DoubleSide
            }});
            wMat.userData = {{ isWindow: true }};
            const winF = new THREE.Mesh(wGeo, wMat);
            winF.userData.isWindow = true;
            winF.position.set(wx, wy, b.depth/2 + 0.15);
            group.add(winF);
            const wMat2 = wMat.clone();
            wMat2.userData = {{ isWindow: true }};
            wMat2.emissiveIntensity = flicker();
            wMat2.opacity = bright();
            const winB = new THREE.Mesh(wGeo, wMat2);
            winB.userData.isWindow = true;
            winB.position.set(wx, wy, -b.depth/2 - 0.15);
            group.add(winB);
          }}

          for (let col = 0; col < winColsS; col++) {{
            const wz = -b.depth/2 + 3 + col * winSpacingX;
            const wGeo = new THREE.PlaneGeometry(winW, winH);
            const wMat = new THREE.MeshStandardMaterial({{
              color: 0x8ac4ed, emissive: 0x3388bb,
              emissiveIntensity: flicker(),
              transparent: true, opacity: bright(),
              side: THREE.DoubleSide
            }});
            wMat.userData = {{ isWindow: true }};
            const winR = new THREE.Mesh(wGeo, wMat);
            winR.userData.isWindow = true;
            winR.rotation.y = Math.PI / 2;
            winR.position.set(b.width/2 + 0.15, wy, wz);
            group.add(winR);
            const wMat2 = wMat.clone();
            wMat2.userData = {{ isWindow: true }};
            wMat2.emissiveIntensity = flicker();
            wMat2.opacity = bright();
            const winL = new THREE.Mesh(wGeo, wMat2);
            winL.userData.isWindow = true;
            winL.rotation.y = Math.PI / 2;
            winL.position.set(-b.width/2 - 0.15, wy, wz);
            group.add(winL);
          }}
        }}
      }}

      // Roof
      if (b.roof_type === 'peaked') {{
        const roofGeo = new THREE.ConeGeometry(Math.max(b.width, b.depth) * 0.7, b.height * 0.3, 4);
        const roofMat = new THREE.MeshStandardMaterial({{ color: 0x884422, roughness: 0.8 }});
        const roofMesh = new THREE.Mesh(roofGeo, roofMat);
        roofMesh.position.y = b.height + b.height * 0.15;
        roofMesh.rotation.y = Math.PI / 4;
        roofMesh.castShadow = true;
        group.add(roofMesh);
      }} else if (b.roof_type === 'dome') {{
        const domeGeo = new THREE.SphereGeometry(Math.min(b.width, b.depth) * 0.4, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2);
        const domeMat = new THREE.MeshStandardMaterial({{ color: 0x99aacc, roughness: 0.3, metalness: 0.5 }});
        const domeMesh = new THREE.Mesh(domeGeo, domeMat);
        domeMesh.position.y = b.height;
        domeMesh.castShadow = true;
        group.add(domeMesh);
      }} else {{
        const eqGeo = new THREE.BoxGeometry(b.width * 0.2, 2, b.depth * 0.15);
        const eqMat = new THREE.MeshStandardMaterial({{ color: 0x777777, roughness: 0.7 }});
        const eq = new THREE.Mesh(eqGeo, eqMat);
        eq.position.y = b.height + 1;
        eq.castShadow = true;
        group.add(eq);
      }}
    }}

    // Foundation (all types)
    const baseGeo = new THREE.BoxGeometry(b.width + 4, 0.5, b.depth + 4);
    const baseMat = new THREE.MeshStandardMaterial({{ color: 0x999999, roughness: 0.9 }});
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = 0.25;
    baseMesh.receiveShadow = true;
    group.add(baseMesh);

    // 3D Name Tag (건물 이름표)
    const tagCanvas = document.createElement('canvas');
    const tagCtx = tagCanvas.getContext('2d');
    const tagName = b.name || '';
    const tagDesc = b.description || '';
    const tagFontSize = 28;
    const descFontSize = 18;
    tagCtx.font = 'bold ' + tagFontSize + 'px Noto Sans KR, sans-serif';
    const nameWidth = tagCtx.measureText(tagName).width;
    tagCtx.font = descFontSize + 'px Noto Sans KR, sans-serif';
    const descWidth = tagDesc ? tagCtx.measureText(tagDesc).width : 0;
    const tagW = Math.max(nameWidth, descWidth) + 40;
    const tagH = tagDesc ? tagFontSize + descFontSize + 30 : tagFontSize + 24;
    tagCanvas.width = tagW;
    tagCanvas.height = tagH;
    // 배경 (검은색 반투명)
    tagCtx.fillStyle = 'rgba(10, 10, 20, 0.85)';
    const tagR = 8;
    tagCtx.beginPath();
    tagCtx.moveTo(tagR, 0);
    tagCtx.lineTo(tagW - tagR, 0);
    tagCtx.quadraticCurveTo(tagW, 0, tagW, tagR);
    tagCtx.lineTo(tagW, tagH - tagR);
    tagCtx.quadraticCurveTo(tagW, tagH, tagW - tagR, tagH);
    tagCtx.lineTo(tagR, tagH);
    tagCtx.quadraticCurveTo(0, tagH, 0, tagH - tagR);
    tagCtx.lineTo(0, tagR);
    tagCtx.quadraticCurveTo(0, 0, tagR, 0);
    tagCtx.fill();
    // 건물명 (큰 흰색 텍스트)
    tagCtx.font = 'bold ' + tagFontSize + 'px Noto Sans KR, sans-serif';
    tagCtx.fillStyle = '#ffffff';
    tagCtx.textAlign = 'center';
    tagCtx.textBaseline = 'middle';
    const nameY = tagDesc ? tagFontSize / 2 + 10 : tagH / 2;
    tagCtx.fillText(tagName, tagW / 2, nameY);
    // 설명 (작은 회색 텍스트)
    if (tagDesc) {{
      tagCtx.font = descFontSize + 'px Noto Sans KR, sans-serif';
      tagCtx.fillStyle = '#aabbcc';
      tagCtx.fillText(tagDesc, tagW / 2, nameY + tagFontSize / 2 + descFontSize / 2 + 6);
    }}
    const tagTex = new THREE.CanvasTexture(tagCanvas);
    tagTex.minFilter = THREE.LinearFilter;
    const tagSpMat = new THREE.SpriteMaterial({{ map: tagTex, transparent: true, depthTest: false }});
    const tagSprite = new THREE.Sprite(tagSpMat);
    const tagBaseScale = Math.max(20, b.width * 0.35) * globalLabelScale;
    tagSprite.scale.set(tagBaseScale, tagBaseScale * (tagH / tagW), 1);
    tagSprite.position.set(0, b.height + tagBaseScale * 0.3, 0);
    tagSprite.userData = {{ baseScale: tagBaseScale / globalLabelScale, aspect: tagH / tagW, baseY: b.height }};
    group.add(tagSprite);
    allNameTagSprites.push(tagSprite);

    group.position.set(b.x, 0, b.z);
    scene.add(group);
    buildingMeshes.push(group);
    allMeshes.push(group);
  }});
}}

// ========== Road creation ==========
function createRoads() {{
  roadsData.forEach(r => {{
    const geo = new THREE.PlaneGeometry(r.length, r.width);
    const mat = new THREE.MeshStandardMaterial({{
      color: r.color || 0x555555,
      roughness: 0.95,
      metalness: 0.0
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(r.x, 0.15, r.z);
    if (r.rotation) mesh.rotation.z = (r.rotation || 0) * Math.PI / 180;
    mesh.receiveShadow = true;
    mesh.userData = {{ ...r, objectType: 'road' }};
    scene.add(mesh);

    // Center line
    const lineGeo = new THREE.PlaneGeometry(r.length - 2, 0.3);
    const lineMat = new THREE.MeshBasicMaterial({{ color: 0xffff44 }});
    const lineMesh = new THREE.Mesh(lineGeo, lineMat);
    lineMesh.rotation.x = -Math.PI / 2;
    lineMesh.position.set(r.x, 0.2, r.z);
    if (r.rotation) lineMesh.rotation.z = (r.rotation || 0) * Math.PI / 180;
    scene.add(lineMesh);
  }});
}}

// ========== Tree creation ==========
function createTrees() {{
  const treeColors = [0x1e5a18, 0x2d6a22, 0x1a5010, 0x2a6830];
  treesData.forEach((t, idx) => {{
    const group = new THREE.Group();

    // Decide tree type: 60% cone (침엽수), 40% sphere (활엽수)
    const isCone = Math.random() < 0.6;

    // Trunk
    const trunkGeo = new THREE.CylinderGeometry(0.3, 0.5, t.trunk_height, 8);
    const trunkMat = new THREE.MeshStandardMaterial({{ color: 0x8B4513, roughness: 0.9 }});
    const trunk = new THREE.Mesh(trunkGeo, trunkMat);
    trunk.position.y = t.trunk_height / 2;
    trunk.castShadow = true;
    group.add(trunk);

    // Canopy
    const canopyColor = treeColors[idx % treeColors.length];
    let canopyGeo, canopy;

    if (isCone) {{
      // Stacked cones for coniferous tree
      canopyGeo = new THREE.ConeGeometry(t.canopy_radius, t.canopy_radius * 2, 8);
      const canopyMat = new THREE.MeshStandardMaterial({{ color: canopyColor, roughness: 0.8 }});
      canopy = new THREE.Mesh(canopyGeo, canopyMat);
      canopy.position.y = t.trunk_height + t.canopy_radius;
      canopy.castShadow = true;
      group.add(canopy);
    }} else {{
      // Sphere for deciduous tree
      canopyGeo = new THREE.SphereGeometry(t.canopy_radius, 8, 8);
      const canopyMat = new THREE.MeshStandardMaterial({{ color: canopyColor, roughness: 0.8 }});
      canopy = new THREE.Mesh(canopyGeo, canopyMat);
      canopy.position.y = t.trunk_height + t.canopy_radius * 0.7;
      canopy.castShadow = true;
      group.add(canopy);
    }}

    group.position.set(t.x, 0, t.z);
    group.userData = {{ ...t, objectType: 'tree' }};
    scene.add(group);
  }});
}}

// ========== Parking lot creation ==========
function createParkings() {{
  parkingsData.forEach(p => {{
    const geo = new THREE.PlaneGeometry(p.width, p.depth);
    const mat = new THREE.MeshStandardMaterial({{
      color: p.color || 0x888888,
      roughness: 0.95
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(p.x, 0.12, p.z);
    mesh.receiveShadow = true;
    mesh.userData = {{ ...p, objectType: 'parking' }};
    scene.add(mesh);

    // Parking lines
    const lineCount = Math.floor(p.width / 3);
    for (let i = 0; i < lineCount; i++) {{
      const lGeo = new THREE.PlaneGeometry(0.15, p.depth * 0.8);
      const lMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
      const lMesh = new THREE.Mesh(lGeo, lMat);
      lMesh.rotation.x = -Math.PI / 2;
      lMesh.position.set(p.x - p.width/2 + i * 3 + 1.5, 0.14, p.z);
      scene.add(lMesh);
    }}
  }});
}}

// ========== Lake creation ==========
function createLakes() {{
  lakesData.forEach(l => {{
    const geo = new THREE.CircleGeometry(l.radius, 32);
    const mat = new THREE.MeshStandardMaterial({{
      color: l.color || 0x4499cc,
      roughness: 0.1,
      metalness: 0.3,
      transparent: true,
      opacity: 0.85
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(l.x, 0.1, l.z);
    mesh.userData = {{ ...l, objectType: 'lake' }};
    scene.add(mesh);
  }});
}}

// ========== Person creation ==========
function createPersons() {{
  personsData.forEach(p => {{
    const group = new THREE.Group();
    let speechSpr = null;

    // Head
    const headGeo = new THREE.SphereGeometry(0.35, 8, 8);
    const skinMat = new THREE.MeshStandardMaterial({{ color: 0xffcc99, roughness: 0.7 }});
    const head = new THREE.Mesh(headGeo, skinMat);
    head.position.y = 1.65;
    head.castShadow = true;
    group.add(head);

    // Body
    const bodyGeo = new THREE.BoxGeometry(0.6, 0.8, 0.35);
    const shirtMat = new THREE.MeshStandardMaterial({{ color: p.shirt_color, roughness: 0.8 }});
    const body = new THREE.Mesh(bodyGeo, shirtMat);
    body.position.y = 1.1;
    body.castShadow = true;
    group.add(body);

    // Legs
    const legGeo = new THREE.BoxGeometry(0.2, 0.7, 0.25);
    const pantsMat = new THREE.MeshStandardMaterial({{ color: p.pants_color, roughness: 0.8 }});
    const leftLeg = new THREE.Mesh(legGeo, pantsMat);
    leftLeg.position.set(-0.15, 0.35, 0);
    leftLeg.castShadow = true;
    group.add(leftLeg);
    const rightLeg = new THREE.Mesh(legGeo, pantsMat);
    rightLeg.position.set(0.15, 0.35, 0);
    rightLeg.castShadow = true;
    group.add(rightLeg);

    // Arms
    const armGeo = new THREE.BoxGeometry(0.18, 0.65, 0.2);
    const leftArm = new THREE.Mesh(armGeo, shirtMat);
    leftArm.position.set(-0.4, 1.05, 0);
    group.add(leftArm);
    const rightArm = new THREE.Mesh(armGeo, shirtMat);
    rightArm.position.set(0.4, 1.05, 0);
    group.add(rightArm);

    // 말풍선 스프라이트
    if (p.speech) {{
      const bc = document.createElement('canvas');
      const bctx = bc.getContext('2d');
      const fontSize = 28;
      bctx.font = '600 ' + fontSize + 'px Noto Sans KR, sans-serif';
      const tw = bctx.measureText(p.speech).width;
      bc.width = tw + 32;
      bc.height = fontSize + 24;
      // 말풍선 배경 (둥근 사각형)
      bctx.fillStyle = 'rgba(255,255,255,0.92)';
      const r = 10;
      bctx.beginPath();
      bctx.moveTo(r, 0);
      bctx.lineTo(bc.width - r, 0);
      bctx.quadraticCurveTo(bc.width, 0, bc.width, r);
      bctx.lineTo(bc.width, bc.height - r);
      bctx.quadraticCurveTo(bc.width, bc.height, bc.width - r, bc.height);
      bctx.lineTo(bc.width/2 + 6, bc.height);
      bctx.lineTo(bc.width/2, bc.height + 8);
      bctx.lineTo(bc.width/2 - 6, bc.height);
      bctx.lineTo(r, bc.height);
      bctx.quadraticCurveTo(0, bc.height, 0, bc.height - r);
      bctx.lineTo(0, r);
      bctx.quadraticCurveTo(0, 0, r, 0);
      bctx.fill();
      // 텍스트
      bctx.font = '600 ' + fontSize + 'px Noto Sans KR, sans-serif';
      bctx.fillStyle = '#333';
      bctx.textAlign = 'center';
      bctx.textBaseline = 'middle';
      bctx.fillText(p.speech, bc.width / 2, bc.height / 2);
      const btex = new THREE.CanvasTexture(bc);
      btex.minFilter = THREE.LinearFilter;
      const bsm = new THREE.SpriteMaterial({{ map: btex, transparent: true, depthTest: false }});
      const bsp = new THREE.Sprite(bsm);
      bsp.scale.set(3, 3 * (bc.height / bc.width), 1);
      bsp.position.y = 3.0;
      bsp.visible = false;  // 주기적으로 깜빡임
      group.add(bsp);
      speechSpr = bsp;
    }}

    group.scale.set(3.0, 3.0, 3.0);
    group.position.set(p.x, 0, p.z);
    group.userData = {{ ...p, objectType: 'person', originX: p.x, originZ: p.z, angle: (p.direction || 0) * Math.PI / 180, leftLeg, rightLeg, leftArm, rightArm, speechSprite: speechSpr }};
    scene.add(group);
    personMeshes.push(group);
  }});
}}

// ========== Gate creation ==========
function createGates() {{
  gatesData.forEach(g => {{
    const group = new THREE.Group();
    const gateMat = new THREE.MeshStandardMaterial({{ color: g.color || 0xaa8855, roughness: 0.5, metalness: 0.3 }});

    // Pillars
    const pillarGeo = new THREE.BoxGeometry(1.5, g.height, g.depth);
    const leftPillar = new THREE.Mesh(pillarGeo, gateMat);
    leftPillar.position.set(-g.width/2, g.height/2, 0);
    leftPillar.castShadow = true;
    group.add(leftPillar);
    const rightPillar = new THREE.Mesh(pillarGeo, gateMat);
    rightPillar.position.set(g.width/2, g.height/2, 0);
    rightPillar.castShadow = true;
    group.add(rightPillar);

    // Top bar
    const topGeo = new THREE.BoxGeometry(g.width + 1.5, 1.5, g.depth);
    const topBar = new THREE.Mesh(topGeo, gateMat);
    topBar.position.set(0, g.height, 0);
    topBar.castShadow = true;
    group.add(topBar);

    // Sign for main gate
    if (g.gate_type === 'main') {{
      const signGeo = new THREE.BoxGeometry(g.width * 0.6, 1.2, 0.3);
      const signMat = new THREE.MeshStandardMaterial({{ color: 0x2255aa, roughness: 0.3, metalness: 0.5 }});
      const sign = new THREE.Mesh(signGeo, signMat);
      sign.position.set(0, g.height + 1.5, 0);
      group.add(sign);
    }}

    // Barrier
    if (g.has_barrier) {{
      const barrierGeo = new THREE.BoxGeometry(g.width * 0.45, 0.15, 0.15);
      const barrierMat = new THREE.MeshStandardMaterial({{ color: 0xff4444, roughness: 0.5 }});
      const barrier = new THREE.Mesh(barrierGeo, barrierMat);
      barrier.position.set(g.width * 0.2, 3.5, 0);
      group.add(barrier);

      const bpGeo = new THREE.CylinderGeometry(0.12, 0.12, 3.5, 8);
      const bpMat = new THREE.MeshStandardMaterial({{ color: 0xcccccc }});
      const bp = new THREE.Mesh(bpGeo, bpMat);
      bp.position.set(-g.width * 0.05, 1.75, 0);
      group.add(bp);
    }}

    // 게이트 3D 네임태그
    const gateTagCanvas = document.createElement('canvas');
    const gateTagCtx = gateTagCanvas.getContext('2d');
    const gateName = g.name || (g.gate_type === 'main' ? '정문' : '후문');
    const gateFs = 26;
    gateTagCtx.font = 'bold ' + gateFs + 'px Noto Sans KR, sans-serif';
    const gateNameW = gateTagCtx.measureText(gateName).width;
    gateTagCanvas.width = gateNameW + 30;
    gateTagCanvas.height = gateFs + 16;
    // 배경
    gateTagCtx.fillStyle = g.gate_type === 'main' ? 'rgba(34, 85, 170, 0.9)' : 'rgba(50, 50, 60, 0.85)';
    gateTagCtx.beginPath();
    const gR = 6;
    gateTagCtx.moveTo(gR, 0);
    gateTagCtx.lineTo(gateTagCanvas.width - gR, 0);
    gateTagCtx.quadraticCurveTo(gateTagCanvas.width, 0, gateTagCanvas.width, gR);
    gateTagCtx.lineTo(gateTagCanvas.width, gateTagCanvas.height - gR);
    gateTagCtx.quadraticCurveTo(gateTagCanvas.width, gateTagCanvas.height, gateTagCanvas.width - gR, gateTagCanvas.height);
    gateTagCtx.lineTo(gR, gateTagCanvas.height);
    gateTagCtx.quadraticCurveTo(0, gateTagCanvas.height, 0, gateTagCanvas.height - gR);
    gateTagCtx.lineTo(0, gR);
    gateTagCtx.quadraticCurveTo(0, 0, gR, 0);
    gateTagCtx.fill();
    // 텍스트
    gateTagCtx.font = 'bold ' + gateFs + 'px Noto Sans KR, sans-serif';
    gateTagCtx.fillStyle = '#ffffff';
    gateTagCtx.textAlign = 'center';
    gateTagCtx.textBaseline = 'middle';
    gateTagCtx.fillText(gateName, gateTagCanvas.width / 2, gateTagCanvas.height / 2);
    const gateTagTex = new THREE.CanvasTexture(gateTagCanvas);
    gateTagTex.minFilter = THREE.LinearFilter;
    const gateSpMat = new THREE.SpriteMaterial({{ map: gateTagTex, transparent: true, depthTest: false }});
    const gateSprite = new THREE.Sprite(gateSpMat);
    const gateBaseScale = 18 * globalLabelScale;
    gateSprite.scale.set(gateBaseScale, gateBaseScale * (gateTagCanvas.height / gateTagCanvas.width), 1);
    gateSprite.position.set(0, g.height + 8, 0);
    gateSprite.userData = {{ baseScale: 18, aspect: gateTagCanvas.height / gateTagCanvas.width, baseY: g.height }};
    group.add(gateSprite);
    allNameTagSprites.push(gateSprite);

    group.position.set(g.x, 0, g.z);
    group.userData = {{ ...g, objectType: 'gate' }};
    scene.add(group);
  }});
}}

// ========== Infrastructure creation ==========
function createInfrastructure() {{
  // Water Tanks (물탱크)
  waterTanksData.forEach(wt => {{
    const group = new THREE.Group();
    const geo = new THREE.CylinderGeometry(wt.radius || 12, wt.radius || 12, wt.height || 28, 16);
    const mat = new THREE.MeshStandardMaterial({{ color: wt.color || 0xcccccc, roughness: 0.4, metalness: 0.6 }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = (wt.height || 28) / 2;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    // Top rim
    const rimGeo = new THREE.TorusGeometry((wt.radius || 12) + 0.5, 0.6, 8, 16);
    const rimMat = new THREE.MeshStandardMaterial({{ color: 0x999999, metalness: 0.7 }});
    const rim = new THREE.Mesh(rimGeo, rimMat);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = wt.height || 28;
    group.add(rim);
    group.position.set(wt.x, 0, wt.z);
    group.userData = {{ ...wt, objectType: 'water_tank' }};
    scene.add(group);
  }});

  // LPG Storage Tanks (LPG 저장탱크)
  lpgTanksData.forEach(lt => {{
    const group = new THREE.Group();
    const geo = new THREE.CylinderGeometry(lt.radius || 7, lt.radius || 7, lt.length || 20, 16);
    const mat = new THREE.MeshStandardMaterial({{ color: lt.color || 0xffffff, roughness: 0.3, metalness: 0.4 }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.z = Math.PI / 2;
    mesh.position.y = (lt.radius || 7) + 3;
    mesh.castShadow = true;
    group.add(mesh);
    // Hazard stripe
    const stripeGeo = new THREE.BoxGeometry((lt.length || 20) * 0.3, 0.5, (lt.radius || 7) * 2.2);
    const stripeMat = new THREE.MeshStandardMaterial({{ color: 0xff3300 }});
    const stripe = new THREE.Mesh(stripeGeo, stripeMat);
    stripe.rotation.z = Math.PI / 2;
    stripe.position.y = (lt.radius || 7) + 3;
    group.add(stripe);
    // Support legs
    for (let i = -1; i <= 1; i += 2) {{
      const legGeo = new THREE.BoxGeometry(1, (lt.radius || 7) + 3, 1);
      const legMat = new THREE.MeshStandardMaterial({{ color: 0x555555 }});
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(i * (lt.length || 20) * 0.3, ((lt.radius || 7) + 3) / 2, 0);
      group.add(leg);
    }}
    group.position.set(lt.x, 0, lt.z);
    group.userData = {{ ...lt, objectType: 'lpg_tank' }};
    scene.add(group);
  }});

  // Chimneys (굴뚝)
  chimneysData.forEach(ch => {{
    const group = new THREE.Group();
    const geo = new THREE.CylinderGeometry(ch.radius || 5, (ch.radius || 5) + 0.5, ch.height || 80, 8);
    const mat = new THREE.MeshStandardMaterial({{ color: ch.color || 0x666666, roughness: 0.7 }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = (ch.height || 80) / 2;
    mesh.castShadow = true;
    group.add(mesh);
    // Red/white stripes at top
    for (let i = 0; i < 3; i++) {{
      const stripeColor = i % 2 === 0 ? 0xff0000 : 0xffffff;
      const stripeGeo = new THREE.TorusGeometry((ch.radius || 5) + 0.5, 0.8, 8, 16);
      const stripeMat = new THREE.MeshStandardMaterial({{ color: stripeColor }});
      const stripe = new THREE.Mesh(stripeGeo, stripeMat);
      stripe.rotation.x = Math.PI / 2;
      stripe.position.y = (ch.height || 80) - 5 - i * 2;
      group.add(stripe);
    }}
    group.position.set(ch.x, 0, ch.z);
    group.userData = {{ ...ch, objectType: 'chimney' }};
    scene.add(group);

    // Smoke particles
    if (ch.has_smoke !== false) {{
      for (let s = 0; s < 6; s++) {{
        const smokeGeo = new THREE.SphereGeometry(2, 6, 6);
        const smokeMat = new THREE.MeshBasicMaterial({{
          color: 0xcccccc, transparent: true, opacity: 0.3, depthWrite: false
        }});
        const smokeMesh = new THREE.Mesh(smokeGeo, smokeMat);
        smokeMesh.position.set(ch.x + (Math.random() - 0.5) * 3, (ch.height || 80) + Math.random() * 10, ch.z + (Math.random() - 0.5) * 3);
        scene.add(smokeMesh);
        smokeParticles.push({{
          mesh: smokeMesh,
          origin: {{ x: ch.x, y: ch.height || 80, z: ch.z }},
          life: Math.random(),
          offset: Math.random() * Math.PI * 2
        }});
      }}
    }}
  }});

  // Walls (벽)
  wallsData.forEach(w => {{
    const group = new THREE.Group();
    const len = w.length || 40;
    const h = w.height || 12;
    const thick = w.thickness || 2;
    const wType = w.wall_type || 'concrete';

    if (wType === 'fence') {{
      // 철조망 펜스
      for (let i = 0; i < len; i += 4) {{
        // 기둥
        const postGeo = new THREE.CylinderGeometry(0.3, 0.3, h, 6);
        const postMat = new THREE.MeshStandardMaterial({{ color: 0x555555, metalness: 0.6 }});
        const post = new THREE.Mesh(postGeo, postMat);
        post.position.set(i - len/2, h/2, 0);
        post.castShadow = true;
        group.add(post);
      }}
      // 가로줄
      for (let j = 1; j <= 3; j++) {{
        const wireGeo = new THREE.BoxGeometry(len, 0.1, 0.1);
        const wireMat = new THREE.MeshStandardMaterial({{ color: 0x888888, metalness: 0.8 }});
        const wire = new THREE.Mesh(wireGeo, wireMat);
        wire.position.y = h * j / 4;
        group.add(wire);
      }}
    }} else {{
      // 콘크리트/벽돌 벽
      const wallColor = wType === 'brick' ? 0xaa5533 : (parseInt((w.color || '#888888').replace('#',''), 16));
      const wallGeo = new THREE.BoxGeometry(len, h, thick);
      const wallMat = new THREE.MeshStandardMaterial({{ color: wallColor, roughness: 0.9 }});
      const wallMesh = new THREE.Mesh(wallGeo, wallMat);
      wallMesh.position.y = h / 2;
      wallMesh.castShadow = true;
      wallMesh.receiveShadow = true;
      group.add(wallMesh);
      // 상단 마감
      const capGeo = new THREE.BoxGeometry(len + 0.5, 1, thick + 0.5);
      const capMat = new THREE.MeshStandardMaterial({{ color: 0x666666, roughness: 0.8 }});
      const cap = new THREE.Mesh(capGeo, capMat);
      cap.position.y = h;
      group.add(cap);
    }}
    group.position.set(w.x, 0, w.z);
    group.rotation.y = (w.rotation || 0) * Math.PI / 180;
    group.userData = {{ ...w, objectType: 'wall' }};
    scene.add(group);
  }});

  // ========== 도로 네트워크 구축 ==========
  // 각 도로의 시작점/끝점 계산 (전역 변수에 할당)
  roadSegments = roadsData.map((r, idx) => {{
    const rad = (r.rotation || 0) * Math.PI / 180;
    const halfLen = (r.length || 100) / 2;
    return {{
      idx: idx,
      cx: r.x, cz: r.z,
      sx: r.x - Math.cos(rad) * halfLen,
      sz: r.z - Math.sin(rad) * halfLen,
      ex: r.x + Math.cos(rad) * halfLen,
      ez: r.z + Math.sin(rad) * halfLen,
      rad: rad,
      length: r.length || 100,
      width: r.width || 8,
      connections: []  // 연결된 도로 인덱스
    }};
  }});

  // 도로 연결점 찾기 (끝점이 가까운 도로끼리 연결, 30유닛 이내)
  const CONNECT_DIST = 30;
  for (let i = 0; i < roadSegments.length; i++) {{
    for (let j = i + 1; j < roadSegments.length; j++) {{
      const ri = roadSegments[i], rj = roadSegments[j];
      const pairs = [
        [ri.sx, ri.sz, rj.sx, rj.sz, 'start', 'start'],
        [ri.sx, ri.sz, rj.ex, rj.ez, 'start', 'end'],
        [ri.ex, ri.ez, rj.sx, rj.sz, 'end', 'start'],
        [ri.ex, ri.ez, rj.ex, rj.ez, 'end', 'end'],
        // 도로 중심이 다른 도로 끝점 근처인 경우도
        [ri.sx, ri.sz, rj.cx, rj.cz, 'start', 'center'],
        [ri.ex, ri.ez, rj.cx, rj.cz, 'end', 'center'],
        [ri.cx, ri.cz, rj.sx, rj.sz, 'center', 'start'],
        [ri.cx, ri.cz, rj.ex, rj.ez, 'center', 'end'],
      ];
      for (const [x1, z1, x2, z2, fromEnd, toEnd] of pairs) {{
        const dist = Math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2);
        if (dist < CONNECT_DIST) {{
          ri.connections.push({{ roadIdx: j, fromEnd, toEnd, dist }});
          rj.connections.push({{ roadIdx: i, fromEnd: toEnd, toEnd: fromEnd, dist }});
          break;  // 이미 연결됨
        }}
      }}
    }}
  }}

  // 벽 충돌 박스 미리 계산
  const wallBoxes = wallsData.map(w => {{
    const rad = (w.rotation || 0) * Math.PI / 180;
    const len = w.length || 40;
    const thick = (w.thickness || 2) + 6;  // 여유 마진
    const cos = Math.cos(rad), sin = Math.sin(rad);
    return {{
      cx: w.x, cz: w.z,
      halfLen: len / 2, halfThick: thick / 2,
      cos: cos, sin: sin
    }};
  }});

  // 벽 충돌 체크 함수 (전역)
  checkWallCollision = function(px, pz) {{
    for (const wb of wallBoxes) {{
      // 로컬 좌표로 변환 (벽 중심 기준)
      const dx = px - wb.cx;
      const dz = pz - wb.cz;
      const localX = dx * wb.cos + dz * wb.sin;
      const localZ = -dx * wb.sin + dz * wb.cos;
      if (Math.abs(localX) < wb.halfLen && Math.abs(localZ) < wb.halfThick) {{
        return true;  // 충돌!
      }}
    }}
    return false;
  }};

  // 건물 충돌 박스 (전역)
  checkBuildingCollision = function(px, pz) {{
    for (const b of buildingsData) {{
      const hw = (b.width || 20) / 2 + 5;
      const hd = (b.depth || 20) / 2 + 5;
      if (Math.abs(px - b.x) < hw && Math.abs(pz - b.z) < hd) return true;
    }}
    return false;
  }};

  // Trucks (운송 트럭)
  trucksData.forEach((t, tIdx) => {{
    const group = new THREE.Group();
    const tType = t.truck_type || 'cargo';
    const truckColor = parseInt((t.color || '#ffffff').replace('#',''), 16);

    // 캐빈 (운전석)
    const cabGeo = new THREE.BoxGeometry(5, 5, 6);
    const cabMat = new THREE.MeshStandardMaterial({{ color: truckColor, roughness: 0.4, metalness: 0.3 }});
    const cab = new THREE.Mesh(cabGeo, cabMat);
    cab.position.set(6, 4.5, 0);
    cab.castShadow = true;
    group.add(cab);

    // 유리창
    const windGeo = new THREE.BoxGeometry(0.3, 2.5, 4.5);
    const windMat = new THREE.MeshStandardMaterial({{ color: 0x88bbee, transparent: true, opacity: 0.6, metalness: 0.8 }});
    const windshield = new THREE.Mesh(windGeo, windMat);
    windshield.position.set(8.6, 5, 0);
    group.add(windshield);

    // 차대 (하체)
    const chassisGeo = new THREE.BoxGeometry(18, 2, 6);
    const chassisMat = new THREE.MeshStandardMaterial({{ color: 0x333333, roughness: 0.7 }});
    const chassis = new THREE.Mesh(chassisGeo, chassisMat);
    chassis.position.set(0, 1.5, 0);
    chassis.castShadow = true;
    group.add(chassis);

    if (tType === 'tanker') {{
      const tankGeo = new THREE.CylinderGeometry(3, 3, 14, 12);
      const tankMat = new THREE.MeshStandardMaterial({{ color: 0xdddddd, metalness: 0.7, roughness: 0.2 }});
      const tank = new THREE.Mesh(tankGeo, tankMat);
      tank.rotation.z = Math.PI / 2;
      tank.position.set(-2, 5.5, 0);
      tank.castShadow = true;
      group.add(tank);
      const signGeo = new THREE.BoxGeometry(3, 3, 0.1);
      const signMat = new THREE.MeshStandardMaterial({{ color: 0xff6600 }});
      const sign = new THREE.Mesh(signGeo, signMat);
      sign.position.set(-2, 5.5, 3.1);
      group.add(sign);
    }} else if (tType === 'flatbed') {{
      const bedGeo = new THREE.BoxGeometry(14, 0.5, 6.5);
      const bedMat = new THREE.MeshStandardMaterial({{ color: 0x555555, roughness: 0.8 }});
      const bed = new THREE.Mesh(bedGeo, bedMat);
      bed.position.set(-2, 2.8, 0);
      bed.castShadow = true;
      group.add(bed);
    }} else {{
      const cargoGeo = new THREE.BoxGeometry(14, 7, 6.2);
      const cargoMat = new THREE.MeshStandardMaterial({{ color: truckColor, roughness: 0.5 }});
      const cargo = new THREE.Mesh(cargoGeo, cargoMat);
      cargo.position.set(-2, 6, 0);
      cargo.castShadow = true;
      group.add(cargo);
    }}

    // 바퀴 (6개)
    const wheelPositions = [[5, 0, 3.2], [5, 0, -3.2], [-4, 0, 3.2], [-4, 0, -3.2], [-7, 0, 3.2], [-7, 0, -3.2]];
    wheelPositions.forEach(wp => {{
      const wheelGeo = new THREE.CylinderGeometry(1.3, 1.3, 0.8, 12);
      const wheelMat = new THREE.MeshStandardMaterial({{ color: 0x222222, roughness: 0.9 }});
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(wp[0], wp[1] + 1.3, wp[2]);
      wheel.castShadow = true;
      group.add(wheel);
    }});

    group.userData = {{ ...t, objectType: 'truck', isMoving: t.speed > 0 }};
    scene.add(group);

    // 트럭을 가장 가까운 도로에 배치하고 이동 설정
    if (t.speed > 0 && roadSegments.length > 0) {{
      // 트럭에서 가장 가까운 도로 찾기
      let bestRoad = 0;
      let bestDist = Infinity;
      roadSegments.forEach((rs, ri) => {{
        const dist = Math.sqrt((t.x - rs.cx) ** 2 + (t.z - rs.cz) ** 2);
        if (dist < bestDist) {{ bestDist = dist; bestRoad = ri; }}
      }});

      const rs = roadSegments[bestRoad];
      // 트럭을 도로 위에 정확히 배치 (시작 progress는 랜덤)
      const startProgress = 0.15 + Math.random() * 0.7;  // 15%~85% 사이 랜덤 위치
      const startX = rs.sx + (rs.ex - rs.sx) * startProgress;
      const startZ = rs.sz + (rs.ez - rs.sz) * startProgress;
      group.position.set(startX, 0, startZ);
      group.rotation.y = rs.rad;

      movingTrucks.push({{
        mesh: group,
        currentRoad: bestRoad,
        progress: startProgress,
        direction: (tIdx % 2 === 0) ? 1 : -1,  // 번갈아 다른 방향
        speed: t.speed || 0.5,
        waitTime: 0  // 도로 끝 대기 시간
      }});
    }} else {{
      // 도로 없으면 그냥 제자리
      group.position.set(t.x, 0, t.z);
      group.rotation.y = (t.direction || 0) * Math.PI / 180;
    }}
  }});

  // Transport Lines (연결통로/리프터/레일)
  transportLinesData.forEach((tl, tlIdx) => {{
    const type = tl.transport_type || 'conveyor';
    const dx = tl.x2 - tl.x1;
    const dz = tl.z2 - tl.z1;
    const len = Math.sqrt(dx * dx + dz * dz);
    const angle = len > 0 ? Math.atan2(dz, dx) : 0;
    const mx = (tl.x1 + tl.x2) / 2;
    const mz = (tl.z1 + tl.z2) / 2;
    const h = tl.height || 20;

    if (type === 'conveyor') {{
      // 컨베이어: 유리 통로 + OHT 레일 + 지지 기둥
      const group = new THREE.Group();
      group.position.set(mx, h, mz);
      group.rotation.y = -angle;

      // 바닥판
      const floorGeo = new THREE.BoxGeometry(len, 0.5, 3);
      const floorMat = new THREE.MeshStandardMaterial({{ color: 0xcccccc, roughness: 0.3 }});
      const floor = new THREE.Mesh(floorGeo, floorMat);
      floor.position.y = -0.5;
      floor.castShadow = true;
      group.add(floor);

      // 천장판
      const ceilGeo = new THREE.BoxGeometry(len, 0.5, 3);
      const ceilMat = new THREE.MeshStandardMaterial({{ color: 0xaaaaaa, roughness: 0.3 }});
      const ceil = new THREE.Mesh(ceilGeo, ceilMat);
      ceil.position.y = 2.5;
      ceil.castShadow = true;
      group.add(ceil);

      // 프레임 기둥 (4개)
      const pillarPositions = [[-len/2 + 1, 0, -1.5], [-len/2 + 1, 0, 1.5], [len/2 - 1, 0, -1.5], [len/2 - 1, 0, 1.5]];
      pillarPositions.forEach(pp => {{
        const pillarGeo = new THREE.BoxGeometry(0.4, 3, 0.4);
        const pillarMat = new THREE.MeshStandardMaterial({{ color: 0x666666 }});
        const pillar = new THREE.Mesh(pillarGeo, pillarMat);
        pillar.position.set(pp[0], pp[1], pp[2]);
        pillar.castShadow = true;
        group.add(pillar);
      }});

      // OHT 레일 2줄 (위쪽)
      for (let r = 0; r < 2; r++) {{
        const railGeo = new THREE.BoxGeometry(len, 0.3, 0.3);
        const railMat = new THREE.MeshStandardMaterial({{ color: 0xffaa00, metalness: 0.8 }});
        const rail = new THREE.Mesh(railGeo, railMat);
        rail.position.set(0, 2.3 + r * 0.5, r === 0 ? -1.2 : 1.2);
        rail.castShadow = true;
        group.add(rail);
      }}

      // 하부 발광판 (파란색)
      const glowGeo = new THREE.BoxGeometry(len, 0.3, 4);
      const glowMat = new THREE.MeshStandardMaterial({{ color: 0x4488ee, emissive: 0x2244bb, emissiveIntensity: 0.5 }});
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.position.y = -0.8;
      group.add(glow);

      group.userData = {{ objectType: 'transport', ...tl }};
      scene.add(group);
    }} else if (type === 'lifter') {{
      // 리프터: 수직 엘리베이터 타워
      const group = new THREE.Group();
      group.position.set(mx, 0, mz);

      // 메인 박스 (엘리베이터 카)
      const boxGeo = new THREE.BoxGeometry(4, h, 4);
      const boxMat = new THREE.MeshStandardMaterial({{ color: 0xee4444, roughness: 0.5 }});
      const box = new THREE.Mesh(boxGeo, boxMat);
      box.position.y = h / 2;
      box.castShadow = true;
      group.add(box);

      // 도어 (파라메트릭)
      for (let d = 0; d < 2; d++) {{
        const doorGeo = new THREE.BoxGeometry(1.8, h - 2, 0.3);
        const doorMat = new THREE.MeshStandardMaterial({{ color: 0x333333, metalness: 0.7 }});
        const door = new THREE.Mesh(doorGeo, doorMat);
        door.position.set(d === 0 ? -1.2 : 1.2, h / 2, 2.2);
        group.add(door);
      }}

      // 상부 빛
      const topGeo = new THREE.BoxGeometry(4.5, 1, 4.5);
      const topMat = new THREE.MeshStandardMaterial({{ color: 0xff8800, emissive: 0xff4400, emissiveIntensity: 0.7 }});
      const top = new THREE.Mesh(topGeo, topMat);
      top.position.y = h + 0.8;
      group.add(top);

      const light = new THREE.PointLight(0xff4444, 2, 50);
      light.position.set(0, h + 2, 0);
      group.add(light);

      group.userData = {{ objectType: 'transport', ...tl }};
      scene.add(group);
    }} else if (type === 'rail') {{
      // 레일: 슬림한 I-beam 레일 트랙 + 행거
      const group = new THREE.Group();
      group.position.set(mx, h, mz);
      group.rotation.y = -angle;

      // I-beam 레일 (수평)
      const railGeo = new THREE.BoxGeometry(len, 0.4, 0.2);
      const railMat = new THREE.MeshStandardMaterial({{ color: 0xffaa22, metalness: 0.9, roughness: 0.2 }});
      const rail = new THREE.Mesh(railGeo, railMat);
      rail.position.y = 0;
      rail.castShadow = true;
      group.add(rail);

      // 행거 지지대 (일정 간격)
      const hangerCount = Math.max(2, Math.floor(len / 15));
      for (let h = 0; h < hangerCount; h++) {{
        const hx = -len / 2 + (h + 1) * (len / (hangerCount + 1));
        const hangerGeo = new THREE.BoxGeometry(0.3, 4, 0.3);
        const hangerMat = new THREE.MeshStandardMaterial({{ color: 0x888888 }});
        const hanger = new THREE.Mesh(hangerGeo, hangerMat);
        hanger.position.set(hx, -2, 0);
        hanger.castShadow = true;
        group.add(hanger);
      }}

      group.userData = {{ objectType: 'transport', ...tl }};
      scene.add(group);
    }}
  }});
}}

// ========== Camera control ==========
function updateCameraPosition() {{
  const s = camSpherical;
  camera.position.x = camTarget.x + s.radius * Math.sin(s.phi) * Math.cos(s.theta);
  camera.position.y = camTarget.y + s.radius * Math.cos(s.phi);
  camera.position.z = camTarget.z + s.radius * Math.sin(s.phi) * Math.sin(s.theta);
  camera.lookAt(camTarget);
}}

function setupControls() {{
  const el = renderer.domElement;

  el.addEventListener('mousedown', (e) => {{
    if (e.button === 0) isDragging = true;
    if (e.button === 2) isPanning = true;
    lastMouse.x = e.clientX;
    lastMouse.y = e.clientY;
  }});

  el.addEventListener('mousemove', (e) => {{
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;

    if (isDragging) {{
      targetSpherical.theta -= dx * 0.005;
      targetSpherical.phi -= dy * 0.005;
      targetSpherical.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, targetSpherical.phi));
    }}

    if (isPanning) {{
      const panFactor = targetSpherical.radius * 0.002;
      const right = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);
      camera.getWorldDirection(right);
      right.cross(up).normalize();
      const forward = new THREE.Vector3();
      forward.crossVectors(up, right).normalize();
      targetTarget.add(right.multiplyScalar(-dx * panFactor));
      targetTarget.add(forward.multiplyScalar(dy * panFactor));
    }}

    lastMouse.x = e.clientX;
    lastMouse.y = e.clientY;
  }});

  el.addEventListener('mouseup', () => {{
    isDragging = false;
    isPanning = false;
  }});

  el.addEventListener('mouseleave', () => {{
    isDragging = false;
    isPanning = false;
  }});

  el.addEventListener('wheel', (e) => {{
    e.preventDefault();
    targetSpherical.radius *= (1 + e.deltaY * 0.001);
    targetSpherical.radius = Math.max(20, Math.min(GROUND_SIZE * 0.8, targetSpherical.radius));
  }}, {{ passive: false }});

  el.addEventListener('contextmenu', (e) => e.preventDefault());

  // Click to select building
  el.addEventListener('click', (e) => {{
    if (Math.abs(e.clientX - lastMouse.x) > 3 || Math.abs(e.clientY - lastMouse.y) > 3) return;
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(scene.children, true);
    let found = null;
    for (const hit of intersects) {{
      let obj = hit.object;
      while (obj.parent && obj.parent !== scene) obj = obj.parent;
      if (obj.userData && obj.userData.objectType === 'building') {{
        found = obj;
        break;
      }}
    }}

    if (found) selectBuilding(found);
    else deselectBuilding();
  }});

  // Keyboard
  document.addEventListener('keydown', (e) => {{
    const panAmount = 10;
    switch(e.key) {{
      case 'ArrowUp':
      case 'w': targetTarget.z -= panAmount; break;
      case 'ArrowDown':
      case 's': targetTarget.z += panAmount; break;
      case 'ArrowLeft':
      case 'a': targetTarget.x -= panAmount; break;
      case 'ArrowRight':
      case 'd': targetTarget.x += panAmount; break;
      case '+':
      case '=': targetSpherical.radius = Math.max(20, targetSpherical.radius * 0.9); break;
      case '-': targetSpherical.radius = Math.min(GROUND_SIZE * 0.8, targetSpherical.radius * 1.1); break;
      case 'Escape': deselectBuilding(); break;
    }}
  }});

  // Resize
  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});
}}

// ========== Camera animation ==========
function flyTo(x, z, dist) {{
  targetTarget.set(x, 0, z);
  targetSpherical.radius = dist || 100;
}}

function resetCamera() {{
  targetTarget.set(0, 0, 0);
  targetSpherical = {{ radius: 300, theta: Math.PI / 4, phi: Math.PI / 3 }};
}}

// ========== Selection ==========
function selectBuilding(mesh) {{
  deselectBuilding();
  selectedMesh = mesh;

  mesh.traverse((child) => {{
    if (child.isMesh && child.material) {{
      child.userData.origEmissive = child.material.emissive ? child.material.emissive.clone() : null;
      if (child.material.emissive) {{
        child.material.emissive.set(0x334466);
      }}
    }}
  }});

  const d = mesh.userData;
  const typeNames = {{
    'office': '오피스', 'factory': '공장', 'lab': '연구소',
    'parking': '주차 건물', 'datacenter': '데이터센터',
    'cafeteria': '카페테리아', 'warehouse': '창고', 'gym': '체육관',
    'water_tank': '물탱크', 'lpg_tank': 'LPG저장탱크', 'chimney': '굴뚝', 'cooling_tower': '냉각탑'
  }};

  document.getElementById('bBadge').style.backgroundColor = d.color;
  document.getElementById('bName').textContent = d.name;
  document.getElementById('bType').textContent = (typeNames[d.building_type] || d.building_type).toUpperCase();

  let panelHTML = '';
  if (d.description) panelHTML += `<div class="building-detail">${{d.description}}</div>`;

  panelHTML += `<div class="spec-grid">
    <div class="spec-item"><div class="spec-label">FLOORS</div><div class="spec-value">${{d.floors}}F</div></div>
    <div class="spec-item"><div class="spec-label">SIZE</div><div class="spec-value">${{d.width}}m</div></div>
    <div class="spec-item"><div class="spec-label">DEPTH</div><div class="spec-value">${{d.depth}}m</div></div>
    <div class="spec-item"><div class="spec-label">HEIGHT</div><div class="spec-value">${{d.height}}m</div></div>
  </div>`;

  document.getElementById('panelBody').innerHTML = panelHTML;
  document.getElementById('infoPanel').classList.add('visible');
}}

function deselectBuilding() {{
  if (selectedMesh) {{
    selectedMesh.traverse((child) => {{
      if (child.isMesh && child.material && child.userData.origEmissive) {{
        child.material.emissive.copy(child.userData.origEmissive);
      }}
    }});
    selectedMesh = null;
  }}
  document.getElementById('infoPanel').classList.remove('visible');
}}

// ========== Time system ==========
function setTimeMode(mode) {{
  timeMode = mode;

  // Update button states
  document.getElementById('btnMorning').classList.toggle('active', mode === 'morning');
  document.getElementById('btnAuto').classList.toggle('active', mode === 'auto');
  document.getElementById('btnNight').classList.toggle('active', mode === 'night');

  updateSky();
}}

function toggleTimePanel() {{
  const body = document.getElementById('timeToggleBody');
  const brow = document.querySelector('.brightness-row');
  const btn = document.getElementById('btnCollapse');
  if (body.style.display === 'none' || !body.style.display) {{
    body.style.display = 'flex';
    if (brow) brow.style.display = 'flex';
    btn.textContent = '▲';
  }} else {{
    body.style.display = 'none';
    if (brow) brow.style.display = 'none';
    btn.textContent = '▼';
  }}
}}

function setBrightness(val) {{
  brightnessMultiplier = val / 100;
  document.getElementById('brightnessVal').textContent = val + '%';
  updateSky();
}}

function setPersonSpeed(val) {{
  globalPersonSpeed = val / 100;
  document.getElementById('personSpeedVal').textContent = val + '%';
}}

function updateSky() {{
  // 시간 결정
  let hours;
  if (timeMode === 'morning') {{
    hours = 10; // 오전 10시 (밝은 낮)
  }} else if (timeMode === 'night') {{
    hours = 0;  // 자정 (한밤중)
  }} else {{
    const now = new Date();
    hours = now.getHours() + now.getMinutes() / 60;
  }}

  // ====== 해 위치 계산 ======
  // 일출 6시, 일몰 18시 기준
  const sunAngle = ((hours - 6) / 12) * Math.PI;
  const sunUp = (hours >= 5.5 && hours <= 18.5);

  if (sunUp && sunMesh) {{
    const sx = skyRadius * Math.cos(sunAngle);
    const sy = skyRadius * Math.sin(sunAngle) * 0.7;
    const sz = -200;
    sunMesh.position.set(sx, Math.max(sy, -20), sz);
    sunMesh.visible = true;
    if (sunGlowMat) sunGlowMat.opacity = 0.2 + 0.15 * Math.sin(sunAngle);
  }} else if (sunMesh) {{
    sunMesh.visible = false;
  }}

  // ====== 달 위치 계산 ======
  // 18시=동쪽, 0시=최고점, 6시=서쪽
  const moonAngle = ((hours - 18 + 24) % 24 / 12) * Math.PI;
  const moonUp = (hours >= 17.5 || hours <= 6.5);

  if (moonUp && moonMesh) {{
    const mx = -skyRadius * Math.cos(moonAngle);
    const my = skyRadius * Math.sin(moonAngle) * 0.6;
    const mz = 150;
    moonMesh.position.set(mx, Math.max(my, -20), mz);
    moonMesh.visible = true;
    // 달 높이에 따라 밝기 조절
    const moonAlt = Math.max(0, my) / (skyRadius * 0.6);
    if (moonLight) moonLight.intensity = 2.5 * moonAlt * brightnessMultiplier;
    if (moonGlowMat) moonGlowMat.opacity = 0.2 + 0.2 * moonAlt;
    if (moonGlow2Mat) moonGlow2Mat.opacity = 0.08 + 0.12 * moonAlt;
  }} else if (moonMesh) {{
    moonMesh.visible = false;
    if (moonLight) moonLight.intensity = 0;
  }}

  // ====== 하늘색 & 조명 계산 ======
  let skyProgress; // 0=밤, 1=낮
  if (hours >= 6 && hours <= 18) {{
    skyProgress = 1 - Math.abs(hours - 12) / 6;
  }} else {{
    skyProgress = 0;
  }}
  // 새벽/석양 전환
  if (hours >= 5 && hours < 6) skyProgress = (hours - 5) * 0.3;
  if (hours > 18 && hours <= 19) skyProgress = (19 - hours) * 0.3;

  // 배경색: 밤 ↔ 낮
  const nightR = 0x10/255, nightG = 0x12/255, nightB = 0x22/255;
  const dayR = 0x55/255, dayG = 0x99/255, dayB = 0xdd/255;
  const dawnR = 0x88/255, dawnG = 0x44/255, dawnB = 0x33/255;
  let r, g, b;
  if (skyProgress > 0.3) {{
    const t = (skyProgress - 0.3) / 0.7;
    r = dayR * t + nightR * (1-t);
    g = dayG * t + nightG * (1-t);
    b = dayB * t + nightB * (1-t);
  }} else if (skyProgress > 0) {{
    const t = skyProgress / 0.3;
    r = dawnR * t + nightR * (1-t);
    g = dawnG * t + nightG * (1-t);
    b = dawnB * t + nightB * (1-t);
  }} else {{
    r = nightR; g = nightG; b = nightB;
  }}
  scene.background.setRGB(r, g, b);

  // 안개
  scene.fog = new THREE.FogExp2(new THREE.Color(r, g, b), 0.4 / GROUND_SIZE);

  // 조명 강도
  const dayAmbient = 1.5, nightAmbient = 0.8;
  const dayDir = 1.8, nightDir = 0.5;
  ambientLight.intensity = (nightAmbient + (dayAmbient - nightAmbient) * skyProgress) * brightnessMultiplier;
  dirLight.intensity = (nightDir + (dayDir - nightDir) * skyProgress) * brightnessMultiplier;
  hemiLight.intensity = (0.3 + 0.7 * skyProgress) * brightnessMultiplier;

  // 해 위치에 맞춰 directional light 이동
  if (sunUp && sunMesh) {{
    dirLight.position.copy(sunMesh.position);
    dirLight.color.setHex(skyProgress > 0.3 ? 0xffeedd : 0xff8844);
  }} else {{
    dirLight.position.set(100, 300, 100);
    dirLight.color.setHex(0x334466);
  }}

  // 톤매핑 밝기
  renderer.toneMappingExposure = (1.2 + skyProgress * 0.6) * brightnessMultiplier;

  // 낮/밤 창문 전환
  const isNight = skyProgress < 0.15;
  buildingMeshes.forEach(bm => {{
    bm.traverse(child => {{
      if (child.material && child.material.userData && child.material.userData.isWindow) {{
        if (isNight) {{
          child.material.color.set(0xffdd88);
          child.material.emissive.set(0xddaa33);
          child.material.emissiveIntensity = 0.6 + Math.random() * 0.35;
          child.material.opacity = 0.8 + Math.random() * 0.2;
        }} else {{
          child.material.color.set(0x88ccff);
          child.material.emissive.set(0x224466);
          child.material.emissiveIntensity = 0.03;
          child.material.opacity = 0.35;
        }}
      }}
    }});
  }});
}}

// ========== UI setup ==========
function setupUI() {{
  buildLegend();
}}

function buildLegend() {{
  const legendItems = document.getElementById('legendItems');
  legendItems.innerHTML = '';

  // 건물별 범례 항목 생성
  const seen = new Set();
  buildingsData.forEach(b => {{
    const key = b.name || b.building_type;
    if (seen.has(key)) return;
    seen.add(key);
    const hexColor = '#' + new THREE.Color(b.color).getHexString();
    createLegendEntry(legendItems, hexColor, key, b.description || '');
  }});

  // 도로
  if (roadsData.length > 0) {{
    createLegendEntry(legendItems, '#555555', '도로', '');
  }}

  // 산업시설
  if (waterTanksData.length > 0) createLegendEntry(legendItems, '#cccccc', '물탱크', '');
  if (lpgTanksData.length > 0) createLegendEntry(legendItems, '#ffffff', 'LPG 저장탱크', '');
  if (chimneysData.length > 0) createLegendEntry(legendItems, '#666666', '굴뚝', '');
  if (typeof wallsData !== 'undefined' && wallsData.length > 0) createLegendEntry(legendItems, '#888888', '벽', '');
  if (typeof trucksData !== 'undefined' && trucksData.length > 0) createLegendEntry(legendItems, '#ffffff', '운송트럭', '');
}}

function createLegendEntry(container, color, name, desc) {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.style.cursor = 'pointer';

  const swatch = document.createElement('div');
  swatch.className = 'swatch';
  swatch.style.background = color;
  swatch.title = '클릭하여 색상 변경';
  swatch.addEventListener('click', (e) => {{
    e.stopPropagation();
    const input = document.createElement('input');
    input.type = 'color';
    input.value = color;
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.addEventListener('input', () => {{
      swatch.style.background = input.value;
    }});
    input.addEventListener('change', () => {{
      document.body.removeChild(input);
    }});
    input.click();
  }});

  const label = document.createElement('span');
  label.textContent = desc ? name + ' (' + desc + ')' : name;
  label.title = '더블클릭하여 편집';
  label.addEventListener('dblclick', () => {{
    const newName = prompt('범례 이름 수정:', label.textContent);
    if (newName !== null) label.textContent = newName;
  }});

  item.appendChild(swatch);
  item.appendChild(label);
  container.appendChild(item);
}}

function addLegendItem() {{
  const name = prompt('범례 항목 이름:');
  if (!name) return;
  const container = document.getElementById('legendItems');
  createLegendEntry(container, '#44aaff', name, '');
}}

function updateStats() {{
  // Update stats as needed
}}

// ========== Minimap ==========
function updateMinimap() {{
  const canvas = document.getElementById('minimapCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = 160;
  const h = canvas.height = 140;
  const scale = Math.min(w / (GROUND_W * 2), h / (GROUND_D * 2));
  const cx = w / 2;
  const cy = h / 2;

  ctx.fillStyle = '#1a2a1a';
  ctx.fillRect(0, 0, w, h);

  // Buildings
  buildingsData.forEach(b => {{
    const hexColor = new THREE.Color(b.color).getHexString();
    ctx.fillStyle = '#' + hexColor;
    ctx.fillRect(
      cx + b.x * scale - b.width * scale / 2,
      cy + b.z * scale - b.depth * scale / 2,
      b.width * scale,
      b.depth * scale
    );
  }});

  // Roads
  roadsData.forEach(r => {{
    ctx.fillStyle = '#555';
    ctx.fillRect(
      cx + r.x * scale - r.length * scale / 2,
      cy + r.z * scale - r.width * scale / 2,
      r.length * scale,
      r.width * scale
    );
  }});

  // Camera position
  ctx.fillStyle = '#ff4444';
  ctx.beginPath();
  ctx.arc(cx + camera.position.x * scale, cy + camera.position.z * scale, 3, 0, Math.PI * 2);
  ctx.fill();
}}

// Export functions moved to Python GUI

// ========== Animation loop ==========
function animate() {{
  requestAnimationFrame(animate);

  // Camera smoothing
  camSpherical.radius += (targetSpherical.radius - camSpherical.radius) * damping;
  camSpherical.theta += (targetSpherical.theta - camSpherical.theta) * damping;
  camSpherical.phi += (targetSpherical.phi - camSpherical.phi) * damping;
  camTarget.lerp(targetTarget, damping);

  // Clamp camera target within ground bounds
  const maxPanX = GROUND_W * 0.8;
  const maxPanZ = GROUND_D * 0.8;
  targetTarget.x = Math.max(-maxPanX, Math.min(maxPanX, targetTarget.x));
  targetTarget.z = Math.max(-maxPanZ, Math.min(maxPanZ, targetTarget.z));

  // Auto rotate
  if (autoRotate) {{
    targetSpherical.theta += 0.0003;
  }}

  updateCameraPosition();

  // FPS calculation
  frameCount++;
  const now = performance.now();
  if (now - lastTime >= 1000) {{
    frameCount = 0;
    lastTime = now;
  }}

  // Update labels and minimap every 2 frames
  if (frameCount % 2 === 0) {{
    updateMinimap();
  }}

  // Person walking animation + speech bubbles + 건물 충돌 방지
  const time = performance.now() * 0.001;
  personMeshes.forEach((pm, idx) => {{
    const d = pm.userData;
    const baseSpeed = (d.speed || 0.5) * globalPersonSpeed;
    const radius = d.walk_radius || 30;
    const angle = d.angle + time * baseSpeed * 0.05;

    // 새 위치 계산
    let newX = d.originX + Math.cos(angle) * radius;
    let newZ = d.originZ + Math.sin(angle) * radius;

    // 건물 충돌 감지
    let collided = false;
    buildingsData.forEach(b => {{
      const margin = 5;
      const hw = (b.width || 30) / 2 + margin;
      const hd = (b.depth || 30) / 2 + margin;
      if (newX > b.x - hw && newX < b.x + hw && newZ > b.z - hd && newZ < b.z + hd) {{
        collided = true;
      }}
    }});

    if (!collided) {{
      pm.position.x = newX;
      pm.position.z = newZ;
    }} else {{
      // 충돌 시 반대 방향으로 살짝 이동
      d.angle += 0.5;
      const escAngle = d.angle + time * baseSpeed * 0.05;
      pm.position.x = d.originX + Math.cos(escAngle) * radius * 0.8;
      pm.position.z = d.originZ + Math.sin(escAngle) * radius * 0.8;
    }}

    pm.rotation.y = -angle + Math.PI / 2;
    const walkAnim = time * baseSpeed * 1.5;
    if (d.leftLeg) d.leftLeg.rotation.x = Math.sin(walkAnim) * 0.4;
    if (d.rightLeg) d.rightLeg.rotation.x = -Math.sin(walkAnim) * 0.4;
    if (d.leftArm) d.leftArm.rotation.x = -Math.sin(walkAnim) * 0.3;
    if (d.rightArm) d.rightArm.rotation.x = Math.sin(walkAnim) * 0.3;
    // 말풍선: 각 사람마다 시간차를 두고 주기적으로 표시 (8초 주기, 3초간 보임)
    if (d.speechSprite) {{
      const cycle = (time + idx * 2.7) % 8;
      d.speechSprite.visible = (cycle < 3);
    }}
  }});

  // Smoke particles update
  const smokeTime = performance.now() * 0.001;
  smokeParticles.forEach(sp => {{
    sp.life += 0.008;
    if (sp.life > 1) {{
      sp.mesh.position.set(sp.origin.x + (Math.random() - 0.5) * 3, sp.origin.y, sp.origin.z + (Math.random() - 0.5) * 3);
      sp.life = 0;
      sp.mesh.scale.set(1, 1, 1);
      sp.mesh.material.opacity = 0.3;
    }}
    sp.mesh.position.y += 0.3;
    sp.mesh.position.x += Math.sin(smokeTime + sp.offset) * 0.1;
    const sc = 1 + sp.life * 3;
    sp.mesh.scale.set(sc, sc, sc);
    sp.mesh.material.opacity = 0.3 * (1 - sp.life);
  }});

  // Moving trucks animation - 도로 네트워크 따라 이동
  movingTrucks.forEach(tr => {{
    // 대기 중이면 카운트다운
    if (tr.waitTime > 0) {{
      tr.waitTime -= 1;
      return;
    }}

    const rs = roadSegments[tr.currentRoad];
    if (!rs) return;

    // 도로 길이에 비례한 이동 속도 (긴 도로에서는 같은 속도로 보이도록)
    const moveStep = tr.speed * 0.002 * (100 / Math.max(rs.length, 30));
    tr.progress += moveStep * tr.direction;

    // 도로 끝 도달 체크
    if (tr.progress >= 1.0 || tr.progress <= 0.0) {{
      tr.progress = Math.max(0, Math.min(1, tr.progress));

      // 연결된 도로 찾기
      const atEnd = tr.direction > 0 ? 'end' : 'start';
      const availConns = rs.connections.filter(c => c.fromEnd === atEnd || c.fromEnd === 'center');

      if (availConns.length > 0 && Math.random() < 0.7) {{
        // 70% 확률로 연결된 도로로 이동
        const conn = availConns[Math.floor(Math.random() * availConns.length)];
        const nextRoad = roadSegments[conn.roadIdx];
        if (nextRoad) {{
          tr.currentRoad = conn.roadIdx;
          // 진입 방향 결정
          if (conn.toEnd === 'start' || conn.toEnd === 'center') {{
            tr.progress = 0.05;
            tr.direction = 1;
          }} else {{
            tr.progress = 0.95;
            tr.direction = -1;
          }}
          tr.waitTime = Math.floor(Math.random() * 20);  // 짧은 대기
        }}
      }} else {{
        // 연결 도로 없거나 30% 확률 → 방향 전환 (U턴)
        tr.direction *= -1;
        tr.waitTime = 30 + Math.floor(Math.random() * 40);  // 잠시 대기 후 U턴
      }}
    }}

    // 현재 도로 위 위치 계산
    const road = roadSegments[tr.currentRoad];
    if (!road) return;
    const nx = road.sx + (road.ex - road.sx) * tr.progress;
    const nz = road.sz + (road.ez - road.sz) * tr.progress;

    // 벽/건물 충돌 체크
    if (checkWallCollision(nx, nz) || checkBuildingCollision(nx, nz)) {{
      // 충돌 시 방향 전환
      tr.direction *= -1;
      tr.progress += moveStep * tr.direction * 3;  // 뒤로 밀어내기
      tr.waitTime = 20;
      return;
    }}

    tr.mesh.position.set(nx, 0, nz);
    // 이동 방향에 따라 트럭 회전 (부드러운 회전)
    const targetAngle = road.rad + (tr.direction > 0 ? 0 : Math.PI);
    const angleDiff = targetAngle - tr.mesh.rotation.y;
    tr.mesh.rotation.y += angleDiff * 0.1;  // 부드러운 회전
  }});

  renderer.render(scene, camera);
}}

// ========== Startup ==========
window.addEventListener('load', init);
</script>

</body>
</html>'''
    return html



# ============================================================
# GUI 애플리케이션
# ============================================================

class CampusBuilderApp:
    """3D 캠퍼스 빌더 메인 애플리케이션"""

    TOOL_SELECT = "select"
    TOOL_BUILDING = "building"
    TOOL_ROAD = "road"
    TOOL_TREE = "tree"
    TOOL_PARKING = "parking"
    TOOL_LAKE = "lake"
    TOOL_PERSON = "person"
    TOOL_GATE = "gate"
    TOOL_WATER_TANK = "water_tank"
    TOOL_LPG_TANK = "lpg_tank"
    TOOL_CHIMNEY = "chimney"
    TOOL_WALL = "wall"
    TOOL_TRUCK = "truck"
    TOOL_TRANSPORT = "transport"

    BUILDING_TYPES = [
        ("office", "오피스"),
        ("factory", "공장/FAB"),
        ("lab", "연구소"),
        ("datacenter", "데이터센터"),
        ("cafeteria", "카페테리아"),
        ("warehouse", "창고"),
        ("gym", "체육관"),
        ("parking", "주차 건물"),
        ("water_tank", "물탱크"),
        ("lpg_tank", "LPG저장탱크"),
        ("chimney", "굴뚝"),
        ("cooling_tower", "냉각탑"),
    ]

    ROOF_TYPES = [
        ("flat", "평지붕"),
        ("peaked", "뾰족지붕"),
        ("dome", "돔"),
    ]

    GATE_TYPES = [
        ("main", "정문"),
        ("side", "후문"),
    ]

    PRESET_COLORS = [
        "#4488cc", "#cc4444", "#44aa44", "#cc8844",
        "#8844cc", "#44cccc", "#cc44aa", "#888888",
        "#2266aa", "#aa2222", "#22aa66", "#aa8822",
        "#6622aa", "#22aaaa", "#aa2288", "#555555",
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("3D Campus Builder v1.0 — SK Hynix Style")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 600)

        # 프로젝트 데이터
        self.project = CampusProject()
        self.buildings: List[Building] = []
        self.roads: List[Road] = []
        self.trees: List[Tree] = []
        self.parking_lots: List[ParkingLot] = []
        self.lakes: List[Lake] = []
        self.persons: List[Person] = []
        self.gates: List[Gate] = []
        self.water_tanks: List[WaterTank] = []
        self.lpg_tanks: List[LPGTank] = []
        self.chimneys: List[Chimney] = []
        self.walls: List[Wall] = []
        self.trucks: List[Truck] = []
        self.transport_lines: List[TransportLine] = []
        self.current_file = None

        # 캔버스 상태
        self.current_tool = self.TOOL_SELECT
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.canvas_scale = 1.0
        self.drag_start = None
        self.selected_item = None
        self.selected_type = None
        self.moving_item = False
        self.move_start = None
        self.resize_handle = None  # 'n','s','e','w','ne','nw','se','sw'
        self.resize_start = None
        self.resize_orig = None
        # 다중 선택
        self.selected_items = []        # [(item, type), ...]
        self.selection_rect_start = None # 선택 사각형 시작 화면좌표
        self.is_box_selecting = False    # 사각형 선택 중?
        # 연결통로 도구
        self.transport_start = None      # (wx, wz) - 연결통로 시작점

        # UI 구축 (의존성 순서: status_bar → toolbar → main_area → example_campus)
        self._build_menu()
        self._build_status_bar()    # status_label, count_label 생성 (toolbar에서 필요)
        self._build_toolbar()       # set_tool → update_status → status_label 사용
        self._build_main_area()     # object_tree, canvas 등 생성

        # 기본 예제 캠퍼스 로드 (UI 구축 후에 호출해야 object_tree, count_label 등 사용 가능)
        self._load_example_campus()

        # 초기 그리기
        self.root.after(100, self.redraw_canvas)

    # ========================
    # 메뉴
    # ========================
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="새 프로젝트  Ctrl+N", command=self.new_project)
        file_menu.add_separator()
        file_menu.add_command(label="열기  Ctrl+O", command=self.open_project)
        file_menu.add_command(label="저장  Ctrl+S", command=self.save_project)
        file_menu.add_command(label="다른 이름으로 저장...", command=self.save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="HTML 내보내기  Ctrl+E", command=self.export_html)
        file_menu.add_command(label="HTML 내보내기 & 열기  Ctrl+Shift+E", command=self.export_and_open)
        file_menu.add_separator()
        file_menu.add_command(label="React (.jsx) 내보내기", command=self.export_react)
        file_menu.add_command(label="Blender OBJ (.obj) 내보내기", command=self.export_obj)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)
        menubar.add_cascade(label="파일", menu=file_menu)

        # 편집 메뉴
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="선택 삭제  Delete", command=self.delete_selected)
        edit_menu.add_command(label="선택 복제  Ctrl+D", command=self.duplicate_selected)
        edit_menu.add_separator()
        edit_menu.add_command(label="전체 선택 해제  Esc", command=self.deselect_all)
        menubar.add_cascade(label="편집", menu=edit_menu)

        # 추가 메뉴
        add_menu = tk.Menu(menubar, tearoff=0)
        add_menu.add_command(label="건물 추가", command=lambda: self.add_object("building"))
        add_menu.add_command(label="도로 추가", command=lambda: self.add_object("road"))
        add_menu.add_command(label="나무 추가", command=lambda: self.add_object("tree"))
        add_menu.add_command(label="주차장 추가", command=lambda: self.add_object("parking"))
        add_menu.add_command(label="호수 추가", command=lambda: self.add_object("lake"))
        add_menu.add_separator()
        add_menu.add_command(label="나무 10개 랜덤 추가", command=self.add_random_trees)
        menubar.add_cascade(label="추가", menu=add_menu)

        # 뷰 메뉴
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="캔버스 리셋  Home", command=self.reset_canvas_view)
        view_menu.add_command(label="확대  +", command=lambda: self.zoom_canvas(1.2))
        view_menu.add_command(label="축소  -", command=lambda: self.zoom_canvas(0.8))
        menubar.add_cascade(label="보기", menu=view_menu)

        # 도움말
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="사용법", command=self.show_help)
        help_menu.add_command(label="정보", command=self.show_about)
        menubar.add_cascade(label="도움말", menu=help_menu)

        self.root.config(menu=menubar)

        # 단축키
        self.root.bind('<Control-n>', lambda e: self.new_project())
        self.root.bind('<Control-o>', lambda e: self.open_project())
        self.root.bind('<Control-s>', lambda e: self.save_project())
        self.root.bind('<Control-e>', lambda e: self.export_html())
        self.root.bind('<Control-E>', lambda e: self.export_and_open())
        self.root.bind('<Control-d>', lambda e: self.duplicate_selected())
        self.root.bind('<Delete>', lambda e: self.delete_selected())
        self.root.bind('<Escape>', lambda e: self.deselect_all())
        self.root.bind('<Home>', lambda e: self.reset_canvas_view())
        self.root.bind('<plus>', lambda e: self.zoom_canvas(1.2))
        self.root.bind('<minus>', lambda e: self.zoom_canvas(0.8))

    # ========================
    # 툴바
    # ========================
    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, padding=3)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        style = ttk.Style()
        style.configure("Tool.TButton", padding=(10, 4))
        style.configure("ActiveTool.TButton", padding=(10, 4), background="#4488cc")

        self.tool_buttons = {}

        tools = [
            (self.TOOL_SELECT, "🖱️ 선택"),
            (self.TOOL_BUILDING, "🏢 건물"),
            (self.TOOL_ROAD, "🛣️ 도로"),
            (self.TOOL_TREE, "🌳 나무"),
            (self.TOOL_PARKING, "🅿️ 주차장"),
            (self.TOOL_LAKE, "💧 호수"),
            (self.TOOL_PERSON, "🚶 사람"),
            (self.TOOL_GATE, "🚪 정문"),
            (self.TOOL_WATER_TANK, "🏗 물탱크"),
            (self.TOOL_LPG_TANK, "⛽ LPG탱크"),
            (self.TOOL_CHIMNEY, "🏭 굴뚝"),
            (self.TOOL_WALL, "🧱 벽"),
            (self.TOOL_TRUCK, "🚛 트럭"),
            (self.TOOL_TRANSPORT, "🔗 연결통로"),
        ]

        for tool_id, label in tools:
            btn = ttk.Button(toolbar, text=label, style="Tool.TButton",
                             command=lambda t=tool_id: self.set_tool(t))
            btn.pack(side=tk.LEFT, padx=2)
            self.tool_buttons[tool_id] = btn

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # 프로젝트 이름
        ttk.Label(toolbar, text="프로젝트:").pack(side=tk.LEFT, padx=(5, 2))
        self.name_var = tk.StringVar(value=self.project.name)
        self.name_entry = ttk.Entry(toolbar, textvariable=self.name_var, width=25)
        self.name_entry.pack(side=tk.LEFT, padx=2)
        self.name_var.trace_add('write', lambda *a: setattr(self.project, 'name', self.name_var.get()))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # 빠른 작업
        ttk.Button(toolbar, text="📤 HTML 내보내기", command=self.export_and_open).pack(side=tk.RIGHT, padx=4)
        ttk.Button(toolbar, text="💾 저장", command=self.save_project).pack(side=tk.RIGHT, padx=4)

        self.set_tool(self.TOOL_SELECT)

    def set_tool(self, tool):
        self.current_tool = tool
        for tid, btn in self.tool_buttons.items():
            if tid == tool:
                btn.configure(style="ActiveTool.TButton")
            else:
                btn.configure(style="Tool.TButton")
        self.update_status(f"도구: {tool}")

    # ========================
    # 메인 영역 (캔버스 + 패널)
    # ========================
    def _build_main_area(self):
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 좌측: 오브젝트 리스트
        left_frame = ttk.LabelFrame(main_frame, text="오브젝트 목록", padding=5)
        main_frame.add(left_frame, weight=1)

        self.object_tree = ttk.Treeview(left_frame, columns=("type",), show="tree headings", height=30)
        self.object_tree.heading("#0", text="이름")
        self.object_tree.heading("type", text="유형")
        self.object_tree.column("#0", width=140)
        self.object_tree.column("type", width=70)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.object_tree.yview)
        self.object_tree.configure(yscrollcommand=scrollbar.set)
        self.object_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.object_tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        # 중앙: 캔버스
        center_frame = ttk.Frame(main_frame)
        main_frame.add(center_frame, weight=4)

        self.canvas = tk.Canvas(center_frame, bg="#2a3a2a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
        self.canvas.bind('<Button-3>', self._on_canvas_right_click)
        self.canvas.bind('<B3-Motion>', self._on_canvas_pan)
        self.canvas.bind('<ButtonRelease-3>', self._on_canvas_pan_end)
        self.canvas.bind('<MouseWheel>', self._on_canvas_scroll)
        self.canvas.bind('<Button-4>', lambda e: self.zoom_canvas(1.1))
        self.canvas.bind('<Button-5>', lambda e: self.zoom_canvas(0.9))
        self.canvas.bind('<Configure>', lambda e: self.redraw_canvas())

        # 드래그 도움말
        self.canvas.bind('<Enter>', lambda e: self.update_status("💡 선택 도구: 클릭=선택, 드래그=이동, 우하단 🗑=삭제, Ctrl+클릭=다중선택, Delete=삭제"))

        # 우측: 속성 패널
        right_frame = ttk.LabelFrame(main_frame, text="속성", padding=5)
        main_frame.add(right_frame, weight=2)

        self.props_frame = ttk.Frame(right_frame)
        self.props_frame.pack(fill=tk.BOTH, expand=True)

        self._show_project_properties()

    def _build_status_bar(self):
        self.status_bar = ttk.Frame(self.root, padding=(5, 2))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(self.status_bar, text="준비")
        self.status_label.pack(side=tk.LEFT)
        self.count_label = ttk.Label(self.status_bar, text="")
        self.count_label.pack(side=tk.RIGHT)
        self._update_counts()

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def _update_counts(self):
        self.count_label.config(
            text=f"건물: {len(self.buildings)} | 도로: {len(self.roads)} | "
                 f"나무: {len(self.trees)} | 주차장: {len(self.parking_lots)} | 호수: {len(self.lakes)} | "
                 f"사람: {len(self.persons)} | 게이트: {len(self.gates)} | "
                 f"물탱크: {len(self.water_tanks)} | LPG: {len(self.lpg_tanks)} | 굴뚝: {len(self.chimneys)} | "
                 f"벽: {len(self.walls)} | 트럭: {len(self.trucks)} | 연결: {len(self.transport_lines)}"
        )

    # ========================
    # 캔버스 렌더링
    # ========================
    def world_to_canvas(self, wx, wz):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = cw / 2 + (wx + self.canvas_offset_x) * self.canvas_scale
        cy = ch / 2 + (wz + self.canvas_offset_y) * self.canvas_scale
        return cx, cy

    def canvas_to_world(self, cx, cy):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        wx = (cx - cw / 2) / self.canvas_scale - self.canvas_offset_x
        wz = (cy - ch / 2) / self.canvas_scale - self.canvas_offset_y
        return wx, wz

    def redraw_canvas(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # 그리드
        grid_spacing = 50
        gs = grid_spacing * self.canvas_scale
        if gs > 5:
            ox = (cw / 2 + self.canvas_offset_x * self.canvas_scale) % gs
            oy = (ch / 2 + self.canvas_offset_y * self.canvas_scale) % gs
            for x in range(int(-gs), cw + int(gs), max(1, int(gs))):
                self.canvas.create_line(x + ox, 0, x + ox, ch, fill="#354535", width=1)
            for y in range(int(-gs), ch + int(gs), max(1, int(gs))):
                self.canvas.create_line(0, y + oy, cw, y + oy, fill="#354535", width=1)

        # 축선
        origin_x, origin_y = self.world_to_canvas(0, 0)
        self.canvas.create_line(origin_x, 0, origin_x, ch, fill="#446644", width=1)
        self.canvas.create_line(0, origin_y, cw, origin_y, fill="#446644", width=1)

        # 지면 경계 표시 (하얀색 테두리)
        gw = self.project.ground_width
        gd = self.project.ground_depth
        gx1, gy1 = self.world_to_canvas(-gw, -gd)
        gx2, gy2 = self.world_to_canvas(gw, gd)
        self.canvas.create_rectangle(gx1, gy1, gx2, gy2,
                                     outline="white", width=2, dash=(8, 4))
        self.canvas.create_text((gx1 + gx2) / 2, gy1 - 10,
                                text=f"지면 ({int(gw*2)}x{int(gd*2)})",
                                fill="white", font=("Arial", 9))

        # 호수
        for lake in self.lakes:
            cx, cy = self.world_to_canvas(lake.x, lake.z)
            r = lake.radius * self.canvas_scale
            is_sel = self.selected_item == lake or self._is_in_multi_selection(lake)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=lake.color, outline="#66bbdd" if is_sel else "#336677", width=3 if is_sel else 1)
            self.canvas.create_text(cx, cy, text=lake.name, fill="white", font=("Arial", 8))

        # 주차장
        for p in self.parking_lots:
            cx, cy = self.world_to_canvas(p.x, p.z)
            hw, hd = p.width / 2 * self.canvas_scale, p.depth / 2 * self.canvas_scale
            is_sel = self.selected_item == p or self._is_in_multi_selection(p)
            self.canvas.create_rectangle(cx - hw, cy - hd, cx + hw, cy + hd,
                                         fill=p.color, outline="#aaaaff" if is_sel else "#666666", width=3 if is_sel else 1)
            self.canvas.create_text(cx, cy, text=p.name, fill="white", font=("Arial", 8))

        # 도로
        for road in self.roads:
            cx, cy = self.world_to_canvas(road.x, road.z)
            hl = road.length / 2 * self.canvas_scale
            hw = road.width / 2 * self.canvas_scale
            rad = math.radians(road.rotation)
            is_sel = self.selected_item == road or self._is_in_multi_selection(road)
            # 간단히 각도 무시하고 가로 도로로 표현 (0도=가로, 90도=세로)
            if abs(road.rotation % 180) < 45 or abs(road.rotation % 180) > 135:
                self.canvas.create_rectangle(cx - hl, cy - hw, cx + hl, cy + hw,
                                             fill=road.color, outline="#aaaa44" if is_sel else "#444444", width=3 if is_sel else 1)
            else:
                self.canvas.create_rectangle(cx - hw, cy - hl, cx + hw, cy + hl,
                                             fill=road.color, outline="#aaaa44" if is_sel else "#444444", width=3 if is_sel else 1)
            self.canvas.create_text(cx, cy, text=road.name, fill="#cccc44", font=("Arial", 7))

        # 나무
        for tree in self.trees:
            cx, cy = self.world_to_canvas(tree.x, tree.z)
            r = tree.canopy_radius * self.canvas_scale
            is_sel = self.selected_item == tree or self._is_in_multi_selection(tree)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=tree.color, outline="#88ff88" if is_sel else "#226622", width=2 if is_sel else 1)

        # 건물
        for b in self.buildings:
            cx, cy = self.world_to_canvas(b.x, b.z)
            hw, hd = b.width / 2 * self.canvas_scale, b.depth / 2 * self.canvas_scale
            is_sel = self.selected_item == b or self._is_in_multi_selection(b)
            outline_color = "#ffff44" if is_sel else "#225588"
            self.canvas.create_rectangle(cx - hw, cy - hd, cx + hw, cy + hd,
                                         fill=b.color, outline=outline_color, width=3 if is_sel else 2)
            self.canvas.create_text(cx, cy, text=b.name, fill="white", font=("Arial", 9, "bold"))
            self.canvas.create_text(cx, cy + 12 * self.canvas_scale * 0.3, text=f"{b.floors}F",
                                    fill="#aaaacc", font=("Arial", 7))

        # 사람
        for person in self.persons:
            cx, cy = self.world_to_canvas(person.x, person.z)
            r = 3 * self.canvas_scale
            is_sel = self.selected_item == person or self._is_in_multi_selection(person)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=person.shirt_color, outline="#ffff44" if is_sel else "#cc9966", width=2 if is_sel else 1)
            self.canvas.create_text(cx, cy + r + 6, text="🚶", font=("Arial", 8))

        # 게이트
        for gate in self.gates:
            cx, cy = self.world_to_canvas(gate.x, gate.z)
            hw = gate.width / 2 * self.canvas_scale
            hd = gate.depth / 2 * self.canvas_scale
            is_sel = self.selected_item == gate or self._is_in_multi_selection(gate)
            self.canvas.create_rectangle(cx - hw, cy - hd, cx + hw, cy + hd,
                                         fill=gate.color, outline="#ffff44" if is_sel else "#886644", width=3 if is_sel else 2)
            self.canvas.create_text(cx, cy, text=gate.name, fill="white", font=("Arial", 8, "bold"))

        # 물탱크
        for tank in self.water_tanks:
            cx, cy = self.world_to_canvas(tank.x, tank.z)
            r = tank.radius * self.canvas_scale
            is_sel = self.selected_item == tank or self._is_in_multi_selection(tank)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=tank.color, outline="#88ccff" if is_sel else "#666666", width=3 if is_sel else 2)
            self.canvas.create_text(cx, cy, text=tank.name, fill="white", font=("Arial", 8))

        # LPG 탱크
        for lpg in self.lpg_tanks:
            cx, cy = self.world_to_canvas(lpg.x, lpg.z)
            hw = (lpg.length / 2) * self.canvas_scale
            hd = (lpg.radius + 2) * self.canvas_scale
            is_sel = self.selected_item == lpg or self._is_in_multi_selection(lpg)
            self.canvas.create_rectangle(cx - hw, cy - hd, cx + hw, cy + hd,
                                         fill=lpg.color, outline="#ffff44" if is_sel else "#ff3300", width=3 if is_sel else 2)
            self.canvas.create_text(cx, cy, text=lpg.name, fill="black", font=("Arial", 7, "bold"))

        # 굴뚝
        for chimney in self.chimneys:
            cx, cy = self.world_to_canvas(chimney.x, chimney.z)
            r = chimney.radius * self.canvas_scale + 2
            is_sel = self.selected_item == chimney or self._is_in_multi_selection(chimney)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=chimney.color, outline="#ffff44" if is_sel else "#888888", width=3 if is_sel else 2)
            self.canvas.create_text(cx, cy, text=chimney.name, fill="white", font=("Arial", 7))

        for wall in self.walls:
            sx, sy = self.world_to_canvas(wall.x, wall.z)
            length = (wall.length or 40) * self.canvas_scale
            rot = math.radians(wall.rotation or 0)
            x1 = sx - length/2 * math.cos(rot)
            y1 = sy - length/2 * math.sin(rot)
            x2 = sx + length/2 * math.cos(rot)
            y2 = sy + length/2 * math.sin(rot)
            self.canvas.create_line(x1, y1, x2, y2, fill=wall.color, width=max(2, 3*self.canvas_scale), tags="object")
            self.canvas.create_text(sx, sy - 8, text=wall.name, fill="white", font=("Arial", 7), tags="object")

        for truck in self.trucks:
            sx, sy = self.world_to_canvas(truck.x, truck.z)
            sz = max(4, 8 * self.canvas_scale)
            self.canvas.create_rectangle(sx - sz, sy - sz/2, sx + sz, sy + sz/2, fill=truck.color, outline="#333", width=1, tags="object")
            self.canvas.create_text(sx, sy - sz/2 - 6, text=truck.name, fill="white", font=("Arial", 7), tags="object")

        # 연결통로
        for tl in self.transport_lines:
            x1, y1 = self.world_to_canvas(tl.x1, tl.z1)
            x2, y2 = self.world_to_canvas(tl.x2, tl.z2)
            is_sel = self.selected_item == tl or self._is_in_multi_selection(tl)
            line_color = tl.color if tl.transport_type == "conveyor" else ("#ee4444" if tl.transport_type == "lifter" else "#eeaa22")
            line_width = 4 if is_sel else 2
            outline_color = "#ffff44" if is_sel else line_color
            self.canvas.create_line(x1, y1, x2, y2, fill=outline_color, width=line_width, tags="object")
            # 라벨 (중점에 표시)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            self.canvas.create_text(mx, my - 10, text=tl.name, fill="white", font=("Arial", 8, "bold"), tags="object")

        # 🗑 휴지통 영역 (우하단)
        cw = self.canvas.winfo_width()
        ch_canvas = self.canvas.winfo_height()
        trash_x1, trash_y1 = cw - 70, ch_canvas - 70
        trash_x2, trash_y2 = cw - 10, ch_canvas - 10
        trash_fill = "#ff4444" if self.moving_item else "#333333"
        self.canvas.create_rectangle(trash_x1, trash_y1, trash_x2, trash_y2,
            fill=trash_fill, outline="#666", width=2, tags="trash")
        self.canvas.create_text((trash_x1 + trash_x2) / 2, (trash_y1 + trash_y2) / 2,
            text="🗑", font=("Arial", 20), fill="white", tags="trash")

        # 선택된 객체 리사이즈 핸들
        if self.selected_item and hasattr(self.selected_item, 'x'):
            item = self.selected_item
            sx, sy = self.world_to_canvas(item.x, item.z)
            hw = 15  # handle half width
            # 크기 기반 바운딩 박스 결정
            if hasattr(item, 'width') and hasattr(item, 'depth'):
                bw = item.width * self.canvas_scale / 2
                bd = item.depth * self.canvas_scale / 2
            elif hasattr(item, 'radius'):
                bw = bd = item.radius * self.canvas_scale
            elif hasattr(item, 'length') and hasattr(item, 'thickness'):
                bw = getattr(item, 'length', 40) * self.canvas_scale / 2
                bd = max(8, getattr(item, 'thickness', 2) * self.canvas_scale * 3)
            elif hasattr(item, 'length'):
                bw = getattr(item, 'length', 20) * self.canvas_scale / 2
                bd = 10 * self.canvas_scale
            else:
                bw = bd = 10 * self.canvas_scale
            # 8방향 핸들 (크게)
            hs = 6  # 핸들 반 크기
            handles = [
                (sx - bw, sy - bd), (sx, sy - bd), (sx + bw, sy - bd),
                (sx - bw, sy), (sx + bw, sy),
                (sx - bw, sy + bd), (sx, sy + bd), (sx + bw, sy + bd),
            ]
            cursors = ['top_left_corner', 'sb_v_double_arrow', 'top_right_corner',
                       'sb_h_double_arrow', 'sb_h_double_arrow',
                       'bottom_left_corner', 'sb_v_double_arrow', 'bottom_right_corner']
            for i, (hx, hy) in enumerate(handles):
                self.canvas.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs,
                    fill="#00aaff", outline="white", width=2, tags="handle")
            # 선택 강조 테두리
            self.canvas.create_rectangle(sx - bw, sy - bd, sx + bw, sy + bd,
                outline="#00aaff", width=2, dash=(4, 4), tags="handle")
            # 크기 표시
            if hasattr(item, 'width') and hasattr(item, 'depth'):
                self.canvas.create_text(sx, sy + bd + 14, text=f"W:{item.width:.0f} D:{item.depth:.0f}",
                    fill="#00aaff", font=("Arial", 8), tags="handle")
            elif hasattr(item, 'radius'):
                self.canvas.create_text(sx, sy + bd + 14, text=f"R:{item.radius:.0f}",
                    fill="#00aaff", font=("Arial", 8), tags="handle")
            elif hasattr(item, 'length'):
                self.canvas.create_text(sx, sy + bd + 14, text=f"L:{item.length:.0f}",
                    fill="#00aaff", font=("Arial", 8), tags="handle")

        # 연결통로 시작점 표시
        if self.transport_start:
            sx, sy = self.world_to_canvas(self.transport_start[0], self.transport_start[1])
            self.canvas.create_oval(sx - 6, sy - 6, sx + 6, sy + 6, fill="#ffff00", outline="#ff8800", width=2)
            self.canvas.create_text(sx, sy - 12, text="시작점", fill="#ff8800", font=("Arial", 8))

        # 범례
        self.canvas.create_text(10, ch - 10, anchor="sw",
                                text="🖱️좌클릭: 선택/배치 | 우클릭드래그: 팬 | 휠: 줌",
                                fill="#668866", font=("Arial", 9))

    # ========================
    # 캔버스 이벤트
    # ========================
    def _on_canvas_click(self, event):
        # 리사이즈 핸들 체크
        if self._on_resize_check(event):
            return

        wx, wz = self.canvas_to_world(event.x, event.y)

        # 연결통로 모드: 두 점 클릭으로 선 그리기
        if self.current_tool == self.TOOL_TRANSPORT:
            if self.transport_start is None:
                # 첫 번째 클릭: 시작점 저장
                self.transport_start = (wx, wz)
                self.update_status("연결 시작점 선택됨, 끝점을 클릭하세요")
                self.redraw_canvas()
            else:
                # 두 번째 클릭: 끝점 저장, TransportLine 생성
                x1, z1 = self.transport_start
                obj = TransportLine(
                    name=f"연결통로 {len(self.transport_lines) + 1}",
                    x1=round(x1), z1=round(z1),
                    x2=round(wx), z2=round(wz)
                )
                self.transport_lines.append(obj)
                self.transport_start = None
                self.selected_item = obj
                self.selected_type = self.TOOL_TRANSPORT
                self._update_object_tree()
                self._update_counts()
                self.redraw_canvas()
                self._show_item_properties(obj, self.TOOL_TRANSPORT)
                self.update_status(f"연결통로 생성됨: ({round(x1)}, {round(z1)}) → ({round(wx)}, {round(wz)})")
            return

        if self.current_tool != self.TOOL_SELECT:
            self._place_object(wx, wz)
            return

        # 선택 모드 — 오브젝트 찾기
        found = self._find_object_at(wx, wz)
        ctrl_held = bool(event.state & 0x4)  # Ctrl 키

        if found:
            item, item_type = found
            # 이미 다중 선택된 항목을 클릭 → 전체 이동 시작
            if self._is_in_multi_selection(item) and not ctrl_held:
                self.moving_item = True
                self.move_start = (wx, wz)
                self.selected_item = item
                self.selected_type = item_type
                return
            # Ctrl+클릭: 다중 선택 토글
            if ctrl_held:
                if self._is_in_multi_selection(item):
                    self.selected_items = [(i, t) for i, t in self.selected_items if i is not item]
                else:
                    self.selected_items.append((item, item_type))
                if self.selected_items:
                    self.selected_item = self.selected_items[-1][0]
                    self.selected_type = self.selected_items[-1][1]
                else:
                    self.selected_item = None
                    self.selected_type = None
            else:
                # 단일 클릭: 이 오브젝트만 선택
                self.selected_item = item
                self.selected_type = item_type
                self.selected_items = [(item, item_type)]
                self.moving_item = True
                self.move_start = (wx, wz)
            self._show_item_properties(item, item_type)
            self._highlight_in_tree(item)
        else:
            if not ctrl_held:
                # 빈 공간 클릭 → 드래그 선택 시작
                self.selection_rect_start = (event.x, event.y)
                self.is_box_selecting = True
                self.selected_items = []
                self.selected_item = None
                self.selected_type = None

        self.redraw_canvas()

    def _is_in_multi_selection(self, item):
        return any(i is item for i, t in self.selected_items)

    def _on_canvas_drag(self, event):
        # 리사이즈 드래그 체크
        if self.resize_handle and self._on_resize_drag(event):
            return

        # 사각형 선택 모드
        if self.is_box_selecting and self.selection_rect_start:
            self.redraw_canvas()
            sx, sy = self.selection_rect_start
            self.canvas.create_rectangle(sx, sy, event.x, event.y,
                                         outline="#44aaff", width=2, dash=(4, 2),
                                         fill="#44aaff", stipple="gray12")
            return

        # 다중 선택 이동
        if self.moving_item and self.move_start:
            wx, wz = self.canvas_to_world(event.x, event.y)
            dx = wx - self.move_start[0]
            dz = wz - self.move_start[1]
            if abs(dx) < 0.5 and abs(dz) < 0.5:
                return
            if self.selected_items:
                for item, _ in self.selected_items:
                    item.x = round(item.x + dx, 1)
                    item.z = round(item.z + dz, 1)
            elif self.selected_item:
                self.selected_item.x = round(self.selected_item.x + dx, 1)
                self.selected_item.z = round(self.selected_item.z + dz, 1)
            self.move_start = (wx, wz)
            self.redraw_canvas()

    def _on_canvas_release(self, event):
        # 리사이즈 완료 체크
        if self._on_resize_release(event):
            return

        # 사각형 선택 완료
        if self.is_box_selecting and self.selection_rect_start:
            sx, sy = self.selection_rect_start
            ex, ey = event.x, event.y
            x1, x2 = min(sx, ex), max(sx, ex)
            y1, y2 = min(sy, ey), max(sy, ey)

            # 사각형이 너무 작으면 단순 클릭으로 처리 → deselect
            if abs(ex - sx) < 5 and abs(ey - sy) < 5:
                self.deselect_all()
            else:
                # 사각형 안에 있는 모든 오브젝트 선택
                wx1, wz1 = self.canvas_to_world(x1, y1)
                wx2, wz2 = self.canvas_to_world(x2, y2)
                min_wx, max_wx = min(wx1, wx2), max(wx1, wx2)
                min_wz, max_wz = min(wz1, wz2), max(wz1, wz2)
                self.selected_items = []
                for b in self.buildings:
                    if min_wx <= b.x <= max_wx and min_wz <= b.z <= max_wz:
                        self.selected_items.append((b, "building"))
                for r in self.roads:
                    if min_wx <= r.x <= max_wx and min_wz <= r.z <= max_wz:
                        self.selected_items.append((r, "road"))
                for t in self.trees:
                    if min_wx <= t.x <= max_wx and min_wz <= t.z <= max_wz:
                        self.selected_items.append((t, "tree"))
                for p in self.parking_lots:
                    if min_wx <= p.x <= max_wx and min_wz <= p.z <= max_wz:
                        self.selected_items.append((p, "parking"))
                for l in self.lakes:
                    if min_wx <= l.x <= max_wx and min_wz <= l.z <= max_wz:
                        self.selected_items.append((l, "lake"))
                for p in self.persons:
                    if min_wx <= p.x <= max_wx and min_wz <= p.z <= max_wz:
                        self.selected_items.append((p, "person"))
                for g in self.gates:
                    if min_wx <= g.x <= max_wx and min_wz <= g.z <= max_wz:
                        self.selected_items.append((g, "gate"))
                if self.selected_items:
                    self.selected_item = self.selected_items[0][0]
                    self.selected_type = self.selected_items[0][1]
                    self.update_status(f"{len(self.selected_items)}개 선택됨")
                else:
                    self.deselect_all()

            self.selection_rect_start = None
            self.is_box_selecting = False
            self.redraw_canvas()
            return

        # 휴지통 영역 드롭 체크
        if self.moving_item and self.selected_item:
            cw = self.canvas.winfo_width()
            ch_canvas = self.canvas.winfo_height()
            if event.x > cw - 70 and event.y > ch_canvas - 70:
                self.delete_selected()
                self.moving_item = False
                self.move_start = None
                return
        self.moving_item = False
        self.move_start = None
        self.redraw_canvas()

    def _on_canvas_right_click(self, event):
        self.drag_start = (event.x, event.y)

    def _on_canvas_pan(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.canvas_offset_x += dx / self.canvas_scale
            self.canvas_offset_y += dy / self.canvas_scale
            self.drag_start = (event.x, event.y)
            self.redraw_canvas()

    def _on_canvas_pan_end(self, event):
        self.drag_start = None

    def _on_canvas_scroll(self, event):
        if event.delta > 0:
            self.zoom_canvas(1.1)
        else:
            self.zoom_canvas(0.9)

    def zoom_canvas(self, factor):
        self.canvas_scale *= factor
        self.canvas_scale = max(0.1, min(10.0, self.canvas_scale))
        self.redraw_canvas()

    def reset_canvas_view(self):
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.canvas_scale = 1.0
        self.redraw_canvas()

    def _on_resize_check(self, event):
        """Check if mouse is near a resize handle of selected item"""
        if not self.selected_item or not hasattr(self.selected_item, 'x'):
            self.resize_handle = None
            return False
        item = self.selected_item
        sx, sy = self.world_to_canvas(item.x, item.z)
        if hasattr(item, 'width') and hasattr(item, 'depth'):
            bw = item.width * self.canvas_scale / 2
            bd = item.depth * self.canvas_scale / 2
        elif hasattr(item, 'radius'):
            bw = bd = item.radius * self.canvas_scale
        elif hasattr(item, 'length') and hasattr(item, 'thickness'):
            # 벽: 길이 x 두께
            bw = getattr(item, 'length', 40) * self.canvas_scale / 2
            bd = max(8, getattr(item, 'thickness', 2) * self.canvas_scale * 3)
        elif hasattr(item, 'length'):
            bw = getattr(item, 'length', 20) * self.canvas_scale / 2
            bd = 10 * self.canvas_scale
        else:
            return False
        handles = {
            'nw': (sx - bw, sy - bd), 'n': (sx, sy - bd), 'ne': (sx + bw, sy - bd),
            'w': (sx - bw, sy), 'e': (sx + bw, sy),
            'sw': (sx - bw, sy + bd), 's': (sx, sy + bd), 'se': (sx + bw, sy + bd),
        }
        for direction, (hx, hy) in handles.items():
            if abs(event.x - hx) < 12 and abs(event.y - hy) < 12:
                self.resize_handle = direction
                self.resize_start = (event.x, event.y)
                self.resize_orig = {
                    'width': getattr(item, 'width', None),
                    'depth': getattr(item, 'depth', None),
                    'height': getattr(item, 'height', None),
                    'radius': getattr(item, 'radius', None),
                    'length': getattr(item, 'length', None),
                    'thickness': getattr(item, 'thickness', None),
                }
                return True
        self.resize_handle = None
        return False

    def _on_resize_drag(self, event):
        """Handle resize dragging"""
        if not self.resize_handle or not self.resize_start or not self.selected_item:
            return False
        dx = (event.x - self.resize_start[0]) / self.canvas_scale
        dy = (event.y - self.resize_start[1]) / self.canvas_scale
        item = self.selected_item
        d = self.resize_handle
        if hasattr(item, 'width') and hasattr(item, 'depth'):
            # 가로 (width)
            if 'e' in d:
                item.width = max(5, round(self.resize_orig['width'] + dx, 1))
            elif 'w' in d:
                item.width = max(5, round(self.resize_orig['width'] - dx, 1))
            # 세로 (depth)
            if 's' in d:
                item.depth = max(5, round(self.resize_orig['depth'] + dy, 1))
            elif 'n' in d:
                item.depth = max(5, round(self.resize_orig['depth'] - dy, 1))
        elif hasattr(item, 'radius'):
            delta = max(abs(dx), abs(dy))
            sign = 1 if (dx + dy) > 0 else -1
            item.radius = max(3, round(self.resize_orig['radius'] + sign * delta * 0.5, 1))
        elif hasattr(item, 'length') and hasattr(item, 'thickness'):
            # 벽: 좌우=길이, 상하=두께
            if 'e' in d:
                item.length = max(5, round(self.resize_orig['length'] + dx, 1))
            elif 'w' in d:
                item.length = max(5, round(self.resize_orig['length'] - dx, 1))
            if 's' in d:
                item.thickness = max(1, round(self.resize_orig['thickness'] + dy * 0.3, 1))
            elif 'n' in d:
                item.thickness = max(1, round(self.resize_orig['thickness'] - dy * 0.3, 1))
        elif hasattr(item, 'length'):
            if 'e' in d:
                item.length = max(5, round(self.resize_orig['length'] + dx, 1))
            elif 'w' in d:
                item.length = max(5, round(self.resize_orig['length'] - dx, 1))
        # 높이 조절: 핸들 위에서 Shift 드래그하면 높이 변경
        if hasattr(item, 'height') and self.resize_orig.get('height'):
            if 'n' in d and hasattr(event, 'state') and (event.state & 0x1):  # Shift 키
                item.height = max(5, round(self.resize_orig['height'] - dy, 1))
        self.redraw_canvas()
        self.update_status(f"리사이즈: {self._get_size_text(item)}")
        return True

    def _get_size_text(self, item):
        parts = []
        if hasattr(item, 'width') and not hasattr(item, 'length'):
            parts.append(f"가로:{item.width:.0f}")
        if hasattr(item, 'depth'):
            parts.append(f"세로:{item.depth:.0f}")
        if hasattr(item, 'height'):
            parts.append(f"높이:{item.height:.0f}")
        if hasattr(item, 'radius'):
            parts.append(f"반경:{item.radius:.0f}")
        if hasattr(item, 'length'):
            parts.append(f"길이:{item.length:.0f}")
        if hasattr(item, 'thickness'):
            parts.append(f"두께:{item.thickness:.1f}")
        return " | ".join(parts)

    def _on_resize_release(self, event):
        """End resize operation"""
        if self.resize_handle:
            self.resize_handle = None
            self.resize_start = None
            self.resize_orig = None
            self._update_object_tree()
            if self.selected_item:
                self._show_item_properties(self.selected_item, self.selected_type)
            return True
        return False

    def _find_object_at(self, wx, wz):
        # 건물 (우선 순위 높음)
        for b in reversed(self.buildings):
            if (b.x - b.width / 2 <= wx <= b.x + b.width / 2 and
                    b.z - b.depth / 2 <= wz <= b.z + b.depth / 2):
                return (b, "building")
        # 주차장
        for p in reversed(self.parking_lots):
            if (p.x - p.width / 2 <= wx <= p.x + p.width / 2 and
                    p.z - p.depth / 2 <= wz <= p.z + p.depth / 2):
                return (p, "parking")
        # 도로
        for r in reversed(self.roads):
            hl, hw = r.length / 2, r.width / 2
            if (r.x - hl <= wx <= r.x + hl and r.z - hw <= wz <= r.z + hw):
                return (r, "road")
        # 호수
        for l in reversed(self.lakes):
            dist = math.sqrt((wx - l.x) ** 2 + (wz - l.z) ** 2)
            if dist <= l.radius:
                return (l, "lake")
        # 나무
        for t in reversed(self.trees):
            dist = math.sqrt((wx - t.x) ** 2 + (wz - t.z) ** 2)
            if dist <= t.canopy_radius + 2:
                return (t, "tree")
        # 사람
        for p in reversed(self.persons):
            dist = math.sqrt((wx - p.x) ** 2 + (wz - p.z) ** 2)
            if dist <= 5:
                return (p, "person")
        # 게이트
        for g in reversed(self.gates):
            if (g.x - g.width / 2 <= wx <= g.x + g.width / 2 and
                    g.z - g.depth / 2 <= wz <= g.z + g.depth / 2):
                return (g, "gate")
        # 물탱크
        for wt in reversed(self.water_tanks):
            dist = math.sqrt((wx - wt.x) ** 2 + (wz - wt.z) ** 2)
            if dist <= wt.radius + 2:
                return (wt, "water_tank")
        # LPG 탱크
        for lt in reversed(self.lpg_tanks):
            if (lt.x - lt.length / 2 <= wx <= lt.x + lt.length / 2 and
                    lt.z - (lt.radius + 2) <= wz <= lt.z + (lt.radius + 2)):
                return (lt, "lpg_tank")
        # 굴뚝
        for ch in reversed(self.chimneys):
            dist = math.sqrt((wx - ch.x) ** 2 + (wz - ch.z) ** 2)
            if dist <= ch.radius + 3:
                return (ch, "chimney")
        for w in reversed(self.walls):
            if abs(wx - w.x) < (w.length or 40)/2 and abs(wz - w.z) < 10:
                return w, "wall"
        for tr in reversed(self.trucks):
            if abs(wx - tr.x) < 12 and abs(wz - tr.z) < 5:
                return tr, "truck"
        # 연결통로 (선분 근처 5 유닛 이내)
        for tl in reversed(self.transport_lines):
            # 선분 (x1,z1)-(x2,z2)까지의 최단거리 계산
            dx = tl.x2 - tl.x1
            dz = tl.z2 - tl.z1
            if dx == 0 and dz == 0:
                dist = math.sqrt((wx - tl.x1) ** 2 + (wz - tl.z1) ** 2)
            else:
                t = max(0, min(1, ((wx - tl.x1) * dx + (wz - tl.z1) * dz) / (dx * dx + dz * dz)))
                px = tl.x1 + t * dx
                pz = tl.z1 + t * dz
                dist = math.sqrt((wx - px) ** 2 + (wz - pz) ** 2)
            if dist <= 5:
                return tl, "transport"
        return None

    def _place_object(self, wx, wz):
        if self.current_tool == self.TOOL_BUILDING:
            obj = Building(name=f"건물 {len(self.buildings) + 1}", x=round(wx), z=round(wz))
            self.buildings.append(obj)
        elif self.current_tool == self.TOOL_ROAD:
            obj = Road(name=f"도로 {len(self.roads) + 1}", x=round(wx), z=round(wz))
            self.roads.append(obj)
        elif self.current_tool == self.TOOL_TREE:
            obj = Tree(name=f"나무 {len(self.trees) + 1}", x=round(wx), z=round(wz))
            self.trees.append(obj)
        elif self.current_tool == self.TOOL_PARKING:
            obj = ParkingLot(name=f"주차장 {len(self.parking_lots) + 1}", x=round(wx), z=round(wz))
            self.parking_lots.append(obj)
        elif self.current_tool == self.TOOL_LAKE:
            obj = Lake(name=f"호수 {len(self.lakes) + 1}", x=round(wx), z=round(wz))
            self.lakes.append(obj)
        elif self.current_tool == self.TOOL_PERSON:
            import random
            obj = Person(name=f"사람 {len(self.persons) + 1}", x=round(wx), z=round(wz),
                        shirt_color=random.choice(["#3366cc", "#cc3333", "#33aa33", "#cc8833", "#8833cc", "#33cccc"]),
                        pants_color=random.choice(["#333333", "#444466", "#554433"]))
            self.persons.append(obj)
        elif self.current_tool == self.TOOL_GATE:
            obj = Gate(name=f"게이트 {len(self.gates) + 1}", x=round(wx), z=round(wz))
            self.gates.append(obj)
        elif self.current_tool == self.TOOL_WATER_TANK:
            obj = WaterTank(name=f"물탱크 {len(self.water_tanks) + 1}", x=round(wx), z=round(wz))
            self.water_tanks.append(obj)
        elif self.current_tool == self.TOOL_LPG_TANK:
            obj = LPGTank(name=f"LPG탱크 {len(self.lpg_tanks) + 1}", x=round(wx), z=round(wz))
            self.lpg_tanks.append(obj)
        elif self.current_tool == self.TOOL_CHIMNEY:
            obj = Chimney(name=f"굴뚝 {len(self.chimneys) + 1}", x=round(wx), z=round(wz))
            self.chimneys.append(obj)
        elif self.current_tool == self.TOOL_WALL:
            obj = Wall(name=f"벽 {len(self.walls) + 1}", x=round(wx), z=round(wz))
            self.walls.append(obj)
        elif self.current_tool == self.TOOL_TRUCK:
            obj = Truck(name=f"트럭 {len(self.trucks) + 1}", x=round(wx), z=round(wz))
            self.trucks.append(obj)
        else:
            return

        self.selected_item = obj
        self.selected_type = self.current_tool
        self._update_object_tree()
        self._update_counts()
        self.redraw_canvas()
        self._show_item_properties(obj, self.current_tool)
        self.update_status(f"{self.current_tool} 추가됨: ({round(wx)}, {round(wz)})")

    # ========================
    # 오브젝트 트리
    # ========================
    def _update_object_tree(self):
        self.object_tree.delete(*self.object_tree.get_children())

        if self.buildings:
            bld_parent = self.object_tree.insert("", "end", text="🏢 건물", open=True, values=("",))
            for b in self.buildings:
                self.object_tree.insert(bld_parent, "end", text=b.name, values=(b.building_type,),
                                        tags=(f"building_{b.id}",))

        if self.roads:
            road_parent = self.object_tree.insert("", "end", text="🛣️ 도로", open=True, values=("",))
            for r in self.roads:
                self.object_tree.insert(road_parent, "end", text=r.name, values=("도로",),
                                        tags=(f"road_{r.id}",))

        if self.trees:
            tree_parent = self.object_tree.insert("", "end", text="🌳 나무", open=False, values=("",))
            for t in self.trees:
                self.object_tree.insert(tree_parent, "end", text=t.name, values=("나무",),
                                        tags=(f"tree_{t.id}",))

        if self.parking_lots:
            park_parent = self.object_tree.insert("", "end", text="🅿️ 주차장", open=True, values=("",))
            for p in self.parking_lots:
                self.object_tree.insert(park_parent, "end", text=p.name, values=("주차장",),
                                        tags=(f"parking_{p.id}",))

        if self.lakes:
            lake_parent = self.object_tree.insert("", "end", text="💧 호수", open=True, values=("",))
            for l in self.lakes:
                self.object_tree.insert(lake_parent, "end", text=l.name, values=("호수",),
                                        tags=(f"lake_{l.id}",))

        if self.persons:
            person_parent = self.object_tree.insert("", "end", text="🚶 사람", open=False, values=("",))
            for p in self.persons:
                self.object_tree.insert(person_parent, "end", text=p.name, values=("사람",),
                                        tags=(f"person_{p.id}",))

        if self.gates:
            gate_parent = self.object_tree.insert("", "end", text="🚪 게이트", open=True, values=("",))
            for g in self.gates:
                self.object_tree.insert(gate_parent, "end", text=g.name, values=(g.gate_type,),
                                        tags=(f"gate_{g.id}",))

        if self.water_tanks:
            tank_parent = self.object_tree.insert("", "end", text="🏗 물탱크", open=True, values=("",))
            for wt in self.water_tanks:
                self.object_tree.insert(tank_parent, "end", text=wt.name, values=("물탱크",),
                                        tags=(f"water_tank_{wt.id}",))

        if self.lpg_tanks:
            lpg_parent = self.object_tree.insert("", "end", text="⛽ LPG탱크", open=True, values=("",))
            for lt in self.lpg_tanks:
                self.object_tree.insert(lpg_parent, "end", text=lt.name, values=("LPG탱크",),
                                        tags=(f"lpg_tank_{lt.id}",))

        if self.chimneys:
            chimney_parent = self.object_tree.insert("", "end", text="🏭 굴뚝", open=True, values=("",))
            for ch in self.chimneys:
                self.object_tree.insert(chimney_parent, "end", text=ch.name, values=("굴뚝",),
                                        tags=(f"chimney_{ch.id}",))

        if self.transport_lines:
            transport_parent = self.object_tree.insert("", "end", text="🔗 연결통로", open=True, values=("",))
            for tl in self.transport_lines:
                self.object_tree.insert(transport_parent, "end", text=tl.name, values=(tl.transport_type,),
                                        tags=(f"transport_{tl.id}",))

    def _on_tree_select(self, event):
        sel = self.object_tree.selection()
        if not sel:
            return
        tags = self.object_tree.item(sel[0], "tags")
        if not tags:
            return
        tag = tags[0]
        parts = tag.split("_", 1)
        if len(parts) != 2:
            return
        obj_type, obj_id = parts

        obj = None
        if obj_type == "building":
            obj = next((b for b in self.buildings if b.id == obj_id), None)
        elif obj_type == "road":
            obj = next((r for r in self.roads if r.id == obj_id), None)
        elif obj_type == "tree":
            obj = next((t for t in self.trees if t.id == obj_id), None)
        elif obj_type == "parking":
            obj = next((p for p in self.parking_lots if p.id == obj_id), None)
        elif obj_type == "lake":
            obj = next((l for l in self.lakes if l.id == obj_id), None)
        elif obj_type == "person":
            obj = next((p for p in self.persons if p.id == obj_id), None)
        elif obj_type == "gate":
            obj = next((g for g in self.gates if g.id == obj_id), None)
        elif obj_type == "transport":
            obj = next((tl for tl in self.transport_lines if tl.id == obj_id), None)

        if obj:
            self.selected_item = obj
            self.selected_type = obj_type
            self._show_item_properties(obj, obj_type)
            self.redraw_canvas()

    def _highlight_in_tree(self, item):
        for iid in self.object_tree.get_children(""):
            for child_iid in self.object_tree.get_children(iid):
                tags = self.object_tree.item(child_iid, "tags")
                if tags and item.id in tags[0]:
                    self.object_tree.selection_set(child_iid)
                    self.object_tree.see(child_iid)
                    return

    # ========================
    # 속성 패널
    # ========================
    def _clear_props(self):
        for widget in self.props_frame.winfo_children():
            widget.destroy()

    def _show_project_properties(self):
        self._clear_props()
        ttk.Label(self.props_frame, text="프로젝트 설정", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 10))

        # 지면 가로 (Width)
        f = ttk.Frame(self.props_frame)
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="지면 가로:").pack(side=tk.LEFT)
        gw_var = tk.DoubleVar(value=self.project.ground_width)
        gw_label = ttk.Label(f, text=f"{int(self.project.ground_width)}")
        gw_label.pack(side=tk.RIGHT)
        gw_scale = ttk.Scale(f, from_=100, to=3000, variable=gw_var, orient=tk.HORIZONTAL)
        gw_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 5))
        def _update_gw(*a):
            try:
                val = gw_var.get()
                if val > 0:
                    self.project.ground_width = val
                    gw_label.config(text=f"{int(val)}")
                    self.redraw_canvas()
            except (tk.TclError, ValueError):
                pass
        gw_var.trace_add('write', _update_gw)

        # 지면 세로 (Depth)
        f = ttk.Frame(self.props_frame)
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="지면 세로:").pack(side=tk.LEFT)
        gd_var = tk.DoubleVar(value=self.project.ground_depth)
        gd_label = ttk.Label(f, text=f"{int(self.project.ground_depth)}")
        gd_label.pack(side=tk.RIGHT)
        gd_scale = ttk.Scale(f, from_=100, to=3000, variable=gd_var, orient=tk.HORIZONTAL)
        gd_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 5))
        def _update_gd(*a):
            try:
                val = gd_var.get()
                if val > 0:
                    self.project.ground_depth = val
                    gd_label.config(text=f"{int(val)}")
                    self.redraw_canvas()
            except (tk.TclError, ValueError):
                pass
        gd_var.trace_add('write', _update_gd)

        # 지면 색상
        f = ttk.Frame(self.props_frame)
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="지면 색상:").pack(side=tk.LEFT)
        gc_btn = tk.Button(f, text="  ", bg=self.project.ground_color, width=4,
                           command=lambda: self._pick_project_color("ground_color", gc_btn))
        gc_btn.pack(side=tk.RIGHT)

        # 하늘 색상
        f = ttk.Frame(self.props_frame)
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="하늘 색상:").pack(side=tk.LEFT)
        sc_btn = tk.Button(f, text="  ", bg=self.project.sky_color, width=4,
                           command=lambda: self._pick_project_color("sky_color", sc_btn))
        sc_btn.pack(side=tk.RIGHT)

        # 이름표 크기
        ttk.Separator(self.props_frame).pack(fill=tk.X, pady=8)
        f = ttk.Frame(self.props_frame)
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="🏷️ 이름표 크기:").pack(side=tk.LEFT)
        ls_var = tk.DoubleVar(value=self.project.label_scale)
        ls_label = ttk.Label(f, text=f"{self.project.label_scale:.1f}x")
        ls_label.pack(side=tk.RIGHT)
        ls_scale = ttk.Scale(f, from_=0.3, to=5.0, variable=ls_var, orient=tk.HORIZONTAL)
        ls_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 5))
        def _update_ls(*a):
            try:
                val = ls_var.get()
                if val > 0:
                    self.project.label_scale = round(val, 1)
                    ls_label.config(text=f"{val:.1f}x")
            except (tk.TclError, ValueError):
                pass
        ls_var.trace_add('write', _update_ls)

    def _pick_project_color(self, attr, btn):
        color = colorchooser.askcolor(getattr(self.project, attr))[1]
        if color:
            setattr(self.project, attr, color)
            btn.config(bg=color)

    def _show_item_properties(self, item, item_type):
        self._clear_props()

        type_labels = {
            "building": "🏢 건물 속성",
            "road": "🛣️ 도로 속성",
            "tree": "🌳 나무 속성",
            "parking": "🅿️ 주차장 속성",
            "lake": "💧 호수 속성",
            "person": "🚶 사람 속성",
            "gate": "🚪 게이트 속성",
            "water_tank": "💧 물탱크 속성",
            "lpg_tank": "⛽ LPG탱크 속성",
            "chimney": "🏭 굴뚝 속성",
            "wall": "🧱 벽 속성",
            "truck": "🚛 트럭 속성",
            "transport": "🔗 연결통로 속성",
        }
        ttk.Label(self.props_frame, text=type_labels.get(item_type, "속성"),
                  font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 10))

        def make_entry(label, attr, var_type=tk.StringVar):
            f = ttk.Frame(self.props_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=f"{label}:").pack(side=tk.LEFT)
            var = var_type(value=getattr(item, attr))
            e = ttk.Entry(f, textvariable=var, width=15)
            e.pack(side=tk.RIGHT)

            def on_change(*args):
                try:
                    val = var.get()
                    setattr(item, attr, val)
                    self.redraw_canvas()
                    self._update_object_tree()
                except (tk.TclError, ValueError):
                    pass

            var.trace_add('write', on_change)
            return var

        def make_color_btn(label, attr):
            f = ttk.Frame(self.props_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=f"{label}:").pack(side=tk.LEFT)
            btn = tk.Button(f, text="  ", bg=getattr(item, attr), width=4)
            btn.config(command=lambda: self._pick_item_color(item, attr, btn))
            btn.pack(side=tk.RIGHT)

        def make_hex_color_entry(label, attr):
            f = ttk.Frame(self.props_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=f"{label}:").pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(item, attr, '#ffffff'))
            e = ttk.Entry(f, textvariable=var, width=10)
            e.pack(side=tk.RIGHT)
            preview = tk.Label(f, text="  ", bg=getattr(item, attr, '#ffffff'), width=3)
            preview.pack(side=tk.RIGHT, padx=2)
            def on_change(*args):
                try:
                    val = var.get().strip()
                    if val.startswith('#') and len(val) in (4, 7):
                        setattr(item, attr, val)
                        preview.config(bg=val)
                        self.redraw_canvas()
                except (tk.TclError, ValueError):
                    pass
            var.trace_add('write', on_change)

        def make_combo(label, attr, values):
            f = ttk.Frame(self.props_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=f"{label}:").pack(side=tk.LEFT)
            # values is list of (english_key, korean_label) tuples
            key_to_label = {v[0]: v[1] for v in values}
            label_to_key = {v[1]: v[0] for v in values}
            current_key = getattr(item, attr)
            display_var = tk.StringVar(value=key_to_label.get(current_key, current_key))
            combo = ttk.Combobox(f, textvariable=display_var, values=[v[1] for v in values], width=12, state="readonly")
            combo.pack(side=tk.RIGHT)

            def on_change(*args):
                korean_label = display_var.get()
                english_key = label_to_key.get(korean_label, korean_label)
                setattr(item, attr, english_key)

            display_var.trace_add('write', on_change)

        # 공통
        make_entry("이름", "name")
        make_entry("X 위치", "x", tk.DoubleVar)
        make_entry("Z 위치", "z", tk.DoubleVar)
        if hasattr(item, 'color'):
            make_color_btn("색상", "color")
            make_hex_color_entry("HEX 코드", "color")

        if item_type == "building":
            make_entry("너비(W)", "width", tk.DoubleVar)
            make_entry("깊이(D)", "depth", tk.DoubleVar)
            make_entry("높이(H)", "height", tk.DoubleVar)
            make_entry("층수", "floors", tk.IntVar)
            make_combo("유형", "building_type", self.BUILDING_TYPES)
            make_combo("지붕", "roof_type", self.ROOF_TYPES)
            make_entry("설명", "description")

        elif item_type == "road":
            make_entry("길이", "length", tk.DoubleVar)
            make_entry("너비", "width", tk.DoubleVar)
            make_entry("회전(도)", "rotation", tk.DoubleVar)

        elif item_type == "tree":
            make_entry("줄기 높이", "trunk_height", tk.DoubleVar)
            make_entry("나뭇잎 반경", "canopy_radius", tk.DoubleVar)

        elif item_type == "parking":
            make_entry("너비", "width", tk.DoubleVar)
            make_entry("깊이", "depth", tk.DoubleVar)

        elif item_type == "lake":
            make_entry("반경", "radius", tk.DoubleVar)

        elif item_type == "person":
            make_entry("이동 속도", "speed", tk.DoubleVar)
            make_entry("방향(도)", "direction", tk.DoubleVar)
            make_entry("이동 반경", "walk_radius", tk.DoubleVar)
            make_entry("말풍선", "speech")
            make_color_btn("셔츠 색상", "shirt_color")
            make_color_btn("바지 색상", "pants_color")

        elif item_type == "gate":
            make_entry("너비", "width", tk.DoubleVar)
            make_entry("높이", "height", tk.DoubleVar)
            make_entry("깊이", "depth", tk.DoubleVar)
            make_combo("유형", "gate_type", self.GATE_TYPES)

        elif item_type == "water_tank":
            make_entry("반경", "radius", tk.DoubleVar)
            make_entry("높이", "height", tk.DoubleVar)

        elif item_type == "lpg_tank":
            make_entry("길이", "length", tk.DoubleVar)
            make_entry("반경", "radius", tk.DoubleVar)

        elif item_type == "chimney":
            make_entry("높이", "height", tk.DoubleVar)
            make_entry("반경", "radius", tk.DoubleVar)
            # 연기 ON/OFF 체크박스
            smoke_f = ttk.Frame(self.props_frame)
            smoke_f.pack(fill=tk.X, pady=2)
            ttk.Label(smoke_f, text="연기:").pack(side=tk.LEFT)
            smoke_var = tk.BooleanVar(value=getattr(item, 'has_smoke', True))
            smoke_cb = ttk.Checkbutton(smoke_f, variable=smoke_var)
            smoke_cb.pack(side=tk.RIGHT)
            def _on_smoke_change(*args):
                item.has_smoke = smoke_var.get()
                self.redraw_canvas()
            smoke_var.trace_add('write', _on_smoke_change)

        elif item_type == "wall":
            make_entry("길이", "length", tk.DoubleVar)
            make_entry("높이", "height", tk.DoubleVar)
            make_entry("두께", "thickness", tk.DoubleVar)
            make_entry("회전(도)", "rotation", tk.DoubleVar)
            make_combo("유형", "wall_type", [("concrete", "콘크리트"), ("brick", "벽돌"), ("fence", "철조망")])

        elif item_type == "truck":
            make_entry("방향(도)", "direction", tk.DoubleVar)
            make_entry("이동 속도", "speed", tk.DoubleVar)
            make_entry("이동 반경", "route_radius", tk.DoubleVar)
            make_combo("유형", "truck_type", [("cargo", "카고"), ("tanker", "탱크로리"), ("flatbed", "평판")])

        elif item_type == "transport":
            make_entry("시작 X", "x1", tk.DoubleVar)
            make_entry("시작 Z", "z1", tk.DoubleVar)
            make_entry("끝 X", "x2", tk.DoubleVar)
            make_entry("끝 Z", "z2", tk.DoubleVar)
            make_entry("높이", "height", tk.DoubleVar)
            make_combo("유형", "transport_type", [("conveyor", "컨베이어"), ("lifter", "리프터"), ("rail", "레일")])

        # 빠른 색상 팔레트
        ttk.Separator(self.props_frame).pack(fill=tk.X, pady=8)
        ttk.Label(self.props_frame, text="빠른 색상:").pack(anchor="w")
        color_frame = ttk.Frame(self.props_frame)
        color_frame.pack(fill=tk.X, pady=3)
        for i, c in enumerate(self.PRESET_COLORS):
            btn = tk.Button(color_frame, bg=c, width=2, height=1,
                            command=lambda color=c: self._set_item_color(item, color))
            btn.grid(row=i // 8, column=i % 8, padx=1, pady=1)

        # 삭제 버튼
        ttk.Separator(self.props_frame).pack(fill=tk.X, pady=8)
        ttk.Button(self.props_frame, text="🗑️ 이 오브젝트 삭제",
                   command=self.delete_selected).pack(fill=tk.X)
        ttk.Button(self.props_frame, text="📋 복제",
                   command=self.duplicate_selected).pack(fill=tk.X, pady=(4, 0))

    def _pick_item_color(self, item, attr, btn):
        color = colorchooser.askcolor(getattr(item, attr))[1]
        if color:
            setattr(item, attr, color)
            btn.config(bg=color)
            self.redraw_canvas()

    def _set_item_color(self, item, color):
        item.color = color
        self._show_item_properties(item, self.selected_type)
        self.redraw_canvas()

    # ========================
    # 오브젝트 조작
    # ========================
    def add_object(self, obj_type):
        if obj_type == "building":
            obj = Building(name=f"건물 {len(self.buildings) + 1}")
            self.buildings.append(obj)
        elif obj_type == "road":
            obj = Road(name=f"도로 {len(self.roads) + 1}")
            self.roads.append(obj)
        elif obj_type == "tree":
            obj = Tree(name=f"나무 {len(self.trees) + 1}")
            self.trees.append(obj)
        elif obj_type == "parking":
            obj = ParkingLot(name=f"주차장 {len(self.parking_lots) + 1}")
            self.parking_lots.append(obj)
        elif obj_type == "lake":
            obj = Lake(name=f"호수 {len(self.lakes) + 1}")
            self.lakes.append(obj)
        elif obj_type == "person":
            obj = Person(name=f"사람 {len(self.persons) + 1}")
            self.persons.append(obj)
        elif obj_type == "gate":
            obj = Gate(name=f"게이트 {len(self.gates) + 1}")
            self.gates.append(obj)
        else:
            return

        self.selected_item = obj
        self.selected_type = obj_type
        self._update_object_tree()
        self._update_counts()
        self.redraw_canvas()
        self._show_item_properties(obj, obj_type)

    def add_random_trees(self):
        import random
        for _ in range(10):
            t = Tree(
                name=f"나무 {len(self.trees) + 1}",
                x=random.randint(-200, 200),
                z=random.randint(-200, 200),
                canopy_radius=random.uniform(2, 5),
                trunk_height=random.uniform(3, 6),
                color=random.choice(["#228B22", "#2E8B57", "#006400", "#32CD32", "#6B8E23"])
            )
            self.trees.append(t)
        self._update_object_tree()
        self._update_counts()
        self.redraw_canvas()

    def delete_selected(self):
        if not self.selected_item:
            return
        item = self.selected_item
        all_lists = [
            self.buildings, self.roads, self.trees, self.parking_lots,
            self.lakes, self.persons, self.gates,
            self.water_tanks, self.lpg_tanks, self.chimneys,
            self.walls, self.trucks, self.transport_lines,
        ]
        for lst in all_lists:
            if item in lst:
                lst.remove(item)
                break

        self.selected_item = None
        self.selected_type = None
        self._update_object_tree()
        self._update_counts()
        self.redraw_canvas()
        self._show_project_properties()

    def duplicate_selected(self):
        if not self.selected_item:
            return
        import copy
        item = copy.deepcopy(self.selected_item)
        item.id = str(uuid.uuid4())[:8]
        item.x += 20
        item.z += 20
        item.name = item.name + " (복사)"

        type_list_map = {
            Building: self.buildings, Road: self.roads, Tree: self.trees,
            ParkingLot: self.parking_lots, Lake: self.lakes, Person: self.persons,
            Gate: self.gates, WaterTank: self.water_tanks, LPGTank: self.lpg_tanks,
            Chimney: self.chimneys, Wall: self.walls, Truck: self.trucks,
            TransportLine: self.transport_lines,
        }
        for cls, lst in type_list_map.items():
            if isinstance(item, cls):
                lst.append(item)
                break

        self.selected_item = item
        self._update_object_tree()
        self._update_counts()
        self.redraw_canvas()
        self._show_item_properties(item, self.selected_type)

    def deselect_all(self):
        self.selected_item = None
        self.selected_type = None
        self.selected_items = []
        self.is_box_selecting = False
        self.selection_rect_start = None
        self.redraw_canvas()
        self._show_project_properties()

    # ========================
    # 파일 I/O
    # ========================
    def _to_project_data(self):
        self.project.name = self.name_var.get()
        self.project.buildings = [asdict(b) for b in self.buildings]
        self.project.roads = [asdict(r) for r in self.roads]
        self.project.trees = [asdict(t) for t in self.trees]
        self.project.parking_lots = [asdict(p) for p in self.parking_lots]
        self.project.lakes = [asdict(l) for l in self.lakes]
        self.project.persons = [asdict(p) for p in self.persons]
        self.project.gates = [asdict(g) for g in self.gates]
        self.project.water_tanks = [asdict(w) for w in self.water_tanks]
        self.project.lpg_tanks = [asdict(l) for l in self.lpg_tanks]
        self.project.chimneys = [asdict(c) for c in self.chimneys]
        self.project.walls = [asdict(w) for w in self.walls]
        self.project.trucks = [asdict(t) for t in self.trucks]
        self.project.transport_lines = [asdict(tl) for tl in self.transport_lines]
        return asdict(self.project)

    def _from_project_data(self, data):
        # Backward compatibility: old projects had ground_size
        if 'ground_size' in data and 'ground_width' not in data:
            data['ground_width'] = data['ground_size']
            data['ground_depth'] = data['ground_size']

        self.project = CampusProject(**{k: v for k, v in data.items()
                                        if k in CampusProject.__dataclass_fields__})
        self.name_var.set(self.project.name)
        self.buildings = [Building(**b) for b in self.project.buildings]
        self.roads = [Road(**r) for r in self.project.roads]
        self.trees = [Tree(**t) for t in self.project.trees]
        self.parking_lots = [ParkingLot(**p) for p in self.project.parking_lots]
        self.lakes = [Lake(**l) for l in self.project.lakes]
        self.persons = [Person(**{k: v for k, v in p.items() if k in Person.__dataclass_fields__})
                        for p in (self.project.persons if hasattr(self.project, 'persons') and self.project.persons else [])]
        self.gates = [Gate(**{k: v for k, v in g.items() if k in Gate.__dataclass_fields__})
                      for g in (self.project.gates if hasattr(self.project, 'gates') and self.project.gates else [])]
        self.water_tanks = [WaterTank(**{k: v for k, v in w.items() if k in WaterTank.__dataclass_fields__})
                            for w in (self.project.water_tanks if hasattr(self.project, 'water_tanks') and self.project.water_tanks else [])]
        self.lpg_tanks = [LPGTank(**{k: v for k, v in l.items() if k in LPGTank.__dataclass_fields__})
                          for l in (self.project.lpg_tanks if hasattr(self.project, 'lpg_tanks') and self.project.lpg_tanks else [])]
        self.chimneys = [Chimney(**{k: v for k, v in c.items() if k in Chimney.__dataclass_fields__})
                         for c in (self.project.chimneys if hasattr(self.project, 'chimneys') and self.project.chimneys else [])]
        self.walls = [Wall(**{k: v for k, v in w.items() if k in Wall.__dataclass_fields__})
                      for w in (self.project.walls if hasattr(self.project, 'walls') and self.project.walls else [])]
        self.trucks = [Truck(**{k: v for k, v in t.items() if k in Truck.__dataclass_fields__})
                       for t in (self.project.trucks if hasattr(self.project, 'trucks') and self.project.trucks else [])]
        self.transport_lines = [TransportLine(**{k: v for k, v in tl.items() if k in TransportLine.__dataclass_fields__})
                                for tl in (self.project.transport_lines if hasattr(self.project, 'transport_lines') and self.project.transport_lines else [])]

    def new_project(self):
        if messagebox.askyesno("새 프로젝트", "현재 프로젝트를 초기화할까요?"):
            self.buildings = []
            self.roads = []
            self.trees = []
            self.parking_lots = []
            self.lakes = []
            self.persons = []
            self.gates = []
            self.water_tanks = []
            self.lpg_tanks = []
            self.chimneys = []
            self.walls = []
            self.trucks = []
            self.transport_lines = []
            self.project = CampusProject()
            self.name_var.set(self.project.name)
            self.current_file = None
            self.selected_item = None
            self._update_object_tree()
            self._update_counts()
            self.redraw_canvas()
            self._show_project_properties()

    def open_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("Campus Project", "*.json"), ("All Files", "*.*")],
            title="프로젝트 열기"
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._from_project_data(data)
            self.current_file = path
            self._update_object_tree()
            self._update_counts()
            self.redraw_canvas()
            self._show_project_properties()
            self.update_status(f"프로젝트 로드: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("오류", f"프로젝트를 열 수 없습니다:\n{e}")

    def save_project(self):
        if self.current_file:
            self._save_to(self.current_file)
        else:
            self.save_project_as()

    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Campus Project", "*.json")],
            title="프로젝트 저장",
            initialfile=f"{self.project.name}.json"
        )
        if not path:
            return
        self._save_to(path)

    def _save_to(self, path):
        try:
            data = self._to_project_data()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.current_file = path
            self.update_status(f"저장 완료: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("오류", f"저장할 수 없습니다:\n{e}")

    def export_html(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML File", "*.html")],
            title="HTML 내보내기",
            initialfile=f"{self.project.name.replace(' ', '_')}.html"
        )
        if not path:
            return
        self._export_html_to(path)

    def export_and_open(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML File", "*.html")],
            title="HTML 내보내기 & 열기",
            initialfile=f"{self.project.name.replace(' ', '_')}.html"
        )
        if not path:
            return
        self._export_html_to(path)
        webbrowser.open('file://' + os.path.abspath(path))

    def _export_html_to(self, path):
        try:
            data = self._to_project_data()
            proj = CampusProject(**{k: v for k, v in data.items()
                                    if k in CampusProject.__dataclass_fields__})
            html = generate_html(proj)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.update_status(f"HTML 내보내기 완료: {os.path.basename(path)}")
            messagebox.showinfo("완료", f"HTML 파일이 생성되었습니다:\n{path}")
        except Exception as e:
            messagebox.showerror("오류", f"HTML 내보내기 실패:\n{e}")

    def export_react(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".jsx",
            filetypes=[("React JSX", "*.jsx")],
            title="React 내보내기",
            initialfile=f"{self.project.name.replace(' ', '_')}_3D.jsx"
        )
        if not path:
            return
        try:
            data = self._to_project_data()
            jsx = self._generate_react(data)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(jsx)
            self.update_status(f"React 내보내기 완료: {os.path.basename(path)}")
            messagebox.showinfo("완료", f"React 파일이 생성되었습니다:\n{path}")
        except Exception as e:
            messagebox.showerror("오류", f"React 내보내기 실패:\n{e}")

    def _generate_react(self, data):
        buildings_json = json.dumps(data.get('buildings', []), ensure_ascii=False, indent=2)
        roads_json = json.dumps(data.get('roads', []), ensure_ascii=False, indent=2)
        trees_json = json.dumps(data.get('trees', []), ensure_ascii=False, indent=2)
        persons_json = json.dumps(data.get('persons', []), ensure_ascii=False, indent=2)
        water_tanks_json = json.dumps(data.get('water_tanks', []), ensure_ascii=False, indent=2)
        lpg_tanks_json = json.dumps(data.get('lpg_tanks', []), ensure_ascii=False, indent=2)
        chimneys_json = json.dumps(data.get('chimneys', []), ensure_ascii=False, indent=2)
        walls_json = json.dumps(data.get('walls', []), ensure_ascii=False, indent=2)
        trucks_json = json.dumps(data.get('trucks', []), ensure_ascii=False, indent=2)
        gates_json = json.dumps(data.get('gates', []), ensure_ascii=False, indent=2)
        name = data.get('name', 'Campus')
        return f'''import React, {{ useRef, useMemo }} from 'react';
import {{ Canvas, useFrame }} from '@react-three/fiber';
import {{ OrbitControls, Sky, Text }} from '@react-three/drei';
import * as THREE from 'three';

// === Data ===
const buildingsData = {buildings_json};
const roadsData = {roads_json};
const treesData = {trees_json};
const personsData = {persons_json};
const waterTanksData = {water_tanks_json};
const lpgTanksData = {lpg_tanks_json};
const chimneysData = {chimneys_json};
const wallsData = {walls_json};
const trucksData = {trucks_json};
const gatesData = {gates_json};

function Building({{ data }}) {{
  return (
    <group position={{[data.x, 0, data.z]}}>
      <mesh position={{[0, data.height / 2, 0]}} castShadow receiveShadow>
        <boxGeometry args={{[data.width, data.height, data.depth]}} />
        <meshStandardMaterial color={{data.color}} roughness={{0.45}} metalness={{0.15}} />
      </mesh>
      <Text position={{[0, data.height + 3, 0]}} fontSize={{3}} color="white" anchorX="center" anchorY="bottom">
        {{data.name}}
      </Text>
    </group>
  );
}}

function Road({{ data }}) {{
  return (
    <mesh position={{[data.x, 0.15, data.z]}} rotation={{[-Math.PI / 2, 0, (data.rotation || 0) * Math.PI / 180]}}>
      <planeGeometry args={{[data.length, data.width]}} />
      <meshStandardMaterial color={{data.color || "#555555"}} roughness={{0.95}} />
    </mesh>
  );
}}

function TreeObj({{ data }}) {{
  return (
    <group position={{[data.x, 0, data.z]}}>
      <mesh position={{[0, data.trunk_height / 2, 0]}} castShadow>
        <cylinderGeometry args={{[0.3, 0.5, data.trunk_height, 8]}} />
        <meshStandardMaterial color="#8B4513" />
      </mesh>
      <mesh position={{[0, data.trunk_height + data.canopy_radius * 0.7, 0]}} castShadow>
        <sphereGeometry args={{[data.canopy_radius, 8, 8]}} />
        <meshStandardMaterial color={{data.color}} />
      </mesh>
    </group>
  );
}}

function Ground() {{
  return (
    <mesh rotation={{[-Math.PI / 2, 0, 0]}} receiveShadow>
      <planeGeometry args={{[2400, 2400]}} />
      <meshStandardMaterial color="#1a2030" roughness={{0.9}} />
    </mesh>
  );
}}

export default function {name.replace(' ', '').replace('-', '')}Scene() {{
  return (
    <div style={{{{ width: '100vw', height: '100vh' }}}}>
      <Canvas shadows camera={{{{ position: [200, 150, 200], fov: 45 }}}}>
        <ambientLight intensity={{1.2}} />
        <hemisphereLight args={{['#aaccee', '#556677', 1.0]}} />
        <directionalLight
          position={{[150, 200, 100]}}
          intensity={{1.6}}
          castShadow
          shadow-mapSize-width={{2048}}
          shadow-mapSize-height={{2048}}
        />
        <Sky sunPosition={{[100, 200, 100]}} />
        <Ground />
        {{buildingsData.map((b, i) => <Building key={{i}} data={{b}} />)}}
        {{roadsData.map((r, i) => <Road key={{i}} data={{r}} />)}}
        {{treesData.map((t, i) => <TreeObj key={{i}} data={{t}} />)}}
        <OrbitControls maxDistance={{1000}} />
        <gridHelper args={{[2400, 60, '#555555', '#333333']}} />
      </Canvas>
    </div>
  );
}}
'''

    def export_obj(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All Files", "*.*")],
            title="Blender OBJ 내보내기",
            initialfile=f"{self.project.name.replace(' ', '_')}.obj"
        )
        if not path:
            return
        try:
            data = self._to_project_data()
            obj_content, mtl_content = self._generate_obj(data)

            # OBJ 파일 저장
            with open(path, 'w', encoding='utf-8') as f:
                f.write(obj_content)

            # MTL 파일 저장
            mtl_path = path.rsplit('.', 1)[0] + '.mtl'
            with open(mtl_path, 'w', encoding='utf-8') as f:
                f.write(mtl_content)

            self.update_status(f"OBJ 내보내기 완료: {os.path.basename(path)}")
            messagebox.showinfo("완료",
                f"OBJ + MTL 파일이 생성되었습니다:\n{path}\n{mtl_path}\n\n"
                f"Blender에서 File > Import > Wavefront (.obj)로 불러오세요.")
        except Exception as e:
            messagebox.showerror("오류", f"OBJ 내보내기 실패:\n{e}")

    def _generate_obj(self, data):
        """Generate Wavefront OBJ + MTL files from project data"""
        vertices = []
        faces = []
        materials = {}
        mtl_lines = ['# SK Hynix Campus Material File', '# Generated by 3D Campus Builder', '']
        obj_lines = ['# SK Hynix 3D Campus', '# Generated by 3D Campus Builder', '']

        mtl_name = data.get('name', 'campus').replace(' ', '_') + '.mtl'
        obj_lines.append(f'mtllib {os.path.basename(mtl_name)}')
        obj_lines.append('')

        vertex_offset = 0

        def add_material(name, hex_color, roughness=0.5):
            if name in materials:
                return
            materials[name] = hex_color
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16) / 255
            g = int(hex_color[2:4], 16) / 255
            b = int(hex_color[4:6], 16) / 255
            mtl_lines.append(f'newmtl {name}')
            mtl_lines.append(f'Kd {r:.4f} {g:.4f} {b:.4f}')
            mtl_lines.append(f'Ka {r*0.2:.4f} {g*0.2:.4f} {b*0.2:.4f}')
            mtl_lines.append(f'Ks 0.1 0.1 0.1')
            mtl_lines.append(f'Ns {(1-roughness)*100:.0f}')
            mtl_lines.append(f'd 1.0')
            mtl_lines.append('')

        def add_box(name, x, y, z, w, h, d, color, roughness=0.5):
            nonlocal vertex_offset
            mat_name = name.replace(' ', '_').replace('/', '_')
            add_material(mat_name, color, roughness)
            obj_lines.append(f'o {mat_name}')
            obj_lines.append(f'usemtl {mat_name}')

            hw, hh, hd = w/2, h/2, d/2
            # 8 vertices of a box
            verts = [
                (x-hw, y-hh, z-hd), (x+hw, y-hh, z-hd),
                (x+hw, y+hh, z-hd), (x-hw, y+hh, z-hd),
                (x-hw, y-hh, z+hd), (x+hw, y-hh, z+hd),
                (x+hw, y+hh, z+hd), (x-hw, y+hh, z+hd),
            ]
            for v in verts:
                obj_lines.append(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')

            # 6 faces (quads)
            vo = vertex_offset + 1
            obj_lines.append(f'f {vo} {vo+1} {vo+2} {vo+3}')  # front
            obj_lines.append(f'f {vo+4} {vo+7} {vo+6} {vo+5}')  # back
            obj_lines.append(f'f {vo} {vo+3} {vo+7} {vo+4}')  # left
            obj_lines.append(f'f {vo+1} {vo+5} {vo+6} {vo+2}')  # right
            obj_lines.append(f'f {vo+3} {vo+2} {vo+6} {vo+7}')  # top
            obj_lines.append(f'f {vo} {vo+4} {vo+5} {vo+1}')  # bottom
            obj_lines.append('')
            vertex_offset += 8

        def add_cylinder(name, x, y, z, radius, height, color, segments=16, roughness=0.5):
            nonlocal vertex_offset
            mat_name = name.replace(' ', '_').replace('/', '_')
            add_material(mat_name, color, roughness)
            obj_lines.append(f'o {mat_name}')
            obj_lines.append(f'usemtl {mat_name}')

            import math
            # Bottom and top circle vertices
            for dy in [0, height]:
                for i in range(segments):
                    angle = 2 * math.pi * i / segments
                    vx = x + radius * math.cos(angle)
                    vz = z + radius * math.sin(angle)
                    obj_lines.append(f'v {vx:.4f} {y + dy:.4f} {vz:.4f}')
            # Center vertices for caps
            obj_lines.append(f'v {x:.4f} {y:.4f} {z:.4f}')
            obj_lines.append(f'v {x:.4f} {y + height:.4f} {z:.4f}')

            vo = vertex_offset + 1
            # Side faces
            for i in range(segments):
                n = (i + 1) % segments
                obj_lines.append(f'f {vo+i} {vo+n} {vo+segments+n} {vo+segments+i}')
            # Bottom cap
            bot_center = vo + segments * 2
            for i in range(segments):
                n = (i + 1) % segments
                obj_lines.append(f'f {bot_center} {vo+n} {vo+i}')
            # Top cap
            top_center = vo + segments * 2 + 1
            for i in range(segments):
                n = (i + 1) % segments
                obj_lines.append(f'f {top_center} {vo+segments+i} {vo+segments+n}')
            obj_lines.append('')
            vertex_offset += segments * 2 + 2

        # Ground plane
        add_box('Ground', 0, -0.25, 0,
                data.get('ground_width', 600) * 2, 0.5, data.get('ground_depth', 600) * 2,
                data.get('ground_color', '#1a2030'))

        # Buildings
        for b in data.get('buildings', []):
            btype = b.get('building_type', 'office')
            if btype in ('water_tank', 'cooling_tower'):
                r = min(b['width'], b['depth']) / 2
                add_cylinder(b.get('name', 'Tank'), b['x'], 0, b['z'], r, b['height'], b.get('color', '#cccccc'))
            else:
                add_box(b.get('name', 'Building'), b['x'], b['height']/2, b['z'],
                        b['width'], b['height'], b['depth'], b.get('color', '#4488cc'))

        # Roads
        for i, r in enumerate(data.get('roads', [])):
            add_box(r.get('name', f'Road_{i}'), r['x'], 0.1, r['z'],
                    r['length'], 0.2, r['width'], r.get('color', '#555555'))

        # Water tanks
        for wt in data.get('water_tanks', []):
            add_cylinder(wt.get('name', 'WaterTank'), wt['x'], 0, wt['z'],
                        wt.get('radius', 12), wt.get('height', 28), wt.get('color', '#cccccc'))

        # LPG tanks
        for lt in data.get('lpg_tanks', []):
            add_cylinder(lt.get('name', 'LPGTank'), lt['x'], 3, lt['z'],
                        lt.get('radius', 7), lt.get('length', 20), lt.get('color', '#ffffff'))

        # Chimneys
        for ch in data.get('chimneys', []):
            add_cylinder(ch.get('name', 'Chimney'), ch['x'], 0, ch['z'],
                        ch.get('radius', 5), ch.get('height', 80), ch.get('color', '#666666'))

        # Walls
        for w in data.get('walls', []):
            add_box(w.get('name', 'Wall'), w['x'], w.get('height', 12)/2, w['z'],
                    w.get('length', 40), w.get('height', 12), w.get('thickness', 2),
                    w.get('color', '#888888'))

        # Trucks (static representation)
        for t in data.get('trucks', []):
            add_box(t.get('name', 'Truck'), t['x'], 3, t['z'],
                    18, 6, 6, t.get('color', '#ffffff'))

        # Trees (simplified as boxes)
        for t in data.get('trees', []):
            # Trunk
            add_box(t.get('name', 'TreeTrunk'), t['x'], t.get('trunk_height', 3)/2, t['z'],
                    0.6, t.get('trunk_height', 3), 0.6, '#8B4513')
            # Canopy
            cr = t.get('canopy_radius', 3)
            add_cylinder(t.get('name', 'TreeCanopy'), t['x'], t.get('trunk_height', 3), t['z'],
                        cr, cr * 1.4, t.get('color', '#228833'), 8)

        # Gates
        for g in data.get('gates', []):
            gw = g.get('width', 8)
            gh = g.get('height', 6)
            gd = g.get('depth', 2)
            # Left pillar
            add_box(g.get('name', 'Gate') + '_L', g['x'] - gw/2, gh/2, g['z'],
                    1.5, gh, gd, g.get('color', '#aa8855'))
            # Right pillar
            add_box(g.get('name', 'Gate') + '_R', g['x'] + gw/2, gh/2, g['z'],
                    1.5, gh, gd, g.get('color', '#aa8855'))
            # Top bar
            add_box(g.get('name', 'Gate') + '_Top', g['x'], gh, g['z'],
                    gw + 1.5, 1.5, gd, g.get('color', '#aa8855'))

        # Persons (simplified as small boxes)
        for p in data.get('persons', []):
            add_box(p.get('name', 'Person'), p['x'], 2.5, p['z'],
                    1.5, 5, 1.5, p.get('shirt_color', '#3366cc'))

        return '\n'.join(obj_lines), '\n'.join(mtl_lines)

    # ========================
    # 예제 캠퍼스
    # ========================
    def _load_example_campus(self):
        """SK Hynix 이천캠퍼스 스타일 예제를 로드합니다."""
        import random
        random.seed(42)

        self.project.name = "SK Hynix 이천캠퍼스"
        self.project.ground_width = 1200
        self.project.ground_depth = 1200
        self.project.ground_color = "#141820"
        self.project.sky_color = "#0a0a12"
        self.name_var.set(self.project.name)

        # === 주요 FAB 건물 ===
        self.buildings = [
            Building(name="M14A/B", x=150, z=280, width=210, depth=160, height=96, color="#ecc94b",
                     floors=8, building_type="factory", description="1a/1b nm DRAM 생산, 최신 공정", roof_type="flat"),
            Building(name="M10A", x=160, z=520, width=200, depth=120, height=65, color="#63b3ed",
                     floors=5, building_type="factory", description="DRAM 생산", roof_type="flat"),
            Building(name="M10B/R3", x=500, z=300, width=140, depth=100, height=60, color="#63b3ed",
                     floors=5, building_type="factory", description="DRAM 생산", roof_type="flat"),
            Building(name="M10C", x=700, z=200, width=160, depth=100, height=60, color="#63b3ed",
                     floors=5, building_type="factory", description="DRAM 생산", roof_type="flat"),
            Building(name="M16A/B", x=530, z=520, width=230, depth=120, height=132, color="#48bb78",
                     floors=11, building_type="factory", description="AMHS 자동물류, 최신 DRAM 양산", roof_type="flat"),
            Building(name="DRAM WT", x=420, z=730, width=150, depth=70, height=45, color="#e07098",
                     floors=4, building_type="lab", description="웨이퍼 테스트동", roof_type="flat"),
            # P&T
            Building(name="P&T1", x=420, z=140, width=110, depth=80, height=35, color="#c4956a",
                     floors=3, building_type="factory", description="패키지 및 테스트", roof_type="flat"),
            Building(name="P&T4", x=830, z=320, width=140, depth=120, height=96, color="#c4956a",
                     floors=8, building_type="factory", description="패키지 및 테스트 (대형)", roof_type="flat"),
            Building(name="P&T5", x=780, z=660, width=130, depth=90, height=48, color="#c4956a",
                     floors=4, building_type="factory", description="패키지 및 테스트", roof_type="flat"),
            # 부속건물
            Building(name="에너지센터", x=-350, z=340, width=50, depth=38, height=38, color="#889988",
                     floors=3, building_type="warehouse", description="", roof_type="flat"),
            Building(name="청운기숙사", x=-355, z=-110, width=55, depth=42, height=52, color="#8899aa",
                     floors=8, building_type="office", description="", roof_type="flat"),
            Building(name="안내센터", x=-230, z=465, width=42, depth=32, height=30, color="#889988",
                     floors=2, building_type="office", description="", roof_type="flat"),
            Building(name="복지관", x=290, z=465, width=45, depth=34, height=35, color="#998888",
                     floors=3, building_type="gym", description="", roof_type="flat"),
            Building(name="물류센터", x=-348, z=170, width=45, depth=35, height=32, color="#889988",
                     floors=2, building_type="warehouse", description="", roof_type="flat"),
            Building(name="연구동", x=-345, z=50, width=48, depth=38, height=40, color="#99887a",
                     floors=4, building_type="lab", description="", roof_type="flat"),
            Building(name="체육관", x=-240, z=-120, width=45, depth=35, height=38, color="#887788",
                     floors=2, building_type="gym", description="", roof_type="dome"),
            Building(name="환경안전동", x=155, z=-120, width=42, depth=34, height=30, color="#888877",
                     floors=2, building_type="office", description="", roof_type="flat"),
        ]

        # === 도로 ===
        self.roads = [
            Road(name="동서도로 1", x=400, z=270, length=800, width=14, rotation=0, color="#333844"),
            Road(name="동서도로 2", x=400, z=470, length=800, width=14, rotation=0, color="#333844"),
            Road(name="동서도로 3", x=400, z=680, length=800, width=14, rotation=0, color="#333844"),
            Road(name="남북도로 1", x=130, z=400, length=600, width=12, rotation=90, color="#333844"),
            Road(name="남북도로 2", x=400, z=400, length=600, width=12, rotation=90, color="#333844"),
            Road(name="남북도로 3", x=680, z=400, length=600, width=12, rotation=90, color="#333844"),
            Road(name="정문 진입로", x=20, z=420, length=80, width=14, rotation=90, color="#333844"),
        ]

        # === 주차장 ===
        self.parking_lots = [
            ParkingLot(name="주차장 A", x=-150, z=300, width=60, depth=40, color="#333844"),
            ParkingLot(name="주차장 B", x=-150, z=150, width=50, depth=35, color="#333844"),
        ]

        self.lakes = []

        # === 나무 ===
        self.trees = []
        tree_cluster_areas = [
            (-230, 340, 15), (-230, 400, 12), (-180, 200, 10), (-180, 100, 10),
            (-60, -40, 8), (60, -60, 8), (-100, 450, 8), (150, 450, 8),
        ]
        for bx, bz, count in tree_cluster_areas:
            for _ in range(count):
                self.trees.append(Tree(
                    name=f"나무 {len(self.trees) + 1}",
                    x=bx + random.randint(-30, 30),
                    z=bz + random.randint(-30, 30),
                    trunk_height=random.uniform(3, 6),
                    canopy_radius=random.uniform(2.5, 5),
                    color=random.choice(["#1e5a18", "#2d6a22", "#1a5010", "#2a6830"])
                ))

        # === 보행자 ===
        shirt_colors = ["#3366cc", "#cc3333", "#33aa33", "#cc8833", "#8833cc", "#33cccc", "#cc6699", "#669933"]
        pants_cols = ["#333333", "#444466", "#554433", "#2a2a3a"]
        person_spots = [
            (100, 280), (200, 350), (400, 400), (500, 450), (150, 450),
            (250, 550), (350, 600), (550, 300), (650, 550), (750, 500),
            (-50, 350), (-100, 250), (50, 500), (100, 650), (20, 400),
            (300, 150), (600, 700), (800, 400), (-200, 400), (700, 350),
        ]
        self.persons = []
        for i, (px, pz) in enumerate(person_spots):
            self.persons.append(Person(
                name=f"직원 {i + 1}",
                x=px + random.randint(-10, 10),
                z=pz + random.randint(-10, 10),
                speed=random.uniform(1.0, 3.0),
                direction=random.randint(0, 360),
                shirt_color=random.choice(shirt_colors),
                pants_color=random.choice(pants_cols),
                walk_radius=random.uniform(15, 40),
            ))

        # === 정문/게이트 ===
        self.gates = [
            Gate(name="정문", x=20, z=375, width=30, height=14, depth=3,
                 color="#aa8855", gate_type="main", has_barrier=True),
            Gate(name="후문", x=400, z=850, width=20, height=8, depth=3,
                 color="#887766", gate_type="side", has_barrier=True),
        ]

        self._update_object_tree()
        self._update_counts()

    # ========================
    # 도움말
    # ========================
    def show_help(self):
        help_text = """
3D Campus Builder v1.0 사용법
=============================

[캔버스 조작]
• 좌클릭: 오브젝트 선택 / 도구로 배치
• 좌클릭 드래그: 선택된 오브젝트 이동
• 우클릭 드래그: 캔버스 팬 (이동)
• 마우스 휠: 줌 인/아웃

[도구바]
• 선택: 오브젝트 선택 및 이동
• 건물/도로/나무/주차장/호수: 클릭하여 배치

[속성 패널]
• 선택된 오브젝트의 속성을 편집합니다
• 빠른 색상 팔레트로 빠르게 색상 변경

[파일]
• Ctrl+S: 프로젝트 저장 (JSON)
• Ctrl+O: 프로젝트 열기
• Ctrl+E: HTML 내보내기

[HTML 3D 뷰어 조작]
• 좌클릭 드래그: 궤도 회전
• 우클릭 드래그: 팬
• 마우스 휠: 줌
• WASD / 화살표: 카메라 이동
• N: 낮/밤 전환
• L: 라벨 토글
• G: 그리드 토글
"""
        messagebox.showinfo("사용법", help_text)

    def show_about(self):
        messagebox.showinfo("정보",
                            "3D Campus Builder v1.0\n\n"
                            "SK Hynix 스타일 3D 캠퍼스 시각화\n"
                            "HTML 생성 프로그램\n\n"
                            "Three.js 기반 인터랙티브 3D 뷰어를 생성합니다.")


# ============================================================
# 메인 엔트리포인트
# ============================================================

def main():
    root = tk.Tk()

    # 스타일 설정
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = CampusBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

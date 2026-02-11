"""
OHT FAB Layout 3D Simulation Server (FastAPI)
Port: 10003

기능:
  - layout_data.json 서빙 (3D 시각화 데이터)
  - 마스터 데이터 CSV 생성/다운로드
  - OHT 시뮬레이션 API (차량 상태, 통계)
  - layout.xml 파싱 (새 FAB 데이터 로드)
  - 웹 UI (oht_3d_layout.html) 서빙

실행: python server.py
또는: uvicorn server:app --host 0.0.0.0 --port 10003 --reload
"""
import os
import sys
import csv
import json
import time
import random
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ===== Configuration =====
BASE_DIR = Path(__file__).parent
LAYOUT_JSON = BASE_DIR / "layout_data.json"
FAB_DATA_DIR = BASE_DIR / "fab_data"
FAB_SETTINGS_FILE = FAB_DATA_DIR / "_fab_settings.json"
MASTER_CSV_DIR = BASE_DIR / "master_csv"
HTML_FILE = BASE_DIR / "oht_3d_layout.html"
CAMPUS_FILE = BASE_DIR / "SK_Hynix_3D_Campus_0.4V.HTML"
DEFAULT_FAB = "M14-Pro"
PORT = 10003
OUTPUT_DIR = BASE_DIR / "output"
CSV_SAVE_INTERVAL = 10       # 10초마다 CSV 저장
OUTPUT_CLEANUP_INTERVAL = 600  # 10분(600초)마다 OUTPUT 파일 삭제

# ===== App Setup =====
app = FastAPI(
    title="AMOS MAP System PRO - OHT FAB Simulator",
    description="OHT 반도체 FAB 레이아웃 3D 시뮬레이션 서버",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Global State =====
layout_data: Optional[Dict] = None
current_fab_name: Optional[str] = None      # 현재 선택된 FAB
current_fab_json: Optional[str] = None      # 현재 FAB JSON 파일 경로
fab_data_registry: Dict[str, dict] = {}     # fab_name -> { json_path, total_nodes, ... }
simulation_state = {
    "running": False,
    "vehicles": [],
    "stats": {
        "total": 0, "running": 0, "loaded": 0, "stopped": 0, "jam": 0,
        "total_in": 0, "total_out": 0,
        "avg_speed": 0, "max_speed": 0,
    },
    "scenario_active": False,
    "start_time": None,
}
fab_registry: Dict[str, dict] = {}  # fab_name -> { json_path, csv_dir, loaded_at }
vehicle_buffer: List[dict] = []  # 샘플 데이터 버퍼
is_running = True


# ===== Models =====
class ParseRequest(BaseModel):
    xml_path: str
    fab_name: str = "M14-Pro"

class SimulationConfig(BaseModel):
    vehicle_count: int = 35
    speed_factor: float = 1.0

class ScenarioConfig(BaseModel):
    jam_rate: float = 0.02
    stop_rate: float = 0.05
    loaded_rate: float = 0.3

class FabSwitchRequest(BaseModel):
    fab_name: str

class MapScanRequest(BaseModel):
    map_dir: str
    output_dir: str = ""

class FabSettingsUpdate(BaseModel):
    oht_count: Optional[int] = None


# ===== Helper: FAB Settings (per-FAB OHT count persistence) =====
def load_fab_settings() -> dict:
    """Load per-FAB settings from _fab_settings.json"""
    if FAB_SETTINGS_FILE.exists():
        try:
            with open(FAB_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load fab settings: {e}")
    return {}


def save_fab_settings(settings: dict):
    """Save per-FAB settings to _fab_settings.json"""
    FAB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(FAB_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to save fab settings: {e}")


# ===== Helper: Load Layout Data =====
def load_layout_data():
    global layout_data
    if LAYOUT_JSON.exists():
        with open(LAYOUT_JSON, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        print(f"[Server] Layout data loaded: {layout_data.get('total_nodes', 0)} nodes, {layout_data.get('total_edges', 0)} edges")
        # Register in fab_registry
        fab_name = layout_data.get('fab_name', DEFAULT_FAB)
        fab_registry[fab_name] = {
            'json_path': str(LAYOUT_JSON),
            'csv_dir': str(MASTER_CSV_DIR),
            'loaded_at': datetime.now().isoformat(),
            'total_nodes': layout_data.get('total_nodes', 0),
            'total_edges': layout_data.get('total_edges', 0),
            'total_stations': layout_data.get('total_stations', 0),
            'total_mcp_zones': layout_data.get('total_mcp_zones', 0),
        }
    else:
        print(f"[Server] Warning: {LAYOUT_JSON} not found")


# ===== Load FAB Data Registry (multi-FAB support) =====
def load_fab_registry():
    """
    Scan fab_data/ directory for available FAB data.

    Expected structure (FAB별 독립 디렉토리):
        fab_data/
            _fab_registry.json
            M14A/
                M14A.json
                master_csv/M14A_HID_Zone_Master.csv
            M14B/
                M14B.json
                master_csv/M14B_HID_Zone_Master.csv
            ...
    Also supports legacy flat structure (fab_data/*.json).
    """
    global fab_data_registry
    if not FAB_DATA_DIR.exists():
        return

    # 1) FAB별 하위 디렉토리 탐색: fab_data/{fab_name}/{fab_name}.json
    for d in sorted(FAB_DATA_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith('_'):
            continue
        fab_name = d.name
        json_file = d / f'{fab_name}.json'
        if not json_file.exists():
            # 디렉토리 안에 아무 .json이라도 있으면
            json_files = list(d.glob('*.json'))
            if json_files:
                json_file = json_files[0]
            else:
                continue
        csv_dir = d / 'master_csv'
        fab_data_registry[fab_name] = {
            'json_path': str(json_file),
            'csv_dir': str(csv_dir) if csv_dir.exists() else '',
            'json_size': json_file.stat().st_size,
            'total_nodes': 0,
            'total_edges': 0,
            'total_stations': 0,
            'total_mcp_zones': 0,
        }

    # 2) 레거시: fab_data/*.json (flat 구조)
    if not fab_data_registry:
        for f in sorted(FAB_DATA_DIR.glob('*.json')):
            if f.name.startswith('_'):
                continue
            fab_name = f.stem
            fab_data_registry[fab_name] = {
                'json_path': str(f),
                'csv_dir': str(FAB_DATA_DIR / 'master_csv'),
                'json_size': f.stat().st_size,
                'total_nodes': 0,
                'total_edges': 0,
                'total_stations': 0,
                'total_mcp_zones': 0,
            }

    # 3) _fab_registry.json 에서 메타데이터 보강
    registry_file = FAB_DATA_DIR / '_fab_registry.json'
    if registry_file.exists():
        try:
            with open(registry_file, 'r', encoding='utf-8') as f:
                reg = json.load(f)
            for entry in reg.get('fabs', []):
                name = entry.get('fab_name', '')
                if name in fab_data_registry:
                    fab_data_registry[name].update({
                        'total_nodes': entry.get('nodes', 0),
                        'total_edges': entry.get('edges', 0),
                        'total_stations': entry.get('stations', 0),
                        'total_mcp_zones': entry.get('mcp_zones', 0),
                    })
                    if entry.get('csv_dir'):
                        fab_data_registry[name]['csv_dir'] = entry['csv_dir']
        except Exception as e:
            print(f"[Warning] Failed to read fab registry: {e}")

    if fab_data_registry:
        print(f"[Server] Found {len(fab_data_registry)} FABs in fab_data/: {list(fab_data_registry.keys())}")


def switch_fab_internal(fab_name):
    """Switch to a different FAB (load its JSON data)"""
    global layout_data, current_fab_json, current_fab_name

    if fab_name in fab_data_registry:
        json_path = fab_data_registry[fab_name]['json_path']
    elif (FAB_DATA_DIR / f'{fab_name}.json').exists():
        json_path = str(FAB_DATA_DIR / f'{fab_name}.json')
    else:
        return False

    print(f"[Server] Switching to FAB: {fab_name} ({json_path})")
    with open(json_path, 'r', encoding='utf-8') as f:
        layout_data = json.load(f)

    current_fab_json = json_path
    current_fab_name = fab_name

    # Update registry metadata
    if fab_name in fab_data_registry:
        fab_data_registry[fab_name].update({
            'total_nodes': layout_data.get('total_nodes', 0),
            'total_edges': layout_data.get('total_edges', 0),
            'total_stations': layout_data.get('total_stations', 0),
            'total_mcp_zones': layout_data.get('total_mcp_zones', 0),
        })

    # Also register in fab_registry (legacy)
    # CSV 경로: FAB별 디렉토리 또는 기본 master_csv
    fab_csv_dir = fab_data_registry.get(fab_name, {}).get('csv_dir', '')
    if not fab_csv_dir:
        fab_csv_dir = str(Path(json_path).parent / 'master_csv')
    fab_registry[fab_name] = {
        'json_path': json_path,
        'csv_dir': fab_csv_dir,
        'loaded_at': datetime.now().isoformat(),
        'total_nodes': layout_data.get('total_nodes', 0),
        'total_edges': layout_data.get('total_edges', 0),
        'total_stations': layout_data.get('total_stations', 0),
        'total_mcp_zones': layout_data.get('total_mcp_zones', 0),
    }

    print(f"[Server] FAB {fab_name}: {layout_data.get('total_nodes', 0):,} nodes, {layout_data.get('total_edges', 0):,} edges")
    return True


# ===== Scan for existing CSV master data =====
def scan_master_data():
    """Scan master_csv directory for existing FAB data"""
    if MASTER_CSV_DIR.exists():
        for f in MASTER_CSV_DIR.glob('*_summary.csv'):
            fab_name = f.stem.replace('_summary', '')
            if fab_name not in fab_registry:
                fab_registry[fab_name] = {
                    'json_path': str(LAYOUT_JSON),
                    'csv_dir': str(MASTER_CSV_DIR),
                    'loaded_at': f.stat().st_mtime,
                }


# ===== Sample Data: Save & Cleanup =====
def record_vehicle_sample():
    """시뮬레이션 차량 데이터를 버퍼에 기록"""
    global vehicle_buffer
    if not simulation_state["vehicles"]:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    fab_name = layout_data.get("fab_name", DEFAULT_FAB) if layout_data else DEFAULT_FAB
    for v in simulation_state["vehicles"]:
        vehicle_buffer.append({
            'createTime': now,
            'fabId': fab_name,
            'vehicleId': v.get('id', ''),
            'state': v.get('state', 'unknown'),
            'speed': round(v.get('speed', 0), 2),
            'hasFoup': 1 if v.get('has_foup', False) else 0,
            'posX': round(v.get('position', {}).get('x', 0), 2),
            'posY': round(v.get('position', {}).get('y', 0), 2),
        })


def save_sample_csv():
    """버퍼 데이터를 CSV 파일로 저장"""
    global vehicle_buffer
    if not vehicle_buffer:
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    veh_file = OUTPUT_DIR / f'SAMPLE_VEHICLE_{timestamp}.csv'
    try:
        with open(veh_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=vehicle_buffer[0].keys())
            writer.writeheader()
            writer.writerows(vehicle_buffer)
        print(f"[샘플 저장] {veh_file.name} ({len(vehicle_buffer)} rows)")
    except Exception as e:
        print(f"[샘플 저장 오류] {e}")
    vehicle_buffer = []


def cleanup_output():
    """OUTPUT 디렉토리 정리 - 모든 파일 삭제"""
    if not OUTPUT_DIR.exists():
        return
    deleted_count = 0
    deleted_size = 0
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            try:
                fsize = f.stat().st_size
                f.unlink()
                deleted_count += 1
                deleted_size += fsize
            except Exception as e:
                print(f"[OUTPUT 정리 오류] {f.name}: {e}")
    if deleted_count > 0:
        size_mb = deleted_size / (1024 * 1024)
        print(f"[OUTPUT 정리] {deleted_count}개 파일 삭제됨 ({size_mb:.2f} MB 확보)")


async def csv_save_loop():
    """CSV 저장 루프 - 10초마다"""
    while is_running:
        await asyncio.sleep(CSV_SAVE_INTERVAL)
        record_vehicle_sample()
        save_sample_csv()


async def output_cleanup_loop():
    """OUTPUT 디렉토리 정리 루프 - 10분마다"""
    while is_running:
        await asyncio.sleep(OUTPUT_CLEANUP_INTERVAL)
        cleanup_output()


# ===== Startup =====
@app.on_event("startup")
async def startup():
    # 1. Load multi-FAB registry from fab_data/
    load_fab_registry()

    # 2. Auto-select first FAB if available
    if fab_data_registry:
        first_fab = next(iter(fab_data_registry))
        switch_fab_internal(first_fab)
    else:
        # Fallback to legacy single layout_data.json
        load_layout_data()

    scan_master_data()
    # 백그라운드 태스크 시작
    asyncio.create_task(csv_save_loop())
    asyncio.create_task(output_cleanup_loop())
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_fabs = list(fab_data_registry.keys()) or list(fab_registry.keys())
    print(f"\n{'='*60}")
    print(f"  AMOS MAP System PRO - OHT FAB Simulator")
    print(f"  Port: {PORT}")
    print(f"  URL: http://localhost:{PORT}")
    print(f"  FABs: {all_fabs}")
    if current_fab_name:
        print(f"  Current FAB: {current_fab_name}")
    print(f"  OUTPUT: {OUTPUT_DIR}")
    print(f"  CSV 저장 간격: {CSV_SAVE_INTERVAL}초 / 정리 간격: {OUTPUT_CLEANUP_INTERVAL}초")
    print(f"{'='*60}\n")


# ===== Routes: HTML Page =====
@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지 - SK Hynix 3D Campus"""
    if CAMPUS_FILE.exists():
        return HTMLResponse(content=CAMPUS_FILE.read_text(encoding='utf-8'))
    # fallback: 캠퍼스 파일 없으면 OHT 레이아웃
    if HTML_FILE.exists():
        return HTMLResponse(content=HTML_FILE.read_text(encoding='utf-8'))
    raise HTTPException(status_code=404, detail="HTML files not found")


@app.get("/oht", response_class=HTMLResponse)
async def oht_page():
    """OHT 3D 레이아웃 페이지"""
    if HTML_FILE.exists():
        return HTMLResponse(content=HTML_FILE.read_text(encoding='utf-8'))
    raise HTTPException(status_code=404, detail="oht_3d_layout.html not found")


@app.get("/oht_3d_layout.html", response_class=HTMLResponse)
async def layout_page():
    """OHT 3D 시각화 페이지 (직접 접근)"""
    if HTML_FILE.exists():
        return HTMLResponse(content=HTML_FILE.read_text(encoding='utf-8'))
    raise HTTPException(status_code=404, detail="oht_3d_layout.html not found")


# ===== Routes: Layout Data =====
@app.get("/layout_data.json")
async def get_layout_json():
    """레이아웃 JSON 데이터 (현재 선택된 FAB)"""
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    # Multi-FAB: serve current FAB's JSON
    if current_fab_json and Path(current_fab_json).exists():
        return FileResponse(current_fab_json, media_type="application/json", headers=headers)
    # Fallback: legacy layout_data.json
    if LAYOUT_JSON.exists():
        return FileResponse(LAYOUT_JSON, media_type="application/json", headers=headers)
    raise HTTPException(status_code=404, detail="No layout data. Parse XML first or check /api/fabs")


@app.get("/api/layout")
async def get_layout():
    """레이아웃 요약 정보"""
    if layout_data is None:
        raise HTTPException(status_code=404, detail="No layout data loaded")
    return {
        "fab_name": layout_data.get("fab_name", DEFAULT_FAB),
        "project": layout_data.get("project", ""),
        "total_nodes": layout_data.get("total_nodes", 0),
        "total_edges": layout_data.get("total_edges", 0),
        "total_stations": layout_data.get("total_stations", 0),
        "total_mcp_zones": layout_data.get("total_mcp_zones", 0),
        "bounds": layout_data.get("bounds", {}),
    }


@app.get("/api/layout/nodes")
async def get_nodes(limit: int = Query(1000, ge=0, le=50000), offset: int = Query(0, ge=0)):
    """노드 목록 (페이지네이션)"""
    if layout_data is None:
        raise HTTPException(status_code=404, detail="No layout data")
    nodes = layout_data.get("nodes", [])
    return {"total": len(nodes), "offset": offset, "limit": limit, "nodes": nodes[offset:offset+limit]}


@app.get("/api/layout/edges")
async def get_edges(limit: int = Query(5000, ge=0, le=50000), offset: int = Query(0, ge=0)):
    """엣지 목록"""
    if layout_data is None:
        raise HTTPException(status_code=404, detail="No layout data")
    edges = layout_data.get("edges", [])
    return {"total": len(edges), "offset": offset, "limit": limit, "edges": edges[offset:offset+limit]}


@app.get("/api/layout/stations")
async def get_stations():
    """스테이션 목록"""
    if layout_data is None:
        raise HTTPException(status_code=404, detail="No layout data")
    return {"total": len(layout_data.get("stations", [])), "stations": layout_data.get("stations", [])}


@app.get("/api/layout/mcp_zones")
async def get_mcp_zones():
    """MCP Zone 목록"""
    if layout_data is None:
        raise HTTPException(status_code=404, detail="No layout data")
    return {"total": len(layout_data.get("mcp_zones", [])), "mcp_zones": layout_data.get("mcp_zones", [])}


@app.get("/api/layout/search")
async def search_layout(q: str = Query(..., min_length=1)):
    """노드/스테이션 검색"""
    if layout_data is None:
        raise HTTPException(status_code=404, detail="No layout data")

    q_upper = q.upper()
    results = []

    for n in layout_data.get("nodes", []):
        if str(n["id"]) == q:
            results.append({"type": "node", "id": n["id"], "x": n["x"], "y": n["y"]})
        for s in n.get("stations", []):
            if q_upper in s["port_id"].upper():
                results.append({"type": "station", "port_id": s["port_id"],
                                "node_id": n["id"], "x": n["x"], "y": n["y"]})

    return {"query": q, "count": len(results), "results": results[:50]}


# ===== Routes: Multi-FAB (FAB 목록 / 전환 / 스캔) =====
@app.get("/api/fabs")
async def list_fabs():
    """등록된 FAB 목록 + 현재 선택된 FAB"""
    fabs = []
    for name, info in fab_data_registry.items():
        # CSV 파일 수 확인
        csv_dir = Path(info.get('csv_dir', ''))
        csv_count = len(list(csv_dir.glob('*.csv'))) if csv_dir.exists() else 0
        fabs.append({
            'fab_name': name,
            'total_nodes': info.get('total_nodes', 0),
            'total_edges': info.get('total_edges', 0),
            'total_stations': info.get('total_stations', 0),
            'total_mcp_zones': info.get('total_mcp_zones', 0),
            'json_size_mb': round(info.get('json_size', 0) / (1024 * 1024), 1),
            'csv_count': csv_count,
        })
    # Legacy single layout_data.json (no fab_data)
    if not fabs and layout_data:
        fabs.append({
            'fab_name': layout_data.get('fab_name', DEFAULT_FAB),
            'total_nodes': layout_data.get('total_nodes', 0),
            'total_edges': layout_data.get('total_edges', 0),
            'total_stations': layout_data.get('total_stations', 0),
            'total_mcp_zones': layout_data.get('total_mcp_zones', 0),
        })
    return {
        'fabs': fabs,
        'current_fab': current_fab_name or (layout_data.get('fab_name') if layout_data else None),
        'total': len(fabs),
    }


@app.post("/api/fab/switch")
async def api_switch_fab(req: FabSwitchRequest):
    """FAB 전환 - 다른 FAB 데이터로 변경"""
    fab_name = req.fab_name
    if fab_name not in fab_data_registry:
        raise HTTPException(status_code=404, detail=f"FAB '{fab_name}' 없음. /api/fabs에서 목록 확인")
    ok = switch_fab_internal(fab_name)
    if not ok:
        raise HTTPException(status_code=500, detail=f"FAB '{fab_name}' 로딩 실패")
    return {
        "status": "switched",
        "fab_name": fab_name,
        "total_nodes": layout_data.get("total_nodes", 0),
        "total_edges": layout_data.get("total_edges", 0),
    }


@app.get("/api/fab/settings")
async def get_fab_settings():
    """현재 FAB의 설정 반환 (OHT 대수 등)"""
    fab_name = current_fab_name
    if not fab_name:
        return {"fab_name": None, "settings": {}}
    settings = load_fab_settings()
    fab_settings = settings.get(fab_name, {})
    return {"fab_name": fab_name, "settings": fab_settings}


@app.post("/api/fab/settings")
async def update_fab_settings(req: FabSettingsUpdate):
    """현재 FAB의 설정 저장 (OHT 대수 등)"""
    fab_name = current_fab_name
    if not fab_name:
        raise HTTPException(status_code=400, detail="No FAB selected")
    settings = load_fab_settings()
    if fab_name not in settings:
        settings[fab_name] = {}
    if req.oht_count is not None:
        settings[fab_name]["oht_count"] = req.oht_count
    save_fab_settings(settings)
    return {"status": "saved", "fab_name": fab_name, "settings": settings[fab_name]}


@app.post("/api/parse/scan")
async def api_scan_map(req: MapScanRequest):
    """MAP 디렉토리 스캔 → 모든 FAB zip 파싱"""
    map_dir = req.map_dir
    if not os.path.isdir(map_dir):
        raise HTTPException(status_code=400, detail=f"디렉토리 없음: {map_dir}")

    output_dir = req.output_dir or str(BASE_DIR)
    try:
        from parse_layout import scan_and_parse_map
        results = scan_and_parse_map(map_dir, output_dir)
        # Reload registry
        load_fab_registry()
        if fab_data_registry:
            first = next(iter(fab_data_registry))
            switch_fab_internal(first)
        return {
            "status": "success",
            "parsed_count": len(results),
            "fabs": [r['fab_name'] for r in results],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Routes: Master Data CSV =====
@app.get("/api/master")
async def list_master_data():
    """등록된 FAB 마스터 데이터 목록"""
    fabs = []
    for name, info in fab_registry.items():
        csv_files = []
        csv_dir = Path(info.get('csv_dir', ''))
        if csv_dir.exists():
            for f in csv_dir.glob(f'{name}_*.csv'):
                csv_files.append({"name": f.name, "size": f.stat().st_size})
        fabs.append({
            "fab_name": name,
            "csv_files": csv_files,
            **{k: v for k, v in info.items() if k != 'csv_dir' and k != 'json_path'}
        })
    return {"fabs": fabs}


@app.get("/api/master/{fab_name}/csv/{filename}")
async def download_csv(fab_name: str, filename: str):
    """CSV 파일 다운로드 (FAB별 디렉토리에서 탐색)"""
    # 1) FAB별 디렉토리: fab_data/{fab_name}/master_csv/
    fab_csv_dir = FAB_DATA_DIR / fab_name / "master_csv"
    csv_path = fab_csv_dir / filename
    if csv_path.exists():
        return FileResponse(csv_path, media_type="text/csv", filename=filename)
    # 2) 레거시: master_csv/
    csv_path = MASTER_CSV_DIR / filename
    if csv_path.exists():
        return FileResponse(csv_path, media_type="text/csv", filename=filename)
    raise HTTPException(status_code=404, detail=f"CSV file not found: {filename}")


@app.get("/api/master/{fab_name}/download_all")
async def download_all_csv(fab_name: str):
    """FAB의 모든 CSV 파일 목록 (개별 다운로드 링크)"""
    csv_files = []
    # 1) FAB별 디렉토리
    fab_csv_dir = FAB_DATA_DIR / fab_name / "master_csv"
    if fab_csv_dir.exists():
        for f in fab_csv_dir.glob('*.csv'):
            csv_files.append({
                "name": f.name, "size": f.stat().st_size,
                "url": f"/api/master/{fab_name}/csv/{f.name}"
            })
    # 2) 레거시 폴더
    if not csv_files and MASTER_CSV_DIR.exists():
        for f in MASTER_CSV_DIR.glob(f'{fab_name}_*.csv'):
            csv_files.append({
                "name": f.name, "size": f.stat().st_size,
                "url": f"/api/master/{fab_name}/csv/{f.name}"
            })
    return {"fab_name": fab_name, "files": csv_files}


# ===== Routes: Parse XML =====
@app.post("/api/parse")
async def parse_xml(req: ParseRequest):
    """layout.xml 파싱 → JSON + CSV 생성"""
    xml_path = req.xml_path
    fab_name = req.fab_name

    if not os.path.exists(xml_path):
        raise HTTPException(status_code=400, detail=f"XML file not found: {xml_path}")

    try:
        from parse_layout import parse_layout_xml
        output_dir = str(BASE_DIR)
        result = parse_layout_xml(xml_path, output_dir, fab_name)

        # Reload
        load_layout_data()

        return {
            "status": "success",
            "fab_name": fab_name,
            "total_nodes": result.get("total_nodes", 0),
            "total_edges": result.get("total_edges", 0),
            "total_stations": result.get("total_stations", 0),
            "total_mcp_zones": result.get("total_mcp_zones", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse/upload")
async def parse_uploaded_xml(file: UploadFile = File(...), fab_name: str = Query(DEFAULT_FAB)):
    """업로드된 XML 파일 파싱"""
    if not file.filename.endswith('.xml'):
        raise HTTPException(status_code=400, detail="XML 파일만 지원합니다")

    # Save uploaded file
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    xml_path = upload_dir / f"{fab_name}_layout.xml"

    content = await file.read()
    with open(xml_path, 'wb') as f:
        f.write(content)

    try:
        from parse_layout import parse_layout_xml
        result = parse_layout_xml(str(xml_path), str(BASE_DIR), fab_name)
        load_layout_data()
        return {
            "status": "success",
            "fab_name": fab_name,
            "file_size": len(content),
            "total_nodes": result.get("total_nodes", 0),
            "total_edges": result.get("total_edges", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Routes: Simulation =====
@app.get("/api/simulation/status")
async def sim_status():
    """시뮬레이션 상태"""
    return {
        "running": simulation_state["running"],
        "vehicle_count": simulation_state["stats"]["total"],
        "scenario_active": simulation_state["scenario_active"],
        "stats": simulation_state["stats"],
        "uptime": (datetime.now() - datetime.fromisoformat(simulation_state["start_time"])).total_seconds()
            if simulation_state["start_time"] else 0
    }


@app.post("/api/simulation/start")
async def sim_start(config: SimulationConfig):
    """시뮬레이션 시작/변경"""
    simulation_state["running"] = True
    simulation_state["start_time"] = datetime.now().isoformat()
    simulation_state["stats"]["total"] = config.vehicle_count

    # Generate vehicle states
    vehicles = []
    for i in range(config.vehicle_count):
        vehicles.append({
            "id": f"V{i+1:05d}",
            "state": "running",
            "speed": random.uniform(150, 300),
            "has_foup": random.random() > 0.3,
            "position": {"x": 0, "y": 0},
        })
    simulation_state["vehicles"] = vehicles
    _update_sim_stats()

    return {"status": "started", "vehicle_count": config.vehicle_count}


@app.post("/api/simulation/scenario")
async def sim_scenario(config: ScenarioConfig):
    """시나리오 생성 (JAM/정지/적재 배분)"""
    simulation_state["scenario_active"] = True
    for v in simulation_state["vehicles"]:
        r = random.random()
        if r < config.jam_rate:
            v["state"] = "jam"
        elif r < config.jam_rate + config.stop_rate:
            v["state"] = "stopped"
        elif r < config.jam_rate + config.stop_rate + config.loaded_rate:
            v["state"] = "loaded"
        else:
            v["state"] = "running"
    _update_sim_stats()
    return {"status": "scenario_created", "stats": simulation_state["stats"]}


@app.post("/api/simulation/reset")
async def sim_reset():
    """시뮬레이션 초기화"""
    simulation_state["scenario_active"] = False
    for v in simulation_state["vehicles"]:
        v["state"] = "running"
    _update_sim_stats()
    return {"status": "reset", "stats": simulation_state["stats"]}


@app.get("/api/simulation/vehicles")
async def sim_vehicles(state: Optional[str] = None):
    """차량 목록 (필터 가능)"""
    vehicles = simulation_state["vehicles"]
    if state:
        vehicles = [v for v in vehicles if v["state"] == state]
    return {"count": len(vehicles), "vehicles": vehicles}


@app.get("/api/simulation/vehicle/{vehicle_id}")
async def sim_vehicle(vehicle_id: str):
    """특정 차량 상세"""
    for v in simulation_state["vehicles"]:
        if v["id"] == vehicle_id:
            return v
    raise HTTPException(status_code=404, detail=f"Vehicle not found: {vehicle_id}")


def _update_sim_stats():
    stats = simulation_state["stats"]
    vehicles = simulation_state["vehicles"]
    stats["total"] = len(vehicles)
    stats["running"] = sum(1 for v in vehicles if v["state"] == "running")
    stats["loaded"] = sum(1 for v in vehicles if v["state"] == "loaded")
    stats["stopped"] = sum(1 for v in vehicles if v["state"] == "stopped")
    stats["jam"] = sum(1 for v in vehicles if v["state"] == "jam")
    if vehicles:
        speeds = [v.get("speed", 0) for v in vehicles]
        stats["avg_speed"] = sum(speeds) / len(speeds)
        stats["max_speed"] = max(speeds)


# ===== Routes: Sample Data =====
@app.get("/api/output/status")
async def output_status():
    """OUTPUT 디렉토리 상태"""
    files = []
    total_size = 0
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.is_file():
                sz = f.stat().st_size
                files.append({"name": f.name, "size": sz})
                total_size += sz
    return {
        "output_dir": str(OUTPUT_DIR),
        "file_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files": files,
        "save_interval": CSV_SAVE_INTERVAL,
        "cleanup_interval": OUTPUT_CLEANUP_INTERVAL,
    }


# ===== Routes: Server Info =====
@app.get("/api/status")
async def server_status():
    """서버 상태"""
    return {
        "server": "AMOS MAP System PRO",
        "version": "2.0.0",
        "port": PORT,
        "layout_loaded": layout_data is not None,
        "fab_name": layout_data.get("fab_name", DEFAULT_FAB) if layout_data else None,
        "registered_fabs": list(fab_registry.keys()),
        "simulation": {
            "running": simulation_state["running"],
            "vehicles": simulation_state["stats"]["total"],
        },
        "sample_data": {
            "output_dir": str(OUTPUT_DIR),
            "save_interval": CSV_SAVE_INTERVAL,
            "cleanup_interval": OUTPUT_CLEANUP_INTERVAL,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.on_event("shutdown")
async def shutdown():
    global is_running
    is_running = False
    print("[Server] 샘플 데이터 백그라운드 태스크 종료")


# ===== Static files (for any additional assets) =====
# Mount at the end so it doesn't override API routes
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ===== Main =====
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  AMOS MAP System PRO - OHT FAB Simulator")
    print(f"  Starting on port {PORT}...")
    print(f"{'='*60}\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )
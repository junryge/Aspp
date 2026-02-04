#!/usr/bin/env python3
"""
SECS/GEM Mock Server
- 설비 Load/Unload 시뮬레이션
- S1F13/S1F14 (Establish Communication)
- S3F17/S3F18 (Carrier Action Request/Acknowledge)

Port: 10012
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="SECS/GEM Mock")

# MCS 서버 주소
MCS_URL = "http://localhost:10011"

# 설비 상태
equipment_ports: Dict[str, dict] = {}
carrier_locations: Dict[str, str] = {}  # carrier_id -> port_id
event_log: List[dict] = []

# WebSocket 클라이언트
ws_clients = set()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>SECS/GEM Mock</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #2e1a2e 0%, #3e162e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; color: #ff66aa; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h2 { color: #ff66aa; margin-bottom: 15px; font-size: 1.1em; }
        .event-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .event-item {
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 13px;
        }
        .event-item .time { color: #888; font-size: 11px; }
        .event-item .msg-id { color: #ff66aa; font-weight: bold; }
        .event-item.load { border-left: 3px solid #00ff88; }
        .event-item.unload { border-left: 3px solid #00d4ff; }
        .carrier-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
        }
        .carrier-card {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .carrier-card .id { font-weight: bold; color: #ff66aa; }
        .carrier-card .port { font-size: 12px; color: #888; margin-top: 5px; }
        .carrier-card .status {
            margin-top: 8px;
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 11px;
            display: inline-block;
        }
        .carrier-card .status.loaded { background: #00ff88; color: #000; }
        .carrier-card .status.transit { background: #ff9900; color: #000; }
        .secs-msg {
            font-family: monospace;
            background: #000;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
            font-size: 11px;
            white-space: pre;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .stat-row .value { font-weight: bold; color: #ff66aa; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SECS/GEM Mock Server</h1>

        <div class="grid">
            <div class="panel">
                <h2>SECS 메시지 로그</h2>
                <div class="event-list" id="eventList"></div>
            </div>

            <div class="panel">
                <h2>통계</h2>
                <div class="stat-row">
                    <span>총 Load 요청</span>
                    <span class="value" id="loadCount">0</span>
                </div>
                <div class="stat-row">
                    <span>총 Unload 요청</span>
                    <span class="value" id="unloadCount">0</span>
                </div>
                <div class="stat-row">
                    <span>현재 Carrier 수</span>
                    <span class="value" id="carrierCount">0</span>
                </div>

                <h2 style="margin-top:20px;">Carrier 현황</h2>
                <div class="carrier-grid" id="carrierGrid">
                    <div style="color:#888;text-align:center;grid-column:1/-1;">Carrier 없음</div>
                </div>
            </div>
        </div>

        <div class="panel" style="margin-top:20px;">
            <h2>SECS 메시지 샘플</h2>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div>
                    <h3 style="font-size:14px;color:#888;margin-bottom:10px;">S3F17 (Carrier Action Request)</h3>
                    <div class="secs-msg">S3F17 W
&lt;L[3]
  &lt;A "FOUP001"&gt;      // Carrier ID
  &lt;A "PTN_LOAD"&gt;     // Action
  &lt;A "PORT_001"&gt;     // Port ID
&gt;</div>
                </div>
                <div>
                    <h3 style="font-size:14px;color:#888;margin-bottom:10px;">S3F18 (Acknowledge)</h3>
                    <div class="secs-msg">S3F18
&lt;L[2]
  &lt;B 0x00&gt;           // CAACK (0=OK)
  &lt;L[0]&gt;             // Error list (empty)
&gt;</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let loadCount = 0, unloadCount = 0;

        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'event') addEvent(msg.data);
                if (msg.type === 'carriers') updateCarriers(msg.data);
                if (msg.type === 'stats') updateStats(msg.data);
            };
            ws.onclose = () => setTimeout(connect, 3000);
        }

        function addEvent(event) {
            const list = document.getElementById('eventList');
            const item = document.createElement('div');
            item.className = 'event-item ' + event.action.toLowerCase();
            item.innerHTML = `
                <div class="time">${event.time}</div>
                <div><span class="msg-id">${event.secsMsg}</span> ${event.action}</div>
                <div>Carrier: ${event.carrierId} | Port: ${event.portId}</div>
            `;
            list.insertBefore(item, list.firstChild);

            // 최대 50개 유지
            while (list.children.length > 50) {
                list.removeChild(list.lastChild);
            }
        }

        function updateCarriers(carriers) {
            const grid = document.getElementById('carrierGrid');
            if (Object.keys(carriers).length === 0) {
                grid.innerHTML = '<div style="color:#888;text-align:center;grid-column:1/-1;">Carrier 없음</div>';
                return;
            }
            grid.innerHTML = Object.entries(carriers).map(([id, port]) => `
                <div class="carrier-card">
                    <div class="id">${id}</div>
                    <div class="port">@ ${port}</div>
                    <div class="status loaded">LOADED</div>
                </div>
            `).join('');
        }

        function updateStats(stats) {
            document.getElementById('loadCount').textContent = stats.loadCount;
            document.getElementById('unloadCount').textContent = stats.unloadCount;
            document.getElementById('carrierCount').textContent = stats.carrierCount;
        }

        connect();
    </script>
</body>
</html>
"""

# 통계
stats = {"loadCount": 0, "unloadCount": 0}

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/health")
async def health():
    return {"status": "ok"}

async def broadcast(message: dict):
    for client in ws_clients.copy():
        try:
            await client.send_json(message)
        except:
            ws_clients.discard(client)

async def add_event(action: str, carrier_id: str, port_id: str, vehicle_id: str = None):
    """이벤트 로그 추가"""
    secs_msg = "S3F17/S3F18" if action == "LOAD" else "S3F17/S3F18"
    event = {
        "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "secsMsg": secs_msg,
        "action": action,
        "carrierId": carrier_id,
        "portId": port_id,
        "vehicleId": vehicle_id
    }
    event_log.append(event)
    await broadcast({"type": "event", "data": event})
    print(f"[SECS] {action}: Carrier={carrier_id}, Port={port_id}")

@app.post("/api/load")
async def load_carrier(request: dict):
    """
    Carrier Load 요청 (S3F17 시뮬레이션)
    OHT가 설비에서 Carrier를 픽업할 때
    """
    carrier_id = request.get("carrierId")
    port_id = request.get("portId")
    vehicle_id = request.get("vehicleId")

    stats["loadCount"] += 1

    # 설비에서 Carrier 제거 (OHT로 이동)
    if carrier_id in carrier_locations:
        del carrier_locations[carrier_id]

    await add_event("LOAD", carrier_id, str(port_id), vehicle_id)
    await broadcast({"type": "carriers", "data": carrier_locations})
    await broadcast({"type": "stats", "data": {**stats, "carrierCount": len(carrier_locations)}})

    # SECS S3F18 응답 시뮬레이션 (CAACK = 0, 성공)
    return {
        "success": True,
        "caack": 0,  # 0 = Acknowledge, command will be performed
        "secsMsg": "S3F18"
    }

@app.post("/api/unload")
async def unload_carrier(request: dict):
    """
    Carrier Unload 요청 (S3F17 시뮬레이션)
    OHT가 설비에 Carrier를 내려놓을 때
    """
    carrier_id = request.get("carrierId")
    port_id = request.get("portId")
    vehicle_id = request.get("vehicleId")

    stats["unloadCount"] += 1

    # 설비에 Carrier 배치
    carrier_locations[carrier_id] = str(port_id)

    await add_event("UNLOAD", carrier_id, str(port_id), vehicle_id)
    await broadcast({"type": "carriers", "data": carrier_locations})
    await broadcast({"type": "stats", "data": {**stats, "carrierCount": len(carrier_locations)}})

    # SECS S3F18 응답 시뮬레이션
    return {
        "success": True,
        "caack": 0,
        "secsMsg": "S3F18"
    }

@app.get("/api/carriers")
async def get_carriers():
    """현재 Carrier 위치 조회"""
    return carrier_locations

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)

    # 초기 데이터 전송
    await websocket.send_json({"type": "carriers", "data": carrier_locations})
    await websocket.send_json({"type": "stats", "data": {**stats, "carrierCount": len(carrier_locations)}})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_clients.discard(websocket)

if __name__ == "__main__":
    print("=" * 50)
    print("SECS/GEM Mock Server")
    print("=" * 50)
    print("URL: http://localhost:10012")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=10012)
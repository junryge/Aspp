#!/usr/bin/env python3
"""
MCS (Material Control System) Simulator Server
- MES에서 Transport 요청 수신
- OHT에 배차 명령 전달
- SECS/GEM Load/Unload 요청

Port: 10011
"""

import asyncio
import json
import httpx
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="MCS Simulator")

# 연결 서버 주소
MES_URL = "http://localhost:10010"
OHT_URL = "http://localhost:10003"
SECS_URL = "http://localhost:10012"

# Transport 큐
transport_queue: List[dict] = []
active_transports: Dict[str, dict] = {}

# WebSocket 클라이언트
ws_clients = set()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>MCS Simulator</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a2e1a 0%, #162e16 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #00ff88; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h2 { color: #00ff88; margin-bottom: 15px; font-size: 1.1em; }
        .status-card {
            display: flex;
            justify-content: space-between;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .status-card .label { color: #888; font-size: 12px; }
        .status-card .value { font-size: 24px; font-weight: bold; }
        .status-card.active .value { color: #00ff88; }
        .status-card.queue .value { color: #ff9900; }
        .status-card.complete .value { color: #00d4ff; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        th { color: #888; font-size: 11px; text-transform: uppercase; }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: bold;
        }
        .badge-queued { background: #666; }
        .badge-dispatched { background: #ff9900; color: #000; }
        .badge-topickup { background: #00d4ff; color: #000; }
        .badge-picking { background: #ff66aa; color: #000; }
        .badge-carrying { background: #00ff88; color: #000; }
        .badge-complete { background: #00aa55; }
        .connection-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        .conn-item {
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            font-size: 12px;
        }
        .conn-ok { background: rgba(0,255,136,0.2); border: 1px solid #00ff88; }
        .conn-err { background: rgba(255,51,102,0.2); border: 1px solid #ff3366; }
        .log-area {
            height: 200px;
            overflow-y: auto;
            background: #000;
            border-radius: 8px;
            padding: 10px;
            font-family: monospace;
            font-size: 11px;
        }
        .log-area .log-line { margin-bottom: 2px; }
        .log-info { color: #00d4ff; }
        .log-success { color: #00ff88; }
        .log-error { color: #ff3366; }
        .log-warn { color: #ff9900; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MCS Simulator (Material Control System)</h1>

        <div class="connection-grid">
            <div class="conn-item" id="connMes">MES: 확인중...</div>
            <div class="conn-item" id="connOht">OHT: 확인중...</div>
            <div class="conn-item" id="connSecs">SECS/GEM: 확인중...</div>
        </div>

        <div class="grid">
            <div class="panel">
                <h2>Transport 현황</h2>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px;">
                    <div class="status-card queue">
                        <div><div class="label">대기중</div><div class="value" id="queueCount">0</div></div>
                    </div>
                    <div class="status-card active">
                        <div><div class="label">진행중</div><div class="value" id="activeCount">0</div></div>
                    </div>
                    <div class="status-card complete">
                        <div><div class="label">완료</div><div class="value" id="completeCount">0</div></div>
                    </div>
                </div>

                <h2>Active Transports</h2>
                <table>
                    <thead>
                        <tr><th>Request</th><th>Carrier</th><th>Route</th><th>Vehicle</th><th>Status</th></tr>
                    </thead>
                    <tbody id="activeList">
                        <tr><td colspan="5" style="color:#888;text-align:center;">없음</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="panel">
                <h2>시스템 로그</h2>
                <div class="log-area" id="logArea"></div>
            </div>
        </div>

        <div class="panel" style="margin-top:20px;">
            <h2>대기 큐 (Queue)</h2>
            <table>
                <thead>
                    <tr><th>Request</th><th>Lot</th><th>Carrier</th><th>From → To</th><th>Priority</th><th>대기시간</th></tr>
                </thead>
                <tbody id="queueList">
                    <tr><td colspan="6" style="color:#888;text-align:center;">대기중인 요청 없음</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let ws;

        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'status') updateStatus(msg.data);
                if (msg.type === 'log') addLog(msg.data);
                if (msg.type === 'connections') updateConnections(msg.data);
            };
            ws.onclose = () => setTimeout(connect, 3000);
        }

        function updateStatus(data) {
            document.getElementById('queueCount').textContent = data.queueCount;
            document.getElementById('activeCount').textContent = data.activeCount;
            document.getElementById('completeCount').textContent = data.completeCount;

            // Active list
            const activeList = document.getElementById('activeList');
            if (data.active.length === 0) {
                activeList.innerHTML = '<tr><td colspan="5" style="color:#888;text-align:center;">없음</td></tr>';
            } else {
                activeList.innerHTML = data.active.map(t => `
                    <tr>
                        <td>${t.requestId}</td>
                        <td>${t.carrierId}</td>
                        <td>${t.fromStation} → ${t.toStation}</td>
                        <td>${t.vehicleId || '-'}</td>
                        <td><span class="badge badge-${t.status.toLowerCase()}">${t.status}</span></td>
                    </tr>
                `).join('');
            }

            // Queue list
            const queueList = document.getElementById('queueList');
            if (data.queue.length === 0) {
                queueList.innerHTML = '<tr><td colspan="6" style="color:#888;text-align:center;">대기중인 요청 없음</td></tr>';
            } else {
                queueList.innerHTML = data.queue.map(t => `
                    <tr>
                        <td>${t.requestId}</td>
                        <td>${t.lotId}</td>
                        <td>${t.carrierId}</td>
                        <td>${t.fromStation} → ${t.toStation}</td>
                        <td>${t.priority}</td>
                        <td>${t.createdAt}</td>
                    </tr>
                `).join('');
            }
        }

        function addLog(log) {
            const area = document.getElementById('logArea');
            const line = document.createElement('div');
            line.className = 'log-line log-' + log.level;
            line.textContent = `[${log.time}] ${log.message}`;
            area.appendChild(line);
            area.scrollTop = area.scrollHeight;
        }

        function updateConnections(conns) {
            ['Mes', 'Oht', 'Secs'].forEach(name => {
                const el = document.getElementById('conn' + name);
                const key = name.toLowerCase();
                if (conns[key]) {
                    el.className = 'conn-item conn-ok';
                    el.textContent = name.toUpperCase() + ': 연결됨';
                } else {
                    el.className = 'conn-item conn-err';
                    el.textContent = name.toUpperCase() + ': 연결안됨';
                }
            });
        }

        connect();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

# 로그 함수
async def log(message: str, level: str = "info"):
    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "level": level
    }
    await broadcast({"type": "log", "data": log_entry})
    print(f"[{log_entry['time']}] [{level.upper()}] {message}")

# 상태 브로드캐스트
async def broadcast_status():
    complete_count = len([t for t in active_transports.values() if t.get("status") == "COMPLETE"])
    status = {
        "queueCount": len(transport_queue),
        "activeCount": len(active_transports) - complete_count,
        "completeCount": complete_count,
        "queue": transport_queue,
        "active": list(active_transports.values())
    }
    await broadcast({"type": "status", "data": status})

async def broadcast(message: dict):
    for client in ws_clients.copy():
        try:
            await client.send_json(message)
        except:
            ws_clients.discard(client)

# MES에서 Transport 요청 수신
@app.post("/api/transport")
async def receive_transport(request: dict):
    """MES에서 Transport 요청 수신 - 즉시 응답, 배차는 비동기"""
    await log(f"MES로부터 Transport 수신: {request['requestId']} ({request['carrierId']})", "info")

    request["status"] = "QUEUED"
    request["vehicleId"] = None
    request["receivedAt"] = datetime.now().strftime("%H:%M:%S")

    # 큐에 추가
    active_transports[request["requestId"]] = request

    # 비동기로 OHT 배차 (MES 응답 안 기다림)
    asyncio.create_task(async_dispatch_to_oht(request))

    # MES에 즉시 응답
    return {"success": True, "vehicleId": None, "status": "QUEUED"}

async def async_dispatch_to_oht(transport: dict):
    """비동기 OHT 배차 - 백그라운드에서 실행"""
    try:
        vehicle_id = await dispatch_to_oht(transport)

        if vehicle_id:
            transport["status"] = "DISPATCHED"
            transport["vehicleId"] = vehicle_id
            await log(f"OHT {vehicle_id} 배차 완료: {transport['fromStation']} → {transport['toStation']}", "success")
        else:
            transport_queue.append(transport)
            await log(f"배차 실패, 큐에 추가: {transport['requestId']}", "warn")

        await broadcast_status()
        await update_mes_status(transport)

    except Exception as e:
        await log(f"배차 오류: {e}", "error")

async def dispatch_to_oht(transport: dict) -> Optional[str]:
    """OHT에 배차 명령 전달"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OHT_URL}/api/dispatch",
                json={
                    "requestId": transport["requestId"],
                    "carrierId": transport["carrierId"],
                    "fromStation": transport["fromStation"],
                    "toStation": transport["toStation"],
                    "priority": transport.get("priority", 2)
                },
                timeout=5.0
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("vehicleId")
    except Exception as e:
        await log(f"OHT 통신 오류: {e}", "error")
    return None

async def update_mes_status(transport: dict):
    """MES에 상태 업데이트 전달"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{MES_URL}/api/transport/{transport['requestId']}/status",
                json={"status": transport["status"], "vehicleId": transport.get("vehicleId")},
                timeout=3.0
            )
    except Exception as e:
        pass  # MES 업데이트 실패는 무시

# OHT에서 상태 업데이트 수신
@app.post("/api/transport/{request_id}/status")
async def oht_status_update(request_id: str, update: dict):
    """OHT에서 Transport 상태 업데이트 수신"""
    if request_id in active_transports:
        transport = active_transports[request_id]
        old_status = transport["status"]
        transport["status"] = update.get("status", transport["status"])

        await log(f"Transport {request_id}: {old_status} → {transport['status']}", "info")

        # SECS/GEM 이벤트 발생
        if transport["status"] == "PICKING":
            await send_secs_load_request(transport)
        elif transport["status"] == "CARRYING":
            await send_secs_unload_request(transport)

        await broadcast_status()
        await update_mes_status(transport)

        return {"success": True}
    return {"success": False, "error": "Not found"}

async def send_secs_load_request(transport: dict):
    """SECS/GEM Load 요청"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{SECS_URL}/api/load",
                json={
                    "carrierId": transport["carrierId"],
                    "portId": transport["fromStation"],
                    "vehicleId": transport.get("vehicleId")
                },
                timeout=3.0
            )
            await log(f"SECS Load 요청: {transport['carrierId']} @ Port {transport['fromStation']}", "info")
    except Exception as e:
        await log(f"SECS Load 요청 실패: {e}", "error")

async def send_secs_unload_request(transport: dict):
    """SECS/GEM Unload 요청"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{SECS_URL}/api/unload",
                json={
                    "carrierId": transport["carrierId"],
                    "portId": transport["toStation"],
                    "vehicleId": transport.get("vehicleId")
                },
                timeout=3.0
            )
            await log(f"SECS Unload 요청: {transport['carrierId']} @ Port {transport['toStation']}", "info")
    except Exception as e:
        await log(f"SECS Unload 요청 실패: {e}", "error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)

    # 초기 상태 전송
    await broadcast_status()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_clients.discard(websocket)

# 연결 상태 체크 (주기적)
async def check_connections():
    while True:
        conns = {"mes": False, "oht": False, "secs": False}
        async with httpx.AsyncClient() as client:
            for name, url in [("mes", MES_URL), ("oht", OHT_URL), ("secs", SECS_URL)]:
                try:
                    r = await client.get(f"{url}/health", timeout=2.0)
                    conns[name] = r.status_code == 200
                except:
                    conns[name] = False

        await broadcast({"type": "connections", "data": conns})
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup():
    asyncio.create_task(check_connections())

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print("=" * 50)
    print("MCS Simulator Server")
    print("=" * 50)
    print("URL: http://localhost:10011")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=10011)
#!/usr/bin/env python3
"""
MES Simulator Server
- Transport 요청 입력 UI
- MCS로 요청 전달

Port: 10010
"""

import asyncio
import json
import httpx
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="MES Simulator")

# MCS 서버 주소
MCS_URL = "http://localhost:10011"

# Transport 요청 목록
transport_requests = []
request_counter = 0

# WebSocket 클라이언트
ws_clients = set()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>MES Simulator</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 30px;
            font-size: 2em;
        }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h2 {
            color: #ff9900;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .form-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .form-group {
            flex: 1;
            min-width: 150px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #888;
            font-size: 12px;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #1a1a3e;
            color: #fff;
            font-size: 14px;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,212,255,0.4); }
        .btn-danger {
            background: #ff3366;
            color: #fff;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }
        .status-queued { background: #666; }
        .status-dispatched { background: #ff9900; color: #000; }
        .status-picking { background: #00d4ff; color: #000; }
        .status-carrying { background: #00ff88; color: #000; }
        .status-complete { background: #00aa55; }
        .status-error { background: #ff3366; }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        th { color: #888; font-size: 12px; text-transform: uppercase; }
        tr:hover { background: rgba(255,255,255,0.05); }
        .arrow { color: #00d4ff; }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 20px;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
        }
        .connected { background: #00ff88; color: #000; }
        .disconnected { background: #ff3366; color: #fff; }
        .info-box {
            background: rgba(0,212,255,0.1);
            border-left: 3px solid #00d4ff;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="connection-status disconnected" id="connStatus">MCS 연결 안됨</div>

    <div class="container">
        <h1>MES Simulator</h1>

        <div class="info-box">
            <strong>시나리오:</strong> MES에서 Transport 요청 입력 → MCS로 전송 → OHT 배차 → SECS/GEM Load/Unload
        </div>

        <div class="panel">
            <h2>Transport 요청 입력</h2>
            <div class="form-row">
                <div class="form-group">
                    <label>Lot ID</label>
                    <input type="text" id="lotId" placeholder="LOT001" value="">
                </div>
                <div class="form-group">
                    <label>Carrier ID (FOUP)</label>
                    <input type="text" id="carrierId" placeholder="FOUP001" value="">
                </div>
                <div class="form-group">
                    <label>From Station</label>
                    <input type="number" id="fromStation" placeholder="14901" value="">
                </div>
                <div class="form-group">
                    <label>To Station</label>
                    <input type="number" id="toStation" placeholder="25001" value="">
                </div>
                <div class="form-group">
                    <label>Priority (1=High)</label>
                    <select id="priority">
                        <option value="1">1 - High</option>
                        <option value="2" selected>2 - Normal</option>
                        <option value="3">3 - Low</option>
                    </select>
                </div>
            </div>
            <button class="btn btn-primary" onclick="sendTransport()">MCS로 전송</button>
        </div>

        <div class="panel">
            <h2>Transport 요청 목록</h2>
            <table>
                <thead>
                    <tr>
                        <th>Request ID</th>
                        <th>Lot ID</th>
                        <th>Carrier</th>
                        <th>From → To</th>
                        <th>Priority</th>
                        <th>Vehicle</th>
                        <th>Status</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="transportList">
                    <tr><td colspan="8" style="text-align:center;color:#888;">요청 없음</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let ws;
        let requestCount = 0;

        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connStatus').textContent = 'MCS 연결됨';
                document.getElementById('connStatus').className = 'connection-status connected';
            };

            ws.onclose = () => {
                document.getElementById('connStatus').textContent = 'MCS 연결 안됨';
                document.getElementById('connStatus').className = 'connection-status disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'transport_list') {
                    updateTransportList(msg.data);
                } else if (msg.type === 'transport_update') {
                    updateTransportItem(msg.data);
                }
            };
        }

        function updateTransportList(list) {
            const tbody = document.getElementById('transportList');
            if (list.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#888;">요청 없음</td></tr>';
                return;
            }

            tbody.innerHTML = list.map(t => `
                <tr>
                    <td>${t.requestId}</td>
                    <td>${t.lotId}</td>
                    <td>${t.carrierId}</td>
                    <td>${t.fromStation} <span class="arrow">→</span> ${t.toStation}</td>
                    <td>${t.priority}</td>
                    <td>${t.vehicleId || '-'}</td>
                    <td><span class="status-badge status-${t.status.toLowerCase()}">${t.status}</span></td>
                    <td>${t.createdAt || ''}</td>
                </tr>
            `).join('');
        }

        function updateTransportItem(item) {
            // 리스트 갱신 요청
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'get_list'}));
            }
        }

        async function sendTransport() {
            const lotId = document.getElementById('lotId').value || `LOT${String(++requestCount).padStart(3, '0')}`;
            const carrierId = document.getElementById('carrierId').value || `FOUP${String(requestCount).padStart(3, '0')}`;
            const fromStation = parseInt(document.getElementById('fromStation').value) || 14901;
            const toStation = parseInt(document.getElementById('toStation').value) || 25001;
            const priority = parseInt(document.getElementById('priority').value);

            const request = {
                lotId,
                carrierId,
                fromStation,
                toStation,
                priority
            };

            try {
                const res = await fetch('/api/transport', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(request)
                });

                const result = await res.json();

                if (result.success) {
                    // 입력 필드 초기화
                    document.getElementById('lotId').value = '';
                    document.getElementById('carrierId').value = '';

                    // 목록 갱신
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({type: 'get_list'}));
                    }
                } else {
                    alert('전송 실패: ' + result.error);
                }
            } catch (e) {
                alert('전송 오류: ' + e.message);
            }
        }

        connect();
    </script>
</body>
</html>
"""

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "MES_Simulator"}

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/api/transport")
async def create_transport(request: dict):
    """Transport 요청 생성 및 MCS로 전달"""
    global request_counter

    request_counter += 1
    transport = {
        "requestId": f"TR{request_counter:05d}",
        "lotId": request.get("lotId", f"LOT{request_counter:03d}"),
        "carrierId": request.get("carrierId", f"FOUP{request_counter:03d}"),
        "fromStation": request.get("fromStation", 14901),
        "toStation": request.get("toStation", 25001),
        "priority": request.get("priority", 2),
        "status": "QUEUED",
        "vehicleId": None,
        "createdAt": datetime.now().strftime("%H:%M:%S"),
        "source": "MES"
    }

    transport_requests.append(transport)

    # MCS로 전달
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MCS_URL}/api/transport",
                json=transport,
                timeout=5.0
            )
            if response.status_code == 200:
                result = response.json()
                transport["status"] = "DISPATCHED"
                transport["vehicleId"] = result.get("vehicleId")
    except Exception as e:
        print(f"MCS 전송 실패: {e}")
        transport["status"] = "MCS_ERROR"

    # WebSocket으로 브로드캐스트
    await broadcast({"type": "transport_update", "data": transport})

    return {"success": True, "transport": transport}

@app.get("/api/transports")
async def get_transports():
    """Transport 목록 조회"""
    return transport_requests

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)

    # 초기 목록 전송
    await websocket.send_json({"type": "transport_list", "data": transport_requests})

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "get_list":
                await websocket.send_json({"type": "transport_list", "data": transport_requests})
    except WebSocketDisconnect:
        ws_clients.discard(websocket)

async def broadcast(message: dict):
    """모든 WebSocket 클라이언트에 메시지 전송"""
    for client in ws_clients.copy():
        try:
            await client.send_json(message)
        except:
            ws_clients.discard(client)

# MCS에서 상태 업데이트 수신
@app.post("/api/transport/{request_id}/status")
async def update_transport_status(request_id: str, update: dict):
    """MCS에서 Transport 상태 업데이트 수신"""
    for t in transport_requests:
        if t["requestId"] == request_id:
            t["status"] = update.get("status", t["status"])
            t["vehicleId"] = update.get("vehicleId", t["vehicleId"])
            await broadcast({"type": "transport_update", "data": t})
            return {"success": True}
    return {"success": False, "error": "Not found"}

if __name__ == "__main__":
    print("=" * 50)
    print("MES Simulator Server")
    print("=" * 50)
    print("URL: http://localhost:10010")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=10010)
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OHT Layout Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0e14; color: #e6edf3; overflow: hidden; }
        
        #dropzone {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            background: #0a0e14; z-index: 2000;
        }
        #dropzone.hidden { display: none; }
        #dropzone.dragover { background: #111820; }
        
        .drop-area {
            border: 2px dashed #00d4ff; border-radius: 16px; padding: 60px 80px;
            text-align: center; cursor: pointer; transition: all 0.3s;
        }
        .drop-area:hover, #dropzone.dragover .drop-area { background: rgba(0,212,255,0.1); border-color: #39ff14; }
        .drop-area h2 { color: #00d4ff; font-size: 24px; margin-bottom: 15px; }
        .drop-area p { color: #8b949e; font-size: 14px; margin-bottom: 20px; }
        .drop-area input { display: none; }
        .drop-area button {
            background: #00d4ff; color: #0a0e14; border: none; padding: 12px 30px;
            border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer;
        }
        .drop-area button:hover { background: #39ff14; }
        
        #header {
            position: fixed; top: 0; left: 0; right: 0; height: 45px;
            background: #111820; border-bottom: 1px solid #21262d;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 15px; z-index: 1000;
        }
        #header h1 { font-size: 15px; color: #00d4ff; font-weight: 500; }
        #header .stats { font-size: 12px; color: #8b949e; }
        
        #toolbar {
            position: fixed; top: 55px; left: 10px;
            background: rgba(17,24,32,0.95); border: 1px solid #21262d;
            border-radius: 8px; padding: 12px; z-index: 1000; font-size: 12px;
        }
        #toolbar label { display: flex; align-items: center; gap: 6px; margin: 6px 0; color: #8b949e; cursor: pointer; }
        #toolbar input[type="checkbox"] { accent-color: #00d4ff; }
        #toolbar .btn-group { display: flex; gap: 5px; margin-top: 10px; }
        #toolbar button {
            flex: 1; padding: 6px 10px; background: #21262d; border: none;
            color: #e6edf3; border-radius: 4px; cursor: pointer; font-size: 11px;
        }
        #toolbar button:hover { background: #00d4ff; color: #0a0e14; }
        
        #search-box {
            margin-top: 10px; padding-top: 10px; border-top: 1px solid #21262d;
        }
        #search-box input {
            width: 100%; padding: 6px 8px; background: #0a0e14; border: 1px solid #21262d;
            color: #e6edf3; border-radius: 4px; font-size: 12px;
        }
        #search-box input:focus { outline: none; border-color: #00d4ff; }
        
        #info {
            position: fixed; bottom: 10px; left: 10px;
            background: rgba(17,24,32,0.95); border: 1px solid #21262d;
            border-radius: 8px; padding: 10px; z-index: 1000;
            font-size: 11px; min-width: 150px;
        }
        #info .title { color: #00d4ff; font-weight: 600; margin-bottom: 5px; }
        #info .content { color: #8b949e; line-height: 1.6; }
        
        #loading {
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: rgba(17,24,32,0.95); border: 1px solid #21262d;
            border-radius: 8px; padding: 20px 40px; z-index: 3000; display: none;
        }
        #loading.show { display: block; }
        
        canvas { display: block; cursor: grab; }
        canvas:active { cursor: grabbing; }
    </style>
</head>
<body>
    <div id="dropzone">
        <div class="drop-area" onclick="document.getElementById('fileInput').click()">
            <h2>📂 OHT Layout XML 파일 선택</h2>
            <p>클릭하거나 파일을 드래그하세요</p>
            <input type="file" id="fileInput" accept=".xml">
            <button type="button">파일 선택</button>
        </div>
    </div>
    
    <div id="loading"><span>⏳ 파싱 중...</span></div>
    
    <div id="header">
        <h1>OHT Layout Viewer</h1>
        <span class="stats" id="stats">-</span>
    </div>
    
    <div id="toolbar">
        <label><input type="checkbox" id="showRails" checked> Rails</label>
        <label><input type="checkbox" id="showStations" checked> Stations</label>
        <div class="btn-group">
            <button onclick="fitView()">Fit</button>
            <button onclick="resetView()">Reset</button>
            <button onclick="loadNew()">New</button>
        </div>
        <div id="search-box">
            <input type="text" id="searchInput" placeholder="Station 검색 (예: ST-10001)">
        </div>
    </div>
    
    <div id="info">
        <div class="title">Info</div>
        <div class="content" id="infoContent">드래그: 이동 | 휠: 줌</div>
    </div>
    
    <canvas id="canvas"></canvas>

    <script>
        // Data
        let addresses = new Map();
        let stations = new Map();
        let connections = [];
        let bounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 };
        
        // Canvas
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        let scale = 0.5, offsetX = 0, offsetY = 0;
        let isDragging = false, lastX, lastY;
        
        // Elements
        const dropzone = document.getElementById('dropzone');
        const loading = document.getElementById('loading');
        const fileInput = document.getElementById('fileInput');
        
        // File handling
        fileInput.addEventListener('change', e => {
            if (e.target.files[0]) loadFile(e.target.files[0]);
        });
        
        dropzone.addEventListener('dragover', e => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });
        
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
        });
        
        function loadFile(file) {
            if (!file.name.endsWith('.xml')) {
                alert('XML 파일만 지원합니다.');
                return;
            }
            
            loading.classList.add('show');
            const reader = new FileReader();
            reader.onload = e => {
                parseXML(e.target.result);
                dropzone.classList.add('hidden');
                loading.classList.remove('show');
                resizeCanvas();
                fitView();
            };
            reader.onerror = () => {
                alert('파일 읽기 실패');
                loading.classList.remove('show');
            };
            reader.readAsText(file);
        }
        
        function parseXML(xmlText) {
            addresses.clear();
            stations.clear();
            connections = [];
            
            const parser = new DOMParser();
            const xml = parser.parseFromString(xmlText, 'text/xml');
            
            // Find all address groups
            const groups = xml.querySelectorAll('group');
            
            groups.forEach(group => {
                const name = group.getAttribute('name') || '';
                const cls = group.getAttribute('class') || '';
                
                // Address parsing
                if (name.includes('Addr') && cls.includes('address.Addr') && !name.includes('NextAddr')) {
                    const match = name.match(/Addr(\d+)/);
                    if (match) {
                        const addrNo = parseInt(match[1]);
                        const addr = { x: 0, y: 0, stations: [], nextAddrs: [] };
                        
                        // Get params
                        group.querySelectorAll(':scope > param').forEach(p => {
                            const key = p.getAttribute('key');
                            const val = p.getAttribute('value');
                            if (key === 'draw-x') addr.x = parseFloat(val) || 0;
                            if (key === 'draw-y') addr.y = parseFloat(val) || 0;
                        });
                        
                        // Get stations
                        group.querySelectorAll('group[class*="Station"]').forEach(sg => {
                            const sMatch = sg.getAttribute('name')?.match(/Station(\d+)/);
                            if (sMatch) {
                                const stNo = parseInt(sMatch[1]);
                                const st = { portId: '', type: 0, addrNo: addrNo };
                                sg.querySelectorAll('param').forEach(p => {
                                    const key = p.getAttribute('key');
                                    const val = p.getAttribute('value');
                                    if (key === 'port-id') st.portId = val;
                                    if (key === 'type') st.type = parseInt(val) || 0;
                                });
                                stations.set(stNo, st);
                                addr.stations.push(stNo);
                            }
                        });
                        
                        // Get next addresses
                        group.querySelectorAll('group[class*="NextAddr"]').forEach(ng => {
                            ng.querySelectorAll('param').forEach(p => {
                                if (p.getAttribute('key') === 'next-address') {
                                    const next = parseInt(p.getAttribute('value'));
                                    if (!isNaN(next)) {
                                        addr.nextAddrs.push(next);
                                        connections.push([addrNo, next]);
                                    }
                                }
                            });
                        });
                        
                        addresses.set(addrNo, addr);
                    }
                }
            });
            
            // Calculate bounds
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            addresses.forEach(a => {
                if (a.x !== 0) { minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x); }
                if (a.y !== 0) { minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y); }
            });
            bounds = { minX, maxX, minY, maxY };
            
            document.getElementById('stats').textContent = 
                `Addr: ${addresses.size.toLocaleString()} | Stn: ${stations.size.toLocaleString()} | Conn: ${connections.length.toLocaleString()}`;
        }
        
        function worldToScreen(wx, wy) {
            return {
                x: (wx - bounds.minX) * scale + offsetX,
                y: height - ((wy - bounds.minY) * scale + offsetY)
            };
        }
        
        function screenToWorld(sx, sy) {
            return {
                x: (sx - offsetX) / scale + bounds.minX,
                y: (height - sy - offsetY) / scale + bounds.minY
            };
        }
        
        function render() {
            ctx.fillStyle = '#0a0e14';
            ctx.fillRect(0, 0, width, height);
            
            if (addresses.size === 0) return;
            
            // Draw rails
            if (document.getElementById('showRails').checked) {
                ctx.strokeStyle = 'rgba(0, 255, 136, 0.4)';
                ctx.lineWidth = Math.max(1, scale * 0.5);
                ctx.beginPath();
                connections.forEach(([from, to]) => {
                    const a = addresses.get(from);
                    const b = addresses.get(to);
                    if (a && b) {
                        const p1 = worldToScreen(a.x, a.y);
                        const p2 = worldToScreen(b.x, b.y);
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                    }
                });
                ctx.stroke();
            }
            
            // Draw stations
            if (document.getElementById('showStations').checked) {
                const r = Math.max(2, scale * 0.8);
                addresses.forEach(addr => {
                    if (addr.stations.length > 0) {
                        const p = worldToScreen(addr.x, addr.y);
                        if (p.x > -20 && p.x < width + 20 && p.y > -20 && p.y < height + 20) {
                            addr.stations.forEach(sNo => {
                                const st = stations.get(sNo);
                                if (st) {
                                    ctx.fillStyle = (st.type >= 5 && st.type <= 9) ? '#ff6b6b' : '#ffd93d';
                                    ctx.beginPath();
                                    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
                                    ctx.fill();
                                }
                            });
                        }
                    }
                });
            }
        }
        
        function fitView() {
            if (addresses.size === 0) return;
            const lw = bounds.maxX - bounds.minX;
            const lh = bounds.maxY - bounds.minY;
            scale = Math.min((width - 100) / lw, (height - 100) / lh);
            offsetX = (width - lw * scale) / 2;
            offsetY = (height - lh * scale) / 2;
            render();
        }
        
        function resetView() {
            scale = 0.5;
            offsetX = 0;
            offsetY = 0;
            render();
        }
        
        function loadNew() {
            dropzone.classList.remove('hidden');
            fileInput.value = '';
        }
        
        function search() {
            const q = document.getElementById('searchInput').value.trim().toUpperCase();
            if (!q) return;
            
            for (const [sNo, st] of stations) {
                if (st.portId && st.portId.toUpperCase().includes(q)) {
                    for (const [aNo, addr] of addresses) {
                        if (addr.stations.includes(sNo)) {
                            const p = worldToScreen(addr.x, addr.y);
                            offsetX += width / 2 - p.x;
                            offsetY += height / 2 - (height - p.y);
                            document.getElementById('infoContent').innerHTML = 
                                `<b>${st.portId}</b><br>No: ${sNo}<br>Type: ${st.type}`;
                            render();
                            return;
                        }
                    }
                }
            }
            document.getElementById('infoContent').textContent = '검색 결과 없음';
        }
        
        function resizeCanvas() {
            width = window.innerWidth;
            height = window.innerHeight - 45;
            canvas.width = width;
            canvas.height = height;
            render();
        }
        
        // Events
        canvas.addEventListener('mousedown', e => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });
        
        canvas.addEventListener('mousemove', e => {
            if (isDragging) {
                offsetX += e.clientX - lastX;
                offsetY -= e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                render();
            }
        });
        
        canvas.addEventListener('mouseup', () => isDragging = false);
        canvas.addEventListener('mouseleave', () => isDragging = false);
        
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const zoom = e.deltaY > 0 ? 0.9 : 1.1;
            const mx = e.clientX;
            const my = e.clientY - 45;
            
            const wb = screenToWorld(mx, my);
            scale *= zoom;
            scale = Math.max(0.05, Math.min(10, scale));
            const sa = worldToScreen(wb.x, wb.y);
            offsetX += mx - sa.x;
            offsetY -= my - (height - sa.y);
            render();
        });
        
        document.getElementById('showRails').addEventListener('change', render);
        document.getElementById('showStations').addEventListener('change', render);
        document.getElementById('searchInput').addEventListener('keypress', e => {
            if (e.key === 'Enter') search();
        });
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    </script>
</body>
</html>
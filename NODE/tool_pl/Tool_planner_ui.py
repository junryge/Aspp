<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Planner 테스트 시스템</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .app-layout {
            display: flex;
            min-height: 100vh;
        }

        /* 사이드바 */
        .sidebar {
            width: 280px;
            background: #0f0f1a;
            border-right: 1px solid #2a2a4a;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }

        .sidebar h2 {
            color: #00d4ff;
            font-size: 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid #2a2a4a;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #2a2a4a;
            font-size: 13px;
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-dot.green { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-dot.red { background: #ff4444; box-shadow: 0 0 10px #ff4444; }
        .status-dot.yellow { background: #ffaa00; box-shadow: 0 0 10px #ffaa00; }

        .tool-list {
            flex: 1;
            overflow-y: auto;
            margin-top: 15px;
        }

        .tool-item {
            padding: 10px;
            background: #1a1a2e;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid #2a2a4a;
        }

        .tool-item:hover {
            background: #2a2a4a;
            border-color: #00d4ff;
        }

        .tool-item .name {
            font-weight: 600;
            color: #00d4ff;
            font-size: 13px;
        }

        .tool-item .type {
            font-size: 11px;
            color: #888;
            margin-top: 3px;
        }

        /* 메인 콘텐츠 */
        .main-content {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            color: #00d4ff;
            font-size: 28px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #888;
            margin-bottom: 30px;
        }

        /* 탭 */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
        }

        .tab-btn {
            padding: 12px 25px;
            background: #1a1a2e;
            border: 2px solid #2a2a4a;
            border-radius: 10px;
            color: #e0e0e0;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .tab-btn:hover {
            border-color: #00d4ff;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            border-color: #00d4ff;
            color: #000;
        }

        /* 패널 */
        .panel {
            display: none;
            background: #1a1a2e;
            border-radius: 15px;
            padding: 25px;
            border: 1px solid #2a2a4a;
        }

        .panel.active {
            display: block;
        }

        /* 입력 폼 */
        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #00d4ff;
        }

        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px 15px;
            background: #0f0f1a;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 14px;
            transition: all 0.3s;
        }

        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #00d4ff;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
        }

        .form-group textarea {
            min-height: 150px;
            resize: vertical;
            font-family: monospace;
        }

        .form-row {
            display: flex;
            gap: 20px;
        }

        .form-row .form-group {
            flex: 1;
        }

        /* 버튼 */
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .btn-primary {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #000;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }

        .btn-secondary {
            background: #2a2a4a;
            color: #e0e0e0;
        }

        .btn-secondary:hover {
            background: #3a3a5a;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* 결과 영역 */
        .result-area {
            margin-top: 25px;
            background: #0f0f1a;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #2a2a4a;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .result-header h3 {
            color: #00d4ff;
            font-size: 16px;
        }

        .result-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }

        .result-status.success {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }

        .result-status.fail {
            background: rgba(255, 68, 68, 0.2);
            color: #ff4444;
        }

        .result-content {
            font-family: 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 500px;
            overflow-y: auto;
        }

        /* 트레이스 아코디언 */
        .trace-item {
            background: #1a1a2e;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid #2a2a4a;
            overflow: hidden;
        }

        .trace-header {
            padding: 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }

        .trace-header:hover {
            background: #2a2a4a;
        }

        .trace-header .iter-badge {
            background: #00d4ff;
            color: #000;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 600;
        }

        .trace-body {
            display: none;
            padding: 15px;
            border-top: 1px solid #2a2a4a;
        }

        .trace-body.open {
            display: block;
        }

        .trace-section {
            margin-bottom: 15px;
        }

        .trace-section-title {
            color: #ffaa00;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .trace-section pre {
            background: #0f0f1a;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            overflow-x: auto;
        }

        /* 예시 태그 */
        .examples {
            margin-top: 20px;
        }

        .examples h4 {
            color: #888;
            font-size: 13px;
            margin-bottom: 10px;
        }

        .example-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .example-tag {
            padding: 8px 15px;
            background: #2a2a4a;
            border-radius: 20px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .example-tag:hover {
            background: #00d4ff;
            color: #000;
        }

        /* 로딩 */
        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #00d4ff;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid #2a2a4a;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* 스크롤바 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #0f0f1a;
        }

        ::-webkit-scrollbar-thumb {
            background: #2a2a4a;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #3a3a5a;
        }

        /* 설정 그리드 */
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        /* JSON 뷰어 */
        .json-viewer {
            background: #0f0f1a;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            max-height: 400px;
            overflow: auto;
        }

        .json-key { color: #ff79c6; }
        .json-string { color: #f1fa8c; }
        .json-number { color: #bd93f9; }
        .json-boolean { color: #50fa7b; }
        .json-null { color: #6272a4; }
    </style>
</head>
<body>
    <div class="app-layout">
        <!-- 사이드바 -->
        <aside class="sidebar">
            <h2>🛠️ Tool Planner</h2>
            
            <div class="status-card">
                <div class="status-item">
                    <span>API 상태</span>
                    <span class="status-dot" id="apiStatus"></span>
                </div>
                <div class="status-item">
                    <span>Registry</span>
                    <span class="status-dot" id="registryStatus"></span>
                </div>
                <div class="status-item">
                    <span>Index</span>
                    <span class="status-dot" id="indexStatus"></span>
                </div>
                <div class="status-item">
                    <span>Tool 수</span>
                    <span id="toolCount">0</span>
                </div>
            </div>
            
            <button class="btn btn-secondary" style="width: 100%;" onclick="refreshStatus()">
                🔄 상태 새로고침
            </button>
            
            <div class="tool-list" id="toolList">
                <p style="color: #666; font-size: 12px; text-align: center; padding: 20px;">
                    빌드 후 Tool 목록이 표시됩니다
                </p>
            </div>
        </aside>

        <!-- 메인 콘텐츠 -->
        <main class="main-content">
            <div class="container">
                <h1>🤖 Tool Planner 테스트 시스템</h1>
                <p class="subtitle">AMHS Tool-RAG 파이프라인 디버깅 및 테스트</p>

                <!-- 탭 -->
                <div class="tabs">
                    <button class="tab-btn active" onclick="showTab('build')">📦 빌드</button>
                    <button class="tab-btn" onclick="showTab('test')">🔍 테스트</button>
                    <button class="tab-btn" onclick="showTab('registry')">📋 Registry</button>
                </div>

                <!-- 빌드 패널 -->
                <div class="panel active" id="panel-build">
                    <h3 style="margin-bottom: 20px; color: #00d4ff;">Tool Registry & Index 빌드</h3>
                    
                    <p style="color: #888; margin-bottom: 20px; font-size: 14px;">
                        Tool 정의를 기반으로 Registry(JSON)와 Embedding Index(joblib)를 생성합니다.
                        <br>기본 Tool을 사용하거나, 아래에 커스텀 Tool을 JSON으로 정의할 수 있습니다.
                    </p>

                    <div class="form-group">
                        <label>Tool 정의 (JSON Array) - 비워두면 기본 Tool 사용</label>
                        <textarea id="buildTools" placeholder='[
  {
    "tool_name": "my_tool",
    "tool_type": "runtime_tool",
    "purpose": "설명",
    "required_inputs": {"key": "type"},
    "optional_inputs": {},
    "outputs": {"key": "type"},
    "preconditions": [],
    "forbidden": [],
    "failure_modes": []
  }
]'></textarea>
                    </div>

                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-primary" onclick="runBuild()" id="buildBtn">
                            🚀 빌드 실행
                        </button>
                        <button class="btn btn-secondary" onclick="loadDefaultRegistry()">
                            📥 기본 Tool 불러오기
                        </button>
                    </div>

                    <div class="result-area" id="buildResult" style="display: none;">
                        <div class="result-header">
                            <h3>빌드 결과</h3>
                            <span class="result-status" id="buildResultStatus"></span>
                        </div>
                        <div class="result-content" id="buildResultContent"></div>
                    </div>
                </div>

                <!-- 테스트 패널 -->
                <div class="panel" id="panel-test">
                    <h3 style="margin-bottom: 20px; color: #00d4ff;">Tool Planner 테스트</h3>

                    <div class="form-group">
                        <label>질의 (Query)</label>
                        <input type="text" id="testQuery" placeholder="예: 공장 14의 반송현황이 어때?" autocomplete="off">
                    </div>

                    <div class="settings-grid">
                        <div class="form-group">
                            <label>모델</label>
                            <select id="testModel">
                                <option value="gemini-2.0-flash">gemini-2.0-flash</option>
                                <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                                <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Top-K</label>
                            <input type="number" id="testK" value="4" min="1" max="10">
                        </div>
                        <div class="form-group">
                            <label>Max Iterations</label>
                            <input type="number" id="testMaxIters" value="3" min="1" max="5">
                        </div>
                        <div class="form-group">
                            <label>FAB List (콤마 구분)</label>
                            <input type="text" id="testFabList" value="M12,M14,M16">
                        </div>
                    </div>

                    <button class="btn btn-primary" onclick="runTest()" id="testBtn">
                        🔍 테스트 실행
                    </button>

                    <div class="examples">
                        <h4>📝 예시 질의 (클릭하면 입력됩니다)</h4>
                        <div class="example-tags">
                            <span class="example-tag" onclick="setTestQuery('공장 14 의 반송현황이몬려냐?')">공장14 반송현황 (오타)</span>
                            <span class="example-tag" onclick="setTestQuery('M12, M14 반송 현황 알려줘')">복수 FAB</span>
                            <span class="example-tag" onclick="setTestQuery('ABC123 캐리어 어디있어?')">캐리어 위치</span>
                            <span class="example-tag" onclick="setTestQuery('M16 공장 셔틀 큐 상태')">셔틀 큐 상태</span>
                        </div>
                    </div>

                    <div class="result-area" id="testResult" style="display: none;">
                        <div class="result-header">
                            <h3>테스트 결과</h3>
                            <span class="result-status" id="testResultStatus"></span>
                        </div>
                        <div id="testResultContent"></div>
                    </div>
                </div>

                <!-- Registry 패널 -->
                <div class="panel" id="panel-registry">
                    <h3 style="margin-bottom: 20px; color: #00d4ff;">현재 Tool Registry</h3>
                    
                    <button class="btn btn-secondary" onclick="loadRegistry()" style="margin-bottom: 20px;">
                        🔄 Registry 새로고침
                    </button>

                    <div class="json-viewer" id="registryViewer">
                        Registry를 불러오는 중...
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        // ========================================
        // 상태 관리
        // ========================================
        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // 상태 표시
                document.getElementById('apiStatus').className = 
                    'status-dot ' + (data.api_available ? 'green' : 'red');
                document.getElementById('registryStatus').className = 
                    'status-dot ' + (data.registry_exists ? 'green' : 'yellow');
                document.getElementById('indexStatus').className = 
                    'status-dot ' + (data.index_exists ? 'green' : 'yellow');
                document.getElementById('toolCount').textContent = data.tool_count;
                
                // Tool 목록 업데이트
                const toolList = document.getElementById('toolList');
                if (data.tools && data.tools.length > 0) {
                    toolList.innerHTML = data.tools.map(name => `
                        <div class="tool-item">
                            <div class="name">${name}</div>
                        </div>
                    `).join('');
                } else {
                    toolList.innerHTML = '<p style="color: #666; font-size: 12px; text-align: center; padding: 20px;">빌드 후 Tool 목록이 표시됩니다</p>';
                }
                
            } catch (error) {
                console.error('상태 조회 실패:', error);
            }
        }

        // ========================================
        // 탭 전환
        // ========================================
        function showTab(tabName) {
            // 탭 버튼 활성화
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // 패널 전환
            document.querySelectorAll('.panel').forEach(panel => panel.classList.remove('active'));
            document.getElementById('panel-' + tabName).classList.add('active');
            
            // Registry 탭이면 자동 로드
            if (tabName === 'registry') {
                loadRegistry();
            }
        }

        // ========================================
        // 빌드
        // ========================================
        async function runBuild() {
            const buildBtn = document.getElementById('buildBtn');
            const resultArea = document.getElementById('buildResult');
            const resultStatus = document.getElementById('buildResultStatus');
            const resultContent = document.getElementById('buildResultContent');
            
            const toolsText = document.getElementById('buildTools').value.trim();
            let tools = null;
            
            if (toolsText) {
                try {
                    tools = JSON.parse(toolsText);
                } catch (e) {
                    alert('JSON 파싱 오류: ' + e.message);
                    return;
                }
            }
            
            buildBtn.disabled = true;
            buildBtn.innerHTML = '<div class="loading"><div class="spinner"></div>빌드 중...</div>';
            resultArea.style.display = 'block';
            resultContent.textContent = '임베딩 생성 중... (약 10초 소요)';
            resultStatus.textContent = '';
            resultStatus.className = 'result-status';
            
            try {
                const response = await fetch('/api/build', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tools: tools })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultStatus.textContent = '성공';
                    resultStatus.className = 'result-status success';
                    resultContent.textContent = JSON.stringify(data, null, 2);
                    refreshStatus();
                } else {
                    resultStatus.textContent = '실패';
                    resultStatus.className = 'result-status fail';
                    resultContent.textContent = data.detail || JSON.stringify(data, null, 2);
                }
                
            } catch (error) {
                resultStatus.textContent = '오류';
                resultStatus.className = 'result-status fail';
                resultContent.textContent = error.message;
            } finally {
                buildBtn.disabled = false;
                buildBtn.innerHTML = '🚀 빌드 실행';
            }
        }

        async function loadDefaultRegistry() {
            try {
                const response = await fetch('/api/default-registry');
                const data = await response.json();
                document.getElementById('buildTools').value = JSON.stringify(data.tools, null, 2);
            } catch (error) {
                alert('기본 Registry 로드 실패: ' + error.message);
            }
        }

        // ========================================
        // 테스트
        // ========================================
        function setTestQuery(query) {
            document.getElementById('testQuery').value = query;
        }

        async function runTest() {
            const testBtn = document.getElementById('testBtn');
            const resultArea = document.getElementById('testResult');
            const resultStatus = document.getElementById('testResultStatus');
            const resultContent = document.getElementById('testResultContent');
            
            const query = document.getElementById('testQuery').value.trim();
            if (!query) {
                alert('질의를 입력하세요');
                return;
            }
            
            const fabListText = document.getElementById('testFabList').value.trim();
            const fabList = fabListText.split(',').map(s => s.trim()).filter(s => s);
            
            testBtn.disabled = true;
            testBtn.innerHTML = '<div class="loading"><div class="spinner"></div>테스트 중...</div>';
            resultArea.style.display = 'block';
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div>ReQuery → Retrieval → Planning...</div>';
            resultStatus.textContent = '';
            resultStatus.className = 'result-status';
            
            try {
                const response = await fetch('/api/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        k: parseInt(document.getElementById('testK').value),
                        max_iters: parseInt(document.getElementById('testMaxIters').value),
                        model: document.getElementById('testModel').value,
                        fab_list: fabList
                    })
                });
                
                const data = await response.json();
                
                if (response.ok && data.success) {
                    resultStatus.textContent = data.final_ok ? 'PASS ✓' : 'FAIL ✗';
                    resultStatus.className = 'result-status ' + (data.final_ok ? 'success' : 'fail');
                    resultContent.innerHTML = renderTrace(data);
                } else {
                    resultStatus.textContent = '오류';
                    resultStatus.className = 'result-status fail';
                    resultContent.innerHTML = `<pre>${data.detail || JSON.stringify(data, null, 2)}</pre>`;
                }
                
            } catch (error) {
                resultStatus.textContent = '오류';
                resultStatus.className = 'result-status fail';
                resultContent.innerHTML = `<pre>${error.message}</pre>`;
            } finally {
                testBtn.disabled = false;
                testBtn.innerHTML = '🔍 테스트 실행';
            }
        }

        function renderTrace(data) {
            let html = `<div style="margin-bottom: 15px;">
                <strong>Query:</strong> ${data.query}<br>
                <strong>Final OK:</strong> <span style="color: ${data.final_ok ? '#00ff88' : '#ff4444'}">${data.final_ok}</span>
            </div>`;
            
            data.trace.forEach((step, idx) => {
                const isLast = idx === data.trace.length - 1;
                const isOk = step.static_validation && step.static_validation.ok;
                
                html += `
                <div class="trace-item">
                    <div class="trace-header" onclick="toggleTrace(${idx})">
                        <div>
                            <span class="iter-badge">ITER ${step.iter}</span>
                            <span style="margin-left: 10px; color: ${isOk ? '#00ff88' : '#ff4444'}">
                                ${isOk ? '✓ PASS' : '✗ FAIL'}
                            </span>
                        </div>
                        <span id="trace-arrow-${idx}">▼</span>
                    </div>
                    <div class="trace-body" id="trace-body-${idx}" ${isLast ? 'style="display:block;"' : ''}>
                        <div class="trace-section">
                            <div class="trace-section-title">📝 ReQuery</div>
                            <pre>${JSON.stringify(step.requery, null, 2)}</pre>
                        </div>
                        <div class="trace-section">
                            <div class="trace-section-title">🔍 Retrieved Tools (Top-K)</div>
                            <pre>${step.retrieved_scores.map(([name, score]) => 
                                `${name.padEnd(24)} score=${score.toFixed(4)}`
                            ).join('\n')}</pre>
                        </div>
                        <div class="trace-section">
                            <div class="trace-section-title">📋 Plan JSON</div>
                            <pre>${step.plan ? JSON.stringify(step.plan, null, 2) : 'null'}</pre>
                        </div>
                        <div class="trace-section">
                            <div class="trace-section-title">✅ Static Validation</div>
                            <pre style="color: ${isOk ? '#00ff88' : '#ff4444'}">${JSON.stringify(step.static_validation, null, 2)}</pre>
                        </div>
                        ${step.feedbacks && step.feedbacks.length > 0 ? `
                        <div class="trace-section">
                            <div class="trace-section-title">💬 Feedbacks</div>
                            <pre>${step.feedbacks.map(f => '- ' + f).join('\n')}</pre>
                        </div>
                        ` : ''}
                    </div>
                </div>`;
            });
            
            return html;
        }

        function toggleTrace(idx) {
            const body = document.getElementById('trace-body-' + idx);
            const arrow = document.getElementById('trace-arrow-' + idx);
            
            if (body.style.display === 'block') {
                body.style.display = 'none';
                arrow.textContent = '▼';
            } else {
                body.style.display = 'block';
                arrow.textContent = '▲';
            }
        }

        // ========================================
        // Registry
        // ========================================
        async function loadRegistry() {
            const viewer = document.getElementById('registryViewer');
            
            try {
                const response = await fetch('/api/registry');
                const data = await response.json();
                
                if (data.tools && data.tools.length > 0) {
                    viewer.innerHTML = syntaxHighlight(JSON.stringify(data.tools, null, 2));
                } else {
                    viewer.innerHTML = '<span style="color: #666;">Registry가 비어있습니다. 먼저 빌드하세요.</span>';
                }
                
            } catch (error) {
                viewer.innerHTML = '<span style="color: #ff4444;">Registry 로드 실패: ' + error.message + '</span>';
            }
        }

        function syntaxHighlight(json) {
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }

        // ========================================
        // 초기화
        // ========================================
        window.addEventListener('load', function() {
            refreshStatus();
            
            // Enter 키로 테스트 실행
            document.getElementById('testQuery').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    runTest();
                }
            });
        });
    </script>
</body>
</html>
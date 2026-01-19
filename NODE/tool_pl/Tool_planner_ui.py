<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Planner 테스트</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #888; font-size: 1.1em; }
        
        /* Status Bar */
        .status-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .status-item {
            background: #252540;
            padding: 12px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .status-dot.green { background: #4CAF50; }
        .status-dot.red { background: #f44336; }
        .status-dot.yellow { background: #ffc107; }
        
        /* Env Toggle */
        .env-toggle {
            display: flex;
            background: #1a1a2e;
            border-radius: 20px;
            padding: 4px;
        }
        .env-btn {
            padding: 8px 16px;
            border: none;
            background: none;
            color: #888;
            cursor: pointer;
            border-radius: 16px;
            transition: all 0.3s;
        }
        .env-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .env-btn:hover:not(.active) { color: #fff; }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab-btn {
            padding: 12px 24px;
            background: #252540;
            border: none;
            color: #888;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            font-size: 1em;
            transition: all 0.3s;
        }
        .tab-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .tab-btn:hover:not(.active) { color: #fff; }
        
        /* Tab Content */
        .tab-content {
            display: none;
            background: #252540;
            padding: 25px;
            border-radius: 0 8px 8px 8px;
        }
        .tab-content.active { display: block; }
        
        /* Form Elements */
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-weight: 500;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 1em;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            min-height: 200px;
            font-family: 'Consolas', monospace;
        }
        
        /* Buttons */
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #333;
            color: #e0e0e0;
        }
        .btn-secondary:hover { background: #444; }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Results */
        .result-box {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            white-space: pre-wrap;
            font-family: 'Consolas', monospace;
            font-size: 0.9em;
            max-height: 600px;
            overflow-y: auto;
        }
        .result-ok { border-left: 4px solid #4CAF50; }
        .result-fail { border-left: 4px solid #f44336; }
        
        /* Accordion */
        .accordion {
            margin-top: 15px;
        }
        .accordion-item {
            background: #1a1a2e;
            border-radius: 8px;
            margin-bottom: 10px;
            overflow: hidden;
        }
        .accordion-header {
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid #667eea;
        }
        .accordion-header:hover { background: #252540; }
        .accordion-header.ok { border-left-color: #4CAF50; }
        .accordion-header.fail { border-left-color: #f44336; }
        .accordion-body {
            display: none;
            padding: 15px 20px;
            border-top: 1px solid #333;
        }
        .accordion-body.show { display: block; }
        
        /* Examples */
        .examples {
            margin-top: 15px;
        }
        .example-tag {
            display: inline-block;
            padding: 8px 16px;
            background: #333;
            border-radius: 20px;
            margin: 5px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }
        .example-tag:hover {
            background: #667eea;
            color: white;
        }
        
        /* Tool List */
        .tool-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .tool-badge {
            background: #333;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        /* Inline Settings */
        .inline-settings {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .inline-settings .form-group {
            margin-bottom: 0;
            flex: 1;
            min-width: 100px;
        }
        .inline-settings input {
            padding: 8px 12px;
        }
        
        /* JSON Display */
        .json-display {
            background: #0d0d1a;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
            overflow-x: auto;
        }
        
        /* Spinner */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #333;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛠️ Tool Planner 테스트</h1>
            <p class="subtitle">ReQuery → Tool Retrieval → Planning → Static Validation</p>
        </header>
        
        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot" id="apiDot"></span>
                <span id="apiStatus">API 상태 확인 중...</span>
            </div>
            <div class="status-item">
                <span class="status-dot" id="registryDot"></span>
                <span id="registryStatus">Registry: -</span>
            </div>
            <div class="status-item">
                <span class="status-dot" id="indexDot"></span>
                <span id="indexStatus">Index: -</span>
            </div>
            <div class="status-item">
                <div class="env-toggle">
                    <button class="env-btn active" id="env-dev" onclick="switchEnv('dev')">개발(30B)</button>
                    <button class="env-btn" id="env-prod" onclick="switchEnv('prod')">운영(80B)</button>
                </div>
            </div>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('build')">🔧 빌드</button>
            <button class="tab-btn" onclick="showTab('test')">🧪 테스트</button>
            <button class="tab-btn" onclick="showTab('registry')">📋 Registry</button>
        </div>
        
        <!-- Build Tab -->
        <div class="tab-content active" id="tab-build">
            <h3 style="margin-bottom: 15px;">Tool Registry & Index 빌드</h3>
            <p style="color: #888; margin-bottom: 20px;">
                기본 Tool Registry를 사용하거나, 커스텀 JSON을 입력하세요.
            </p>
            
            <div class="form-group">
                <label>Tool Registry JSON (비워두면 기본값 사용)</label>
                <textarea id="buildJson" placeholder='[{"tool_name": "...", "tool_type": "...", ...}]'></textarea>
            </div>
            
            <button class="btn btn-primary" onclick="buildArtifacts()">
                🚀 빌드 실행
            </button>
            <button class="btn btn-secondary" onclick="loadDefaultRegistry()">
                📥 기본 Registry 불러오기
            </button>
            
            <div class="result-box" id="buildResult" style="display: none;"></div>
        </div>
        
        <!-- Test Tab -->
        <div class="tab-content" id="tab-test">
            <h3 style="margin-bottom: 15px;">Tool Planner 테스트</h3>
            
            <div class="inline-settings">
                <div class="form-group">
                    <label>Top-K</label>
                    <input type="number" id="testK" value="4" min="1" max="10">
                </div>
                <div class="form-group">
                    <label>Max Iters</label>
                    <input type="number" id="testMaxIters" value="3" min="1" max="5">
                </div>
                <div class="form-group">
                    <label>FAB List</label>
                    <input type="text" id="testFabList" value="M12,M14,M16">
                </div>
            </div>
            
            <div class="form-group">
                <label>질의 입력</label>
                <input type="text" id="testQuery" placeholder="예: 공장 14의 반송현황 알려줘">
            </div>
            
            <button class="btn btn-primary" id="testBtn" onclick="runTest()">
                ▶️ 테스트 실행
            </button>
            
            <div class="examples">
                <span style="color: #888;">예시:</span>
                <span class="example-tag" onclick="setQuery('공장 14의 반송현황이몬려냐?')">오타 테스트</span>
                <span class="example-tag" onclick="setQuery('M14 셔틀 현황 알려줘')">FAB 직접 입력</span>
                <span class="example-tag" onclick="setQuery('ABC123 캐리어 어디있어?')">ID 추출</span>
                <span class="example-tag" onclick="setQuery('M12, M16 반송 비교해줘')">복수 FAB</span>
            </div>
            
            <div id="testResult"></div>
        </div>
        
        <!-- Registry Tab -->
        <div class="tab-content" id="tab-registry">
            <h3 style="margin-bottom: 15px;">현재 Tool Registry</h3>
            <button class="btn btn-secondary" onclick="loadRegistry()">🔄 새로고침</button>
            
            <div id="registryContent" style="margin-top: 20px;">
                <p style="color: #888;">빌드 후 확인 가능합니다.</p>
            </div>
        </div>
    </div>
    
    <script>
        // Tab 전환
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');
        }
        
        // 상태 확인
        async function checkStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                // API 상태
                const apiDot = document.getElementById('apiDot');
                const apiStatus = document.getElementById('apiStatus');
                if (data.api_available) {
                    apiDot.className = 'status-dot green';
                    apiStatus.textContent = `API: ${data.env_name}`;
                } else {
                    apiDot.className = 'status-dot red';
                    apiStatus.textContent = 'API: ' + data.api_message;
                }
                
                // 환경 버튼 업데이트
                document.querySelectorAll('.env-btn').forEach(b => b.classList.remove('active'));
                document.getElementById('env-' + data.env_mode).classList.add('active');
                
                // Registry 상태
                const regDot = document.getElementById('registryDot');
                const regStatus = document.getElementById('registryStatus');
                if (data.registry_exists) {
                    regDot.className = 'status-dot green';
                    regStatus.textContent = `Registry: ${data.tool_count} tools`;
                } else {
                    regDot.className = 'status-dot yellow';
                    regStatus.textContent = 'Registry: 없음';
                }
                
                // Index 상태
                const idxDot = document.getElementById('indexDot');
                const idxStatus = document.getElementById('indexStatus');
                if (data.index_exists && data.retriever_ready) {
                    idxDot.className = 'status-dot green';
                    idxStatus.textContent = 'Index: 준비됨';
                } else {
                    idxDot.className = 'status-dot yellow';
                    idxStatus.textContent = 'Index: 없음';
                }
                
            } catch (e) {
                console.error('Status check failed:', e);
            }
        }
        
        // 환경 전환
        async function switchEnv(env) {
            try {
                const res = await fetch('/api/switch_env', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({env})
                });
                const data = await res.json();
                if (data.success) {
                    checkStatus();
                }
            } catch (e) {
                alert('환경 전환 실패: ' + e);
            }
        }
        
        // 빌드
        async function buildArtifacts() {
            const jsonText = document.getElementById('buildJson').value.trim();
            const resultDiv = document.getElementById('buildResult');
            
            resultDiv.style.display = 'block';
            resultDiv.className = 'result-box';
            resultDiv.innerHTML = '<span class="spinner"></span> 빌드 중...';
            
            try {
                let body = {};
                if (jsonText) {
                    body.tools = JSON.parse(jsonText);
                }
                
                const res = await fetch('/api/build', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                
                const data = await res.json();
                
                if (data.success) {
                    resultDiv.className = 'result-box result-ok';
                    resultDiv.textContent = `✅ ${data.message}\n\nTools: ${data.tools.join(', ')}\nHash: ${data.registry_hash}`;
                } else {
                    resultDiv.className = 'result-box result-fail';
                    resultDiv.textContent = '❌ 빌드 실패: ' + (data.detail || JSON.stringify(data));
                }
                
                checkStatus();
                
            } catch (e) {
                resultDiv.className = 'result-box result-fail';
                resultDiv.textContent = '❌ 오류: ' + e.message;
            }
        }
        
        // 기본 Registry 로드
        async function loadDefaultRegistry() {
            try {
                const res = await fetch('/api/default-registry');
                const data = await res.json();
                document.getElementById('buildJson').value = JSON.stringify(data.tools, null, 2);
            } catch (e) {
                alert('로드 실패: ' + e);
            }
        }
        
        // 테스트 실행
        async function runTest() {
            const query = document.getElementById('testQuery').value.trim();
            if (!query) {
                alert('질의를 입력하세요');
                return;
            }
            
            const k = parseInt(document.getElementById('testK').value) || 4;
            const maxIters = parseInt(document.getElementById('testMaxIters').value) || 3;
            const fabList = document.getElementById('testFabList').value.split(',').map(s => s.trim()).filter(s => s);
            
            const resultDiv = document.getElementById('testResult');
            const testBtn = document.getElementById('testBtn');
            
            testBtn.disabled = true;
            testBtn.innerHTML = '<span class="spinner"></span> 테스트 중...';
            resultDiv.innerHTML = '';
            
            try {
                const res = await fetch('/api/test', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query,
                        k,
                        max_iters: maxIters,
                        fab_list: fabList
                    })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    renderTestResult(data);
                } else {
                    resultDiv.innerHTML = `<div class="result-box result-fail">❌ 오류: ${data.detail || JSON.stringify(data)}</div>`;
                }
                
            } catch (e) {
                resultDiv.innerHTML = `<div class="result-box result-fail">❌ 오류: ${e.message}</div>`;
            } finally {
                testBtn.disabled = false;
                testBtn.innerHTML = '▶️ 테스트 실행';
            }
        }
        
        // 테스트 결과 렌더링
        function renderTestResult(data) {
            const resultDiv = document.getElementById('testResult');
            
            const finalOk = data.final_ok;
            const statusClass = finalOk ? 'result-ok' : 'result-fail';
            const statusIcon = finalOk ? '✅' : '❌';
            
            let html = `
                <div class="result-box ${statusClass}" style="margin-top: 20px;">
                    ${statusIcon} 최종 결과: ${finalOk ? 'PASS' : 'FAIL'}
                </div>
                <div class="accordion">
            `;
            
            for (const step of data.trace) {
                const stepOk = step.static_validation?.ok;
                const headerClass = stepOk ? 'ok' : 'fail';
                const stepIcon = stepOk ? '✅' : '❌';
                
                html += `
                    <div class="accordion-item">
                        <div class="accordion-header ${headerClass}" onclick="toggleAccordion(this)">
                            <span>${stepIcon} Iteration ${step.iter}</span>
                            <span>▼</span>
                        </div>
                        <div class="accordion-body">
                            <h4 style="margin-bottom: 10px; color: #667eea;">1. ReQuery</h4>
                            <div class="json-display">${JSON.stringify(step.requery, null, 2)}</div>
                            
                            <h4 style="margin: 15px 0 10px; color: #667eea;">2. Retrieved Tools</h4>
                            <div class="tool-list">
                                ${step.retrieved_scores.map(([name, score]) => 
                                    `<span class="tool-badge">${name} (${score.toFixed(3)})</span>`
                                ).join('')}
                            </div>
                            
                            <h4 style="margin: 15px 0 10px; color: #667eea;">3. Plan</h4>
                            <div class="json-display">${step.plan ? JSON.stringify(step.plan, null, 2) : 'null'}</div>
                            
                            <h4 style="margin: 15px 0 10px; color: #667eea;">4. Static Validation</h4>
                            <div class="json-display">${JSON.stringify(step.static_validation, null, 2)}</div>
                            
                            ${step.feedbacks.length > 0 ? `
                                <h4 style="margin: 15px 0 10px; color: #f44336;">Feedbacks</h4>
                                <ul style="padding-left: 20px; color: #f44336;">
                                    ${step.feedbacks.map(fb => `<li>${fb}</li>`).join('')}
                                </ul>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            resultDiv.innerHTML = html;
        }
        
        // 아코디언 토글
        function toggleAccordion(header) {
            const body = header.nextElementSibling;
            body.classList.toggle('show');
        }
        
        // 예시 질의 설정
        function setQuery(q) {
            document.getElementById('testQuery').value = q;
        }
        
        // Registry 로드
        async function loadRegistry() {
            const content = document.getElementById('registryContent');
            
            try {
                const res = await fetch('/api/registry');
                const data = await res.json();
                
                if (data.tools && data.tools.length > 0) {
                    let html = '';
                    for (const tool of data.tools) {
                        html += `
                            <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">${tool.tool_name}</h4>
                                <p style="color: #888; margin-bottom: 10px;">${tool.purpose}</p>
                                <div class="json-display" style="font-size: 0.8em;">${JSON.stringify(tool, null, 2)}</div>
                            </div>
                        `;
                    }
                    content.innerHTML = html;
                } else {
                    content.innerHTML = '<p style="color: #888;">Registry가 비어있습니다. 빌드를 먼저 실행하세요.</p>';
                }
                
            } catch (e) {
                content.innerHTML = '<p style="color: #f44336;">로드 실패: ' + e.message + '</p>';
            }
        }
        
        // 초기화
        window.addEventListener('load', () => {
            checkStatus();
            setInterval(checkStatus, 30000);
        });
        
        // Enter 키로 테스트 실행
        document.getElementById('testQuery').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runTest();
        });
    </script>
</body>
</html>
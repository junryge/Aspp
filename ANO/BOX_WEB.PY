"""
Box Position Anomaly Detection - Web Version
깔끔한 디자인
"""

import os, cv2, numpy as np, requests, re, base64
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)
reference_data = {'center': None, 'angle': None}

# LLM 설정
LLM_URL = "http://common.llm.skhynix.com/v1/chat/completions"
LLM_MODEL = "gpt-oss-20b"
API_TOKEN = None

for p in ["token.txt", "../token.txt"]:
    if os.path.exists(p):
        API_TOKEN = open(p).read().strip()
        if API_TOKEN: print(f"✅ 토큰 로드: {p}"); break

def call_llm(prompt, sys=""):
    if not API_TOKEN: return {"success": False, "error": "토큰없음"}
    try:
        msgs = []
        if sys: msgs.append({"role":"system","content":sys})
        msgs.append({"role":"user","content":prompt})
        
        print(f"🚀 LLM 호출: {prompt[:50]}...")
        r = requests.post(LLM_URL, headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
            json={"model": LLM_MODEL, "messages": msgs, "max_tokens": 2000, "temperature": 0.3}, timeout=120)
        
        print(f"📡 응답 코드: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            c = data["choices"][0]["message"]["content"]
            c = re.sub(r'<think>.*?</think>', '', c, flags=re.DOTALL).strip()
            print(f"✅ LLM 응답: {c[:100]}...")
            return {"success": True, "content": c}
        else:
            print(f"❌ API 오류: {r.text}")
            return {"success": False, "error": f"API오류:{r.status_code} - {r.text[:100]}"}
    except Exception as e:
        print(f"❌ 예외: {e}")
        return {"success": False, "error": str(e)}

def find_center(g):
    _, t = cv2.threshold(g, 80, 255, cv2.THRESH_BINARY)
    c, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c: return None
    M = cv2.moments(max(c, key=cv2.contourArea))
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] else None

def find_angle(g):
    _, t = cv2.threshold(g, 80, 255, cv2.THRESH_BINARY)
    c, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c: return 0
    a = cv2.minAreaRect(max(c, key=cv2.contourArea))[2]
    return a + 90 if a < -45 else a

def detect(img, pt=30, at=10):
    if not reference_data['center']: return {'error': '기준 미설정'}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c, a = find_center(g), find_angle(g)
    if not c: return {'is_anomaly': True, 'type': '박스없음', 'shift': -1, 'angle': -1}
    ps = np.sqrt((c[0]-reference_data['center'][0])**2 + (c[1]-reference_data['center'][1])**2)
    aa = abs(a - reference_data['angle'])
    ip, ia = ps > pt, aa > at
    return {'is_anomaly': bool(ip or ia), 'type': '위치+회전' if ip and ia else '위치이탈' if ip else '회전' if ia else '정상',
            'shift': round(ps,1), 'angle': round(aa,1), 'center': c, 'ref': reference_data['center']}

def draw(img, r):
    img = img.copy()
    col = (0,0,255) if r.get('is_anomaly') else (0,255,0)
    cv2.rectangle(img, (3,3), (img.shape[1]-3, img.shape[0]-3), col, 3)
    if r.get('center'): cv2.circle(img, r['center'], 8, col, -1)
    if r.get('ref'): cv2.circle(img, r['ref'], 6, (255,0,0), 2)
    return img

def to_b64(img):
    _, b = cv2.imencode('.png', img)
    return base64.b64encode(b).decode()

HTML = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>박스 위치 이상 탐지</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; overflow:hidden; }
body { font-family:'Malgun Gothic',sans-serif; background:#111827; color:#fff; }

/* 스크롤바 숨김 */
::-webkit-scrollbar { display:none; }
* { -ms-overflow-style:none; scrollbar-width:none; }

/* 헤더 */
header { background:#1f2937; padding:12px 24px; display:flex; justify-content:space-between; align-items:center; }
header h1 { font-size:18px; color:#60a5fa; }
header h1 span { color:#6b7280; font-size:12px; margin-left:8px; }
.stats { display:flex; gap:24px; }
.stats > div { text-align:center; }
.stats .num { font-size:24px; font-weight:bold; }
.stats .num.blue { color:#60a5fa; }
.stats .num.green { color:#34d399; }
.stats .num.red { color:#f87171; }
.stats .label { font-size:10px; color:#9ca3af; }

/* 메인 */
main { display:flex; height:calc(100vh - 56px); }

/* 사이드바 */
.sidebar { width:260px; background:#1f2937; padding:14px; border-right:1px solid #374151; overflow-y:auto; }
.sidebar h2 { font-size:13px; color:#9ca3af; margin-bottom:10px; display:flex; align-items:center; gap:6px; }
.sidebar h2 .dot { width:8px; height:8px; border-radius:50%; }
.sidebar h2 .dot.red { background:#f87171; }
.sidebar h2 .dot.green { background:#34d399; }

.upload-box { border:2px dashed #374151; border-radius:8px; padding:16px; text-align:center; cursor:pointer; margin-bottom:10px; transition:0.2s; }
.upload-box:hover { border-color:#60a5fa; background:rgba(96,165,250,0.1); }
.upload-box .icon { font-size:28px; }
.upload-box p { font-size:11px; color:#6b7280; }
.upload-box input { display:none; }

.preview { width:100%; max-height:80px; object-fit:contain; border-radius:6px; margin-bottom:8px; display:none; }

.btn { width:100%; padding:10px; border:none; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; margin-bottom:8px; transition:0.2s; }
.btn:disabled { opacity:0.4; cursor:not-allowed; }
.btn-blue { background:#3b82f6; color:#fff; }
.btn-green { background:#10b981; color:#fff; }
.btn-purple { background:#8b5cf6; color:#fff; }

.input-row { display:flex; gap:6px; margin-bottom:10px; }
.input-row > div { flex:1; }
.input-row label { display:block; font-size:9px; color:#6b7280; margin-bottom:2px; }
.input-row input { width:100%; padding:6px; background:#374151; border:1px solid #4b5563; border-radius:4px; color:#fff; font-size:11px; }

/* 컨텐츠 */
.content { flex:1; display:flex; flex-direction:column; }

/* 결과 영역 */
.result-area { flex:1; display:flex; gap:12px; padding:12px; overflow:hidden; }

.result-card { flex:1; background:#1f2937; border-radius:10px; padding:12px; display:flex; flex-direction:column; }
.result-card h3 { font-size:13px; color:#9ca3af; margin-bottom:10px; }
.result-card.success { border:2px solid #34d399; }
.result-card.fail { border:2px solid #f87171; }

.result-status { font-size:20px; font-weight:bold; margin-bottom:12px; }
.result-status.green { color:#34d399; }
.result-status.red { color:#f87171; }

.result-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:10px; }
.result-item { background:#374151; padding:8px; border-radius:6px; text-align:center; }
.result-item .label { font-size:9px; color:#6b7280; }
.result-item .value { font-size:14px; font-weight:bold; }

.result-img { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
.result-img img { max-width:100%; max-height:280px; object-fit:contain; border-radius:6px; }

.empty-state { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#4b5563; }
.empty-state .icon { font-size:36px; margin-bottom:8px; }
.empty-state p { font-size:12px; }

/* AI 분석 */
.ai-card { flex:1; background:#1f2937; border-radius:10px; padding:12px; display:flex; flex-direction:column; border:1px solid #4b5563; }
.ai-card h3 { font-size:13px; color:#a78bfa; margin-bottom:10px; }
.ai-content { flex:1; overflow-y:auto; font-size:11px; line-height:1.5; }
.ai-btns { display:flex; gap:6px; margin-top:10px; }
.ai-btns .btn { flex:1; padding:6px; font-size:10px; }

/* 마크다운 */
.md h1,.md h2,.md h3,.md h4 { color:#60a5fa; font-size:12px; margin:10px 0 5px; }
.md h3 { color:#a78bfa; }
.md table { width:100%; border-collapse:collapse; margin:6px 0; font-size:10px; }
.md th,.md td { border:1px solid #4b5563; padding:4px 6px; text-align:left; }
.md th { background:#374151; color:#60a5fa; }
.md ul,.md ol { padding-left:16px; margin:4px 0; }
.md strong { color:#34d399; }
.md hr { border:none; border-top:1px solid #4b5563; margin:8px 0; }

/* 히스토리 */
.history { height:110px; background:#1f2937; border-top:1px solid #374151; padding:12px 16px; margin-bottom:10px; }
.history h3 { font-size:12px; color:#9ca3af; margin-bottom:8px; }
.history-list { display:flex; gap:10px; overflow-x:auto; }
.history-item { flex-shrink:0; display:flex; align-items:center; gap:8px; background:#374151; padding:8px 12px; border-radius:6px; cursor:pointer; transition:0.2s; }
.history-item:hover { background:#4b5563; transform:translateY(-2px); }
.history-item.green { border-left:3px solid #34d399; }
.history-item.red { border-left:3px solid #f87171; }
.history-item img { width:45px; height:45px; object-fit:cover; border-radius:4px; }
.history-item .info { font-size:11px; }
.history-item .info strong { display:block; }
.history-item .info span { color:#6b7280; }
.history-empty { color:#4b5563; font-size:11px; padding:12px; }

/* 히스토리 상세 모달 */
.hist-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:200; align-items:center; justify-content:center; }
.hist-modal-box { background:#1f2937; border-radius:12px; padding:20px; width:90%; max-width:700px; max-height:85vh; overflow-y:auto; }
.hist-modal-box h2 { font-size:18px; margin-bottom:16px; display:flex; align-items:center; gap:10px; }
.hist-modal-box h2.green { color:#34d399; }
.hist-modal-box h2.red { color:#f87171; }
.hist-modal-content { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.hist-modal-img { text-align:center; }
.hist-modal-img img { max-width:100%; max-height:300px; border-radius:8px; }
.hist-modal-info { }
.hist-modal-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }
.hist-modal-item { background:#374151; padding:12px; border-radius:8px; text-align:center; }
.hist-modal-item .label { font-size:11px; color:#6b7280; margin-bottom:4px; }
.hist-modal-item .value { font-size:18px; font-weight:bold; }
.hist-close { width:100%; margin-top:16px; }

/* 모달 */
.modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.8); z-index:100; align-items:center; justify-content:center; }
.modal-box { background:#1f2937; border-radius:10px; padding:20px; width:90%; max-width:450px; }
.modal-box h2 { color:#a78bfa; font-size:14px; margin-bottom:12px; }
.modal-box label { display:block; font-size:10px; color:#6b7280; margin-bottom:4px; }
.modal-box textarea { width:100%; background:#374151; border:1px solid #4b5563; border-radius:6px; color:#fff; padding:8px; font-size:10px; resize:none; margin-bottom:10px; }
.modal-btns { display:flex; gap:8px; }
.modal-btns .btn { flex:1; }
</style>
</head>
<body>

<header>
    <h1>📦 박스 위치 이상 탐지 <span>BATA_VER_V0.1</span></h1>
    <div class="stats">
        <div><div class="num blue" id="sT">0</div><div class="label">총 검사</div></div>
        <div><div class="num green" id="sN">0</div><div class="label">정상</div></div>
        <div><div class="num red" id="sA">0</div><div class="label">이상</div></div>
    </div>
</header>

<main>
    <div class="sidebar">
        <h2><span class="dot" id="refDot"></span> 기준 이미지</h2>
        <div class="upload-box" onclick="$('#refIn').click()">
            <input type="file" id="refIn" accept="image/*">
            <div class="icon">📷</div>
            <p>정상 박스 이미지 업로드</p>
        </div>
        <img id="refPv" class="preview">
        <button class="btn btn-green" id="refBtn" disabled onclick="setRef()">기준으로 설정</button>
        
        <div class="input-row">
            <div><label>위치 오차 (px)</label><input type="number" id="posT" value="30"></div>
            <div><label>각도 오차 (°)</label><input type="number" id="angT" value="10"></div>
        </div>
        
        <h2 style="margin-top:16px"><span class="dot red" id="testDot"></span> 테스트 이미지</h2>
        <div class="upload-box" onclick="$('#testIn').click()">
            <input type="file" id="testIn" accept="image/*">
            <div class="icon">🔍</div>
            <p>분석할 이미지 업로드</p>
        </div>
        <img id="testPv" class="preview">
        <button class="btn btn-blue" id="anaBtn" disabled onclick="analyze()">이상 탐지 분석</button>
    </div>
    
    <div class="content">
        <div class="result-area">
            <div class="result-card" id="resultCard">
                <h3>📊 분석 결과</h3>
                <div class="empty-state" id="emptyResult">
                    <div class="icon">📋</div>
                    <p>이미지를 분석하면 결과가 표시됩니다</p>
                </div>
                <div id="resultContent" style="display:none;flex:1;display:none;flex-direction:column;">
                    <div class="result-status" id="resultStatus"></div>
                    <div class="result-grid">
                        <div class="result-item"><div class="label">유형</div><div class="value" id="rType">-</div></div>
                        <div class="result-item"><div class="label">이탈거리</div><div class="value" id="rShift">-</div></div>
                        <div class="result-item"><div class="label">회전각도</div><div class="value" id="rAngle">-</div></div>
                        <div class="result-item"><div class="label">판정</div><div class="value" id="rJudge">-</div></div>
                    </div>
                    <div class="result-img"><img id="rImg"></div>
                </div>
            </div>
            
            <div class="ai-card" id="aiCard">
                <h3>🤖 AI 원인 분석</h3>
                <div class="empty-state" id="emptyAI">
                    <div class="icon">🤖</div>
                    <p>분석 완료 후 AI 분석을 실행하세요</p>
                </div>
                <div class="ai-content md" id="aiContent" style="display:none;"></div>
                <div class="ai-btns" id="aiBtns" style="display:none;">
                    <button class="btn btn-purple" id="aiBtn" onclick="aiAnalyze()">🤖 AI 분석</button>
                    <button class="btn" style="background:#4b5563" onclick="openModal()">⚙️</button>
                </div>
            </div>
        </div>
        
        <div class="history">
            <h3>📜 검사 히스토리</h3>
            <div class="history-list" id="hist">
                <div class="history-empty">아직 검사 기록이 없습니다</div>
            </div>
        </div>
    </div>
</main>

<div class="modal" id="modal">
    <div class="modal-box">
        <h2>⚙️ 프롬프트 설정</h2>
        <label>시스템 프롬프트</label>
        <textarea id="sysP" rows="3">당신은 FAB 공정 품질관리 전문가입니다. 박스 위치 이상 탐지 결과를 분석하세요.</textarea>
        <label>사용자 프롬프트</label>
        <textarea id="usrP" rows="4">결과: {is_anomaly}, {type}, 이탈:{shift}px, 회전:{angle}°
분석: 원인, 영향, 조치, 예방</textarea>
        <div class="modal-btns">
            <button class="btn btn-green" onclick="saveP()">저장</button>
            <button class="btn" style="background:#4b5563" onclick="closeModal()">닫기</button>
        </div>
    </div>
</div>

<!-- 히스토리 상세보기 모달 -->
<div class="hist-modal" id="histModal">
    <div class="hist-modal-box">
        <h2 id="histModalTitle">📊 검사 상세</h2>
        <div class="hist-modal-content">
            <div class="hist-modal-img">
                <img id="histModalImg">
            </div>
            <div class="hist-modal-info">
                <div class="hist-modal-grid">
                    <div class="hist-modal-item"><div class="label">유형</div><div class="value" id="histType">-</div></div>
                    <div class="hist-modal-item"><div class="label">이탈거리</div><div class="value" id="histShift">-</div></div>
                    <div class="hist-modal-item"><div class="label">회전각도</div><div class="value" id="histAngle">-</div></div>
                    <div class="hist-modal-item"><div class="label">판정</div><div class="value" id="histJudge">-</div></div>
                </div>
                <p style="color:#6b7280;font-size:11px;" id="histTime"></p>
            </div>
        </div>
        <button class="btn hist-close" style="background:#4b5563" onclick="closeHistModal()">닫기</button>
    </div>
</div>

<script>
const $=s=>document.querySelector(s);
const $$=s=>document.querySelectorAll(s);
let refD,testD,lastR,st={t:0,n:0,a:0};
let historyData = []; // 히스토리 데이터 저장

$('#refIn').onchange=e=>{if(e.target.files[0])loadImg(e.target.files[0],'ref')};
$('#testIn').onchange=e=>{if(e.target.files[0])loadImg(e.target.files[0],'test')};

function loadImg(f,t){
    const r=new FileReader();
    r.onload=e=>{
        if(t==='ref'){
            refD=e.target.result;
            $('#refPv').src=refD;
            $('#refPv').style.display='block';
            $('#refBtn').disabled=false;
        }else{
            testD=e.target.result;
            $('#testPv').src=testD;
            $('#testPv').style.display='block';
            $('#anaBtn').disabled=false;
        }
    };
    r.readAsDataURL(f);
}

async function setRef(){
    const r=await fetch('/set_ref',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({img:refD})});
    const d=await r.json();
    if(d.ok){
        $('#refDot').className='dot green';
        $('#refBtn').textContent='✓ 설정완료';
        $('#refBtn').disabled=true;
    }else alert('오류: '+d.error);
}

async function analyze(){
    const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        img:testD, pt:+$('#posT').value, at:+$('#angT').value
    })});
    const d=await r.json();
    if(d.error){alert('오류: '+d.error);return;}
    
    lastR=d;
    
    // 결과 표시
    $('#emptyResult').style.display='none';
    $('#resultContent').style.display='flex';
    $('#resultCard').className='result-card '+(d.is_anomaly?'fail':'success');
    $('#resultStatus').textContent=d.is_anomaly?'⚠️ 이상 감지':'✅ 정상';
    $('#resultStatus').className='result-status '+(d.is_anomaly?'red':'green');
    $('#rType').textContent=d.type;
    $('#rShift').textContent=d.shift+'px';
    $('#rAngle').textContent=d.angle+'°';
    $('#rJudge').textContent=d.is_anomaly?'FAIL':'PASS';
    $('#rJudge').style.color=d.is_anomaly?'#f87171':'#34d399';
    $('#rImg').src='data:image/png;base64,'+d.img;
    
    // AI 버튼 표시
    $('#emptyAI').style.display='none';
    $('#aiContent').style.display='block';
    $('#aiContent').innerHTML='<p style="color:#6b7280">AI 분석 버튼을 클릭하세요</p>';
    $('#aiBtns').style.display='flex';
    
    // 통계
    d.is_anomaly?st.a++:st.n++;
    st.t++;
    $('#sT').textContent=st.t;
    $('#sN').textContent=st.n;
    $('#sA').textContent=st.a;
    
    addHist(d);
}

async function aiAnalyze(){
    if(!lastR)return;
    $('#aiBtn').disabled=true;
    $('#aiBtn').textContent='분석중...';
    $('#aiContent').innerHTML='<p style="color:#6b7280">AI가 분석 중입니다...</p>';
    
    let sys=localStorage.getItem('sysP')||$('#sysP').value;
    let usr=(localStorage.getItem('usrP')||$('#usrP').value)
        .replace('{is_anomaly}',lastR.is_anomaly?'이상':'정상')
        .replace('{type}',lastR.type)
        .replace('{shift}',lastR.shift)
        .replace('{angle}',lastR.angle);
    
    try{
        const r=await fetch('/llm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sys,usr})});
        const d=await r.json();
        console.log('LLM 응답:', d);
        if(d.ok && d.content){
            $('#aiContent').innerHTML=marked.parse(d.content);
        }else{
            $('#aiContent').innerHTML='<p style="color:#f87171">오류: '+(d.error||'응답 없음')+'</p>';
        }
    }catch(e){
        console.error('LLM 오류:', e);
        $('#aiContent').innerHTML='<p style="color:#f87171">오류: '+e.message+'</p>';
    }
    $('#aiBtn').disabled=false;
    $('#aiBtn').textContent='🤖 AI 분석';
}

function addHist(d){
    // 히스토리 데이터 저장
    const histItem = {
        ...d,
        time: new Date().toLocaleString('ko-KR'),
        id: historyData.length
    };
    historyData.unshift(histItem);
    
    const h=$('#hist');
    if(st.t===1)h.innerHTML='';
    const item=document.createElement('div');
    item.className='history-item '+(d.is_anomaly?'red':'green');
    item.innerHTML=`<img src="data:image/png;base64,${d.img}"><div class="info"><strong>${d.is_anomaly?'⚠️ 이상':'✅ 정상'}</strong><span>${d.type} | ${d.shift}px</span></div>`;
    item.onclick = () => showHistDetail(histItem);
    h.insertBefore(item,h.firstChild);
}

function showHistDetail(data){
    $('#histModalTitle').textContent = data.is_anomaly ? '⚠️ 이상 감지 상세' : '✅ 정상 상세';
    $('#histModalTitle').className = data.is_anomaly ? 'red' : 'green';
    $('#histModalImg').src = 'data:image/png;base64,' + data.img;
    $('#histType').textContent = data.type;
    $('#histShift').textContent = data.shift + 'px';
    $('#histAngle').textContent = data.angle + '°';
    $('#histJudge').textContent = data.is_anomaly ? 'FAIL' : 'PASS';
    $('#histJudge').style.color = data.is_anomaly ? '#f87171' : '#34d399';
    $('#histTime').textContent = '검사 시간: ' + data.time;
    $('#histModal').style.display = 'flex';
}

function closeHistModal(){
    $('#histModal').style.display = 'none';
}

function openModal(){$('#modal').style.display='flex'}
function closeModal(){$('#modal').style.display='none'}
function saveP(){
    localStorage.setItem('sysP',$('#sysP').value);
    localStorage.setItem('usrP',$('#usrP').value);
    alert('저장!');
    closeModal();
}
</script>
</body>
</html>
'''

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/set_ref', methods=['POST'])
def set_ref():
    try:
        d = request.json['img'].split(',')[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(d), np.uint8), cv2.IMREAD_COLOR)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reference_data['center'] = find_center(g)
        reference_data['angle'] = find_angle(g)
        return jsonify({'ok': reference_data['center'] is not None, 'error': '박스없음' if not reference_data['center'] else ''})
    except Exception as e: return jsonify({'ok': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        d = request.json
        img = cv2.imdecode(np.frombuffer(base64.b64decode(d['img'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        r = detect(img, d.get('pt',30), d.get('at',10))
        if 'error' in r: return jsonify(r)
        r['img'] = to_b64(draw(img, r))
        return jsonify(r)
    except Exception as e: return jsonify({'error': str(e)})

@app.route('/llm', methods=['POST'])
def llm():
    d = request.json
    r = call_llm(d['usr'], d['sys'])
    return jsonify({'ok': r['success'], 'content': r.get('content',''), 'error': r.get('error','')})

if __name__ == '__main__':
    print("="*40)
    print("📦 박스 위치 이상 탐지")
    print("🌐 http://localhost:10005")
    print("="*40)
    app.run(host='0.0.0.0', port=10005, debug=True)
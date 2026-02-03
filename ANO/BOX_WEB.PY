"""
Box Position Anomaly Detection - Web Version V3
화면 꽉 참 + 스크롤 숨김 + 히스토리 하단
"""

import os
import cv2
import numpy as np
import requests
import re
from flask import Flask, render_template_string, request, jsonify
import base64

app = Flask(__name__)

reference_data = {'center': None, 'angle': None, 'image': None}

LLM_CONFIG = {
    "url": "http://common.llm.skhynix.com/v1/chat/completions",
    "model": "gpt-oss-20b"
}
API_TOKEN = None

def load_api_token():
    global API_TOKEN
    for path in ["token.txt", "../token.txt", os.path.expanduser("~/token.txt")]:
        if os.path.exists(path):
            with open(path, "r", encoding='utf-8') as f:
                API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    print(f"✅ 토큰 로드: {path}")
                    return True
    print("⚠️ 토큰 없음")
    return False

def call_llm_api(prompt, system_prompt=""):
    if not API_TOKEN:
        return {"success": False, "error": "토큰 없음"}
    try:
        response = requests.post(
            LLM_CONFIG["url"],
            headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
            json={
                "model": LLM_CONFIG["model"],
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}] if system_prompt else [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.3
            },
            timeout=120
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return {"success": True, "content": content}
        return {"success": False, "error": f"API 오류: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

load_api_token()

def find_box_center(gray):
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def find_box_angle(gray):
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[2]
    return angle + 90 if angle < -45 else angle

def detect_anomaly(img, pos_th=30, ang_th=10):
    if reference_data['center'] is None:
        return {'error': '기준 미설정'}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center = find_box_center(gray)
    angle = find_box_angle(gray)
    if center is None:
        return {'is_anomaly': True, 'anomaly_type': '박스 없음', 'position_shift': -1, 'angle_shift': -1}
    pos_shift = np.sqrt((center[0]-reference_data['center'][0])**2 + (center[1]-reference_data['center'][1])**2)
    ang_shift = abs(angle - reference_data['angle'])
    is_pos = pos_shift > pos_th
    is_ang = ang_shift > ang_th
    if is_pos and is_ang: atype = '위치+회전'
    elif is_pos: atype = '위치 이탈'
    elif is_ang: atype = '회전됨'
    else: atype = '정상'
    return {
        'is_anomaly': bool(is_pos or is_ang),
        'anomaly_type': atype,
        'position_shift': float(round(pos_shift, 1)),
        'angle_shift': float(round(ang_shift, 1)),
        'test_center': center,
        'ref_center': reference_data['center']
    }

def draw_result(img, result):
    img = img.copy()
    color = (0,0,255) if result.get('is_anomaly') else (0,255,0)
    cv2.rectangle(img, (5,5), (img.shape[1]-5, img.shape[0]-5), color, 4)
    if result.get('test_center'): cv2.circle(img, result['test_center'], 10, color, -1)
    if result.get('ref_center'): cv2.circle(img, result['ref_center'], 8, (255,0,0), 2)
    return img

def img_to_base64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')

HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>박스 위치 이상 탐지</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        html,body{height:100%;overflow:hidden}
        body{font-family:'Malgun Gothic',sans-serif;background:#1a1a2e;color:#fff}
        
        /* 스크롤바 숨김 */
        ::-webkit-scrollbar{display:none}
        *{-ms-overflow-style:none;scrollbar-width:none}
        
        /* 헤더 */
        .header{background:rgba(0,0,0,.5);padding:10px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #00d4ff}
        .header h1{font-size:20px;color:#00d4ff}
        .header span{color:#666;font-size:11px;margin-left:10px}
        .stats{display:flex;gap:25px}
        .stat{text-align:center}
        .stat .n{font-size:28px;font-weight:bold}
        .stat.t .n{color:#00d4ff}
        .stat.g .n{color:#0f8}
        .stat.r .n{color:#f44}
        .stat .l{font-size:10px;color:#888}
        
        /* 메인 - 상단 2열 + 하단 히스토리 */
        .wrapper{height:calc(100vh - 60px);display:flex;flex-direction:column;padding:8px;gap:8px}
        .top{display:grid;grid-template-columns:240px 1fr;gap:8px;flex:1;min-height:0}
        .bottom{height:100px;flex-shrink:0}
        
        /* 카드 */
        .card{background:rgba(255,255,255,.03);border-radius:10px;padding:15px;border:1px solid rgba(255,255,255,.1);overflow:hidden}
        .card h3{color:#00d4ff;font-size:13px;margin-bottom:8px;display:flex;align-items:center;gap:6px}
        .card h3 .badge{font-size:9px;padding:2px 6px;border-radius:8px;background:#f44;color:#fff}
        .card h3 .badge.ok{background:#0f8}
        
        /* 왼쪽 - 설정 */
        .left{display:flex;flex-direction:column;gap:10px;overflow-y:auto}
        .left::-webkit-scrollbar{width:4px}
        .left::-webkit-scrollbar-thumb{background:#444;border-radius:2px}
        .upload{border:2px dashed #444;border-radius:8px;padding:10px;text-align:center;cursor:pointer;transition:all .2s}
        .upload:hover{border-color:#00d4ff;background:rgba(0,212,255,.05)}
        .upload .ico{font-size:24px}
        .upload p{font-size:10px;color:#666;margin-top:3px}
        .upload input{display:none}
        .preview{width:100%;max-height:60px;object-fit:contain;border-radius:5px;margin-top:5px;display:none}
        .btn{width:100%;padding:8px;border:none;border-radius:15px;cursor:pointer;font-size:12px;font-weight:bold;color:#fff;transition:all .2s}
        .btn:hover{transform:translateY(-2px);filter:brightness(1.1)}
        .btn:disabled{background:#333!important;cursor:not-allowed;transform:none;filter:none}
        .btn-p{background:linear-gradient(135deg,#00d4ff,#0099cc)}
        .btn-g{background:linear-gradient(135deg,#0f8,#0c6)}
        .btn-v{background:linear-gradient(135deg,#9b59b6,#8e44ad)}
        .sets{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:5px}
        .sets label{font-size:9px;color:#888}
        .sets input{width:100%;padding:6px;border:1px solid #444;border-radius:4px;background:rgba(0,0,0,.3);color:#fff;font-size:11px;margin-top:2px}
        
        /* 오른쪽 - 결과 (2열) */
        .right{display:grid;grid-template-columns:1fr 1fr;gap:15px}
        .result-section,.llm-section{display:flex;flex-direction:column}
        
        /* 결과 박스 */
        .rbox{flex:1;padding:15px;border-radius:10px;display:flex;flex-direction:column}
        .rbox.g{background:rgba(0,255,136,.08);border:2px solid #0f8}
        .rbox.r{background:rgba(255,68,68,.08);border:2px solid #f44}
        .rbox.empty{background:rgba(255,255,255,.02);border:2px dashed #333;justify-content:center;align-items:center}
        .rbox.empty p{color:#555;font-size:13px}
        .rtitle{font-size:22px;font-weight:bold;margin-bottom:15px}
        .rbox.g .rtitle{color:#0f8}
        .rbox.r .rtitle{color:#f44}
        .rgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:15px}
        .ritem{background:rgba(0,0,0,.2);padding:12px;border-radius:8px;text-align:center}
        .ritem .lb{font-size:11px;color:#888;margin-bottom:5px}
        .ritem .vl{font-size:18px;font-weight:bold}
        .rimg{max-height:200px;object-fit:contain;border-radius:8px;margin-top:auto}
        
        /* LLM 결과 */
        .llm-box{flex:1;padding:15px;border-radius:10px;background:rgba(155,89,182,.08);border:1px solid #9b59b6;display:flex;flex-direction:column}
        .llm-box.empty{background:rgba(255,255,255,.02);border:2px dashed #333;justify-content:center;align-items:center}
        .llm-box h4{color:#9b59b6;font-size:14px;margin-bottom:10px}
        .llm-content{flex:1;overflow-y:auto;font-size:12px;line-height:1.6}
        .llm-btns{display:flex;gap:8px;margin-top:10px}
        .llm-btns .btn{flex:1}
        
        /* 마크다운 */
        .md h1,.md h2,.md h3,.md h4{color:#00d4ff;margin:15px 0 8px;font-size:14px;border-bottom:1px solid #333;padding-bottom:5px}
        .md h3{color:#9b59b6;border:none}
        .md table{width:100%;border-collapse:collapse;margin:10px 0;font-size:11px}
        .md th,.md td{border:1px solid #444;padding:8px}
        .md th{background:rgba(0,212,255,.1);color:#00d4ff}
        .md ul,.md ol{padding-left:20px;margin:8px 0}
        .md li{margin:4px 0}
        .md strong{color:#0f8}
        .md hr{border:none;border-top:1px solid #444;margin:15px 0}
        
        /* 하단 히스토리 */
        .bottom .card{height:100%;display:flex;flex-direction:column}
        .bottom h3{margin-bottom:8px;flex-shrink:0}
        .hist{display:flex;gap:10px;overflow-x:auto;flex:1;align-items:center;padding:5px 0}
        .hitem{flex-shrink:0;display:flex;align-items:center;gap:8px;padding:8px 12px;background:rgba(0,0,0,.2);border-radius:8px;font-size:11px}
        .hitem.g{border-left:3px solid #0f8}
        .hitem.r{border-left:3px solid #f44}
        .hitem img{width:50px;height:50px;object-fit:cover;border-radius:5px}
        .hempty{color:#555;width:100%;text-align:center;font-size:12px}
        
        /* 모달 */
        .modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.9);z-index:1000;justify-content:center;align-items:center}
        .mcontent{background:#1a1a2e;border-radius:12px;padding:25px;width:90%;max-width:600px;border:2px solid #9b59b6}
        .mcontent h2{color:#9b59b6;font-size:18px;margin-bottom:20px}
        .mcontent label{font-size:12px;color:#888;display:block;margin:15px 0 5px}
        .mcontent textarea{width:100%;background:rgba(0,0,0,.3);border:1px solid #444;border-radius:8px;color:#fff;padding:12px;font-size:12px;resize:none}
        .mbtns{display:flex;gap:10px;margin-top:20px}
        .mbtns .btn{flex:1}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>📦 박스 위치 이상 탐지 시스템 <span>BATA_VER_V0.1</span></h1>
        </div>
        <div class="stats">
            <div class="stat t"><div class="n" id="sT">0</div><div class="l">총 검사</div></div>
            <div class="stat g"><div class="n" id="sN">0</div><div class="l">정상</div></div>
            <div class="stat r"><div class="n" id="sA">0</div><div class="l">이상</div></div>
        </div>
    </div>
    
    <div class="wrapper">
        <div class="top">
            <!-- 왼쪽: 설정 -->
            <div class="card left">
                <div>
                    <h3>📷 기준 이미지 <span class="badge" id="refBadge">미설정</span></h3>
                    <div class="upload" onclick="document.getElementById('refIn').click()">
                        <input type="file" id="refIn" accept="image/*">
                        <div class="ico">📷</div>
                        <p>정상 박스 이미지 업로드</p>
                    </div>
                    <img id="refPv" class="preview">
                    <button class="btn btn-g" id="refBtn" disabled onclick="setRef()">✓ 기준으로 설정</button>
                </div>
                
                <div class="sets">
                    <div>
                        <label>위치 오차 (px)</label>
                        <input type="number" id="posT" value="30">
                    </div>
                    <div>
                        <label>각도 오차 (°)</label>
                        <input type="number" id="angT" value="10">
                    </div>
                </div>
                
                <div style="margin-top:15px">
                    <h3>🔍 테스트 이미지</h3>
                    <div class="upload" onclick="document.getElementById('testIn').click()">
                        <input type="file" id="testIn" accept="image/*">
                        <div class="ico">🔍</div>
                        <p>분석할 이미지 업로드</p>
                    </div>
                    <img id="testPv" class="preview">
                    <button class="btn btn-p" id="anaBtn" disabled onclick="analyze()">🔎 이상 탐지 분석</button>
                </div>
            </div>
            
            <!-- 오른쪽: 결과 -->
            <div class="right">
                <!-- 분석 결과 -->
                <div class="result-section">
                    <h3 style="color:#00d4ff;font-size:14px;margin-bottom:10px">📊 분석 결과</h3>
                    <div class="rbox empty" id="rBox">
                        <p>이미지를 업로드하고<br>분석 버튼을 클릭하세요</p>
                    </div>
                </div>
                
                <!-- AI 분석 -->
                <div class="llm-section">
                    <h3 style="color:#9b59b6;font-size:14px;margin-bottom:10px">🤖 AI 원인 분석</h3>
                    <div class="llm-box empty" id="llmBox">
                        <p>분석 결과 확인 후<br>AI 분석 버튼을 클릭하세요</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 하단: 히스토리 -->
        <div class="bottom">
            <div class="card">
                <h3>📜 검사 히스토리</h3>
                <div class="hist" id="hist">
                    <div class="hempty">아직 검사 기록이 없습니다</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 프롬프트 모달 -->
    <div class="modal" id="modal">
        <div class="mcontent">
            <h2>⚙️ LLM 프롬프트 설정</h2>
            <label>시스템 프롬프트</label>
            <textarea id="sysP" rows="3">당신은 반도체 FAB 공정의 품질 관리 전문가입니다. 박스 위치 이상 탐지 결과를 분석하고, 원인과 조치 방안을 제시합니다. 한국어로 간결하게 답변하세요.</textarea>
            <label>사용자 프롬프트 (변수: {is_anomaly}, {anomaly_type}, {position_shift}, {angle_shift})</label>
            <textarea id="usrP" rows="6">박스 위치 이상 탐지 결과:
- 판정: {is_anomaly}
- 유형: {anomaly_type}
- 위치 이탈: {position_shift}px
- 회전 각도: {angle_shift}°

분석해주세요:
1. 이상 원인 추정
2. 공정에 미치는 영향
3. 권장 조치 사항
4. 예방 대책</textarea>
            <div class="mbtns">
                <button class="btn btn-g" onclick="saveP()">💾 저장</button>
                <button class="btn" style="background:#555" onclick="resetP()">🔄 초기화</button>
                <button class="btn" style="background:#333" onclick="closeModal()">✕ 닫기</button>
            </div>
        </div>
    </div>

<script>
let refD=null,testD=null,lastR=null,st={t:0,n:0,a:0};

document.getElementById('refIn').onchange=e=>{if(e.target.files[0])loadImg(e.target.files[0],'ref')};
document.getElementById('testIn').onchange=e=>{if(e.target.files[0])loadImg(e.target.files[0],'test')};

function loadImg(f,t){
    const r=new FileReader();
    r.onload=e=>{
        if(t==='ref'){
            refD=e.target.result;
            document.getElementById('refPv').src=refD;
            document.getElementById('refPv').style.display='block';
            document.getElementById('refBtn').disabled=false;
        }else{
            testD=e.target.result;
            document.getElementById('testPv').src=testD;
            document.getElementById('testPv').style.display='block';
            document.getElementById('anaBtn').disabled=false;
        }
    };
    r.readAsDataURL(f);
}

async function setRef(){
    const r=await fetch('/set_reference',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:refD})});
    const d=await r.json();
    if(d.success){
        document.getElementById('refBadge').textContent='설정완료';
        document.getElementById('refBadge').className='badge ok';
    }else{
        alert('오류: '+d.error);
    }
}

async function analyze(){
    const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        image:testD,
        position_threshold:+document.getElementById('posT').value,
        angle_threshold:+document.getElementById('angT').value
    })});
    const d=await r.json();
    if(d.error){alert('오류: '+d.error);return}
    
    lastR=d;
    
    // 결과 박스 업데이트
    const box=document.getElementById('rBox');
    box.className='rbox '+(d.is_anomaly?'r':'g');
    box.innerHTML=`
        <div class="rtitle">${d.is_anomaly?'⚠️ 이상 감지!':'✅ 정상'}</div>
        <div class="rgrid">
            <div class="ritem"><div class="lb">유형</div><div class="vl">${d.anomaly_type}</div></div>
            <div class="ritem"><div class="lb">이탈거리</div><div class="vl">${d.position_shift}px</div></div>
            <div class="ritem"><div class="lb">회전각도</div><div class="vl">${d.angle_shift}°</div></div>
            <div class="ritem"><div class="lb">판정</div><div class="vl" style="color:${d.is_anomaly?'#f44':'#0f8'}">${d.is_anomaly?'FAIL':'PASS'}</div></div>
        </div>
        <img class="rimg" src="data:image/png;base64,${d.result_image}">
    `;
    
    // LLM 박스 초기화 + 버튼 추가
    const llmBox=document.getElementById('llmBox');
    llmBox.className='llm-box';
    llmBox.innerHTML=`
        <h4>🤖 AI 원인 분석</h4>
        <div class="llm-content md" id="llmC">분석 버튼을 클릭하세요</div>
        <div class="llm-btns">
            <button class="btn btn-v" id="llmBtn" onclick="llmAna()">🤖 AI 분석 시작</button>
            <button class="btn" style="background:#555" onclick="openModal()">⚙️ 설정</button>
        </div>
    `;
    
    // 통계
    if(d.is_anomaly)st.a++;else st.n++;
    st.t++;
    document.getElementById('sT').textContent=st.t;
    document.getElementById('sN').textContent=st.n;
    document.getElementById('sA').textContent=st.a;
    
    addHist(d);
}

async function llmAna(){
    if(!lastR){alert('먼저 분석하세요');return}
    
    const btn=document.getElementById('llmBtn');
    const c=document.getElementById('llmC');
    
    btn.disabled=true;
    btn.textContent='🔄 분석 중...';
    c.innerHTML='<p style="color:#888">AI가 분석 중입니다...</p>';
    
    let sP=localStorage.getItem('sysP')||document.getElementById('sysP').value;
    let uP=(localStorage.getItem('usrP')||document.getElementById('usrP').value)
        .replace('{is_anomaly}',lastR.is_anomaly?'이상':'정상')
        .replace('{anomaly_type}',lastR.anomaly_type)
        .replace('{position_shift}',lastR.position_shift)
        .replace('{angle_shift}',lastR.angle_shift);
    
    try{
        const r=await fetch('/llm_analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({result:lastR,system_prompt:sP,user_prompt:uP})});
        const d=await r.json();
        c.innerHTML=d.success?marked.parse(d.analysis):'<p style="color:#f44">❌ '+d.error+'</p>';
    }catch(e){
        c.innerHTML='<p style="color:#f44">❌ '+e.message+'</p>';
    }
    
    btn.disabled=false;
    btn.textContent='🤖 다시 분석';
}

function addHist(d){
    const h=document.getElementById('hist');
    if(st.t===1)h.innerHTML='';
    
    const item=document.createElement('div');
    item.className='hitem '+(d.is_anomaly?'r':'g');
    item.innerHTML=`
        <img src="data:image/png;base64,${d.result_image}">
        <div>
            <strong>${d.is_anomaly?'⚠️ 이상':'✅ 정상'}</strong><br>
            <span style="color:#888">${d.anomaly_type} | ${d.position_shift}px | ${d.angle_shift}°</span>
        </div>
    `;
    h.insertBefore(item,h.firstChild);
}

function openModal(){document.getElementById('modal').style.display='flex'}
function closeModal(){document.getElementById('modal').style.display='none'}
function saveP(){
    localStorage.setItem('sysP',document.getElementById('sysP').value);
    localStorage.setItem('usrP',document.getElementById('usrP').value);
    alert('저장됨!');
    closeModal();
}
function resetP(){
    document.getElementById('sysP').value='당신은 반도체 FAB 공정의 품질 관리 전문가입니다. 박스 위치 이상 탐지 결과를 분석하고, 원인과 조치 방안을 제시합니다. 한국어로 간결하게 답변하세요.';
    document.getElementById('usrP').value=`박스 위치 이상 탐지 결과:
- 판정: {is_anomaly}
- 유형: {anomaly_type}
- 위치 이탈: {position_shift}px
- 회전 각도: {angle_shift}°

분석해주세요:
1. 이상 원인 추정
2. 공정에 미치는 영향
3. 권장 조치 사항
4. 예방 대책`;
    localStorage.removeItem('sysP');
    localStorage.removeItem('usrP');
    alert('초기화됨!');
}
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/set_reference', methods=['POST'])
def set_reference():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'success': False, 'error': '이미지 로드 실패'})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reference_data['center'] = find_box_center(gray)
        reference_data['angle'] = find_box_angle(gray)
        reference_data['image'] = img
        if reference_data['center'] is None: return jsonify({'success': False, 'error': '박스 없음'})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error': '이미지 로드 실패'})
        result = detect_anomaly(img, data.get('position_threshold', 30), data.get('angle_threshold', 10))
        if 'error' in result: return jsonify(result)
        result['result_image'] = img_to_base64(draw_result(img, result))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/llm_analyze', methods=['POST'])
def llm_analyze():
    try:
        data = request.json
        result = call_llm_api(data.get('user_prompt', ''), data.get('system_prompt', ''))
        return jsonify({"success": result["success"], "analysis": result.get("content", ""), "error": result.get("error", "")})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("="*50)
    print("📦 박스 위치 이상 탐지 시스템 V3")
    print("="*50)
    print("🌐 http://localhost:10005")
    print("="*50)
    app.run(host='0.0.0.0', port=10005, debug=True)
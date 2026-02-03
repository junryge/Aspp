"""
Box Position Anomaly Detection - Web Version
Flask 기반 웹 서비스 + LLM 분석

실행: python box_anomaly_web.py
접속: http://localhost:10005
"""

import os
import cv2
import numpy as np
import requests
import re
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 전역 변수: 기준 이미지
reference_data = {
    'center': None,
    'angle': None,
    'image': None
}

# ========================================
# LLM API 설정
# ========================================
LLM_CONFIG = {
    "url": "http://common.llm.skhynix.com/v1/chat/completions",
    "model": "gpt-oss-20b",
    "name": "COMMON(20B)"
}

API_TOKEN = None

def load_api_token():
    """API 토큰 로드"""
    global API_TOKEN
    
    token_paths = [
        "token.txt",
        "../token.txt",
        os.path.expanduser("~/token.txt")
    ]
    
    for token_path in token_paths:
        if os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    print(f"✅ API 토큰 로드 완료: {token_path}")
                    return True
            except Exception as e:
                print(f"❌ 토큰 로드 실패: {e}")
    
    print("⚠️ 토큰 파일을 찾을 수 없습니다")
    return False


def call_llm_api(prompt: str, system_prompt: str = "") -> dict:
    """LLM API 호출"""
    global API_TOKEN
    
    if not API_TOKEN:
        return {"success": False, "error": "API 토큰이 설정되지 않았습니다."}
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": LLM_CONFIG["model"],
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.3
    }
    
    try:
        print("🚀 LLM API 호출 중...")
        response = requests.post(LLM_CONFIG["url"], headers=headers, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # <think> 태그 제거
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()
            
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": f"API 오류: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "API 요청 시간 초과"}
    except Exception as e:
        return {"success": False, "error": f"API 호출 실패: {str(e)}"}


# 토큰 로드
load_api_token()


def find_box_center(gray_img):
    """박스 중심점 찾기"""
    _, thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_box_angle(gray_img):
    """박스 회전 각도 찾기"""
    _, thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    
    if angle < -45:
        angle += 90
    
    return angle


def detect_anomaly(test_img, position_threshold=30, angle_threshold=10):
    """이상 탐지"""
    if reference_data['center'] is None:
        return {'error': '기준 이미지가 설정되지 않았습니다'}
    
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_center = find_box_center(gray)
    test_angle = find_box_angle(gray)
    
    if test_center is None:
        return {
            'is_anomaly': True,
            'anomaly_type': '박스 없음',
            'position_shift': -1,
            'angle_shift': -1
        }
    
    # 거리 계산
    position_shift = np.sqrt(
        (test_center[0] - reference_data['center'][0])**2 + 
        (test_center[1] - reference_data['center'][1])**2
    )
    angle_shift = abs(test_angle - reference_data['angle'])
    
    # 판정
    is_position_anomaly = position_shift > position_threshold
    is_angle_anomaly = angle_shift > angle_threshold
    is_anomaly = is_position_anomaly or is_angle_anomaly
    
    if is_position_anomaly and is_angle_anomaly:
        anomaly_type = '위치+회전'
    elif is_position_anomaly:
        anomaly_type = '위치 이탈'
    elif is_angle_anomaly:
        anomaly_type = '회전됨'
    else:
        anomaly_type = '정상'
    
    return {
        'is_anomaly': bool(is_anomaly),
        'anomaly_type': anomaly_type,
        'position_shift': float(round(position_shift, 1)),
        'angle_shift': float(round(angle_shift, 1)),
        'test_center': test_center,
        'ref_center': reference_data['center']
    }


def draw_result(img, result):
    """결과 이미지에 표시"""
    img = img.copy()
    
    if result.get('is_anomaly'):
        color = (0, 0, 255)  # 빨강
    else:
        color = (0, 255, 0)  # 초록
    
    # 테두리
    cv2.rectangle(img, (5, 5), (img.shape[1]-5, img.shape[0]-5), color, 4)
    
    # 중심점 표시
    if result.get('test_center'):
        cv2.circle(img, result['test_center'], 10, color, -1)
    if result.get('ref_center'):
        cv2.circle(img, result['ref_center'], 8, (255, 0, 0), 2)
    
    return img


def img_to_base64(img):
    """이미지를 base64로 변환"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


# HTML 템플릿
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>박스 위치 이상 탐지 시스템</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Malgun Gothic', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #00d4ff;
        }
        .header h1 {
            font-size: 28px;
            color: #00d4ff;
            margin-bottom: 5px;
        }
        .header p { color: #888; font-size: 14px; }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card h2::before {
            content: '▶';
            font-size: 12px;
        }
        
        .upload-area {
            border: 2px dashed #444;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 15px;
        }
        .upload-area:hover {
            border-color: #00d4ff;
            background: rgba(0,212,255,0.05);
        }
        .upload-area.dragover {
            border-color: #00ff88;
            background: rgba(0,255,136,0.1);
        }
        .upload-area input { display: none; }
        .upload-area p { color: #888; margin-top: 10px; }
        
        .preview-img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            display: none;
            margin-top: 15px;
        }
        
        .btn {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #fff;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
            margin-top: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,212,255,0.4);
        }
        .btn:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .btn-success { background: linear-gradient(135deg, #00ff88, #00cc6a); }
        
        .result-box {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .result-box.normal {
            background: rgba(0,255,136,0.1);
            border: 2px solid #00ff88;
        }
        .result-box.anomaly {
            background: rgba(255,68,68,0.1);
            border: 2px solid #ff4444;
        }
        .result-box h3 {
            font-size: 24px;
            margin-bottom: 15px;
        }
        .result-box.normal h3 { color: #00ff88; }
        .result-box.anomaly h3 { color: #ff4444; }
        
        .result-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .detail-item {
            background: rgba(0,0,0,0.2);
            padding: 10px 15px;
            border-radius: 8px;
        }
        .detail-item label {
            color: #888;
            font-size: 12px;
            display: block;
        }
        .detail-item span {
            font-size: 18px;
            font-weight: bold;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }
        .status-badge.set { background: #00ff88; color: #000; }
        .status-badge.unset { background: #ff4444; color: #fff; }
        
        .settings {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        .setting-item label {
            display: block;
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .setting-item input {
            width: 100%;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 16px;
        }
        
        .history {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .history-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .history-item img {
            width: 60px;
            height: 60px;
            object-fit: cover;
            border-radius: 5px;
        }
        .history-item.normal { border-left: 3px solid #00ff88; }
        .history-item.anomaly { border-left: 3px solid #ff4444; }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }
        .stat-item .number {
            font-size: 32px;
            font-weight: bold;
        }
        .stat-item .label { color: #888; font-size: 12px; }
        .stat-item.total .number { color: #00d4ff; }
        .stat-item.normal .number { color: #00ff88; }
        .stat-item.anomaly .number { color: #ff4444; }
        
        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📦 박스 위치 이상 탐지 시스템</h1>
        <p>Box Position Anomaly Detection System - BATA_VER_V0.1</p>
    </div>
    
    <div class="container">
        <!-- 왼쪽: 기준 이미지 설정 -->
        <div class="card">
            <h2>기준 이미지 설정
                <span class="status-badge unset" id="refStatus">미설정</span>
            </h2>
            
            <div class="upload-area" id="refUpload" onclick="document.getElementById('refInput').click()">
                <input type="file" id="refInput" accept="image/*">
                <div style="font-size: 48px;">📷</div>
                <p>클릭하거나 이미지를 드래그하세요</p>
                <p style="font-size: 12px; color: #666;">정상 상태의 박스 이미지</p>
            </div>
            <img id="refPreview" class="preview-img">
            <button class="btn btn-success" id="setRefBtn" disabled onclick="setReference()">
                기준 이미지로 설정
            </button>
            
            <div class="settings">
                <div class="setting-item">
                    <label>위치 허용 오차 (px)</label>
                    <input type="number" id="posThreshold" value="30" min="1" max="200">
                </div>
                <div class="setting-item">
                    <label>각도 허용 오차 (°)</label>
                    <input type="number" id="angleThreshold" value="10" min="1" max="45">
                </div>
            </div>
        </div>
        
        <!-- 오른쪽: 테스트 이미지 -->
        <div class="card">
            <h2>테스트 이미지 분석</h2>
            
            <div class="upload-area" id="testUpload" onclick="document.getElementById('testInput').click()">
                <input type="file" id="testInput" accept="image/*">
                <div style="font-size: 48px;">🔍</div>
                <p>분석할 이미지를 업로드하세요</p>
            </div>
            <img id="testPreview" class="preview-img">
            <button class="btn" id="analyzeBtn" disabled onclick="analyzeImage()">
                이상 탐지 분석
            </button>
            
            <div class="result-box" id="resultBox">
                <h3 id="resultTitle"></h3>
                <div class="result-details">
                    <div class="detail-item">
                        <label>유형</label>
                        <span id="resultType">-</span>
                    </div>
                    <div class="detail-item">
                        <label>이탈 거리</label>
                        <span id="resultShift">-</span>
                    </div>
                    <div class="detail-item">
                        <label>회전 각도</label>
                        <span id="resultAngle">-</span>
                    </div>
                    <div class="detail-item">
                        <label>판정</label>
                        <span id="resultJudge">-</span>
                    </div>
                </div>
                <img id="resultImage" class="result-image">
                
                <!-- LLM 분석 버튼 -->
                <button class="btn" id="llmBtn" onclick="llmAnalyze()" style="margin-top: 15px; background: linear-gradient(135deg, #9b59b6, #8e44ad);">
                    🤖 AI 원인 분석 (LLM)
                </button>
                <div id="llmResult" style="display: none; margin-top: 15px; padding: 15px; background: rgba(155,89,182,0.1); border: 1px solid #9b59b6; border-radius: 10px;">
                    <h4 style="color: #9b59b6; margin-bottom: 10px;">🤖 AI 분석 결과</h4>
                    <div id="llmContent" style="white-space: pre-wrap; line-height: 1.6; font-size: 14px;"></div>
                </div>
            </div>
        </div>
        
        <!-- 하단: 통계 및 히스토리 -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>분석 통계</h2>
            <div class="stats">
                <div class="stat-item total">
                    <div class="number" id="statTotal">0</div>
                    <div class="label">총 검사</div>
                </div>
                <div class="stat-item normal">
                    <div class="number" id="statNormal">0</div>
                    <div class="label">정상</div>
                </div>
                <div class="stat-item anomaly">
                    <div class="number" id="statAnomaly">0</div>
                    <div class="label">이상</div>
                </div>
            </div>
            
            <h2>검사 히스토리</h2>
            <div class="history" id="history">
                <p style="color: #666; text-align: center; padding: 20px;">
                    아직 검사 기록이 없습니다
                </p>
            </div>
        </div>
    </div>
    
    <script>
        let refImageData = null;
        let testImageData = null;
        let stats = { total: 0, normal: 0, anomaly: 0 };
        
        // 드래그앤드롭 설정
        ['refUpload', 'testUpload'].forEach(id => {
            const area = document.getElementById(id);
            area.addEventListener('dragover', e => {
                e.preventDefault();
                area.classList.add('dragover');
            });
            area.addEventListener('dragleave', () => {
                area.classList.remove('dragover');
            });
            area.addEventListener('drop', e => {
                e.preventDefault();
                area.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    if (id === 'refUpload') handleRefImage(file);
                    else handleTestImage(file);
                }
            });
        });
        
        // 파일 선택 이벤트
        document.getElementById('refInput').addEventListener('change', e => {
            if (e.target.files[0]) handleRefImage(e.target.files[0]);
        });
        document.getElementById('testInput').addEventListener('change', e => {
            if (e.target.files[0]) handleTestImage(e.target.files[0]);
        });
        
        function handleRefImage(file) {
            const reader = new FileReader();
            reader.onload = e => {
                refImageData = e.target.result;
                document.getElementById('refPreview').src = refImageData;
                document.getElementById('refPreview').style.display = 'block';
                document.getElementById('setRefBtn').disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        function handleTestImage(file) {
            const reader = new FileReader();
            reader.onload = e => {
                testImageData = e.target.result;
                document.getElementById('testPreview').src = testImageData;
                document.getElementById('testPreview').style.display = 'block';
                document.getElementById('analyzeBtn').disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        async function setReference() {
            const response = await fetch('/set_reference', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: refImageData })
            });
            const result = await response.json();
            
            if (result.success) {
                document.getElementById('refStatus').textContent = '설정됨';
                document.getElementById('refStatus').className = 'status-badge set';
                alert('기준 이미지가 설정되었습니다!');
            } else {
                alert('오류: ' + result.error);
            }
        }
        
        async function analyzeImage() {
            const posThreshold = document.getElementById('posThreshold').value;
            const angleThreshold = document.getElementById('angleThreshold').value;
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    image: testImageData,
                    position_threshold: parseFloat(posThreshold),
                    angle_threshold: parseFloat(angleThreshold)
                })
            });
            const result = await response.json();
            
            if (result.error) {
                alert('오류: ' + result.error);
                return;
            }
            
            // 결과 표시
            const resultBox = document.getElementById('resultBox');
            resultBox.style.display = 'block';
            
            if (result.is_anomaly) {
                resultBox.className = 'result-box anomaly';
                document.getElementById('resultTitle').textContent = '⚠️ 이상 감지!';
                document.getElementById('resultJudge').textContent = 'FAIL';
                document.getElementById('resultJudge').style.color = '#ff4444';
                stats.anomaly++;
            } else {
                resultBox.className = 'result-box normal';
                document.getElementById('resultTitle').textContent = '✅ 정상';
                document.getElementById('resultJudge').textContent = 'PASS';
                document.getElementById('resultJudge').style.color = '#00ff88';
                stats.normal++;
            }
            stats.total++;
            
            document.getElementById('resultType').textContent = result.anomaly_type;
            document.getElementById('resultShift').textContent = result.position_shift + ' px';
            document.getElementById('resultAngle').textContent = result.angle_shift + '°';
            document.getElementById('resultImage').src = 'data:image/png;base64,' + result.result_image;
            
            // 통계 업데이트
            document.getElementById('statTotal').textContent = stats.total;
            document.getElementById('statNormal').textContent = stats.normal;
            document.getElementById('statAnomaly').textContent = stats.anomaly;
            
            // 히스토리 추가
            addHistory(result);
        }
        
        let lastResult = null;  // 마지막 분석 결과 저장
        
        // analyzeImage 함수 수정 - 결과 저장
        const originalAnalyze = analyzeImage;
        analyzeImage = async function() {
            const posThreshold = document.getElementById('posThreshold').value;
            const angleThreshold = document.getElementById('angleThreshold').value;
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    image: testImageData,
                    position_threshold: parseFloat(posThreshold),
                    angle_threshold: parseFloat(angleThreshold)
                })
            });
            const result = await response.json();
            
            if (result.error) {
                alert('오류: ' + result.error);
                return;
            }
            
            lastResult = result;  // 결과 저장
            
            // 결과 표시
            const resultBox = document.getElementById('resultBox');
            resultBox.style.display = 'block';
            
            if (result.is_anomaly) {
                resultBox.className = 'result-box anomaly';
                document.getElementById('resultTitle').textContent = '⚠️ 이상 감지!';
                document.getElementById('resultJudge').textContent = 'FAIL';
                document.getElementById('resultJudge').style.color = '#ff4444';
                stats.anomaly++;
            } else {
                resultBox.className = 'result-box normal';
                document.getElementById('resultTitle').textContent = '✅ 정상';
                document.getElementById('resultJudge').textContent = 'PASS';
                document.getElementById('resultJudge').style.color = '#00ff88';
                stats.normal++;
            }
            stats.total++;
            
            document.getElementById('resultType').textContent = result.anomaly_type;
            document.getElementById('resultShift').textContent = result.position_shift + ' px';
            document.getElementById('resultAngle').textContent = result.angle_shift + '°';
            document.getElementById('resultImage').src = 'data:image/png;base64,' + result.result_image;
            
            // LLM 결과 초기화
            document.getElementById('llmResult').style.display = 'none';
            
            // 통계 업데이트
            document.getElementById('statTotal').textContent = stats.total;
            document.getElementById('statNormal').textContent = stats.normal;
            document.getElementById('statAnomaly').textContent = stats.anomaly;
            
            // 히스토리 추가
            addHistory(result);
        }
        
        async function llmAnalyze() {
            if (!lastResult) {
                alert('먼저 이미지를 분석해주세요!');
                return;
            }
            
            const btn = document.getElementById('llmBtn');
            const resultDiv = document.getElementById('llmResult');
            const contentDiv = document.getElementById('llmContent');
            
            btn.disabled = true;
            btn.textContent = '🤖 AI 분석 중...';
            contentDiv.textContent = '분석 중입니다. 잠시만 기다려주세요...';
            resultDiv.style.display = 'block';
            
            try {
                const response = await fetch('/llm_analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ result: lastResult })
                });
                const data = await response.json();
                
                if (data.success) {
                    contentDiv.textContent = data.analysis;
                } else {
                    contentDiv.textContent = '❌ 분석 실패: ' + data.error;
                }
            } catch (e) {
                contentDiv.textContent = '❌ 오류: ' + e.message;
            }
            
            btn.disabled = false;
            btn.textContent = '🤖 AI 원인 분석 (LLM)';
        }
        
        function addHistory(result) {
            const history = document.getElementById('history');
            if (stats.total === 1) history.innerHTML = '';
            
            const item = document.createElement('div');
            item.className = 'history-item ' + (result.is_anomaly ? 'anomaly' : 'normal');
            item.innerHTML = `
                <img src="data:image/png;base64,${result.result_image}">
                <div>
                    <strong>${result.is_anomaly ? '⚠️ 이상' : '✅ 정상'}</strong>
                    <span style="color: #888; font-size: 12px; margin-left: 10px;">
                        ${result.anomaly_type} | ${result.position_shift}px | ${result.angle_shift}°
                    </span>
                </div>
            `;
            history.insertBefore(item, history.firstChild);
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/set_reference', methods=['POST'])
def set_reference():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # numpy 배열로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': '이미지를 읽을 수 없습니다'})
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        reference_data['center'] = find_box_center(gray)
        reference_data['angle'] = find_box_angle(gray)
        reference_data['image'] = img
        
        if reference_data['center'] is None:
            return jsonify({'success': False, 'error': '박스를 찾을 수 없습니다'})
        
        return jsonify({
            'success': True,
            'center': reference_data['center'],
            'angle': round(reference_data['angle'], 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        position_threshold = data.get('position_threshold', 30)
        angle_threshold = data.get('angle_threshold', 10)
        
        # numpy 배열로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다'})
        
        # 분석
        result = detect_anomaly(img, position_threshold, angle_threshold)
        
        if 'error' in result:
            return jsonify(result)
        
        # 결과 이미지 생성
        result_img = draw_result(img, result)
        result_b64 = img_to_base64(result_img)
        
        result['result_image'] = result_b64
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/llm_analyze', methods=['POST'])
def llm_analyze():
    """LLM으로 이상 탐지 결과 분석"""
    try:
        data = request.json
        result = data.get('result', {})
        
        system_prompt = """당신은 반도체 FAB 공정의 품질 관리 전문가입니다.
박스 위치 이상 탐지 결과를 분석하고, 원인과 조치 방안을 제시합니다.
한국어로 간결하고 명확하게 답변하세요."""
        
        prompt = f"""박스 위치 이상 탐지 결과를 분석해주세요:

- 판정: {'이상' if result.get('is_anomaly') else '정상'}
- 유형: {result.get('anomaly_type', '알 수 없음')}
- 위치 이탈: {result.get('position_shift', 0)}px
- 회전 각도: {result.get('angle_shift', 0)}°

다음 항목을 분석해주세요:
1. 이상 원인 추정
2. 공정에 미치는 영향
3. 권장 조치 사항
4. 예방 대책"""
        
        llm_result = call_llm_api(prompt, system_prompt)
        
        if llm_result["success"]:
            return jsonify({"success": True, "analysis": llm_result["content"]})
        else:
            return jsonify({"success": False, "error": llm_result["error"]})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/llm_status')
def llm_status():
    """LLM 상태 확인"""
    return jsonify({
        "available": API_TOKEN is not None,
        "model": LLM_CONFIG["name"]
    })


if __name__ == '__main__':
    print("="*50)
    print("📦 박스 위치 이상 탐지 시스템 - 웹 버전")
    print("="*50)
    print()
    print("🌐 접속 주소:")
    print("   로컬: http://localhost:10005")
    print("   네트워크: http://<IP주소>:10005")
    print()
    print("📌 사용법:")
    print("   1. 기준 이미지(정상 박스) 업로드 → '기준 이미지로 설정' 클릭")
    print("   2. 테스트 이미지 업로드 → '이상 탐지 분석' 클릭")
    print()
    print("종료: Ctrl+C")
    print("="*50)
    
    app.run(host='0.0.0.0', port=10005, debug=True)
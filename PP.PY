<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RNN, LSTM, ARIMA 시각적 설명</title>
    <style>
        body {
            font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 40px;
            font-size: 2.5em;
        }
        
        h2 {
            color: #3498db;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 2em;
        }
        
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        
        .visual-box {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .formula {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
        }
        
        .explanation {
            background: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196f3;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .rnn-cell, .lstm-cell {
            display: inline-block;
            width: 80px;
            height: 80px;
            background: #3498db;
            border-radius: 10px;
            margin: 10px;
            position: relative;
            text-align: center;
            line-height: 80px;
            color: white;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .rnn-cell:hover, .lstm-cell:hover {
            transform: scale(1.1);
            background: #2980b9;
        }
        
        .arrow {
            display: inline-block;
            width: 40px;
            height: 2px;
            background: #34495e;
            position: relative;
            margin: 0 5px;
        }
        
        .arrow::after {
            content: '';
            position: absolute;
            right: 0;
            top: -4px;
            width: 0;
            height: 0;
            border-left: 8px solid #34495e;
            border-top: 5px solid transparent;
            border-bottom: 5px solid transparent;
        }
        
        .gate {
            background: #e74c3c;
            width: 60px;
            height: 40px;
            display: inline-block;
            margin: 5px;
            border-radius: 5px;
            text-align: center;
            line-height: 40px;
            color: white;
            font-size: 0.9em;
        }
        
        .memory-cell {
            background: #f39c12;
            width: 100px;
            height: 60px;
            margin: 10px auto;
            border-radius: 10px;
            text-align: center;
            line-height: 60px;
            color: white;
            font-weight: bold;
        }
        
        .arima-component {
            display: inline-block;
            background: #16a085;
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            min-width: 120px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .arima-component:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .example-data {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
        }
        
        .data-point {
            background: #9b59b6;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .highlight {
            background: #f1c40f;
            color: #2c3e50;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin: 10px 5px;
            transition: all 0.3s ease;
        }
        
        .button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        
        .demo-container {
            border: 2px dashed #95a5a6;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        @keyframes flow {
            0% { transform: translateX(0); }
            100% { transform: translateX(20px); }
        }
        
        .flowing {
            animation: flow 1s ease-in-out infinite alternate;
        }
    </style>
</head>
<body>
    <h1>🧠 RNN, LSTM, ARIMA 쉽게 이해하기</h1>
    
    <div class="container">
        <h2>1. RNN (순환 신경망) - 기억하는 신경망</h2>
        
        <div class="explanation">
            💡 <strong>핵심 아이디어:</strong> RNN은 "이전 정보를 기억하는" 신경망입니다. 
            마치 책을 읽을 때 앞의 내용을 기억하면서 읽는 것처럼요!
        </div>
        
        <h3>🎯 작동 원리</h3>
        <div class="visual-box">
            <div style="text-align: center;">
                <div class="rnn-cell">입력1</div>
                <span class="arrow"></span>
                <div class="rnn-cell">RNN</div>
                <span class="arrow"></span>
                <div class="rnn-cell">출력1</div>
                <br><br>
                <div style="margin-left: 140px;">↓ <em>기억 전달</em></div>
                <br>
                <div class="rnn-cell">입력2</div>
                <span class="arrow"></span>
                <div class="rnn-cell">RNN</div>
                <span class="arrow"></span>
                <div class="rnn-cell">출력2</div>
            </div>
        </div>
        
        <h3>📐 핵심 공식</h3>
        <div class="formula">
            h_t = tanh(W_h × h_{t-1} + W_x × x_t + b)
        </div>
        
        <div class="explanation">
            <strong>공식 해석:</strong><br>
            • <span class="highlight">h_t</span> = 현재 시점의 기억(숨은 상태)<br>
            • <span class="highlight">h_{t-1}</span> = 이전 시점의 기억<br>
            • <span class="highlight">x_t</span> = 현재 입력<br>
            • <span class="highlight">W</span> = 가중치(중요도)<br>
            • <span class="highlight">tanh</span> = 활성화 함수(-1~1 사이로 조정)
        </div>
        
        <h3>🎬 실생활 예시</h3>
        <div class="demo-container">
            <p><strong>"나는 학교에 갔다"</strong>라는 문장을 이해할 때:</p>
            <div class="example-data">
                <div class="data-point">나는</div>
                <div class="flowing">→</div>
                <div class="data-point">학교에</div>
                <div class="flowing">→</div>
                <div class="data-point">갔다</div>
            </div>
            <p>RNN은 "나는"을 기억하고 있어서 "갔다"가 누가 간 것인지 압니다!</p>
        </div>
    </div>
    
    <div class="container">
        <h2>2. LSTM (장단기 기억 신경망) - 똑똑한 기억 관리자</h2>
        
        <div class="explanation">
            💡 <strong>핵심 아이디어:</strong> LSTM은 RNN의 업그레이드 버전입니다. 
            중요한 정보는 오래 기억하고, 불필요한 정보는 잊어버립니다!
        </div>
        
        <h3>🎯 작동 원리 - 3개의 문(Gate)</h3>
        <div class="visual-box">
            <div style="text-align: center;">
                <div class="memory-cell">장기 기억</div>
                <br>
                <div class="gate">잊기 문</div>
                <div class="gate">입력 문</div>
                <div class="gate">출력 문</div>
                <br><br>
                <div style="background: #ecf0f1; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <strong>🚪 각 문의 역할:</strong><br>
                    • <strong>잊기 문:</strong> "이 정보는 이제 필요 없어" (0~1)<br>
                    • <strong>입력 문:</strong> "이 새 정보는 기억할 만해" (0~1)<br>
                    • <strong>출력 문:</strong> "지금 이 정보를 사용해야 해" (0~1)
                </div>
            </div>
        </div>
        
        <h3>📐 핵심 공식들</h3>
        <div class="formula">
            잊기 문: f_t = σ(W_f × [h_{t-1}, x_t] + b_f)<br>
            입력 문: i_t = σ(W_i × [h_{t-1}, x_t] + b_i)<br>
            출력 문: o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
        </div>
        
        <div class="explanation">
            <strong>공식 해석:</strong><br>
            • <span class="highlight">σ</span> = 시그모이드 함수(0~1 사이 값으로 "얼마나 열지" 결정)<br>
            • <span class="highlight">[h_{t-1}, x_t]</span> = 이전 기억과 현재 입력을 합침<br>
            • 각 문은 0(완전 닫힘)~1(완전 열림) 사이의 값을 가집니다
        </div>
        
        <h3>🎬 실생활 예시</h3>
        <div class="demo-container">
            <p><strong>"철수는 학교에 갔다. 그는 수학을 공부했다."</strong></p>
            <button class="button" onclick="alert('잊기 문: \"철수\"를 계속 기억할까? → 예(1에 가까운 값)')">잊기 문 작동</button>
            <button class="button" onclick="alert('입력 문: \"수학\"이라는 새 정보를 기억할까? → 예(1에 가까운 값)')">입력 문 작동</button>
            <button class="button" onclick="alert('출력 문: 지금 \"철수가 수학 공부\"라는 정보를 사용할까? → 예(1에 가까운 값)')">출력 문 작동</button>
        </div>
    </div>
    
    <div class="container">
        <h2>3. ARIMA - 시계열 데이터 예측 전문가</h2>
        
        <div class="explanation">
            💡 <strong>핵심 아이디어:</strong> ARIMA는 과거 데이터의 패턴을 분석해서 미래를 예측합니다.
            주가, 날씨, 판매량 같은 시간에 따른 변화를 예측할 때 사용해요!
        </div>
        
        <h3>🎯 ARIMA의 3요소</h3>
        <div class="visual-box" style="text-align: center;">
            <div class="arima-component">
                <h4>AR (자기회귀)</h4>
                <p>과거 값들의<br>영향</p>
            </div>
            <div class="arima-component">
                <h4>I (차분)</h4>
                <p>데이터를<br>안정화</p>
            </div>
            <div class="arima-component">
                <h4>MA (이동평균)</h4>
                <p>과거 오차의<br>영향</p>
            </div>
        </div>
        
        <h3>📐 핵심 공식</h3>
        <div class="formula">
            ARIMA(p,d,q): y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + θ₁ε_{t-1} + ... + θ_qε_{t-q} + ε_t
        </div>
        
        <div class="explanation">
            <strong>공식 해석:</strong><br>
            • <span class="highlight">p</span> = AR 차수 (과거 몇 개의 값을 볼 것인가)<br>
            • <span class="highlight">d</span> = 차분 횟수 (데이터를 몇 번 차분할 것인가)<br>
            • <span class="highlight">q</span> = MA 차수 (과거 몇 개의 오차를 볼 것인가)<br>
            • <span class="highlight">φ</span> = AR 계수 (과거 값의 영향력)<br>
            • <span class="highlight">θ</span> = MA 계수 (과거 오차의 영향력)<br>
            • <span class="highlight">ε</span> = 오차
        </div>
        
        <h3>🎬 실생활 예시: 아이스크림 판매량 예측</h3>
        <div class="demo-container">
            <p><strong>지난 5일간 판매량:</strong></p>
            <div class="example-data">
                <div class="data-point">월: 100개</div>
                <div class="data-point">화: 120개</div>
                <div class="data-point">수: 110개</div>
                <div class="data-point">목: 130개</div>
                <div class="data-point">금: 140개</div>
            </div>
            <br>
            <div style="background: #e8f5e9; padding: 15px; border-radius: 10px;">
                <strong>ARIMA 분석:</strong><br>
                • <strong>AR:</strong> "최근 며칠 판매량이 증가 추세네" → 토요일도 증가 예상<br>
                • <strong>I:</strong> "전체적으로 상승 트렌드가 있네" → 차분으로 안정화<br>
                • <strong>MA:</strong> "예측 오차가 있었네" → 오차도 고려해서 조정<br>
                <br>
                <strong>🎯 예측 결과: 토요일 약 145~150개</strong>
            </div>
        </div>
    </div>
    
    <div class="container">
        <h2>📊 세 모델 비교</h2>
        <div class="visual-box">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #3498db; color: white;">
                    <th style="padding: 15px; border: 1px solid #ddd;">특징</th>
                    <th style="padding: 15px; border: 1px solid #ddd;">RNN</th>
                    <th style="padding: 15px; border: 1px solid #ddd;">LSTM</th>
                    <th style="padding: 15px; border: 1px solid #ddd;">ARIMA</th>
                </tr>
                <tr>
                    <td style="padding: 15px; border: 1px solid #ddd;"><strong>장점</strong></td>
                    <td style="padding: 15px; border: 1px solid #ddd;">간단하고 직관적</td>
                    <td style="padding: 15px; border: 1px solid #ddd;">장기 기억 우수</td>
                    <td style="padding: 15px; border: 1px solid #ddd;">통계적 해석 가능</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 15px; border: 1px solid #ddd;"><strong>단점</strong></td>
                    <td style="padding: 15px; border: 1px solid #ddd;">장기 기억 약함</td>
                    <td style="padding: 15px; border: 1px solid #ddd;">복잡하고 느림</td>
                    <td style="padding: 15px; border: 1px solid #ddd;">비선형 패턴 약함</td>
                </tr>
                <tr>
                    <td style="padding: 15px; border: 1px solid #ddd;"><strong>사용 예</strong></td>
                    <td style="padding: 15px; border: 1px solid #ddd;">짧은 문장 분석</td>
                    <td style="padding: 15px; border: 1px solid #ddd;">긴 문서, 번역</td>
                    <td style="padding: 15px; border: 1px solid #ddd;">주가, 판매 예측</td>
                </tr>
            </table>
        </div>
    </div>
    
    <div class="container" style="background: #fff3cd; border-left: 5px solid #ffc107;">
        <h3>🎓 핵심 요약</h3>
        <ul>
            <li><strong>RNN:</strong> "이전 것을 기억하는" 기본 순환 신경망</li>
            <li><strong>LSTM:</strong> "중요한 것만 오래 기억하는" 똑똑한 순환 신경망</li>
            <li><strong>ARIMA:</strong> "과거 패턴을 분석해서 미래를 예측하는" 통계 모델</li>
        </ul>
        <br>
        <p><strong>💡 선택 가이드:</strong></p>
        <ul>
            <li>텍스트나 음성 같은 순서가 중요한 데이터 → RNN/LSTM</li>
            <li>시간에 따른 수치 데이터 예측 → ARIMA</li>
            <li>짧은 순서 데이터 → RNN</li>
            <li>긴 순서 데이터 → LSTM</li>
        </ul>
    </div>
</body>
</html>
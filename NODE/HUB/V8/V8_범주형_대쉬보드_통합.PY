# -*- coding: utf-8 -*-
"""
V8 실시간 대시보드 - 범주형 전용
10분, 15분, 25분 범주형 예측을 시각화
"""

import subprocess
import json
import os
from datetime import datetime
import webbrowser

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_categorical_predictions():
    """범주형 통합 예측 실행"""
    try:
        result = subprocess.run(
            ['python', os.path.join(script_dir, 'V8_실시간_범주형_통합.py')],
            capture_output=True,
            text=True,
            timeout=30
        )
        data = json.loads(result.stdout)
        return data
    except Exception as e:
        print(f"❌ 범주형 예측 실패: {e}")
        return []

def generate_html_dashboard(data):
    """범주형 HTML 대시보드 생성"""
    
    if not data or len(data) == 0:
        return "<html><body><h1>데이터 없음</h1></body></html>"
    
    # 데이터 추출
    current_time = data[0].get('CURRENT_TIME', '')
    current_value = data[0].get('CURRENT_VALUE', 0)
    
    pred_10 = data[0] if len(data) > 0 else {}
    pred_15 = data[1] if len(data) > 1 else {}
    pred_25 = data[2] if len(data) > 2 else {}
    
    # 급증 확률 최대값
    max_surge_prob = max(
        pred_10.get('CLASS_2_PROB', 0),
        pred_15.get('CLASS_2_PROB', 0),
        pred_25.get('CLASS_2_PROB', 0)
    )
    
    # 추세 판단 (급증 확률 기준)
    surge_10 = pred_10.get('CLASS_2_PROB', 0)
    surge_15 = pred_15.get('CLASS_2_PROB', 0)
    surge_25 = pred_25.get('CLASS_2_PROB', 0)
    
    if surge_10 < surge_15 < surge_25:
        trend = "📈 급증 확률 증가 추세"
        trend_color = "#e53e3e"
        trend_alert = "시간이 지날수록 급증 가능성이 높아집니다"
    elif surge_10 > surge_15 > surge_25:
        trend = "📉 급증 확률 감소 추세"
        trend_color = "#38a169"
        trend_alert = "급증 위험이 줄어들고 있습니다"
    else:
        trend = "📊 급증 확률 변동 추세"
        trend_color = "#d69e2e"
        trend_alert = "급증 확률이 불규칙하게 변화합니다"
    
    # 위험 레벨 (최대 급증 확률 기준)
    if max_surge_prob >= 70:
        risk_level = "CRITICAL"
        risk_color = "#c53030"
        risk_icon = "🔴"
        risk_desc = "매우 높은 급증 위험"
    elif max_surge_prob >= 50:
        risk_level = "HIGH"
        risk_color = "#dd6b20"
        risk_icon = "🟠"
        risk_desc = "높은 급증 위험"
    elif max_surge_prob >= 30:
        risk_level = "MEDIUM"
        risk_color = "#d69e2e"
        risk_icon = "🟡"
        risk_desc = "중간 수준 급증 위험"
    else:
        risk_level = "LOW"
        risk_color = "#38a169"
        risk_icon = "🟢"
        risk_desc = "낮은 급증 위험"
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V8 범주형 예측 대시보드</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            font-size: 36px;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #718096;
            font-size: 16px;
        }}
        
        .header-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        
        .header-item {{
            text-align: center;
        }}
        
        .header-label {{
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .header-value {{
            font-size: 32px;
            font-weight: 700;
            color: #2d3748;
        }}
        
        .current-value {{
            color: #4299e1;
        }}
        
        .surge-prob {{
            color: {risk_color};
        }}
        
        .summary-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .risk-banner {{
            background: {risk_color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .risk-banner-icon {{
            font-size: 48px;
            margin-bottom: 10px;
        }}
        
        .risk-banner-text {{
            font-size: 24px;
            font-weight: 700;
        }}
        
        .risk-banner-desc {{
            margin-top: 10px;
            font-size: 16px;
        }}
        
        .trend-card {{
            background: #f7fafc;
            border-left: 4px solid {trend_color};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .trend-title {{
            font-size: 20px;
            font-weight: 700;
            color: {trend_color};
            margin-bottom: 10px;
        }}
        
        .trend-desc {{
            font-size: 16px;
            color: #4a5568;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .chart-title {{
            font-size: 20px;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 20px;
        }}
        
        .probability-bars {{
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            height: 250px;
            background: #f7fafc;
            border-radius: 10px;
            padding: 20px;
        }}
        
        .prob-bar-item {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
        }}
        
        .prob-bar {{
            width: 80%;
            background: linear-gradient(to top, #f093fb, #f5576c);
            border-radius: 8px 8px 0 0;
            transition: all 0.3s;
        }}
        
        .prob-bar:hover {{
            transform: scale(1.05);
        }}
        
        .prob-bar-label {{
            margin-top: 15px;
            font-weight: 700;
            font-size: 16px;
            color: #2d3748;
        }}
        
        .prob-bar-value {{
            margin-top: 5px;
            font-size: 24px;
            font-weight: 700;
            color: #f5576c;
        }}
        
        .predictions-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .prediction-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .card-header {{
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .time-badge {{
            font-size: 24px;
            font-weight: 700;
            color: #f5576c;
        }}
        
        .time-range {{
            font-size: 14px;
            color: #718096;
            margin-top: 5px;
        }}
        
        .result-highlight {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 20px;
        }}
        
        .class-section {{
            margin-bottom: 15px;
        }}
        
        .class-label {{
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .class-bar-container {{
            background: #f7fafc;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }}
        
        .class-bar-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .class-name {{
            font-size: 14px;
            font-weight: 600;
            color: #2d3748;
        }}
        
        .class-prob {{
            font-size: 18px;
            font-weight: 700;
            color: #f5576c;
        }}
        
        .class-bar {{
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .class-bar-fill {{
            height: 100%;
            background: linear-gradient(to right, #f093fb, #f5576c);
            transition: width 0.5s;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 16px;
            margin-top: 15px;
        }}
        
        .status-normal {{
            background: #c6f6d5;
            color: #2f855a;
        }}
        
        .status-caution {{
            background: #fefcbf;
            color: #b7791f;
        }}
        
        .status-warning {{
            background: #feebc8;
            color: #c05621;
        }}
        
        .status-critical {{
            background: #fed7d7;
            color: #c53030;
        }}
        
        .alert-box {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: 700;
            margin-top: 15px;
        }}
        
        .alert-surge {{
            background: #fed7d7;
            color: #c53030;
            border: 2px solid #fc8181;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 V8 범주형 예측 대시보드</h1>
            <p class="subtitle">3-Class Classification | 10분/15분/25분 시간대 분석</p>
            <div class="header-grid">
                <div class="header-item">
                    <div class="header-label">현재값</div>
                    <div class="header-value current-value">{current_value}</div>
                </div>
                <div class="header-item">
                    <div class="header-label">최대 급증 확률</div>
                    <div class="header-value surge-prob">{max_surge_prob:.1f}%</div>
                </div>
                <div class="header-item">
                    <div class="header-label">위험 레벨</div>
                    <div class="header-value" style="color: {risk_color};">{risk_icon} {risk_level}</div>
                </div>
            </div>
        </div>

        <!-- Risk Banner -->
        <div class="risk-banner">
            <div class="risk-banner-icon">{risk_icon}</div>
            <div class="risk-banner-text">{risk_desc}</div>
            <div class="risk-banner-desc">데이터 시간: {current_time}</div>
        </div>

        <!-- Trend Card -->
        <div class="summary-card">
            <div class="trend-card">
                <div class="trend-title">{trend}</div>
                <div class="trend-desc">{trend_alert}</div>
            </div>
        </div>

        <!-- Probability Chart -->
        <div class="chart-container">
            <div class="chart-title">📊 시간대별 급증(Class 2) 확률 변화</div>
            <div class="probability-bars">
                <div class="prob-bar-item">
                    <div class="prob-bar" style="height: {surge_10}%;"></div>
                    <div class="prob-bar-label">10분 후</div>
                    <div class="prob-bar-value">{surge_10:.1f}%</div>
                </div>
                <div class="prob-bar-item">
                    <div class="prob-bar" style="height: {surge_15}%;"></div>
                    <div class="prob-bar-label">15분 후</div>
                    <div class="prob-bar-value">{surge_15:.1f}%</div>
                </div>
                <div class="prob-bar-item">
                    <div class="prob-bar" style="height: {surge_25}%;"></div>
                    <div class="prob-bar-label">25분 후</div>
                    <div class="prob-bar-value">{surge_25:.1f}%</div>
                </div>
            </div>
        </div>

        <!-- Predictions Grid -->
        <div class="predictions-grid">
            <!-- 10분 -->
            <div class="prediction-card">
                <div class="card-header">
                    <div class="time-badge">⏱️ 10분 후</div>
                    <div class="time-range">{pred_10.get('PREDICTION_TIME', '')}</div>
                </div>
                
                <div class="result-highlight">
                    {pred_10.get('RESULT', '')}
                </div>
                
                <div class="class-section">
                    <div class="class-label">클래스별 확률</div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🔴 Class 2 (급증)</span>
                            <span class="class-prob">{pred_10.get('CLASS_2_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_10.get('CLASS_2_PROB', 0)}%;"></div>
                        </div>
                    </div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🟡 Class 1 (소폭증가)</span>
                            <span class="class-prob">{pred_10.get('CLASS_1_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_10.get('CLASS_1_PROB', 0)}%;"></div>
                        </div>
                    </div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🟢 Class 0 (하락/정체)</span>
                            <span class="class-prob">{pred_10.get('CLASS_0_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_10.get('CLASS_0_PROB', 0)}%;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <span class="status-badge status-{pred_10.get('STATUS', '').lower()}">{pred_10.get('STATUS', '')}</span>
                </div>
                
                {'<div class="alert-box alert-surge">🚨 급증 경고 발령</div>' if pred_10.get('SURGE_ALERT', 0) == 1 else ''}
            </div>

            <!-- 15분 -->
            <div class="prediction-card">
                <div class="card-header">
                    <div class="time-badge">⏱️ 15분 후</div>
                    <div class="time-range">{pred_15.get('PREDICTION_TIME', '')}</div>
                </div>
                
                <div class="result-highlight">
                    {pred_15.get('RESULT', '')}
                </div>
                
                <div class="class-section">
                    <div class="class-label">클래스별 확률</div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🔴 Class 2 (급증)</span>
                            <span class="class-prob">{pred_15.get('CLASS_2_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_15.get('CLASS_2_PROB', 0)}%;"></div>
                        </div>
                    </div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🟡 Class 1 (소폭증가)</span>
                            <span class="class-prob">{pred_15.get('CLASS_1_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_15.get('CLASS_1_PROB', 0)}%;"></div>
                        </div>
                    </div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🟢 Class 0 (하락/정체)</span>
                            <span class="class-prob">{pred_15.get('CLASS_0_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_15.get('CLASS_0_PROB', 0)}%;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <span class="status-badge status-{pred_15.get('STATUS', '').lower()}">{pred_15.get('STATUS', '')}</span>
                </div>
                
                {'<div class="alert-box alert-surge">🚨 급증 경고 발령</div>' if pred_15.get('SURGE_ALERT', 0) == 1 else ''}
            </div>

            <!-- 25분 -->
            <div class="prediction-card">
                <div class="card-header">
                    <div class="time-badge">⏱️ 25분 후</div>
                    <div class="time-range">{pred_25.get('PREDICTION_TIME', '')}</div>
                </div>
                
                <div class="result-highlight">
                    {pred_25.get('RESULT', '')}
                </div>
                
                <div class="class-section">
                    <div class="class-label">클래스별 확률</div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🔴 Class 2 (급증)</span>
                            <span class="class-prob">{pred_25.get('CLASS_2_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_25.get('CLASS_2_PROB', 0)}%;"></div>
                        </div>
                    </div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🟡 Class 1 (소폭증가)</span>
                            <span class="class-prob">{pred_25.get('CLASS_1_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_25.get('CLASS_1_PROB', 0)}%;"></div>
                        </div>
                    </div>
                    
                    <div class="class-bar-container">
                        <div class="class-bar-header">
                            <span class="class-name">🟢 Class 0 (하락/정체)</span>
                            <span class="class-prob">{pred_25.get('CLASS_0_PROB', 0):.1f}%</span>
                        </div>
                        <div class="class-bar">
                            <div class="class-bar-fill" style="width: {pred_25.get('CLASS_0_PROB', 0)}%;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <span class="status-badge status-{pred_25.get('STATUS', '').lower()}">{pred_25.get('STATUS', '')}</span>
                </div>
                
                {'<div class="alert-box alert-surge">🚨 급증 경고 발령</div>' if pred_25.get('SURGE_ALERT', 0) == 1 else ''}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>🤖 V8 Categorical Prediction System | XGBoost Classifier</p>
            <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
    
    return html

def main():
    """메인 실행"""
    print("="*60)
    print("V8 범주형 대시보드 생성 중...")
    print("="*60)
    
    print("\n[1/2] 범주형 예측 실행 중...")
    data = get_categorical_predictions()
    
    if not data:
        print("\n❌ 예측 실패!")
        return
    
    print("[2/2] HTML 대시보드 생성 중...")
    html_content = generate_html_dashboard(data)
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'V8_범주형_대시보드_{timestamp}.html'
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ 대시보드 생성 완료!")
    print(f"📂 파일: {filename}")
    print(f"📍 위치: {filepath}")
    
    # 브라우저 열기
    try:
        webbrowser.open('file://' + os.path.abspath(filepath))
        print(f"\n🌐 브라우저에서 대시보드를 열었습니다!")
    except Exception as e:
        print(f"\n⚠️ 브라우저 자동 열기 실패: {e}")
        print(f"💡 수동으로 파일을 열어주세요: {filepath}")

if __name__ == '__main__':
    main()
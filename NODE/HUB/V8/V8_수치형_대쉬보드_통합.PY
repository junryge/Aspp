# -*- coding: utf-8 -*-
"""
V8 실시간 대시보드 - 수치형 전용
10분, 15분, 25분 수치형 예측을 시각화
"""

import subprocess
import json
import os
from datetime import datetime
import webbrowser

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_numerical_predictions():
    """수치형 통합 예측 실행"""
    try:
        result = subprocess.run(
            ['python', os.path.join(script_dir, 'V8_Numerical_Real_time_DO.py')],
            capture_output=True,
            text=True,
            timeout=30
        )
        data = json.loads(result.stdout)
        return data
    except Exception as e:
        print(f"❌ 수치형 예측 실패: {e}")
        return []

def generate_html_dashboard(data):
    """수치형 HTML 대시보드 생성"""
    
    if not data or len(data) == 0:
        return "<html><body><h1>데이터 없음</h1></body></html>"
    
    # 데이터 추출
    current_time = data[0].get('CURRENT_TIME', '')
    current_value = data[0].get('CURRENT_VALUE', 0)
    
    pred_10 = data[0] if len(data) > 0 else {}
    pred_15 = data[1] if len(data) > 1 else {}
    pred_25 = data[2] if len(data) > 2 else {}
    
    # 최대값 찾기
    max_value = max(
        pred_10.get('PREDICT_MAX', 0),
        pred_15.get('PREDICT_MAX', 0),
        pred_25.get('PREDICT_MAX', 0)
    )
    
    # 추세 판단
    if pred_10.get('PREDICT_MAX', 0) < pred_15.get('PREDICT_MAX', 0) < pred_25.get('PREDICT_MAX', 0):
        trend = "📈 지속 상승 추세"
        trend_color = "#e53e3e"
        trend_alert = "위험도가 시간에 따라 증가합니다"
    elif pred_10.get('PREDICT_MAX', 0) > pred_15.get('PREDICT_MAX', 0) > pred_25.get('PREDICT_MAX', 0):
        trend = "📉 지속 하락 추세"
        trend_color = "#38a169"
        trend_alert = "안정화 중입니다"
    else:
        trend = "📊 변동 추세"
        trend_color = "#d69e2e"
        trend_alert = "변화가 불규칙합니다"
    
    # 위험 레벨
    if max_value >= 300:
        risk_level = "CRITICAL"
        risk_color = "#c53030"
        risk_icon = "🔴"
    elif max_value >= 280:
        risk_level = "WARNING"
        risk_color = "#dd6b20"
        risk_icon = "🟠"
    elif max_value >= 270:
        risk_level = "CAUTION"
        risk_color = "#d69e2e"
        risk_icon = "🟡"
    else:
        risk_level = "NORMAL"
        risk_color = "#38a169"
        risk_icon = "🟢"
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V8 수치형 예측 대시보드</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        
        .max-value {{
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
            color: #667eea;
        }}
        
        .time-range {{
            font-size: 14px;
            color: #718096;
            margin-top: 5px;
        }}
        
        .metric-section {{
            margin-bottom: 20px;
        }}
        
        .metric-title {{
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-big {{
            font-size: 48px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }}
        
        .metric-max {{
            color: #e53e3e;
        }}
        
        .metric-avg {{
            color: #ed8936;
        }}
        
        .metric-min {{
            color: #48bb78;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: #f7fafc;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #4a5568;
        }}
        
        .metric-value {{
            font-size: 18px;
            font-weight: 700;
            color: #2d3748;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 14px;
            margin-top: 10px;
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
        
        .timeline {{
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            height: 250px;
            background: #f7fafc;
            border-radius: 10px;
            padding: 20px;
        }}
        
        .timeline-item {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
        }}
        
        .bar {{
            width: 80%;
            background: linear-gradient(to top, #667eea, #764ba2);
            border-radius: 8px 8px 0 0;
            transition: all 0.3s;
        }}
        
        .bar:hover {{
            transform: scale(1.05);
        }}
        
        .bar-label {{
            margin-top: 15px;
            font-weight: 700;
            font-size: 16px;
            color: #2d3748;
        }}
        
        .bar-value {{
            margin-top: 5px;
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
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
            <h1>📊 V8 수치형 예측 대시보드</h1>
            <p class="subtitle">MIN/MAX/AVG 변화량 예측 | 10분/15분/25분 시간대 분석</p>
            <div class="header-grid">
                <div class="header-item">
                    <div class="header-label">현재값</div>
                    <div class="header-value current-value">{current_value}</div>
                </div>
                <div class="header-item">
                    <div class="header-label">최대 예측값</div>
                    <div class="header-value max-value">{max_value:.1f}</div>
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
            <div class="risk-banner-text">위험 레벨: {risk_level}</div>
            <div style="margin-top: 10px; font-size: 16px;">데이터 시간: {current_time}</div>
        </div>

        <!-- Trend Card -->
        <div class="summary-card">
            <div class="trend-card">
                <div class="trend-title">{trend}</div>
                <div class="trend-desc">{trend_alert}</div>
            </div>
        </div>

        <!-- Timeline Chart -->
        <div class="chart-container">
            <div class="chart-title">📈 시간대별 MAX 값 변화</div>
            <div class="timeline">
                <div class="timeline-item">
                    <div class="bar" style="height: {pred_10.get('PREDICT_MAX', 0) / max_value * 100 if max_value > 0 else 0}%;"></div>
                    <div class="bar-label">10분 후</div>
                    <div class="bar-value">{pred_10.get('PREDICT_MAX', 0):.1f}</div>
                </div>
                <div class="timeline-item">
                    <div class="bar" style="height: {pred_15.get('PREDICT_MAX', 0) / max_value * 100 if max_value > 0 else 0}%;"></div>
                    <div class="bar-label">15분 후</div>
                    <div class="bar-value">{pred_15.get('PREDICT_MAX', 0):.1f}</div>
                </div>
                <div class="timeline-item">
                    <div class="bar" style="height: {pred_25.get('PREDICT_MAX', 0) / max_value * 100 if max_value > 0 else 0}%;"></div>
                    <div class="bar-label">25분 후</div>
                    <div class="bar-value">{pred_25.get('PREDICT_MAX', 0):.1f}</div>
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
                
                <div class="metric-section">
                    <div class="metric-title">MAX 예측값</div>
                    <div class="metric-big metric-max">{pred_10.get('PREDICT_MAX', 0):.1f}</div>
                </div>
                
                <div class="metric-row">
                    <span class="metric-label">MIN</span>
                    <span class="metric-value">{pred_10.get('PREDICT_MIN', 0):.1f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">AVG</span>
                    <span class="metric-value">{pred_10.get('PREDICT_AVG', 0):.1f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">변화량 (MAX)</span>
                    <span class="metric-value">{pred_10.get('CHANGE_MAX', 0):+.1f}</span>
                </div>
                
                <div style="text-align: center; margin-top: 15px;">
                    <span class="status-badge status-{pred_10.get('STATUS', '').lower()}">{pred_10.get('STATUS', '')}</span>
                </div>
                
                {'<div style="margin-top: 15px; padding: 10px; background: #fed7d7; border-radius: 8px; text-align: center; color: #c53030; font-weight: 700;">🚨 급증 경고</div>' if pred_10.get('SURGE_WARNING', 0) == 1 else ''}
            </div>

            <!-- 15분 -->
            <div class="prediction-card">
                <div class="card-header">
                    <div class="time-badge">⏱️ 15분 후</div>
                    <div class="time-range">{pred_15.get('PREDICTION_TIME', '')}</div>
                </div>
                
                <div class="metric-section">
                    <div class="metric-title">MAX 예측값</div>
                    <div class="metric-big metric-max">{pred_15.get('PREDICT_MAX', 0):.1f}</div>
                </div>
                
                <div class="metric-row">
                    <span class="metric-label">MIN</span>
                    <span class="metric-value">{pred_15.get('PREDICT_MIN', 0):.1f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">AVG</span>
                    <span class="metric-value">{pred_15.get('PREDICT_AVG', 0):.1f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">변화량 (MAX)</span>
                    <span class="metric-value">{pred_15.get('CHANGE_MAX', 0):+.1f}</span>
                </div>
                
                <div style="text-align: center; margin-top: 15px;">
                    <span class="status-badge status-{pred_15.get('STATUS', '').lower()}">{pred_15.get('STATUS', '')}</span>
                </div>
                
                {'<div style="margin-top: 15px; padding: 10px; background: #fed7d7; border-radius: 8px; text-align: center; color: #c53030; font-weight: 700;">🚨 급증 경고</div>' if pred_15.get('SURGE_WARNING', 0) == 1 else ''}
            </div>

            <!-- 25분 -->
            <div class="prediction-card">
                <div class="card-header">
                    <div class="time-badge">⏱️ 25분 후</div>
                    <div class="time-range">{pred_25.get('PREDICTION_TIME', '')}</div>
                </div>
                
                <div class="metric-section">
                    <div class="metric-title">MAX 예측값</div>
                    <div class="metric-big metric-max">{pred_25.get('PREDICT_MAX', 0):.1f}</div>
                </div>
                
                <div class="metric-row">
                    <span class="metric-label">MIN</span>
                    <span class="metric-value">{pred_25.get('PREDICT_MIN', 0):.1f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">AVG</span>
                    <span class="metric-value">{pred_25.get('PREDICT_AVG', 0):.1f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">변화량 (MAX)</span>
                    <span class="metric-value">{pred_25.get('CHANGE_MAX', 0):+.1f}</span>
                </div>
                
                <div style="text-align: center; margin-top: 15px;">
                    <span class="status-badge status-{pred_25.get('STATUS', '').lower()}">{pred_25.get('STATUS', '')}</span>
                </div>
                
                {'<div style="margin-top: 15px; padding: 10px; background: #fed7d7; border-radius: 8px; text-align: center; color: #c53030; font-weight: 700;">🚨 급증 경고</div>' if pred_25.get('SURGE_WARNING', 0) == 1 else ''}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>🤖 V8 Numerical Prediction System | XGBoost Regression Model</p>
            <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
    
    return html

def main():
    """메인 실행"""
    print("="*60)
    print("V8 수치형 대시보드 생성 중...")
    print("="*60)
    
    print("\n[1/2] 수치형 예측 실행 중...")
    data = get_numerical_predictions()
    
    if not data:
        print("\n❌ 예측 실패!")
        return
    
    print("[2/2] HTML 대시보드 생성 중...")
    html_content = generate_html_dashboard(data)
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'V8_수치형_대시보드_{timestamp}.html'
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
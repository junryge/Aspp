# -*- coding: utf-8 -*-
"""
V6 실시간 예측 시스템 - 10분/15분/25분 통합 비교
- 30분 시퀀스 → 3개 예측 시간 동시 예측
- Momentum Feature 포함
- 300 이상 달성 확률 계산
- 3개 모델 결과 비교 대시보드
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'HUBROOM_PIVOT_DATA.csv')

# V6 Feature 컬럼 정의
FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def calculate_exceed_300_probability(prediction, features):
    """300 이상 달성 확률 계산"""
    if prediction >= 310:
        base_prob = 95
    elif prediction >= 300:
        base_prob = 85
    elif prediction >= 290:
        base_prob = 60
    elif prediction >= 280:
        base_prob = 35
    elif prediction >= 270:
        base_prob = 20
    else:
        base_prob = 5
    
    momentum_boost = 0
    if features.get('surge_imminent', 0) == 1:
        momentum_boost += 15
    if features.get('target_acceleration', 0) > 1.0:
        momentum_boost += 10
    elif features.get('target_acceleration', 0) > 0.5:
        momentum_boost += 5
    
    risk_boost = 0
    if features.get('hub_high', 0) == 1:
        risk_boost += 10
    if features.get('storage_util_critical', 0) == 1:
        risk_boost += 5
    if features.get('total_cmd_low', 0) == 1:
        risk_boost += 5
    
    final_prob = min(100, max(0, base_prob + momentum_boost + risk_boost))
    return final_prob

def analyze_surge_cause(features, seq_target):
    """상승 원인 분석"""
    causes = []
    
    hub_last = features.get('HUBROOMTOTAL_last_value', 650)
    if hub_last < 590:
        causes.append(f"HUB심각병목({int(hub_last)})")
    elif hub_last < 610:
        causes.append(f"HUB병목({int(hub_last)})")
    
    acceleration = features.get('target_acceleration', 0)
    if acceleration > 1.0:
        causes.append(f"매우높은가속도({acceleration:.1f})")
    elif acceleration > 0.5:
        causes.append(f"높은가속도({acceleration:.1f})")
    
    total_cmd = features.get('total_cmd', 300)
    if total_cmd < 200:
        causes.append(f"CMD심각부족({int(total_cmd)})")
    elif total_cmd < 220:
        causes.append(f"CMD부족({int(total_cmd)})")
    
    storage_util = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    if storage_util >= 207:
        causes.append(f"스토리지심각({int(storage_util)})")
    elif storage_util >= 205:
        causes.append(f"스토리지과부하({int(storage_util)})")
    
    fs_util = features.get('CD_M163FSTORAGEUTIL_last_value', 0)
    if fs_util >= 10:
        causes.append(f"FS심각부족({fs_util:.1f})")
    elif fs_util >= 7:
        causes.append(f"FS부족({fs_util:.1f})")
    
    if features.get('target_rapid_rise', 0) == 1:
        causes.append("5분급등패턴")
    
    if features.get('surge_imminent', 0) == 1:
        causes.append("급증임박신호")
    
    if not causes:
        current_value = seq_target[-1]
        if current_value < 250:
            causes.append("정상범위")
        else:
            causes.append("완만한상승추세")
    
    return " + ".join(causes)

def create_features_realtime(df_30):
    """실시간 Feature 생성"""
    seq_target = df_30[TARGET_COL].values
    
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_last_value': seq_target[-1],
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
    }
    
    features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
    features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
    features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
    features['target_last_10_mean'] = np.mean(seq_target[-10:])
    
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in df_30.columns:
                continue
            
            col_seq = df_30[col].values
            
            if group_name == 'maxcapa':
                features[f'{col}_last_value'] = col_seq[-1]
            
            elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_std'] = np.std(col_seq)
                features[f'{col}_max'] = np.max(col_seq)
                features[f'{col}_min'] = np.min(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
            
            else:
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
    
    if 'CD_M163FSTORAGEUSE' in df_30.columns and 'CD_M163FSTORAGETOTAL' in df_30.columns:
        storage_use = df_30['CD_M163FSTORAGEUSE'].values
        storage_total = df_30['CD_M163FSTORAGETOTAL'].values
        storage_util = df_30['CD_M163FSTORAGEUTIL'].values
        
        features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
        features['storage_remaining'] = storage_total[-1] - storage_use[-1]
        features['storage_util_last'] = storage_util[-1]
        features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
    
    if 'HUBROOMTOTAL' in df_30.columns:
        hub_seq = df_30['HUBROOMTOTAL'].values
        hub_last = hub_seq[-1]
        
        features['hub_critical'] = 1 if hub_last < 590 else 0
        features['hub_high'] = 1 if hub_last < 610 else 0
        features['hub_warning'] = 1 if hub_last < 620 else 0
        features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        
        if 'CD_M163FSTORAGEUTIL' in df_30.columns:
            storage_util_last = df_30['CD_M163FSTORAGEUTIL'].iloc[-1]
            features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
    
    inflow_sum = sum(df_30[col].iloc[-1] for col in FEATURE_COLS['inflow'] if col in df_30.columns)
    outflow_sum = sum(df_30[col].iloc[-1] for col in FEATURE_COLS['outflow'] if col in df_30.columns)
    features['net_flow'] = inflow_sum - outflow_sum
    
    cmd_sum = sum(df_30[col].iloc[-1] for col in FEATURE_COLS['cmd'] if col in df_30.columns)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
    
    if 'HUBROOMTOTAL' in df_30.columns:
        hub_last = df_30['HUBROOMTOTAL'].iloc[-1]
        features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
    
    if 'M16A_3F_STORAGE_UTIL' in df_30.columns:
        storage_util = df_30['M16A_3F_STORAGE_UTIL'].iloc[-1]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
    
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 +
        features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 +
        features.get('storage_util_high', 0) * 1
    )
    
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 272 and
        features.get('target_acceleration', 0) > 0.2 and
        features.get('hub_high', 0) == 1
    ) else 0
    
    return features, seq_target

def predict_all_horizons():
    """3개 예측 시간 모델 모두 실행"""
    results = {}
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except FileNotFoundError:
        return None, "DATA_FILE_NOT_FOUND"
    except Exception as e:
        return None, f"DATA_LOAD_ERROR: {str(e)}"
    
    if TARGET_COL not in df.columns:
        return None, "TARGET_COLUMN_NOT_FOUND"
    
    if len(df) < 30:
        return None, "INSUFFICIENT_DATA"
    
    df_30 = df.tail(30).copy()
    df_30[TARGET_COL] = pd.to_numeric(df_30[TARGET_COL], errors='coerce')
    df_30 = df_30.dropna(subset=[TARGET_COL])
    
    if len(df_30) < 30:
        return None, "INSUFFICIENT_VALID_DATA"
    
    # STAT_DT 처리
    if 'STAT_DT' in df_30.columns:
        try:
            df_30['STAT_DT'] = pd.to_datetime(df_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime.now() - timedelta(minutes=29)
            df_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    else:
        base_time = datetime.now() - timedelta(minutes=29)
        df_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    
    current_time = df_30['STAT_DT'].iloc[-1]
    
    # Feature 생성
    features, seq_target = create_features_realtime(df_30)
    X_pred = pd.DataFrame([features])
    current_value = seq_target[-1]
    
    # 3개 모델 예측
    for horizon in [10, 15, 25]:
        model_path = os.path.join(script_dir, 'model', f'xgboost_30min_{horizon}min_D3.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            return None, f"MODEL_FILE_NOT_FOUND: {horizon}분"
        except Exception as e:
            return None, f"MODEL_LOAD_ERROR_{horizon}분: {str(e)}"
        
        prediction = model.predict(X_pred)[0]
        increase_rate = (prediction - current_value) / current_value * 100
        
        if prediction >= 300:
            status = "DANGEROUS"
        elif prediction >= 280:
            status = "WARNING"
        else:
            status = "NORMAL"
        
        risk_score = features.get('surge_risk_score', 0)
        if risk_score >= 5:
            surge_risk = "CRITICAL"
        elif risk_score >= 3:
            surge_risk = "HIGH"
        elif risk_score >= 1:
            surge_risk = "MEDIUM"
        else:
            surge_risk = "LOW"
        
        exceed_300_prob = calculate_exceed_300_probability(prediction, features)
        
        prediction_start = current_time + timedelta(minutes=1)
        prediction_end = current_time + timedelta(minutes=horizon)
        
        results[horizon] = {
            "PREDICTED_VALUE": int(round(prediction)),
            "INCREASE_RATE": f"{increase_rate:+.1f}%",
            "EXCEED_300_PROBABILITY": exceed_300_prob,
            "STATUS": status,
            "SURGE_RISK": surge_risk,
            "PREDICTION_TIME": f"{prediction_start.strftime('%Y-%m-%d %H:%M')} ~ {prediction_end.strftime('%Y-%m-%d %H:%M')}"
        }
    
    # 공통 정보
    common_info = {
        "CURRENT_VALUE": round(current_value, 1),
        "CURRENT_TIME": current_time.strftime('%Y-%m-%d %H:%M:%S'),
        "SEQUENCE_START_TIME": (current_time - timedelta(minutes=29)).strftime('%Y-%m-%d %H:%M:%S'),
        "CAUSE_ANALYSIS": analyze_surge_cause(features, seq_target),
        "SEQUENCE_DATA": seq_target.tolist(),  # 시퀀스 그래프용 데이터
        "MOMENTUM_SIGNAL": {
            "acceleration": round(features.get('target_acceleration', 0), 2),
            "is_rising": bool(features.get('target_is_rising', 0)),
            "rapid_rise": bool(features.get('target_rapid_rise', 0)),
            "surge_imminent": bool(features.get('surge_imminent', 0))
        },
        "KEY_FACTORS": {
            "HUBROOMTOTAL": int(features.get('HUBROOMTOTAL_last_value', 0)),
            "FS_STORAGE_UTIL": round(features.get('CD_M163FSTORAGEUTIL_last_value', 0), 1),
            "TOTAL_CMD": int(features.get('total_cmd', 0)),
            "STORAGE_UTIL": int(features.get('M16A_3F_STORAGE_UTIL_last_value', 0))
        },
        "MODEL_VERSION": "V6_MULTI_HORIZON_MOMENTUM_W10X_T280"
    }
    
    return {**common_info, "predictions": results}, None

def create_sequence_graph(seq_data, features, predictions, current_time):
    """30분 시퀀스 + 예측값 인터랙티브 그래프 생성"""
    
    # 실제 시간 라벨 생성 (30분 시퀀스)
    time_labels = []
    for i in range(30):
        time_point = current_time - timedelta(minutes=(29-i))
        time_labels.append(time_point.strftime('%Y-%m-%d %H:%M'))
    
    # 예측 시간 라벨
    pred_time_10 = current_time + timedelta(minutes=10)
    pred_time_15 = current_time + timedelta(minutes=15)
    pred_time_25 = current_time + timedelta(minutes=25)
    
    # 호버 텍스트 생성 (각 포인트별 상세 정보)
    hover_texts = []
    for i, value in enumerate(seq_data):
        status = "🟢 안전" if value < 280 else ("🟡 경고" if value < 300 else "🔴 위험")
        
        # 변화량 계산
        if i > 0:
            change = value - seq_data[i-1]
            change_text = f"변화: {change:+.1f}"
        else:
            change_text = "변화: -"
        
        # 추세 분석
        if i >= 5:
            recent_avg = np.mean(seq_data[i-5:i+1])
            trend = "상승 추세" if recent_avg > np.mean(seq_data[:i-4]) else "하락 추세"
        else:
            trend = "분석 중"
        
        hover_text = (
            f"<b>{time_labels[i]}</b><br>"
            f"값: <b>{value:.1f}</b><br>"
            f"{status}<br>"
            f"{change_text}<br>"
            f"추세: {trend}"
        )
        hover_texts.append(hover_text)
    
    # 그래프 생성
    fig = go.Figure()
    
    # 배경 색상 영역
    fig.add_hrect(y0=300, y1=max(seq_data)+50, 
                  fillcolor="rgba(254, 215, 215, 0.3)", 
                  line_width=0, 
                  annotation_text="위험", 
                  annotation_position="top right")
    
    fig.add_hrect(y0=280, y1=300, 
                  fillcolor="rgba(254, 235, 200, 0.3)", 
                  line_width=0,
                  annotation_text="경고",
                  annotation_position="top right")
    
    fig.add_hrect(y0=min(seq_data)-20, y1=280, 
                  fillcolor="rgba(198, 246, 213, 0.2)", 
                  line_width=0,
                  annotation_text="안전",
                  annotation_position="bottom right")
    
    # 메인 라인 (30분 시퀀스)
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=seq_data,
        mode='lines+markers',
        name='실제 시퀀스 (30분)',
        line=dict(color='#4299e1', width=3),
        marker=dict(size=8, color='#4299e1', line=dict(color='white', width=2)),
        hovertext=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=14,
            font_color="white"
        )
    ))
    
    # 현재값에서 예측값으로 연결선
    current_value = seq_data[-1]
    
    # 10분 예측
    pred_10 = predictions[10]['PREDICTED_VALUE']
    status_10 = "🟢" if pred_10 < 280 else ("🟡" if pred_10 < 300 else "🔴")
    fig.add_trace(go.Scatter(
        x=[time_labels[-1], pred_time_10.strftime('%Y-%m-%d %H:%M')],
        y=[current_value, pred_10],
        mode='lines+markers',
        name=f'10분 예측: {pred_10}',
        line=dict(color='#3182ce', width=2, dash='dash'),
        marker=dict(size=12, color='#3182ce', symbol='diamond'),
        hovertext=f"<b>10분 후 예측</b><br>{pred_time_10.strftime('%Y-%m-%d %H:%M')}<br>예측값: <b>{pred_10}</b><br>{status_10} {predictions[10]['STATUS']}<br>확률: {predictions[10]['EXCEED_300_PROBABILITY']}%",
        hoverinfo='text'
    ))
    
    # 15분 예측
    pred_15 = predictions[15]['PREDICTED_VALUE']
    status_15 = "🟢" if pred_15 < 280 else ("🟡" if pred_15 < 300 else "🔴")
    fig.add_trace(go.Scatter(
        x=[time_labels[-1], pred_time_15.strftime('%Y-%m-%d %H:%M')],
        y=[current_value, pred_15],
        mode='lines+markers',
        name=f'15분 예측: {pred_15}',
        line=dict(color='#38a169', width=2, dash='dash'),
        marker=dict(size=12, color='#38a169', symbol='diamond'),
        hovertext=f"<b>15분 후 예측</b><br>{pred_time_15.strftime('%Y-%m-%d %H:%M')}<br>예측값: <b>{pred_15}</b><br>{status_15} {predictions[15]['STATUS']}<br>확률: {predictions[15]['EXCEED_300_PROBABILITY']}%",
        hoverinfo='text'
    ))
    
    # 25분 예측
    pred_25 = predictions[25]['PREDICTED_VALUE']
    status_25 = "🟢" if pred_25 < 280 else ("🟡" if pred_25 < 300 else "🔴")
    fig.add_trace(go.Scatter(
        x=[time_labels[-1], pred_time_25.strftime('%Y-%m-%d %H:%M')],
        y=[current_value, pred_25],
        mode='lines+markers',
        name=f'25분 예측: {pred_25}',
        line=dict(color='#dd6b20', width=2, dash='dash'),
        marker=dict(size=12, color='#dd6b20', symbol='diamond'),
        hovertext=f"<b>25분 후 예측</b><br>{pred_time_25.strftime('%Y-%m-%d %H:%M')}<br>예측값: <b>{pred_25}</b><br>{status_25} {predictions[25]['STATUS']}<br>확률: {predictions[25]['EXCEED_300_PROBABILITY']}%",
        hoverinfo='text'
    ))
    
    # 기준선
    fig.add_hline(y=300, line_dash="dot", line_color="#f56565", line_width=2,
                  annotation_text="위험선 (300)", annotation_position="right")
    
    fig.add_hline(y=280, line_dash="dot", line_color="#ed8936", line_width=2,
                  annotation_text="경고선 (280)", annotation_position="right")
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="<b>30분 시퀀스 + 3개 예측값 분석</b><br><sub>클릭하여 확대/축소 가능 | 호버하면 상세 정보 표시</sub>",
            font=dict(size=18, color='#2d3748'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="<b>Job Queue Level</b>",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickfont=dict(size=12)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=None,
        autosize=True,
        margin=dict(l=80, r=80, t=120, b=100),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )
    
    # HTML로 변환
    graph_html = fig.to_html(
        include_plotlyjs='cdn',
        div_id='sequence-graph',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'sequence_graph',
                'height': 600,
                'width': 1400
            }
        }
    )
    
    # 원인 분석 텍스트 생성
    analysis_text = ""
    
    # 가속도 분석
    if features.get('target_acceleration', 0) > 1.0:
        analysis_text += "• <b>매우 높은 가속도</b> 감지 (급격한 상승 중)<br>"
    elif features.get('target_acceleration', 0) > 0.5:
        analysis_text += "• <b>높은 가속도</b> 감지 (빠른 상승 추세)<br>"
    elif features.get('target_acceleration', 0) > 0.2:
        analysis_text += "• 가속도 증가 중 (상승 추세)<br>"
    
    # 추세 분석
    if features.get('target_is_rising', 0) == 1:
        analysis_text += "• 현재 <b>상승 추세</b> 지속 중<br>"
    
    # 급등 패턴
    if features.get('target_rapid_rise', 0) == 1:
        analysis_text += "• ⚠️ <b>5분 급등 패턴</b> 감지됨<br>"
    
    # HUB 상태
    hub_last = features.get('HUBROOMTOTAL_last_value', 650)
    if hub_last < 590:
        analysis_text += f"• 🔴 <b>HUB 심각 병목</b> (현재: {int(hub_last)})<br>"
    elif hub_last < 610:
        analysis_text += f"• 🟡 <b>HUB 병목 위험</b> (현재: {int(hub_last)})<br>"
    
    # CMD 상태
    total_cmd = features.get('total_cmd', 300)
    if total_cmd < 200:
        analysis_text += f"• 🔴 <b>CMD 심각 부족</b> (현재: {int(total_cmd)})<br>"
    elif total_cmd < 220:
        analysis_text += f"• 🟡 <b>CMD 부족</b> (현재: {int(total_cmd)})<br>"
    
    # 스토리지 상태
    storage_util = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    if storage_util >= 207:
        analysis_text += f"• 🔴 <b>스토리지 심각 과부하</b> (현재: {int(storage_util)})<br>"
    elif storage_util >= 205:
        analysis_text += f"• 🟡 <b>스토리지 과부하</b> (현재: {int(storage_util)})<br>"
    
    # 급증 임박
    if features.get('surge_imminent', 0) == 1:
        analysis_text += "• 🚨 <b>급증 임박 신호</b> 발생!<br>"
    
    if not analysis_text:
        analysis_text = "• 현재 정상 범위 내에서 운영 중입니다."
    
    return graph_html, analysis_text

def generate_html_dashboard(result):
    """3개 모델 비교 HTML 대시보드 생성"""
    
    # 현재 시간을 datetime 객체로 변환
    from datetime import datetime
    current_time = datetime.strptime(result['CURRENT_TIME'], '%Y-%m-%d %H:%M:%S')
    
    # 시퀀스 그래프 생성 (예측값 포함)
    graph_html, analysis_text = create_sequence_graph(result['SEQUENCE_DATA'], 
                                                       {**result['KEY_FACTORS'], 
                                                        **result['MOMENTUM_SIGNAL'],
                                                        'target_acceleration': result['MOMENTUM_SIGNAL']['acceleration'],
                                                        'target_is_rising': 1 if result['MOMENTUM_SIGNAL']['is_rising'] else 0,
                                                        'target_rapid_rise': 1 if result['MOMENTUM_SIGNAL']['rapid_rise'] else 0,
                                                        'surge_imminent': 1 if result['MOMENTUM_SIGNAL']['surge_imminent'] else 0,
                                                        'HUBROOMTOTAL_last_value': result['KEY_FACTORS']['HUBROOMTOTAL'],
                                                        'total_cmd': result['KEY_FACTORS']['TOTAL_CMD'],
                                                        'M16A_3F_STORAGE_UTIL_last_value': result['KEY_FACTORS']['STORAGE_UTIL']},
                                                       result['predictions'],
                                                       current_time)
    
    # 원인 분석 HTML
    causes = result['CAUSE_ANALYSIS'].split(' + ')
    cause_items_html = ""
    for cause in causes:
        if any(word in cause for word in ['심각', '매우높은', '급증임박']):
            icon = "🔴"
        else:
            icon = "⚠️"
        cause_items_html += f'<div class="cause-item"><div class="cause-icon">{icon}</div><div class="cause-text">{cause}</div></div>'
    
    # Momentum 체크
    momentum_checks = {
        'acceleration': '✅' if result['MOMENTUM_SIGNAL']['acceleration'] > 0.5 else '❌',
        'is_rising': '✅' if result['MOMENTUM_SIGNAL']['is_rising'] else '❌',
        'rapid_rise': '✅' if result['MOMENTUM_SIGNAL']['rapid_rise'] else '❌',
        'surge_imminent': '✅' if result['MOMENTUM_SIGNAL']['surge_imminent'] else '❌'
    }
    
    # KEY_FACTORS 색상
    hub_class = "factor-critical" if result['KEY_FACTORS']['HUBROOMTOTAL'] < 610 else ""
    fs_class = "factor-warning" if result['KEY_FACTORS']['FS_STORAGE_UTIL'] >= 7 else ""
    storage_class = "factor-warning" if result['KEY_FACTORS']['STORAGE_UTIL'] >= 205 else ""
    
    # 3개 모델 카드 생성
    predictions_cards = ""
    for horizon in [10, 15, 25]:
        pred = result['predictions'][horizon]
        prob = pred['EXCEED_300_PROBABILITY']
        
        # 색상 결정
        if horizon == 10:
            card_color = "#4299e1"
            badge_color = "#3182ce"
        elif horizon == 15:
            card_color = "#48bb78"
            badge_color = "#38a169"
        else:
            card_color = "#ed8936"
            badge_color = "#dd6b20"
        
        status_class = f"status-{pred['STATUS'].lower()}"
        risk_class = f"risk-{pred['SURGE_RISK'].lower()}"
        
        predictions_cards += f'''
        <div class="prediction-card" style="border-left: 5px solid {card_color};">
            <div class="prediction-header" style="background: {card_color};">
                <h3>{horizon}분 예측</h3>
                <div class="time-badge">{pred['PREDICTION_TIME']}</div>
            </div>
            <div class="prediction-body">
                <div class="big-pred-value" style="color: {card_color};">{pred['PREDICTED_VALUE']}</div>
                <div class="pred-increase">{pred['INCREASE_RATE']}</div>
                <div class="prob-meter">
                    <div class="prob-bar" style="width: {prob}%; background: {card_color};"></div>
                    <div class="prob-text">{prob}% 확률로 300 초과</div>
                </div>
                <div class="pred-badges">
                    <span class="status-badge {status_class}">{pred['STATUS']}</span>
                    <span class="risk-badge {risk_class}">{pred['SURGE_RISK']}</span>
                </div>
            </div>
        </div>
        '''
    
    # 최대 위험도 모델 표시
    max_risk_horizon = max([10, 15, 25], key=lambda h: result['predictions'][h]['EXCEED_300_PROBABILITY'])
    max_risk_prob = result['predictions'][max_risk_horizon]['EXCEED_300_PROBABILITY']
    
    alert_html = ""
    if max_risk_prob >= 60:
        alert_html = f'''
        <div class="alert-box">
            <div class="alert-title">
                <span class="alert-icon">🚨</span> 긴급 알림: 급증 위험 감지
            </div>
            <div class="alert-message">
                <strong>{max_risk_horizon}분 후</strong> 예측에서 가장 높은 위험도가 감지되었습니다. 
                300 초과 확률이 <strong>{max_risk_prob}%</strong>로 즉시 대응이 필요합니다.
            </div>
        </div>
        '''
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V6 실시간 예측 대시보드 - 3개 모델 비교</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header h1 {{ font-size: 32px; color: #2d3748; margin-bottom: 10px; }}
        .header .subtitle {{ color: #718096; font-size: 16px; }}
        .model-info {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }}
        .timestamp {{ float: right; color: #a0aec0; font-size: 14px; }}
        
        .alert-box {{
            background: linear-gradient(135deg, #f56565 0%, #ed8936 100%);
            border-radius: 15px;
            padding: 25px;
            color: white;
            box-shadow: 0 10px 30px rgba(245, 101, 101, 0.3);
            margin-bottom: 20px;
        }}
        .alert-title {{ font-size: 24px; font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }}
        .alert-message {{ font-size: 16px; line-height: 1.6; }}
        .alert-icon {{ animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        
        .current-section {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        .current-value {{ font-size: 72px; font-weight: 700; color: #4299e1; margin: 20px 0; }}
        .current-label {{ font-size: 18px; color: #718096; margin-bottom: 10px; }}
        
        .predictions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .prediction-card {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .prediction-header {{
            padding: 20px;
            color: white;
            text-align: center;
        }}
        .prediction-header h3 {{ font-size: 24px; margin-bottom: 10px; }}
        .time-badge {{
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 13px;
            line-height: 1.4;
        }}
        .prediction-body {{ padding: 25px; text-align: center; }}
        .big-pred-value {{ font-size: 56px; font-weight: 700; margin-bottom: 10px; }}
        .pred-increase {{ font-size: 18px; color: #718096; margin-bottom: 20px; }}
        .prob-meter {{
            background: #e2e8f0;
            border-radius: 10px;
            height: 40px;
            position: relative;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .prob-bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        .prob-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: 700;
            color: #2d3748;
            font-size: 16px;
        }}
        .pred-badges {{ display: flex; gap: 10px; justify-content: center; }}
        .status-badge, .risk-badge {{
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 14px;
        }}
        .status-normal {{ background: #c6f6d5; color: #2f855a; }}
        .status-warning {{ background: #feebc8; color: #c05621; }}
        .status-dangerous {{ background: #fed7d7; color: #c53030; }}
        .risk-low {{ background: #c6f6d5; color: #2f855a; }}
        .risk-medium {{ background: #feebc8; color: #c05621; }}
        .risk-high {{ background: #ffd4b3; color: #dd6b20; }}
        .risk-critical {{ background: #fed7d7; color: #c53030; }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .card-title {{
            font-size: 14px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .cause-analysis {{
            background: #f7fafc;
            border-radius: 10px;
            padding: 15px;
        }}
        .cause-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .cause-icon {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
            background: #fed7d7;
        }}
        .cause-text {{ font-size: 15px; color: #2d3748; font-weight: 500; }}
        
        .momentum-grid, .factors-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        .momentum-item, .factor-item {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .momentum-label, .factor-label {{
            font-size: 12px;
            color: #718096;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .momentum-value, .factor-value {{
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
        }}
        .momentum-active {{ color: #f56565; }}
        .factor-critical {{ color: #f56565; }}
        .factor-warning {{ color: #ed8936; }}
        
        .footer {{ text-align: center; color: white; margin-top: 30px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="timestamp">데이터 시간: {result['CURRENT_TIME']}</div>
            <h1>🚀 V6 실시간 예측 대시보드 - 3개 모델 비교</h1>
            <p class="subtitle">10분/15분/25분 예측 동시 모니터링</p>
            <span class="model-info">{result['MODEL_VERSION']}</span>
        </div>

        {alert_html}

        <div class="current-section">
            <div class="current-label">현재값 (30분 시퀀스 마지막)</div>
            <div class="current-value">{result['CURRENT_VALUE']}</div>
            <div style="color: #718096; font-size: 14px;">시퀀스 범위: {result['SEQUENCE_START_TIME']} ~ {result['CURRENT_TIME']}</div>
        </div>

        <div class="predictions-grid">
            {predictions_cards}
        </div>

        <div class="info-grid">
            <div class="card">
                <div class="card-title">🔍 상승 원인 분석</div>
                <div class="cause-analysis">
                    {cause_items_html}
                </div>
            </div>

            <div class="card">
                <div class="card-title">🎯 Momentum 신호</div>
                <div class="momentum-grid">
                    <div class="momentum-item">
                        <div class="momentum-label">가속도</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['acceleration'] > 0.5 else ''}">{result['MOMENTUM_SIGNAL']['acceleration']}</div>
                    </div>
                    <div class="momentum-item">
                        <div class="momentum-label">상승 중</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['is_rising'] else ''}">{momentum_checks['is_rising']}</div>
                    </div>
                    <div class="momentum-item">
                        <div class="momentum-label">급등 여부</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['rapid_rise'] else ''}">{momentum_checks['rapid_rise']}</div>
                    </div>
                    <div class="momentum-item">
                        <div class="momentum-label">급증 임박</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['surge_imminent'] else ''}">{momentum_checks['surge_imminent']}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">📊 주요 지표</div>
                <div class="factors-grid">
                    <div class="factor-item">
                        <div class="factor-label">HUBROOMTOTAL</div>
                        <div class="factor-value {hub_class}">{result['KEY_FACTORS']['HUBROOMTOTAL']}</div>
                    </div>
                    <div class="factor-item">
                        <div class="factor-label">FS STORAGE UTIL</div>
                        <div class="factor-value {fs_class}">{result['KEY_FACTORS']['FS_STORAGE_UTIL']}</div>
                    </div>
                    <div class="factor-item">
                        <div class="factor-label">TOTAL CMD</div>
                        <div class="factor-value">{result['KEY_FACTORS']['TOTAL_CMD']}</div>
                    </div>
                    <div class="factor-item">
                        <div class="factor-label">STORAGE UTIL</div>
                        <div class="factor-value {storage_class}">{result['KEY_FACTORS']['STORAGE_UTIL']}</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <div class="card-title">📈 30분 시퀀스 + 예측값 추이 분석</div>
            <div style="margin-top: 15px;">
                <div style="background: #f7fafc; border-radius: 10px; padding: 20px;">
                    {graph_html}
                </div>
                <div style="margin-top: 15px; padding: 15px; background: #e6f7ff; border-radius: 8px; border-left: 4px solid #1890ff;">
                    <strong>🔍 증가 원인 분석:</strong>
                    <div style="margin-top: 10px; color: #0050b3; line-height: 1.8;">
                        {analysis_text}
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <strong>📍 그래프 사용법:</strong>
                    <ul style="margin: 10px 0 0 20px; color: #856404;">
                        <li><b>호버:</b> 마우스를 올리면 해당 시점의 상세 데이터 표시</li>
                        <li><b>클릭 & 드래그:</b> 특정 영역 확대</li>
                        <li><b>더블클릭:</b> 원래 크기로 복원</li>
                        <li><b>범례 클릭:</b> 특정 라인 표시/숨김 전환</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>🤖 V6 Multi-Horizon Model | 10분/15분/25분 동시 예측</p>
            <p>Powered by XGBoost + Momentum Features</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

def predict_realtime():
    """실시간 예측 실행 및 HTML 대시보드 생성"""
    result, error = predict_all_horizons()
    
    if error:
        print([{"ERROR": error}])
    else:
        print([result])
        
        try:
            html_content = generate_html_dashboard(result)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_filename = f'V6_실시간예측_3모델비교_{timestamp}.html'
            html_path = os.path.join(script_dir, html_filename)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\n✅ HTML 대시보드 생성 완료: {html_filename}")
            print(f"📂 파일 위치: {html_path}")
            
            try:
                webbrowser.open('file://' + os.path.abspath(html_path))
                print(f"🌐 브라우저에서 대시보드를 열었습니다!")
            except Exception as browser_error:
                print(f"⚠️ 브라우저 자동 열기 실패: {str(browser_error)}")
                print(f"💡 수동으로 파일을 열어주세요: {html_path}")
            
        except Exception as e:
            print(f"\n⚠️ HTML 생성 실패: {str(e)}")

if __name__ == '__main__':
    predict_realtime()
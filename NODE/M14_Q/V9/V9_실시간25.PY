#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 실시간 예측 시스템 - V9.0 25분 (queue_gap 복원!)
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

def create_features_v9(row_dict):
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_last30'] = np.mean(seq_totalcnt[-30:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]
    
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]
    
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_current'] = seq_gap[-1]
    features['gap_last5'] = np.mean(seq_gap[-5:])
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_last30'] = np.mean(seq_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_total'] = seq_gap[-1] * seq_totalcnt[-1] / 1000
    features['gap_x_trans'] = seq_gap[-1] * seq_transport[-1] / 100
    features['ratio_gap_total'] = seq_gap[-1] / (seq_totalcnt[-1] + 1)
    
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['gap_over_200'] = np.sum(seq_gap > 200)
    features['gap_over_250'] = np.sum(seq_gap > 250)
    features['gap_over_300'] = np.sum(seq_gap > 300)
    features['gap_over_350'] = np.sum(seq_gap > 350)
    
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250 and seq_transport[-1] > 145) else 0
    
    return features

def adjust_prediction_v9(pred, current_total, m14b, m14bsum, gap, trans):
    # 25분 하한선 = -130
    floor = current_total - 130
    if pred < floor:
        pred = floor
    
    if 1650 <= pred < 1700:
        boost = 0
        if m14b > 540 and m14bsum > 620:
            boost += 50
        elif m14b > 520 and m14bsum > 600:
            boost += 40
        if gap > 350:
            boost += 40
        elif gap > 300:
            boost += 35
        elif gap > 250:
            boost += 25
        if trans > 180:
            boost += 30
        elif trans > 151:
            boost += 20
        pred = pred + boost
    
    # 25분 위험시 = -100
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - 100)
    
    return pred

def get_status_info(value):
    if value < 900:
        return 'LOW'
    elif value < 1600:
        return 'NORMAL'
    elif value < 1700:
        return 'CAUTION'
    else:
        return 'CRITICAL'

def predict_latest():
    model_file = 'model_v9_25min.pkl'
    if not os.path.exists(model_file):
        model_file = 'model_v9.pkl'
    
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        return {'prediction': 0, 'status': 'Model operation failure', 'prediction_time': '', 'danger_probability': 0, 'error_message': str(e)}
    
    csv_file = 'data/2025_DATA.CSV'
    if not os.path.exists(csv_file):
        return {'prediction': 0, 'status': 'No data', 'prediction_time': '', 'danger_probability': 0, 'error_message': f'No data: {csv_file}'}
    
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip', dtype={'CURRTIME': str})
    except Exception as e:
        return {'prediction': 0, 'status': 'No data', 'prediction_time': '', 'danger_probability': 0, 'error_message': str(e)}
    
    if len(df) == 0:
        return {'prediction': 0, 'status': 'No data', 'prediction_time': '', 'danger_probability': 0, 'error_message': 'CSV empty'}
    
    required_cols = ['M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT',
                     'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
                     'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {'prediction': 0, 'status': 'No data', 'prediction_time': '', 'danger_probability': 0, 'error_message': f'Missing: {missing_cols}'}
    
    if len(df) < 280:
        return {'prediction': 0, 'status': 'Lack of data', 'prediction_time': '', 'danger_probability': 0, 'error_message': f'Need 280, got {len(df)}'}
    
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = df['CURRTIME'].astype(str).str.strip()
            df = df[df['CURRTIME'].str.len() == 12].copy()
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
            df = df.dropna(subset=['CURRTIME']).copy()
        except:
            df['CURRTIME'] = [datetime.now() - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    else:
        df['CURRTIME'] = [datetime.now() - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    
    if len(df) < 280:
        return {'prediction': 0, 'status': 'Lack of data', 'prediction_time': '', 'danger_probability': 0, 'error_message': f'Need 280'}
    
    try:
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[-280:].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[-280:].values,
            'M10AM14A': df['M10AM14A'].iloc[-280:].values,
            'TOTALCNT': df['TOTALCNT'].iloc[-280:].values,
            'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-280:].values,
            'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[-280:].values,
            'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[-280:].values,
            'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[-280:].values,
        }
        
        current_time = df['CURRTIME'].iloc[-1]
        if pd.isna(current_time):
            current_time = datetime.now()
        prediction_time = current_time + timedelta(minutes=25)
        
        seq_totalcnt = row_dict['TOTALCNT']
        seq_m14b = row_dict['M14AM14B']
        seq_m14bsum = row_dict['M14AM14BSUM']
        seq_gap = row_dict['Q_CREATED'] - row_dict['Q_COMPLETED']
        seq_trans = row_dict['TRANSPORT']
        
        current_totalcnt = seq_totalcnt[-1]
        current_m14b = seq_m14b[-1]
        current_m14bsum = seq_m14bsum[-1]
        current_gap = seq_gap[-1]
        current_trans = seq_trans[-1]
        
        features = create_features_v9(row_dict)
        X_pred = pd.DataFrame([features])
        pred_raw = model.predict(X_pred)[0]
        pred = adjust_prediction_v9(pred_raw, current_totalcnt, current_m14b, current_m14bsum, current_gap, current_trans)
        pred_status = get_status_info(pred)
        
        gold_strict = (current_m14b > 540 and current_m14bsum > 620)
        gold_normal = (current_m14b > 520 and current_m14bsum > 600)
        danger_gap = current_gap > 300
        triple_check = (current_m14b > 520 and current_m14bsum > 600 and current_gap > 250)
        
        danger_prob = 5
        if pred >= 1750: danger_prob = 100
        elif pred >= 1700: danger_prob = 95
        elif pred >= 1680: danger_prob = 75
        elif pred >= 1650: danger_prob = 50
        elif pred >= 1620: danger_prob = 30
        elif pred >= 1600: danger_prob = 15
        
        if gold_strict: danger_prob = min(100, danger_prob + 20)
        elif gold_normal: danger_prob = min(100, danger_prob + 15)
        if danger_gap: danger_prob = min(100, danger_prob + 15)
        if triple_check: danger_prob = min(100, danger_prob + 10)
        if current_totalcnt >= 1700: danger_prob = max(danger_prob, 85)
        elif current_totalcnt >= 1650: danger_prob = max(danger_prob, 60)
        
        danger_prob = max(0, min(100, danger_prob))
        
        return {
            'prediction': int(pred),
            'status': pred_status,
            'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M'),
            'danger_probability': danger_prob
        }
        
    except Exception as e:
        return {'prediction': 0, 'status': 'Model operation failure', 'prediction_time': '', 'danger_probability': 0, 'error_message': str(e)}

if __name__ == '__main__':
    result = predict_latest()
    if result:
        print(f"prediction: {result['prediction']:,}")
        print(f"status: {result['status']}")
        print(f"prediction_time: {result['prediction_time']}")
        print(f"1700+ danger_probability: {result['danger_probability']}%")
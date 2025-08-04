import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

def generate_integrated_mcs_data(num_records=10000, days=30):
    """
    통합 MCS 데이터 생성 - 딥러닝 예측용
    시계열 패턴이 강화된 데이터 생성
    """
    
    # 기준 시작 시간
    base_time = datetime(2025, 7, 1, 0, 0, 0)
    
    # FAB 라인 및 베이 정보
    fab_lines = ['FAB1', 'FAB2', 'FAB3']
    bays = {
        'PHOTO': '포토공정',
        'ETCH': '식각공정',
        'DIFF': '확산공정',
        'IMPLANT': '이온주입',
        'CVD': 'CVD공정',
        'PVD': 'PVD공정',
        'CMP': 'CMP공정',
        'CLEAN': '세정공정',
        'METROLOGY': '계측공정',
        'TEST': '테스트공정'
    }
    
    # 장비 정보
    equipment_info = {
        'PHOTO': ['ASML_NXT_1980', 'ASML_NXT_2050', 'ASML_XT_1460'],
        'ETCH': ['LAM_KIYO_01', 'LAM_VERSYS_01', 'TEL_ETCH_01'],
        'CVD': ['AMAT_CENTURA_01', 'AMAT_PRODUCER_01'],
        'PVD': ['AMAT_ENDURA_01', 'AMAT_ENDURA_02'],
        'CMP': ['AMAT_REFLEXION_01', 'EBARA_F_REX_01'],
        'CLEAN': ['TEL_CLEAN_01', 'DNS_CLEAN_01'],
        'DIFF': ['TEL_UNITY_01', 'HITACHI_DD_01'],
        'IMPLANT': ['AMAT_VARIAN_01', 'AXCELIS_PURION_01'],
        'METROLOGY': ['KLA_2935', 'KLA_SP5', 'ASML_YS350'],
        'TEST': ['TEL_PRECIO_01', 'ACCRETECH_UF3000']
    }
    
    # 캐리어 타입
    carrier_types = ['FOUP', 'FOSB', 'OPEN_CASSETTE']
    
    # 웨이퍼 사이즈
    wafer_sizes = ['200mm', '300mm']
    
    # 제품 코드
    product_codes = ['DRAM_16G', 'DRAM_32G', 'NAND_512G', 'NAND_1T', 
                     'LOGIC_7NM', 'LOGIC_5NM', 'ANALOG_28NM']
    
    # 우선순위
    priorities = ['NORMAL', 'HOT_LOT', 'SUPER_HOT', 'ENGINEERING', 'PILOT']
    
    # MCS 이벤트 타입
    event_types = [
        'LOAD_REQUEST',      # 적재 요청
        'UNLOAD_REQUEST',    # 하역 요청
        'TRANSFER_START',    # 이송 시작
        'TRANSFER_COMPLETE', # 이송 완료
        'PROCESS_START',     # 공정 시작
        'PROCESS_END',       # 공정 종료
        'STOCKER_IN',       # 스토커 입고
        'STOCKER_OUT',      # 스토커 출고
        'SENSOR_UPDATE',    # 센서 데이터 업데이트
        'ALARM_OCCURRED',   # 알람 발생
        'ALARM_CLEARED'     # 알람 해제
    ]
    
    # 운반 장비
    transport_vehicles = ['OHT', 'OHS', 'AGV', 'RGV', 'MANUAL']
    
    # 알람 코드
    alarm_codes = ['E84_TIMEOUT', 'CARRIER_ID_MISMATCH', 'VEHICLE_COLLISION_RISK', 
                   'LOADPORT_ERROR', 'STOCKER_FULL', 'PATH_BLOCKED', 'SENSOR_OUT_OF_RANGE',
                   'VIBRATION_HIGH', 'TEMPERATURE_HIGH', 'PRESSURE_LOW', 'PARTICLE_HIGH']
    
    # 센서 타입
    sensor_types = ['VIBRATION', 'TEMPERATURE', 'PRESSURE', 'PARTICLE', 'HUMIDITY', 'FLOW_RATE']
    
    records = []
    current_time = base_time
    
    # 장비별 센서 기준값 설정
    equipment_sensor_baseline = {}
    for bay, equipments in equipment_info.items():
        for eq in equipments:
            equipment_sensor_baseline[eq] = {
                'vibration': random.uniform(0.5, 1.5),  # mm/s
                'temperature': random.uniform(20, 25),   # °C
                'pressure': random.uniform(750, 760),    # Torr
                'particle': random.randint(10, 50),      # count
                'humidity': random.uniform(40, 50),      # %
                'flow_rate': random.uniform(90, 100)     # %
            }
    
    # 시계열 패턴 생성을 위한 파라미터
    daily_pattern = lambda h: 1.0 + 0.3 * np.sin(2 * np.pi * h / 24 - np.pi/2)  # 일일 패턴
    weekly_pattern = lambda d: 1.0 if d < 5 else 0.7  # 주말 효과
    
    # 병목 현상 시뮬레이션을 위한 상태
    bottleneck_state = {bay: 0 for bay in bays.keys()}
    
    # 장비별 누적 사용 시간 (센서 이상 시뮬레이션용)
    equipment_usage_hours = {eq: 0 for eqs in equipment_info.values() for eq in eqs}
    
    # 시간별 레코드 생성
    for day in range(days):
        for hour in range(24):
            # 시간대별 이벤트 수 조정
            hour_factor = daily_pattern(hour)
            day_factor = weekly_pattern(day % 7)
            events_this_hour = int(num_records / (days * 24) * hour_factor * day_factor)
            
            for _ in range(events_this_hour):
                # 시간 증가
                current_time += timedelta(seconds=random.randint(10, 300))
                
                # 이벤트 타입 선택 (시간대별 패턴 반영)
                if 6 <= hour <= 22:  # 주간
                    event_weights = [10, 10, 20, 20, 15, 15, 5, 5, 30, 5, 5]
                else:  # 야간
                    event_weights = [5, 5, 15, 15, 10, 10, 10, 10, 40, 10, 10]
                
                event_type = random.choices(event_types, weights=event_weights)[0]
                
                # 기본 MCS 데이터
                base_data = {
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_type': event_type,
                    'lot_id': f"LOT{2025070000 + (day * 100 + hour):010d}",
                    'carrier_id': f"CAR{random.randint(10000, 99999)}",
                    'carrier_type': random.choice(carrier_types),
                    'product_code': random.choice(product_codes),
                    'wafer_size': random.choice(wafer_sizes),
                    'wafer_count': 25 if random.choice(wafer_sizes) == '300mm' else 50,
                    'priority': random.choice(priorities),
                    'fab_line': random.choice(fab_lines),
                    'status': 'NORMAL',
                    'alarm_code': None,
                }
                
                # 이벤트별 추가 데이터
                if event_type in ['LOAD_REQUEST', 'UNLOAD_REQUEST', 'PROCESS_START', 'PROCESS_END']:
                    # 병목 현상을 고려한 베이 선택
                    bay_weights = [1.0 / (1.0 + bottleneck_state[bay]) for bay in bays.keys()]
                    bay = random.choices(list(bays.keys()), weights=bay_weights)[0]
                    equipment = random.choice(equipment_info[bay])
                    
                    # 장비 사용 시간 누적
                    equipment_usage_hours[equipment] += 0.5
                    
                    base_data.update({
                        'location': bay,
                        'equipment_id': equipment,
                        'recipe': f"{bay}_RECIPE_{random.randint(1, 5)}",
                        'step_number': f"{random.randint(1, 10) * 100}",
                    })
                    
                    # 병목 상태 업데이트
                    if event_type == 'PROCESS_START':
                        bottleneck_state[bay] += 1
                    elif event_type == 'PROCESS_END':
                        bottleneck_state[bay] = max(0, bottleneck_state[bay] - 1)
                    
                    # 공정 시작/종료 시 센서 데이터 포함
                    if event_type in ['PROCESS_START', 'PROCESS_END']:
                        baseline = equipment_sensor_baseline[equipment]
                        usage_factor = 1.0 + equipment_usage_hours[equipment] / 1000  # 사용 시간에 따른 열화
                        
                        base_data.update({
                            'sensor_vibration_mm_s': round(baseline['vibration'] * usage_factor + random.gauss(0, 0.1), 3),
                            'sensor_temperature_c': round(baseline['temperature'] + usage_factor * 2 + random.gauss(0, 1), 1),
                            'sensor_pressure_torr': round(baseline['pressure'] + random.gauss(0, 2), 1),
                            'sensor_particle_count': int(baseline['particle'] * usage_factor + random.randint(-5, 10)),
                            'sensor_humidity_pct': round(baseline['humidity'] + random.gauss(0, 2), 1),
                            'sensor_flow_rate_pct': round(baseline['flow_rate'] - usage_factor + random.gauss(0, 3), 1),
                        })
                        
                elif event_type in ['TRANSFER_START', 'TRANSFER_COMPLETE']:
                    # 이송 관련 이벤트
                    from_bay = random.choice(list(bays.keys()) + ['STOCKER_01', 'STOCKER_02'])
                    to_bay = random.choice(list(bays.keys()) + ['STOCKER_01', 'STOCKER_02'])
                    
                    # 병목 구간으로의 이송은 시간이 더 걸림
                    transfer_time = 120  # 기본 이송 시간
                    if to_bay in bays and bottleneck_state.get(to_bay, 0) > 5:
                        transfer_time += bottleneck_state[to_bay] * 10
                    
                    base_data.update({
                        'from_location': from_bay,
                        'to_location': to_bay,
                        'vehicle_id': f"{random.choice(transport_vehicles)}_{random.randint(100, 999)}",
                        'transfer_time_sec': transfer_time + random.randint(-20, 50) if event_type == 'TRANSFER_COMPLETE' else None,
                    })
                    
                    # OHT의 경우 진동 데이터 추가
                    if 'OHT' in base_data.get('vehicle_id', ''):
                        congestion_factor = sum(bottleneck_state.values()) / len(bottleneck_state)
                        base_data['oht_vibration_mm_s'] = round(0.3 + congestion_factor * 0.1 + random.uniform(0, 0.2), 3)
                        
                elif event_type == 'SENSOR_UPDATE':
                    # 센서 데이터 업데이트 이벤트
                    bay = random.choice(list(bays.keys()))
                    equipment = random.choice(equipment_info[bay])
                    baseline = equipment_sensor_baseline[equipment]
                    usage_factor = 1.0 + equipment_usage_hours[equipment] / 1000
                    
                    # 시간대별 이상 발생 확률
                    anomaly_prob = 0.05 if 6 <= hour <= 22 else 0.15  # 야간에 이상 발생 확률 높음
                    anomaly = random.random() < anomaly_prob
                    anomaly_factor = random.uniform(1.5, 3.0) if anomaly else 1.0
                    
                    base_data.update({
                        'location': bay,
                        'equipment_id': equipment,
                        'sensor_type': random.choice(sensor_types),
                        'sensor_vibration_mm_s': round(baseline['vibration'] * usage_factor * anomaly_factor + random.gauss(0, 0.1), 3),
                        'sensor_temperature_c': round(baseline['temperature'] + usage_factor * 2 + 
                                                    (random.uniform(5, 15) if anomaly else random.gauss(0, 1)), 1),
                        'sensor_pressure_torr': round(baseline['pressure'] + random.gauss(0, 2), 1),
                        'sensor_particle_count': int(baseline['particle'] * usage_factor + 
                                                   (random.randint(50, 200) if anomaly else random.randint(-5, 10))),
                        'sensor_humidity_pct': round(baseline['humidity'] + random.gauss(0, 2), 1),
                        'sensor_flow_rate_pct': round(baseline['flow_rate'] - usage_factor + random.gauss(0, 3), 1),
                        'sensor_status': 'ANOMALY' if anomaly else 'NORMAL'
                    })
                    
                    # 이상치 발생 시 상태 업데이트
                    if anomaly:
                        base_data['status'] = 'WARNING'
                        
                elif event_type == 'ALARM_OCCURRED':
                    # 알람 발생
                    bay = random.choice(list(bays.keys()))
                    equipment = random.choice(equipment_info[bay])
                    
                    base_data.update({
                        'location': bay,
                        'equipment_id': equipment,
                        'alarm_code': random.choice(alarm_codes),
                        'status': 'ALARM',
                        'alarm_severity': random.choice(['WARNING', 'CRITICAL', 'INFO']),
                    })
                    
                    # 알람 발생 시 센서 데이터도 포함
                    baseline = equipment_sensor_baseline[equipment]
                    usage_factor = 1.0 + equipment_usage_hours[equipment] / 1000
                    
                    if 'VIBRATION' in base_data['alarm_code']:
                        base_data['sensor_vibration_mm_s'] = round(baseline['vibration'] * usage_factor * random.uniform(2, 4), 3)
                    elif 'TEMPERATURE' in base_data['alarm_code']:
                        base_data['sensor_temperature_c'] = round(baseline['temperature'] + usage_factor * 2 + random.uniform(10, 20), 1)
                    elif 'PARTICLE' in base_data['alarm_code']:
                        base_data['sensor_particle_count'] = int(baseline['particle'] * usage_factor + random.randint(100, 500))
                        
                elif event_type in ['STOCKER_IN', 'STOCKER_OUT']:
                    # 스토커 입출고
                    base_data.update({
                        'location': f"STOCKER_{random.randint(1, 5):02d}",
                        'stocker_capacity_pct': random.randint(40, 95),
                    })
                    
                records.append(base_data)
    
    # DataFrame 생성
    df = pd.DataFrame(records)
    
    # 시간순 정렬
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 추가 특성 엔지니어링 (딥러닝 모델용)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['shift'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], labels=['Night', 'Morning', 'Evening'])
    
    return df


def create_time_series_features(df):
    """
    딥러닝 모델을 위한 시계열 특성 생성
    """
    # 이송 시간 통계 (LSTM용)
    transfer_times = df[df['transfer_time_sec'].notna()].groupby(
        [pd.Grouper(key='timestamp', freq='H'), 'from_location', 'to_location']
    )['transfer_time_sec'].agg(['mean', 'std', 'count']).reset_index()
    
    # 병목 지표 (RNN용)
    bottleneck_metrics = df[df['event_type'] == 'PROCESS_START'].groupby(
        [pd.Grouper(key='timestamp', freq='H'), 'location']
    ).size().reset_index(name='queue_size')
    
    # 처리량 데이터 (ARIMA용)
    throughput = df[df['event_type'] == 'PROCESS_END'].groupby(
        pd.Grouper(key='timestamp', freq='H')
    ).size().reset_index(name='hourly_throughput')
    
    # 센서 이상 패턴 (이상탐지용)
    sensor_anomalies = df[df['sensor_status'] == 'ANOMALY'].groupby(
        [pd.Grouper(key='timestamp', freq='H'), 'equipment_id']
    ).size().reset_index(name='anomaly_count')
    
    return {
        'transfer_times': transfer_times,
        'bottleneck_metrics': bottleneck_metrics,
        'throughput': throughput,
        'sensor_anomalies': sensor_anomalies
    }


def analyze_mcs_data(df):
    """
    MCS 데이터 분석 및 요약 - 딥러닝 관점
    """
    print("\n📊 MCS 데이터 분석 결과 (딥러닝 예측용):")
    print(f"- 총 레코드 수: {len(df):,}")
    print(f"- 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"- 고유 LOT 수: {df['lot_id'].nunique()}")
    print(f"- 고유 캐리어 수: {df['carrier_id'].nunique()}")
    
    print("\n📈 이벤트 타입별 분포:")
    event_dist = df['event_type'].value_counts()
    for event, count in event_dist.items():
        print(f"  - {event}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # 시계열 패턴 분석
    print("\n⏰ 시계열 패턴 분석:")
    hourly_events = df.groupby(df['timestamp'].dt.hour).size()
    peak_hour = hourly_events.idxmax()
    print(f"  - 피크 시간대: {peak_hour}시 ({hourly_events[peak_hour]:,}건)")
    print(f"  - 평균 시간당 이벤트: {hourly_events.mean():.1f}건")
    
    # 이송 시간 분석 (LSTM용)
    transfer_data = df[df['transfer_time_sec'].notna()]
    if len(transfer_data) > 0:
        print(f"\n🚚 이송 시간 분석 (LSTM 예측 대상):")
        print(f"  - 평균 이송 시간: {transfer_data['transfer_time_sec'].mean():.1f}초")
        print(f"  - 표준편차: {transfer_data['transfer_time_sec'].std():.1f}초")
        print(f"  - 최대 이송 시간: {transfer_data['transfer_time_sec'].max()}초")
        
    # 병목 분석 (RNN용)
    process_starts = df[df['event_type'] == 'PROCESS_START']
    if len(process_starts) > 0:
        bay_loads = process_starts['location'].value_counts()
        print(f"\n🔄 베이별 부하 분석 (RNN 병목 예측 대상):")
        for bay, count in bay_loads.head(5).items():
            print(f"  - {bay}: {count:,}건 (부하율: {count/len(process_starts)*100:.1f}%)")
    
    # 처리량 분석 (ARIMA용)
    process_ends = df[df['event_type'] == 'PROCESS_END']
    if len(process_ends) > 0:
        hourly_throughput = process_ends.groupby(df['timestamp'].dt.floor('H')).size()
        print(f"\n📊 처리량 분석 (ARIMA 예측 대상):")
        print(f"  - 평균 시간당 처리량: {hourly_throughput.mean():.1f} 웨이퍼")
        print(f"  - 최대 시간당 처리량: {hourly_throughput.max()} 웨이퍼")
        print(f"  - 처리량 변동계수: {hourly_throughput.std()/hourly_throughput.mean():.3f}")
    
    # 센서 이상 분석
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col.endswith(('_mm_s', '_c', '_torr', '_count', '_pct'))]
    if sensor_cols:
        print("\n🔍 센서 이상 탐지 분석:")
        
        # 각 센서별 이상치 비율
        for col in sensor_cols[:5]:  # 주요 5개만
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    # IQR 방식으로 이상치 탐지
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
                    print(f"  - {col}: {outliers}개 이상치 ({outliers/len(data)*100:.2f}%)")
    
    # 알람 패턴 분석
    alarm_data = df[df['event_type'] == 'ALARM_OCCURRED']
    if len(alarm_data) > 0:
        print(f"\n⚠️ 알람 패턴 분석:")
        print(f"  - 총 알람 수: {len(alarm_data):,}")
        print(f"  - 알람 발생률: {len(alarm_data)/len(df)*100:.2f}%")
        print(f"  - 주요 알람 타입:")
        for alarm, count in alarm_data['alarm_code'].value_counts().head(3).items():
            print(f"    - {alarm}: {count}건")
    
    return df


def save_integrated_mcs_data(df, time_series_features):
    """
    통합 MCS 데이터 저장 - 딥러닝 학습용
    """
    # 출력 디렉토리 생성
    output_dir = "integrated_mcs_data_dl"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 메인 데이터 저장
    csv_path = os.path.join(output_dir, 'mcs_events.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 메인 데이터 저장: {csv_path}")
    
    # 시계열 특성 데이터 저장
    for name, data in time_series_features.items():
        feature_path = os.path.join(output_dir, f'{name}.csv')
        data.to_csv(feature_path, index=False, encoding='utf-8-sig')
        print(f"✅ {name} 저장: {feature_path}")
    
    # JSON 형식으로도 저장
    df_json = df.copy()
    df_json['timestamp'] = df_json['timestamp'].astype(str)
    json_path = os.path.join(output_dir, 'mcs_events.json')
    df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"✅ JSON 파일 저장: {json_path}")
    
    # Excel 저장 (상세 분석 포함)
    excel_path = os.path.join(output_dir, 'mcs_analysis.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 전체 데이터 (샘플)
        df.head(1000).to_excel(writer, sheet_name='Sample_Data', index=False)
        
        # 이벤트 요약
        event_summary = df['event_type'].value_counts().reset_index()
        event_summary.columns = ['이벤트_타입', '건수']
        event_summary.to_excel(writer, sheet_name='Event_Summary', index=False)
        
        # 시간대별 패턴
        hourly_pattern = df.groupby([df['timestamp'].dt.hour, 'event_type']).size().unstack(fill_value=0)
        hourly_pattern.to_excel(writer, sheet_name='Hourly_Pattern')
        
        # 장비별 센서 통계
        sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col.endswith(('_mm_s', '_c', '_torr', '_count', '_pct'))]
        if sensor_cols:
            sensor_stats = df.groupby('equipment_id')[sensor_cols].agg(['mean', 'std', 'min', 'max'])
            sensor_stats.to_excel(writer, sheet_name='Sensor_Stats')
        
        # 알람 분석
        alarm_data = df[df['event_type'] == 'ALARM_OCCURRED']
        if len(alarm_data) > 0:
            alarm_summary = alarm_data.groupby(['equipment_id', 'alarm_code']).size().reset_index(name='count')
            alarm_summary.to_excel(writer, sheet_name='Alarm_Analysis', index=False)
    
    print(f"✅ Excel 분석 파일 저장: {excel_path}")
    
    return output_dir


def create_dl_training_guides():
    """
    딥러닝 모델 학습 가이드 생성
    """
    guides = {
        'lstm_guide.md': """# LSTM 이송시간 예측 모델 가이드

## 데이터 준비
- 입력 파일: `transfer_times.csv`
- 주요 특성: timestamp, from_location, to_location, mean, std, count

## 모델 구조
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
```

## 학습 파라미터
- Sequence Length: 24 (24시간 이력)
- Batch Size: 32
- Epochs: 100
- Learning Rate: 0.001

## 예측 대상
- 다음 1시간 이송 시간
- 신뢰 구간 포함
""",

        'rnn_guide.md': """# RNN 병목 예측 모델 가이드

## 데이터 준비
- 입력 파일: `bottleneck_metrics.csv`
- 주요 특성: timestamp, location, queue_size

## 모델 구조
```python
model = Sequential([
    SimpleRNN(128, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.3),
    SimpleRNN(64),
    Dense(n_locations, activation='softmax')
])
```

## 학습 파라미터
- Sequence Length: 48 (48시간 이력)
- Batch Size: 64
- Epochs: 150
- Learning Rate: 0.0005

## 예측 대상
- 각 베이별 병목 발생 확률
- 예상 대기 시간
""",

        'arima_guide.md': """# ARIMA 처리량 예측 모델 가이드

## 데이터 준비
- 입력 파일: `throughput.csv`
- 주요 특성: timestamp, hourly_throughput

## 모델 선택
```python
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# 자동 파라미터 탐색
model = auto_arima(data, 
                   start_p=1, start_q=1,
                   max_p=5, max_q=5, 
                   seasonal=True, m=24,
                   stepwise=True)
```

## 최적 파라미터 (예시)
- ARIMA(2,1,2)(1,1,1)[24]
- AIC: 최소값 기준

## 예측 대상
- 향후 24시간 처리량
- 95% 신뢰 구간
""",

        'anomaly_guide.md': """# 센서 이상탐지 모델 가이드

## 데이터 준비
- 입력 파일: `mcs_events.csv` (센서 데이터 필터링)
- 주요 특성: 모든 sensor_* 컬럼

## 모델 옵션

### 1. Isolation Forest
```python
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.1, random_state=42)
```

### 2. LSTM Autoencoder
```python
encoder = Sequential([
    LSTM(64, activation='relu', input_shape=(timesteps, n_features)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu')
])

decoder = Sequential([
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(n_features)
])
```

## 평가 지표
- Precision, Recall, F1-Score
- 조기 경보 성공률
"""
    }
    
    # 가이드 파일 저장
    output_dir = "integrated_mcs_data_dl"
    for filename, content in guides.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 가이드 생성: {filepath}")


def main():
    """
    메인 실행 함수
    """
    print("="*60)
    print("🏭 반도체 FAB 통합 MCS 데이터 생성 (딥러닝 예측용)")
    print("="*60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 데이터 생성
    print("\n1️⃣ 통합 MCS 데이터 생성 중...")
    df = generate_integrated_mcs_data(num_records=10000, days=30)
    print(f"   - 생성된 레코드: {len(df):,}개")
    
    # 2. 시계열 특성 생성
    print("\n2️⃣ 딥러닝용 시계열 특성 생성 중...")
    time_series_features = create_time_series_features(df)
    for name, data in time_series_features.items():
        print(f"   - {name}: {len(data):,}개 레코드")
    
    # 3. 데이터 분석
    print("\n3️⃣ 데이터 분석 중...")
    analyze_mcs_data(df)
    
    # 4. 파일 저장
    print("\n4️⃣ 데이터 파일 저장 중...")
    output_dir = save_integrated_mcs_data(df, time_series_features)
    
    # 5. 학습 가이드 생성
    print("\n5️⃣ 딥러닝 모델 학습 가이드 생성 중...")
    create_dl_training_guides()
    
    # 6. 결과 요약
    print("\n" + "="*60)
    print("✨ 딥러닝용 MCS 데이터 생성 완료!")
    print("="*60)
    print(f"\n📁 생성된 파일 위치: {os.path.abspath(output_dir)}")
    print("\n📄 생성된 파일 목록:")
    print("  - mcs_events.csv : 전체 MCS 이벤트 데이터")
    print("  - transfer_times.csv : LSTM용 이송시간 데이터")
    print("  - bottleneck_metrics.csv : RNN용 병목 지표")
    print("  - throughput.csv : ARIMA용 처리량 데이터")
    print("  - sensor_anomalies.csv : 센서 이상 패턴")
    print("  - mcs_analysis.xlsx : 종합 분석 리포트")
    print("  - *_guide.md : 각 모델별 학습 가이드")
    
    print("\n💡 데이터 특징:")
    print("  - 시계열 패턴이 강화된 MCS 이벤트 로그")
    print("  - 일일/주간 주기성 반영")
    print("  - 병목 현상 시뮬레이션")
    print("  - 장비 사용에 따른 센서 값 변화")
    print("  - 딥러닝 모델별 최적화된 특성 데이터")
    
    # 샘플 데이터 미리보기
    print("\n📊 샘플 데이터 (처음 5개):")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    sample_cols = ['timestamp', 'event_type', 'equipment_id', 'sensor_vibration_mm_s', 'transfer_time_sec']
    display_cols = [col for col in sample_cols if col in df.columns]
    print(df[display_cols].head())
    
    return df


if __name__ == "__main__":
    # pandas 옵션 설정
    pd.set_option('display.max_colwidth', 50)
    
    # 메인 함수 실행
    mcs_df = main()
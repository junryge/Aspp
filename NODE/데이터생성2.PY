import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

def generate_integrated_mcs_data(num_records=100):
    """
    통합 MCS 데이터 생성 - 이벤트 로그 + 장비 센서 데이터
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
    
    # 레코드 생성
    lot_id = f"LOT{2025070000}"
    
    for i in range(num_records):
        # 시간 증가 (30초 ~ 5분 간격)
        current_time += timedelta(seconds=random.randint(30, 300))
        
        # 이벤트 타입 선택 (센서 업데이트 이벤트 비중 높임)
        if random.random() < 0.3:  # 30% 확률로 센서 데이터
            event_type = 'SENSOR_UPDATE'
        else:
            event_type = random.choice([et for et in event_types if et != 'SENSOR_UPDATE'])
        
        # 기본 MCS 데이터
        base_data = {
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'event_type': event_type,
            'lot_id': f"LOT{2025070000 + i // 10:010d}",
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
            # 장비 관련 이벤트
            bay = random.choice(list(bays.keys()))
            equipment = random.choice(equipment_info[bay])
            
            base_data.update({
                'location': bay,
                'equipment_id': equipment,
                'recipe': f"{bay}_RECIPE_{random.randint(1, 5)}",
                'step_number': f"{random.randint(1, 10) * 100}",
            })
            
            # 공정 시작/종료 시 센서 데이터 포함
            if event_type in ['PROCESS_START', 'PROCESS_END']:
                baseline = equipment_sensor_baseline[equipment]
                base_data.update({
                    'sensor_vibration_mm_s': round(baseline['vibration'] + random.gauss(0, 0.1), 3),
                    'sensor_temperature_c': round(baseline['temperature'] + random.gauss(0, 1), 1),
                    'sensor_pressure_torr': round(baseline['pressure'] + random.gauss(0, 2), 1),
                    'sensor_particle_count': baseline['particle'] + random.randint(-5, 10),
                    'sensor_humidity_pct': round(baseline['humidity'] + random.gauss(0, 2), 1),
                    'sensor_flow_rate_pct': round(baseline['flow_rate'] + random.gauss(0, 3), 1),
                })
                
        elif event_type in ['TRANSFER_START', 'TRANSFER_COMPLETE']:
            # 이송 관련 이벤트
            base_data.update({
                'from_location': random.choice(list(bays.keys()) + ['STOCKER_01', 'STOCKER_02']),
                'to_location': random.choice(list(bays.keys()) + ['STOCKER_01', 'STOCKER_02']),
                'vehicle_id': f"{random.choice(transport_vehicles)}_{random.randint(100, 999)}",
                'transfer_time_sec': random.randint(30, 300) if event_type == 'TRANSFER_COMPLETE' else None,
            })
            
            # OHT의 경우 진동 데이터 추가
            if 'OHT' in base_data.get('vehicle_id', ''):
                base_data['oht_vibration_mm_s'] = round(random.uniform(0.1, 0.5) + 
                                                       (random.uniform(0.5, 1.5) if random.random() < 0.1 else 0), 3)
                
        elif event_type == 'SENSOR_UPDATE':
            # 센서 데이터 업데이트 이벤트
            bay = random.choice(list(bays.keys()))
            equipment = random.choice(equipment_info[bay])
            baseline = equipment_sensor_baseline[equipment]
            
            # 가끔 이상치 발생 (10% 확률)
            anomaly = random.random() < 0.1
            anomaly_factor = random.uniform(1.5, 3.0) if anomaly else 1.0
            
            base_data.update({
                'location': bay,
                'equipment_id': equipment,
                'sensor_type': random.choice(sensor_types),
                'sensor_vibration_mm_s': round(baseline['vibration'] * anomaly_factor + random.gauss(0, 0.1), 3),
                'sensor_temperature_c': round(baseline['temperature'] + 
                                            (random.uniform(5, 15) if anomaly else random.gauss(0, 1)), 1),
                'sensor_pressure_torr': round(baseline['pressure'] + random.gauss(0, 2), 1),
                'sensor_particle_count': baseline['particle'] + 
                                       (random.randint(50, 200) if anomaly else random.randint(-5, 10)),
                'sensor_humidity_pct': round(baseline['humidity'] + random.gauss(0, 2), 1),
                'sensor_flow_rate_pct': round(baseline['flow_rate'] + random.gauss(0, 3), 1),
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
            if 'VIBRATION' in base_data['alarm_code']:
                base_data['sensor_vibration_mm_s'] = round(baseline['vibration'] * random.uniform(2, 4), 3)
            elif 'TEMPERATURE' in base_data['alarm_code']:
                base_data['sensor_temperature_c'] = round(baseline['temperature'] + random.uniform(10, 20), 1)
            elif 'PARTICLE' in base_data['alarm_code']:
                base_data['sensor_particle_count'] = baseline['particle'] + random.randint(100, 500)
                
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
    
    return df


def analyze_mcs_data(df):
    """
    MCS 데이터 분석 및 요약
    """
    print("\n📊 MCS 데이터 분석 결과:")
    print(f"- 총 레코드 수: {len(df)}")
    print(f"- 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"- 고유 LOT 수: {df['lot_id'].nunique()}")
    print(f"- 고유 캐리어 수: {df['carrier_id'].nunique()}")
    
    print("\n📈 이벤트 타입별 분포:")
    event_dist = df['event_type'].value_counts()
    for event, count in event_dist.items():
        print(f"  - {event}: {count} ({count/len(df)*100:.1f}%)")
    
    # 센서 데이터 분석
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    if sensor_cols:
        print("\n🔍 센서 데이터 분석:")
        
        # 진동 데이터
        if 'sensor_vibration_mm_s' in df.columns:
            vibration_data = df['sensor_vibration_mm_s'].dropna()
            if len(vibration_data) > 0:
                print(f"  진동 센서:")
                print(f"    - 평균: {vibration_data.mean():.3f} mm/s")
                print(f"    - 최대: {vibration_data.max():.3f} mm/s")
                print(f"    - 이상치 (>2.0): {(vibration_data > 2.0).sum()}건")
        
        # 온도 데이터
        if 'sensor_temperature_c' in df.columns:
            temp_data = df['sensor_temperature_c'].dropna()
            if len(temp_data) > 0:
                print(f"  온도 센서:")
                print(f"    - 평균: {temp_data.mean():.1f} °C")
                print(f"    - 최대: {temp_data.max():.1f} °C")
                print(f"    - 이상치 (>30°C): {(temp_data > 30).sum()}건")
        
        # 파티클 데이터
        if 'sensor_particle_count' in df.columns:
            particle_data = df['sensor_particle_count'].dropna()
            if len(particle_data) > 0:
                print(f"  파티클 센서:")
                print(f"    - 평균: {particle_data.mean():.0f} count")
                print(f"    - 최대: {particle_data.max()} count")
                print(f"    - 이상치 (>100): {(particle_data > 100).sum()}건")
    
    # 알람 분석
    alarm_data = df[df['event_type'] == 'ALARM_OCCURRED']
    if len(alarm_data) > 0:
        print(f"\n⚠️ 알람 분석:")
        print(f"  - 총 알람 수: {len(alarm_data)}")
        print(f"  - 알람 타입별:")
        for alarm, count in alarm_data['alarm_code'].value_counts().items():
            print(f"    - {alarm}: {count}건")
    
    # 장비별 이벤트
    if 'equipment_id' in df.columns:
        equipment_events = df[df['equipment_id'].notna()]['equipment_id'].value_counts().head(5)
        print(f"\n🏭 상위 5개 장비 이벤트:")
        for eq, count in equipment_events.items():
            print(f"  - {eq}: {count}건")
    
    return df


def save_integrated_mcs_data(df):
    """
    통합 MCS 데이터 저장
    """
    # 출력 디렉토리 생성
    output_dir = "integrated_mcs_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # CSV 저장
    csv_path = os.path.join(output_dir, 'integrated_mcs_data.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV 파일 저장: {csv_path}")
    
    # JSON 저장
    df_json = df.copy()
    df_json['timestamp'] = df_json['timestamp'].astype(str)
    json_path = os.path.join(output_dir, 'integrated_mcs_data.json')
    df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"✅ JSON 파일 저장: {json_path}")
    
    # Excel 저장 (상세 분석 포함)
    excel_path = os.path.join(output_dir, 'integrated_mcs_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 전체 데이터
        df.to_excel(writer, sheet_name='MCS_Data', index=False)
        
        # 이벤트 요약
        event_summary = df['event_type'].value_counts().reset_index()
        event_summary.columns = ['이벤트_타입', '건수']
        event_summary.to_excel(writer, sheet_name='Event_Summary', index=False)
        
        # 센서 데이터만 추출
        sensor_cols = ['timestamp', 'equipment_id', 'location'] + [col for col in df.columns if 'sensor_' in col]
        sensor_data = df[df['event_type'].isin(['SENSOR_UPDATE', 'PROCESS_START', 'PROCESS_END'])][
            [col for col in sensor_cols if col in df.columns]
        ].dropna(how='all')
        
        if len(sensor_data) > 0:
            sensor_data.to_excel(writer, sheet_name='Sensor_Data', index=False)
        
        # 알람 데이터
        alarm_data = df[df['event_type'] == 'ALARM_OCCURRED']
        if len(alarm_data) > 0:
            alarm_data.to_excel(writer, sheet_name='Alarm_Data', index=False)
        
        # 통계 요약
        stats_data = {
            '항목': ['총 레코드 수', '고유 LOT 수', '고유 캐리어 수', '센서 업데이트 수', 
                    '알람 발생 수', '평균 이송 시간(초)'],
            '값': [
                len(df),
                df['lot_id'].nunique(),
                df['carrier_id'].nunique(),
                len(df[df['event_type'] == 'SENSOR_UPDATE']),
                len(df[df['event_type'] == 'ALARM_OCCURRED']),
                df[df['transfer_time_sec'].notna()]['transfer_time_sec'].mean() if 'transfer_time_sec' in df.columns else 'N/A'
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    print(f"✅ Excel 파일 저장: {excel_path}")
    
    return output_dir


def create_sample_queries():
    """
    데이터 활용을 위한 샘플 쿼리 생성
    """
    queries = """
📌 MCS 데이터 활용 샘플 쿼리:

1. 특정 장비의 센서 데이터 조회:
   df[(df['equipment_id'] == 'ASML_NXT_1980') & (df['sensor_vibration_mm_s'].notna())]

2. 진동 이상치 탐지:
   df[df['sensor_vibration_mm_s'] > 2.0]

3. 시간대별 알람 발생 패턴:
   df[df['event_type'] == 'ALARM_OCCURRED'].groupby(df['timestamp'].dt.hour)['alarm_code'].count()

4. LOT별 처리 시간 분석:
   df.groupby('lot_id')['transfer_time_sec'].agg(['mean', 'sum', 'count'])

5. 특정 베이의 장비 가동 현황:
   df[(df['location'] == 'PHOTO') & (df['event_type'].isin(['PROCESS_START', 'PROCESS_END']))]

6. 센서 데이터 시계열 분석:
   sensor_data = df[df['sensor_temperature_c'].notna()].set_index('timestamp')
   sensor_data['sensor_temperature_c'].resample('1H').mean()

7. 파티클 카운트 이상 패턴:
   df[(df['sensor_particle_count'] > 100) & (df['event_type'] == 'SENSOR_UPDATE')]

8. 장비별 알람 빈도:
   df[df['event_type'] == 'ALARM_OCCURRED'].groupby('equipment_id')['alarm_code'].count().sort_values(ascending=False)
"""
    
    # 쿼리 파일 저장
    with open('integrated_mcs_data/sample_queries.txt', 'w', encoding='utf-8') as f:
        f.write(queries)
    
    print("\n📄 샘플 쿼리 파일 생성: integrated_mcs_data/sample_queries.txt")


def main():
    """
    메인 실행 함수
    """
    print("="*60)
    print("🏭 반도체 FAB 통합 MCS 데이터 생성")
    print("="*60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 데이터 생성
    print("\n1️⃣ 통합 MCS 데이터 생성 중...")
    df = generate_integrated_mcs_data(100)
    print(f"   - 생성된 레코드: {len(df)}개")
    
    # 2. 데이터 분석
    print("\n2️⃣ 데이터 분석 중...")
    analyze_mcs_data(df)
    
    # 3. 파일 저장
    print("\n3️⃣ 데이터 파일 저장 중...")
    output_dir = save_integrated_mcs_data(df)
    
    # 4. 샘플 쿼리 생성
    create_sample_queries()
    
    # 5. 결과 요약
    print("\n" + "="*60)
    print("✨ 통합 MCS 데이터 생성 완료!")
    print("="*60)
    print(f"\n📁 생성된 파일 위치: {os.path.abspath(output_dir)}")
    print("\n📄 생성된 파일 목록:")
    print("  - integrated_mcs_data.csv : 전체 MCS 로그 데이터")
    print("  - integrated_mcs_data.json : JSON 형식")
    print("  - integrated_mcs_data.xlsx : Excel (시트별 분석 포함)")
    print("  - sample_queries.txt : 데이터 활용 샘플 쿼리")
    
    print("\n💡 데이터 특징:")
    print("  - MCS 이벤트 로그 (이송, 공정, 알람 등)")
    print("  - 장비 센서 데이터 (진동, 온도, 압력, 파티클 등)")
    print("  - OHT 진동 데이터")
    print("  - 실시간 센서 업데이트 이벤트")
    print("  - 알람 발생 시 관련 센서 데이터 포함")
    
    # 샘플 데이터 미리보기
    print("\n📊 샘플 데이터 (처음 3개):")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    print(df[['timestamp', 'event_type', 'equipment_id', 'sensor_vibration_mm_s', 'sensor_temperature_c']].head(3))
    
    return df


if __name__ == "__main__":
    # pandas 옵션 설정
    pd.set_option('display.max_colwidth', 50)
    
    # 메인 함수 실행
    mcs_df = main()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

def generate_semiconductor_mcs_data(num_records=100):
    """
    반도체 FAB MCS(Material Control System) 물류 로그 샘플 데이터 생성
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
    
    # 장비 타입
    equipment_types = {
        'ASML': ['ASML_NXT_1980', 'ASML_NXT_2050', 'ASML_XT_1460'],
        'AMAT': ['AMAT_ENDURA', 'AMAT_CENTURA', 'AMAT_PRODUCER'],
        'LAM': ['LAM_KIYO', 'LAM_VERSYS', 'LAM_FLEX'],
        'TEL': ['TEL_CLEAN', 'TEL_COAT', 'TEL_ETCH'],
        'KLA': ['KLA_2935', 'KLA_SP5', 'KLA_AIT']
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
        'STOCKER_IN',       # 스토커 입고
        'STOCKER_OUT',      # 스토커 출고
        'LOADPORT_LOAD',    # 로드포트 적재
        'LOADPORT_UNLOAD',  # 로드포트 하역
        'CARRIER_WAIT',     # 캐리어 대기
        'ALARM_OCCURRED'    # 알람 발생
    ]
    
    # 운반 장비
    transport_vehicles = ['OHT', 'OHS', 'AGV', 'RGV', 'MANUAL']
    
    # 알람 코드 (실제 MCS에서 사용하는 형식)
    alarm_codes = ['E84_TIMEOUT', 'CARRIER_ID_MISMATCH', 'VEHICLE_COLLISION_RISK', 
                   'LOADPORT_ERROR', 'STOCKER_FULL', 'PATH_BLOCKED']
    
    records = []
    
    # LOT 단위로 데이터 생성
    lot_count = num_records // 5  # 평균 5개 이벤트 per LOT
    
    for lot_idx in range(lot_count):
        # LOT 정보 생성
        lot_id = f"LOT{2025070000 + lot_idx:010d}"
        carrier_id = f"CAR{random.randint(10000, 99999)}"
        product = random.choice(product_codes)
        priority = random.choice(priorities)
        wafer_size = random.choice(wafer_sizes)
        wafer_count = 25 if wafer_size == '300mm' else 50
        
        # LOT의 라우팅 (공정 순서)
        num_steps = random.randint(3, 7)
        route = random.sample(list(bays.keys()), num_steps)
        
        # 시작 시간
        current_time = base_time + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # 각 공정 단계별 이벤트 생성
        for step_idx, bay in enumerate(route):
            # 이송 요청
            records.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': 'TRANSFER_START',
                'lot_id': lot_id,
                'carrier_id': carrier_id,
                'carrier_type': random.choice(carrier_types),
                'product_code': product,
                'wafer_size': wafer_size,
                'wafer_count': wafer_count,
                'priority': priority,
                'from_location': route[step_idx-1] if step_idx > 0 else 'STOCKER_01',
                'to_location': bay,
                'vehicle_id': f"{random.choice(transport_vehicles)}_{random.randint(100, 999)}",
                'fab_line': random.choice(fab_lines),
                'step_number': f"{(step_idx + 1) * 100}",
                'recipe': f"{bay}_RECIPE_{random.randint(1, 5)}",
                'temperature': round(20 + random.random() * 5, 1),
                'humidity': round(45 + random.random() * 10, 1),
                'particle_count': random.randint(0, 50),
                'status': 'IN_TRANSIT',
                'alarm_code': None,
                'transfer_time_sec': None
            })
            
            # 이송 시간 (초)
            transfer_time = random.randint(30, 300)
            current_time += timedelta(seconds=transfer_time)
            
            # 이송 완료
            records.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': 'TRANSFER_COMPLETE',
                'lot_id': lot_id,
                'carrier_id': carrier_id,
                'carrier_type': random.choice(carrier_types),
                'product_code': product,
                'wafer_size': wafer_size,
                'wafer_count': wafer_count,
                'priority': priority,
                'from_location': route[step_idx-1] if step_idx > 0 else 'STOCKER_01',
                'to_location': bay,
                'vehicle_id': f"{random.choice(transport_vehicles)}_{random.randint(100, 999)}",
                'fab_line': random.choice(fab_lines),
                'step_number': f"{(step_idx + 1) * 100}",
                'recipe': f"{bay}_RECIPE_{random.randint(1, 5)}",
                'temperature': round(20 + random.random() * 5, 1),
                'humidity': round(45 + random.random() * 10, 1),
                'particle_count': random.randint(0, 50),
                'status': 'ARRIVED',
                'alarm_code': None,
                'transfer_time_sec': transfer_time
            })
            
            # 장비에 로드
            current_time += timedelta(seconds=random.randint(10, 60))
            
            # 가끔 알람 발생
            if random.random() < 0.1:  # 10% 확률
                records.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_type': 'ALARM_OCCURRED',
                    'lot_id': lot_id,
                    'carrier_id': carrier_id,
                    'carrier_type': random.choice(carrier_types),
                    'product_code': product,
                    'wafer_size': wafer_size,
                    'wafer_count': wafer_count,
                    'priority': priority,
                    'from_location': bay,
                    'to_location': bay,
                    'vehicle_id': None,
                    'fab_line': random.choice(fab_lines),
                    'step_number': f"{(step_idx + 1) * 100}",
                    'recipe': f"{bay}_RECIPE_{random.randint(1, 5)}",
                    'temperature': round(20 + random.random() * 5, 1),
                    'humidity': round(45 + random.random() * 10, 1),
                    'particle_count': random.randint(0, 50),
                    'status': 'ALARM',
                    'alarm_code': random.choice(alarm_codes),
                    'transfer_time_sec': None
                })
                current_time += timedelta(minutes=random.randint(5, 30))
            
            # 공정 시간
            process_time = random.randint(30, 180)  # 30분 ~ 3시간
            current_time += timedelta(minutes=process_time)
    
    # DataFrame 생성
    df = pd.DataFrame(records[:num_records])  # 정확히 100개만
    
    # 시간순 정렬
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def generate_vibration_sensor_data(num_records=100):
    """
    반도체 장비 진동센서 데이터 생성
    OHT(Overhead Hoist Transport) 및 주요 장비의 진동 데이터
    """
    
    # 기준 시작 시간
    base_time = datetime(2025, 7, 1, 0, 0, 0)
    
    # 센서 위치 (장비 타입별)
    sensor_locations = {
        'OHT_RAIL': ['OHT_RAIL_A1', 'OHT_RAIL_A2', 'OHT_RAIL_B1', 'OHT_RAIL_B2'],
        'OHT_VEHICLE': ['OHT_001', 'OHT_002', 'OHT_003', 'OHT_004', 'OHT_005'],
        'STOCKER': ['STK_01_TOP', 'STK_01_MID', 'STK_02_TOP', 'STK_02_MID'],
        'EQUIPMENT': ['ASML_1980_01', 'AMAT_ENDURA_01', 'LAM_KIYO_01', 'TEL_CLEAN_01'],
        'LOADPORT': ['LP_PHOTO_01', 'LP_ETCH_01', 'LP_CVD_01', 'LP_CMP_01']
    }
    
    # 진동 상태
    vibration_states = {
        'NORMAL': {'x_range': (0.1, 0.5), 'y_range': (0.1, 0.5), 'z_range': (0.2, 0.8)},
        'WARNING': {'x_range': (0.5, 1.5), 'y_range': (0.5, 1.5), 'z_range': (0.8, 2.0)},
        'CRITICAL': {'x_range': (1.5, 3.0), 'y_range': (1.5, 3.0), 'z_range': (2.0, 4.0)},
        'MAINTENANCE': {'x_range': (0.05, 0.2), 'y_range': (0.05, 0.2), 'z_range': (0.1, 0.3)}
    }
    
    records = []
    
    # 각 센서별로 시계열 데이터 생성
    all_sensors = []
    for location_type, sensors in sensor_locations.items():
        all_sensors.extend([(location_type, sensor) for sensor in sensors])
    
    # 레코드 수를 센서별로 분배
    records_per_sensor = num_records // len(all_sensors)
    remaining_records = num_records % len(all_sensors)
    
    for idx, (location_type, sensor_id) in enumerate(all_sensors):
        # 이 센서의 레코드 수
        sensor_records = records_per_sensor + (1 if idx < remaining_records else 0)
        
        # 센서별 초기 상태
        current_state = 'NORMAL'
        base_vibration_level = random.uniform(0.2, 0.4)
        
        for i in range(sensor_records):
            # 시간 증가 (1분 간격)
            timestamp = base_time + timedelta(minutes=i + idx * records_per_sensor)
            
            # 가끔 상태 변경 (10% 확률)
            if random.random() < 0.1:
                current_state = random.choice(list(vibration_states.keys()))
            
            # 진동 값 생성
            state_config = vibration_states[current_state]
            
            # 기본 진동 + 노이즈
            vibration_x = random.uniform(*state_config['x_range']) + random.gauss(0, 0.05)
            vibration_y = random.uniform(*state_config['y_range']) + random.gauss(0, 0.05)
            vibration_z = random.uniform(*state_config['z_range']) + random.gauss(0, 0.08)
            
            # OHT 이동 중일 때 진동 증가
            if location_type == 'OHT_VEHICLE' and random.random() < 0.3:
                vibration_x *= 1.5
                vibration_y *= 1.5
                vibration_z *= 2.0
            
            # RMS 값 계산
            rms_value = np.sqrt((vibration_x**2 + vibration_y**2 + vibration_z**2) / 3)
            
            # 주파수 스펙트럼 (간단한 시뮬레이션)
            dominant_frequency = random.uniform(10, 500)
            
            # FFT 피크 값들
            fft_peaks = {
                'peak_1_freq': dominant_frequency,
                'peak_1_amp': random.uniform(0.1, 0.5),
                'peak_2_freq': dominant_frequency * 2,  # 하모닉
                'peak_2_amp': random.uniform(0.05, 0.2),
                'peak_3_freq': dominant_frequency * 3,
                'peak_3_amp': random.uniform(0.02, 0.1)
            }
            
            # 온도 (진동과 약간의 상관관계)
            temperature = 25 + base_vibration_level * 10 + random.gauss(0, 2)
            
            record = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'sensor_id': sensor_id,
                'location_type': location_type,
                'vibration_x_mm_s': round(vibration_x, 4),
                'vibration_y_mm_s': round(vibration_y, 4),
                'vibration_z_mm_s': round(vibration_z, 4),
                'rms_mm_s': round(rms_value, 4),
                'dominant_frequency_hz': round(dominant_frequency, 2),
                'peak_1_frequency_hz': round(fft_peaks['peak_1_freq'], 2),
                'peak_1_amplitude': round(fft_peaks['peak_1_amp'], 4),
                'peak_2_frequency_hz': round(fft_peaks['peak_2_freq'], 2),
                'peak_2_amplitude': round(fft_peaks['peak_2_amp'], 4),
                'temperature_c': round(temperature, 1),
                'vibration_state': current_state,
                'threshold_exceeded': current_state in ['WARNING', 'CRITICAL'],
                'maintenance_required': current_state == 'CRITICAL',
                'sensor_health': 'GOOD' if random.random() > 0.05 else 'CHECK'
            }
            
            records.append(record)
    
    # DataFrame 생성 및 정렬
    df = pd.DataFrame(records[:num_records])  # 정확히 요청된 수만큼
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['sensor_id', 'timestamp']).reset_index(drop=True)
    
    return df


def save_data_files(mcs_df, vibration_df):
    """
    데이터를 다양한 형식으로 저장
    """
    # 출력 디렉토리 생성
    output_dir = "semiconductor_test_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # MCS 데이터 저장
    mcs_path = os.path.join(output_dir, "mcs_data")
    if not os.path.exists(mcs_path):
        os.makedirs(mcs_path)
    
    # CSV
    mcs_df.to_csv(os.path.join(mcs_path, 'semiconductor_mcs_data.csv'), 
                  index=False, encoding='utf-8-sig')
    
    # JSON
    mcs_df_json = mcs_df.copy()
    mcs_df_json['timestamp'] = mcs_df_json['timestamp'].astype(str)
    mcs_df_json.to_json(os.path.join(mcs_path, 'semiconductor_mcs_data.json'), 
                        orient='records', force_ascii=False, indent=2)
    
    # Excel
    with pd.ExcelWriter(os.path.join(mcs_path, 'semiconductor_mcs_data.xlsx'), 
                        engine='openpyxl') as writer:
        mcs_df.to_excel(writer, sheet_name='MCS_LOG', index=False)
        
        # 요약 통계
        summary_df = pd.DataFrame({
            '항목': ['총 레코드 수', '고유 LOT 수', '고유 캐리어 수', '알람 발생 수', 
                    '평균 이송 시간(초)', '가장 많은 공정'],
            '값': [
                len(mcs_df),
                mcs_df['lot_id'].nunique(),
                mcs_df['carrier_id'].nunique(),
                len(mcs_df[mcs_df['event_type'] == 'ALARM_OCCURRED']),
                mcs_df[mcs_df['transfer_time_sec'].notna()]['transfer_time_sec'].mean(),
                mcs_df['to_location'].mode()[0] if len(mcs_df) > 0 else 'N/A'
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print("✅ MCS 데이터 저장 완료")
    
    # 진동센서 데이터 저장
    vibration_path = os.path.join(output_dir, "vibration_data")
    if not os.path.exists(vibration_path):
        os.makedirs(vibration_path)
    
    # CSV
    vibration_df.to_csv(os.path.join(vibration_path, 'vibration_sensor_data.csv'), 
                        index=False, encoding='utf-8-sig')
    
    # JSON
    vibration_df_json = vibration_df.copy()
    vibration_df_json['timestamp'] = vibration_df_json['timestamp'].astype(str)
    vibration_df_json.to_json(os.path.join(vibration_path, 'vibration_sensor_data.json'), 
                              orient='records', force_ascii=False, indent=2)
    
    # Excel
    with pd.ExcelWriter(os.path.join(vibration_path, 'vibration_sensor_data.xlsx'), 
                        engine='openpyxl') as writer:
        vibration_df.to_excel(writer, sheet_name='Vibration_Data', index=False)
        
        # 센서별 통계
        sensor_stats = vibration_df.groupby('sensor_id').agg({
            'rms_mm_s': ['mean', 'max', 'std'],
            'threshold_exceeded': 'sum',
            'maintenance_required': 'sum'
        }).round(4)
        sensor_stats.columns = ['평균_RMS', '최대_RMS', 'RMS_표준편차', '임계값_초과_횟수', '유지보수_필요_횟수']
        sensor_stats.to_excel(writer, sheet_name='Sensor_Statistics')
    
    print("✅ 진동센서 데이터 저장 완료")
    
    # 통합 분석 리포트
    print("\n📊 통합 분석 리포트 생성 중...")
    
    # 시간대별 상관관계 분석
    mcs_hourly = mcs_df.set_index('timestamp').resample('H').size()
    vibration_hourly = vibration_df[vibration_df['threshold_exceeded']].set_index('timestamp').resample('H').size()
    
    correlation_df = pd.DataFrame({
        'MCS_이벤트_수': mcs_hourly,
        '진동_임계값_초과': vibration_hourly
    }).fillna(0)
    
    with pd.ExcelWriter(os.path.join(output_dir, 'integrated_analysis_report.xlsx'), 
                        engine='openpyxl') as writer:
        correlation_df.to_excel(writer, sheet_name='Hourly_Correlation')
        
        # 요약 정보
        summary_data = {
            '항목': ['MCS 총 이벤트', 'MCS 알람 발생', '진동센서 임계값 초과', 
                    '진동센서 유지보수 필요', '분석 기간(일)'],
            '값': [
                len(mcs_df),
                len(mcs_df[mcs_df['event_type'] == 'ALARM_OCCURRED']),
                vibration_df['threshold_exceeded'].sum(),
                vibration_df['maintenance_required'].sum(),
                (mcs_df['timestamp'].max() - mcs_df['timestamp'].min()).days
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print("✅ 통합 분석 리포트 저장 완료")
    
    return output_dir


def main():
    """
    메인 실행 함수
    """
    print("="*60)
    print("🏭 반도체 물류 예측 시스템 - 테스트 데이터 생성")
    print("="*60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. MCS 데이터 생성
    print("1️⃣ MCS 데이터 생성 중...")
    mcs_df = generate_semiconductor_mcs_data(100)
    print(f"   - 생성된 레코드: {len(mcs_df)}개")
    print(f"   - 기간: {mcs_df['timestamp'].min()} ~ {mcs_df['timestamp'].max()}")
    
    # 2. 진동센서 데이터 생성
    print("\n2️⃣ 진동센서 데이터 생성 중...")
    vibration_df = generate_vibration_sensor_data(100)
    print(f"   - 생성된 레코드: {len(vibration_df)}개")
    print(f"   - 센서 수: {vibration_df['sensor_id'].nunique()}개")
    
    # 3. 파일 저장
    print("\n3️⃣ 데이터 파일 저장 중...")
    output_dir = save_data_files(mcs_df, vibration_df)
    
    # 4. 결과 요약
    print("\n" + "="*60)
    print("✨ 테스트 데이터 생성 완료!")
    print("="*60)
    print(f"\n📁 생성된 파일 위치: {os.path.abspath(output_dir)}")
    print("\n📄 생성된 파일 목록:")
    print("  MCS 데이터:")
    print("    - semiconductor_mcs_data.csv")
    print("    - semiconductor_mcs_data.json")
    print("    - semiconductor_mcs_data.xlsx")
    print("\n  진동센서 데이터:")
    print("    - vibration_sensor_data.csv")
    print("    - vibration_sensor_data.json")
    print("    - vibration_sensor_data.xlsx")
    print("\n  통합 분석:")
    print("    - integrated_analysis_report.xlsx")
    
    # 5. 사용 안내
    print("\n📌 사용 방법:")
    print("1. 노드 에디터를 실행하세요")
    print("2. '데이터 입력' 노드를 추가하고 생성된 CSV 파일을 선택하세요")
    print("3. 전처리 → 분석 → 프롬프트 → LLM 순서로 노드를 연결하세요")
    print("4. 파이프라인을 실행하여 예측 결과를 확인하세요")
    
    # 6. 샘플 데이터 미리보기
    print("\n📊 MCS 샘플 데이터 (처음 3개):")
    print(mcs_df[['timestamp', 'event_type', 'lot_id', 'from_location', 'to_location']].head(3))
    
    print("\n📊 진동센서 샘플 데이터 (처음 3개):")
    print(vibration_df[['timestamp', 'sensor_id', 'rms_mm_s', 'vibration_state']].head(3))
    
    return mcs_df, vibration_df


if __name__ == "__main__":
    # pandas 옵션 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # 메인 함수 실행
    mcs_df, vibration_df = main()
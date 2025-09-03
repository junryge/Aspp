#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HUBROOM 300 임계값 Sensing 분석 시스템
과거 20분 데이터 기반 예측 성능 평가
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_sensing_performance(csv_file_path, output_file_path=None):
    """
    CSV 파일을 읽어 Sensing 성능을 분류하고 새 컬럼 추가
    
    분류 기준:
    1. 300_Sensing_OK: 과거 20분 300이하 → 예측 300이상 → 실제 300이상 (성공)
    2. 300_Sensing_NG: 과거 20분 300이하 → 예측 300이상 → 실제 300미만 (실패)
    3. 200_Sensing_OK: 과거 20분 300이상 → 예측 300미만 → 실제 300미만 (성공)
    4. 200_Sensing_NG: 과거 20분 300이상 → 예측 300미만 → 실제 300이상 (실패)
    """
    
    print("="*80)
    print("🏭 HUBROOM 300 임계값 Sensing 분석 시스템")
    print("="*80)
    
    # 1. CSV 파일 읽기 (탭 구분자 사용)
    print("\n📂 CSV 파일 로드 중...")
    try:
        # 먼저 탭 구분자로 시도
        df = pd.read_csv(csv_file_path, sep='\t')
        print(f"✅ 데이터 로드 완료 (탭 구분): {len(df):,} 행")
    except:
        # 실패하면 쉼표 구분자로 시도
        try:
            df = pd.read_csv(csv_file_path, sep=',')
            print(f"✅ 데이터 로드 완료 (쉼표 구분): {len(df):,} 행")
        except:
            # 그래도 실패하면 공백 구분자로 시도
            df = pd.read_csv(csv_file_path, delim_whitespace=True)
            print(f"✅ 데이터 로드 완료 (공백 구분): {len(df):,} 행")
    
    # 2. timestamp를 datetime으로 변환
    print("\n⏰ 시간 데이터 처리 중...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 3. Sensing 컬럼 생성
    print("\n🔍 Sensing 분류 시작...")
    threshold = 300
    lookback_minutes = 20
    
    sensing_results = []
    
    for idx in range(len(df)):
        if idx % 1000 == 0:
            print(f"  진행 중: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)", end='\r')
        
        current_time = df.loc[idx, 'timestamp']
        current_actual = df.loc[idx, 'actual']
        current_predicted = df.loc[idx, 'predicted']
        
        # 과거 20분 데이터 추출
        past_time = current_time - timedelta(minutes=lookback_minutes)
        past_data = df[(df['timestamp'] > past_time) & (df['timestamp'] < current_time)]
        
        # 과거 데이터가 없으면 분류 불가
        if len(past_data) == 0:
            sensing_results.append('No_Past_Data')
            continue
        
        # 과거 20분 데이터의 최대값
        past_max = past_data['actual'].max()
        
        # 분류 로직
        if past_max <= threshold:  # 과거 20분이 모두 300 이하
            if current_predicted >= threshold:  # 예측이 300 이상
                if current_actual >= threshold:  # 실제도 300 이상
                    sensing_results.append('300_Sensing_OK')
                else:  # 실제는 300 미만
                    sensing_results.append('300_Sensing_NG')
            else:
                sensing_results.append('Normal')  # 예측도 300 미만
        else:  # 과거 20분 중 300 이상이 있음
            if current_predicted < threshold:  # 예측이 300 미만
                if current_actual < threshold:  # 실제도 300 미만
                    sensing_results.append('200_Sensing_OK')
                else:  # 실제는 300 이상
                    sensing_results.append('200_Sensing_NG')
            else:
                sensing_results.append('High_Maintaining')  # 계속 높음
    
    # Sensing 컬럼 추가
    df['Sensing'] = sensing_results
    
    print(f"\n✅ Sensing 분류 완료!")
    
    # 4. 통계 출력
    print("\n" + "="*60)
    print("📊 분류 결과 통계")
    print("="*60)
    
    sensing_counts = df['Sensing'].value_counts()
    total = len(df)
    
    for category, count in sensing_counts.items():
        percentage = (count / total) * 100
        print(f"  {category:20}: {count:6,} 건 ({percentage:5.2f}%)")
    
    # 5. 성능 지표 계산
    print("\n" + "="*60)
    print("🎯 감지 성능 분석")
    print("="*60)
    
    # 300 상승 감지 성능
    ok_300 = sensing_counts.get('300_Sensing_OK', 0)
    ng_300 = sensing_counts.get('300_Sensing_NG', 0)
    total_300 = ok_300 + ng_300
    
    if total_300 > 0:
        accuracy_300 = (ok_300 / total_300) * 100
        print(f"\n📈 300 상승 감지 성능:")
        print(f"  - 전체 시도: {total_300:,} 건")
        print(f"  - 성공 (OK): {ok_300:,} 건")
        print(f"  - 실패 (NG): {ng_300:,} 건")
        print(f"  - 정확도: {accuracy_300:.2f}%")
    
    # 300 하락 감지 성능
    ok_200 = sensing_counts.get('200_Sensing_OK', 0)
    ng_200 = sensing_counts.get('200_Sensing_NG', 0)
    total_200 = ok_200 + ng_200
    
    if total_200 > 0:
        accuracy_200 = (ok_200 / total_200) * 100
        print(f"\n📉 300 하락 감지 성능:")
        print(f"  - 전체 시도: {total_200:,} 건")
        print(f"  - 성공 (OK): {ok_200:,} 건")
        print(f"  - 실패 (NG): {ng_200:,} 건")
        print(f"  - 정확도: {accuracy_200:.2f}%")
    
    # 전체 정확도
    total_attempts = total_300 + total_200
    total_success = ok_300 + ok_200
    
    if total_attempts > 0:
        overall_accuracy = (total_success / total_attempts) * 100
        print(f"\n📊 전체 감지 정확도: {overall_accuracy:.2f}%")
    
    # 6. 결과 저장
    if output_file_path is None:
        output_file_path = csv_file_path.replace('.csv', '_sensing_analyzed.csv')
    
    print(f"\n💾 결과 저장 중...")
    df.to_csv(output_file_path, index=False)
    print(f"✅ 저장 완료: {output_file_path}")
    
    # 7. 샘플 출력
    print("\n" + "="*60)
    print("📝 분류 샘플 (각 카테고리별 2개)")
    print("="*60)
    
    for category in ['300_Sensing_OK', '300_Sensing_NG', '200_Sensing_OK', '200_Sensing_NG']:
        sample = df[df['Sensing'] == category].head(2)
        if len(sample) > 0:
            print(f"\n🔹 {category}:")
            for _, row in sample.iterrows():
                print(f"  시간: {row['timestamp']}")
                print(f"  예측: {row['predicted']:.0f}, 실제: {row['actual']:.0f}")
    
    return df

# 메인 실행
if __name__ == "__main__":
    # 파일 경로 설정 (여기에 실제 파일 경로 입력)
    input_file = "your_data.csv"  
    output_file = "result_sensing.csv"  # 옵션: None으로 두면 자동 생성
    
    # 실행
    try:
        result = analyze_sensing_performance(input_file, output_file)
        print("\n✨ 분석 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
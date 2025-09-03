#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HUBROOM 300 임계값 Sensing 분류 시스템
과거 20분 데이터의 300 임계값 상태와 예측/실제값 비교를 통한 감지 성능 분류
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def classify_sensing_performance(csv_file_path, output_file_path=None):
    """
    CSV 파일을 읽어 Sensing 성능을 분류하고 새 컬럼 추가
    
    Parameters:
    -----------
    csv_file_path : str
        입력 CSV 파일 경로
    output_file_path : str, optional
        출력 CSV 파일 경로 (없으면 '_sensing_analyzed.csv' 추가)
    
    Returns:
    --------
    pd.DataFrame : Sensing 컬럼이 추가된 데이터프레임
    """
    
    print("="*80)
    print("🏭 HUBROOM 300 임계값 Sensing 분류 시스템")
    print("="*80)
    
    # 1. CSV 파일 읽기
    print("\n📂 CSV 파일 로드 중...")
    df = pd.read_csv(csv_file_path)
    print(f"✅ 데이터 로드 완료: {len(df):,} 행")
    
    # 2. timestamp 컬럼을 datetime으로 변환
    print("\n⏰ 시간 데이터 처리 중...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 3. Sensing 컬럼 초기화
    df['Sensing'] = ''
    
    # 4. 각 행에 대해 분류 수행
    print("\n🔍 Sensing 분류 시작...")
    threshold = 300
    lookback_minutes = 20
    
    total_rows = len(df)
    classified_count = 0
    
    # 분류 카운터
    sensing_counts = {
        '300_Sensing_OK': 0,
        '300_Sensing_NG': 0,
        '200_Sensing_OK': 0,
        '200_Sensing_NG': 0,
        'No_Classification': 0
    }
    
    for idx in range(len(df)):
        if idx % 100 == 0:
            print(f"  진행: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)", end='\r')
        
        current_time = df.loc[idx, 'timestamp']
        current_actual = df.loc[idx, 'actual']
        current_predicted = df.loc[idx, 'predicted']
        
        # 과거 20분 데이터 찾기
        past_time = current_time - timedelta(minutes=lookback_minutes)
        past_data = df[(df['timestamp'] > past_time) & (df['timestamp'] < current_time)]
        
        if len(past_data) == 0:
            df.loc[idx, 'Sensing'] = 'No_Past_Data'
            sensing_counts['No_Classification'] += 1
            continue
        
        # 과거 20분 데이터의 최대값 확인
        past_max = past_data['actual'].max()
        
        # 분류 로직
        if past_max <= threshold:  # 과거 20분이 300 이하
            if current_predicted >= threshold:  # 예측이 300 이상
                if current_actual >= threshold:  # 실제도 300 이상
                    df.loc[idx, 'Sensing'] = '300_Sensing_OK'
                    sensing_counts['300_Sensing_OK'] += 1
                else:  # 실제는 300 미만
                    df.loc[idx, 'Sensing'] = '300_Sensing_NG'
                    sensing_counts['300_Sensing_NG'] += 1
            else:
                df.loc[idx, 'Sensing'] = 'No_Alert_Needed'
                sensing_counts['No_Classification'] += 1
                
        else:  # 과거 20분에 300 이상 존재
            if current_predicted < threshold:  # 예측이 300 미만
                if current_actual < threshold:  # 실제도 300 미만
                    df.loc[idx, 'Sensing'] = '200_Sensing_OK'
                    sensing_counts['200_Sensing_OK'] += 1
                else:  # 실제는 300 이상
                    df.loc[idx, 'Sensing'] = '200_Sensing_NG'
                    sensing_counts['200_Sensing_NG'] += 1
            else:
                df.loc[idx, 'Sensing'] = 'Maintaining_High'
                sensing_counts['No_Classification'] += 1
    
    print(f"\n✅ Sensing 분류 완료!")
    
    # 5. 분류 결과 통계 출력
    print("\n" + "="*60)
    print("📊 분류 결과 통계")
    print("="*60)
    
    for category, count in sensing_counts.items():
        if count > 0:
            percentage = (count / total_rows) * 100
            print(f"  {category:20}: {count:6,} 건 ({percentage:5.2f}%)")
    
    # 주요 4가지 카테고리의 합계
    main_categories = ['300_Sensing_OK', '300_Sensing_NG', '200_Sensing_OK', '200_Sensing_NG']
    main_total = sum(sensing_counts[cat] for cat in main_categories)
    print(f"\n  {'주요 분류 합계':20}: {main_total:6,} 건 ({main_total/total_rows*100:5.2f}%)")
    
    # 6. 성능 지표 계산
    print("\n" + "="*60)
    print("🎯 감지 성능 분석")
    print("="*60)
    
    # 300 상승 감지 성능
    up_total = sensing_counts['300_Sensing_OK'] + sensing_counts['300_Sensing_NG']
    if up_total > 0:
        up_accuracy = (sensing_counts['300_Sensing_OK'] / up_total) * 100
        print(f"\n📈 300 상승 감지:")
        print(f"  - 전체 감지 시도: {up_total:,} 건")
        print(f"  - 정확 감지 (OK): {sensing_counts['300_Sensing_OK']:,} 건")
        print(f"  - 오감지 (NG): {sensing_counts['300_Sensing_NG']:,} 건")
        print(f"  - 정확도: {up_accuracy:.2f}%")
    
    # 300 하락 감지 성능
    down_total = sensing_counts['200_Sensing_OK'] + sensing_counts['200_Sensing_NG']
    if down_total > 0:
        down_accuracy = (sensing_counts['200_Sensing_OK'] / down_total) * 100
        print(f"\n📉 300 하락 감지:")
        print(f"  - 전체 감지 시도: {down_total:,} 건")
        print(f"  - 정확 감지 (OK): {sensing_counts['200_Sensing_OK']:,} 건")
        print(f"  - 오감지 (NG): {sensing_counts['200_Sensing_NG']:,} 건")
        print(f"  - 정확도: {down_accuracy:.2f}%")
    
    # 7. 결과 저장
    if output_file_path is None:
        output_file_path = csv_file_path.replace('.csv', '_sensing_analyzed.csv')
    
    print(f"\n💾 결과 저장 중...")
    df.to_csv(output_file_path, index=False)
    print(f"✅ 저장 완료: {output_file_path}")
    
    # 8. 샘플 데이터 출력
    print("\n" + "="*60)
    print("📝 분류 결과 샘플 (주요 카테고리만)")
    print("="*60)
    
    for category in main_categories:
        sample = df[df['Sensing'] == category].head(2)
        if len(sample) > 0:
            print(f"\n🔹 {category}:")
            for _, row in sample.iterrows():
                print(f"  시간: {row['timestamp']}")
                print(f"  실제값: {row['actual']:.1f}, 예측값: {row['predicted']:.1f}")
                print()
    
    return df

# 실행 함수
def main():
    """메인 실행 함수"""
    
    # 파일 경로 설정
    input_file = 'your_data.csv'  # 여기에 실제 CSV 파일 경로 입력
    output_file = 'your_data_sensing_analyzed.csv'  # 출력 파일명 (옵션)
    
    try:
        # 분석 실행
        result_df = classify_sensing_performance(input_file, output_file)
        
        print("\n" + "="*80)
        print("✨ 분석 완료!")
        print("="*80)
        print(f"📊 총 {len(result_df):,} 행 처리 완료")
        print(f"📁 결과 파일: {output_file}")
        
        # 추가 분석 (옵션)
        print("\n💡 추가 분석 팁:")
        print("  - Sensing 컬럼으로 그룹화하여 상세 분석 가능")
        print("  - 시간대별 성능 변화 추적 가능")
        print("  - 특정 구간의 감지 성능 집중 분석 가능")
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 사용 예시
    print("📌 사용법:")
    print("  1. input_file 변수에 CSV 파일 경로 설정")
    print("  2. main() 함수 실행")
    print("\n또는 직접 함수 호출:")
    print('  df = classify_sensing_performance("your_file.csv")')
    
    # 실제 실행하려면 아래 주석 해제
    # main()
    
    # 또는 직접 실행
    # df = classify_sensing_performance("your_data.csv", "output_sensing.csv")
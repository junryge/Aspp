import pandas as pd
from datetime import datetime, timedelta

def validate_input_max_min(file_path):
    """
    과거 20분간의 actual 값에서 계산한 max/min과 
    input_max/input_min이 일치하는지 검증
    """
    # CSV 파일 읽기 (탭으로 구분)
    df = pd.read_csv(file_path, sep='\t')
    
    # timestamp를 datetime으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 검증 결과를 저장할 리스트
    validation_results = []
    
    print("=" * 80)
    print("Input Max/Min 검증 결과")
    print("=" * 80)
    print(f"{'시간':<20} {'Actual':<8} {'Input_Max':<10} {'Input_Min':<10} {'계산_Max':<10} {'계산_Min':<10} {'Max_일치':<10} {'Min_일치':<10}")
    print("-" * 80)
    
    # 각 행에 대해 검증
    for idx, row in df.iterrows():
        current_time = row['timestamp']
        
        # 과거 20분 범위 계산 (현재 시간 - 20분 ~ 현재 시간 - 1분)
        start_time = current_time - timedelta(minutes=20)
        end_time = current_time - timedelta(minutes=1)
        
        # 과거 20분 데이터 필터링
        past_20min = df[(df['timestamp'] > start_time) & (df['timestamp'] <= end_time)]
        
        if len(past_20min) > 0:
            # 과거 20분간의 actual 최대/최소값 계산
            calculated_max = past_20min['actual'].max()
            calculated_min = past_20min['actual'].min()
            
            # 현재 input_max, input_min과 비교
            input_max = row['input_max']
            input_min = row['input_min']
            
            # 일치 여부 확인
            max_match = (calculated_max == input_max)
            min_match = (calculated_min == input_min)
            
            # 결과 저장
            result = {
                'timestamp': current_time,
                'actual': row['actual'],
                'input_max': input_max,
                'input_min': input_min,
                'calculated_max': calculated_max,
                'calculated_min': calculated_min,
                'max_match': max_match,
                'min_match': min_match,
                'past_20min_count': len(past_20min)
            }
            validation_results.append(result)
            
            # 불일치하는 경우만 출력 (또는 모든 결과 출력)
            if not max_match or not min_match:
                print(f"{current_time.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{row['actual']:<8} "
                      f"{input_max:<10} "
                      f"{input_min:<10} "
                      f"{calculated_max:<10} "
                      f"{calculated_min:<10} "
                      f"{'O' if max_match else 'X':<10} "
                      f"{'O' if min_match else 'X':<10}")
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(validation_results)
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("검증 통계")
    print("=" * 80)
    
    if len(results_df) > 0:
        total_count = len(results_df)
        max_correct = results_df['max_match'].sum()
        min_correct = results_df['min_match'].sum()
        both_correct = ((results_df['max_match']) & (results_df['min_match'])).sum()
        
        print(f"전체 검증 건수: {total_count}")
        print(f"Max 일치: {max_correct}/{total_count} ({max_correct/total_count*100:.1f}%)")
        print(f"Min 일치: {min_correct}/{total_count} ({min_correct/total_count*100:.1f}%)")
        print(f"둘 다 일치: {both_correct}/{total_count} ({both_correct/total_count*100:.1f}%)")
        
        # 불일치 케이스 상세 분석
        max_errors = results_df[~results_df['max_match']]
        min_errors = results_df[~results_df['min_match']]
        
        if len(max_errors) > 0:
            print(f"\n최대값 불일치 케이스: {len(max_errors)}건")
            print("평균 오차:", (max_errors['input_max'] - max_errors['calculated_max']).abs().mean())
            
        if len(min_errors) > 0:
            print(f"\n최소값 불일치 케이스: {len(min_errors)}건")
            print("평균 오차:", (min_errors['input_min'] - min_errors['calculated_min']).abs().mean())
    
    return results_df

def check_specific_time_range(file_path, target_time):
    """
    특정 시점의 과거 20분 데이터를 상세히 확인
    """
    df = pd.read_csv(file_path, sep='\t')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 타겟 시간을 datetime으로 변환
    target = pd.to_datetime(target_time)
    
    # 해당 시점 찾기
    target_row = df[df['timestamp'] == target]
    
    if target_row.empty:
        print(f"{target_time} 시점의 데이터가 없습니다.")
        return
    
    # 과거 20분 데이터 추출
    start_time = target - timedelta(minutes=20)
    end_time = target - timedelta(minutes=1)
    past_20min = df[(df['timestamp'] > start_time) & (df['timestamp'] <= end_time)]
    
    print(f"\n{'='*60}")
    print(f"{target_time} 시점의 과거 20분 데이터 분석")
    print(f"{'='*60}")
    print(f"분석 범위: {start_time.strftime('%H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')}")
    print(f"데이터 개수: {len(past_20min)}개")
    
    if len(past_20min) > 0:
        print(f"\n과거 20분 actual 값들:")
        for _, row in past_20min.iterrows():
            print(f"  {row['timestamp'].strftime('%H:%M')}: {row['actual']}")
        
        calculated_max = past_20min['actual'].max()
        calculated_min = past_20min['actual'].min()
        
        print(f"\n계산된 Max: {calculated_max}")
        print(f"계산된 Min: {calculated_min}")
        print(f"\n기록된 input_max: {target_row.iloc[0]['input_max']}")
        print(f"기록된 input_min: {target_row.iloc[0]['input_min']}")
        
        if calculated_max != target_row.iloc[0]['input_max']:
            print(f"⚠️ Max 불일치! 차이: {abs(calculated_max - target_row.iloc[0]['input_max'])}")
        else:
            print("✓ Max 일치")
            
        if calculated_min != target_row.iloc[0]['input_min']:
            print(f"⚠️ Min 불일치! 차이: {abs(calculated_min - target_row.iloc[0]['input_min'])}")
        else:
            print("✓ Min 일치")

# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    file_path = 'your_data.csv'  # 실제 파일 경로로 변경
    
    # 전체 검증 실행
    results = validate_input_max_min(file_path)
    
    # 특정 시점 상세 분석 (예: 문제가 있는 시점)
    # check_specific_time_range(file_path, '2025-08-01 01:00:00')
    
    # 결과를 CSV로 저장 (선택사항)
    # results.to_csv('validation_results.csv', index=False)
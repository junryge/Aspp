import pandas as pd
import numpy as np

def add_sensing_column(input_file, output_file=None):
    """
    탭으로 구분된 CSV 파일에 Sensing 컬럼 추가
    """
    
    # 1. 데이터 읽기 (탭 구분)
    print("파일 읽는 중...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"✓ 데이터 로드 완료: {len(df)} 행")
    
    # 2. Sensing 컬럼 생성
    print("\nSensing 컬럼 생성 중...")
    sensing = []
    
    for i in range(len(df)):
        if i < 20:
            # 처음 20개는 과거 데이터 부족
            sensing.append('')
        else:
            # 과거 20개 행의 actual 값
            past_actual = df['actual'].iloc[i-20:i]
            past_max = past_actual.max()
            past_min = past_actual.min()
            
            # 현재 값
            current_actual = df['actual'].iloc[i]
            current_predicted = df['predicted'].iloc[i]
            
            # 분류
            if past_max < 300:  # 과거 20개 모두 300 미만
                if current_predicted >= 300:
                    if current_actual >= 300:
                        sensing.append('300_SENSING_OK')
                    else:
                        sensing.append('300_SENSING_NG')
                else:
                    sensing.append('')
            elif past_min >= 300:  # 과거 20개 모두 300 이상
                if current_predicted < 300:
                    if current_actual < 300:
                        sensing.append('200_SENSING_OK')
                    else:
                        sensing.append('200_SENSING_NG')
                else:
                    sensing.append('')
            else:
                sensing.append('')
    
    # 3. Sensing 컬럼 추가
    df['Sensing'] = sensing
    
    # 4. 결과 통계
    print("\n=== 분류 결과 ===")
    
    ok_300 = (df['Sensing'] == '300_SENSING_OK').sum()
    ng_300 = (df['Sensing'] == '300_SENSING_NG').sum()
    ok_200 = (df['Sensing'] == '200_SENSING_OK').sum()
    ng_200 = (df['Sensing'] == '200_SENSING_NG').sum()
    
    print(f"300_SENSING_OK: {ok_300}개")
    print(f"300_SENSING_NG: {ng_300}개")
    print(f"200_SENSING_OK: {ok_200}개")
    print(f"200_SENSING_NG: {ng_200}개")
    
    # 정확도
    if ok_300 + ng_300 > 0:
        acc_300 = ok_300 / (ok_300 + ng_300) * 100
        print(f"\n300 감지 정확도: {acc_300:.1f}%")
    
    if ok_200 + ng_200 > 0:
        acc_200 = ok_200 / (ok_200 + ng_200) * 100
        print(f"200 감지 정확도: {acc_200:.1f}%")
    
    # 5. 파일 저장 (탭 구분)
    if output_file is None:
        output_file = input_file.replace('.csv', '_sensing.csv')
    
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ 저장 완료: {output_file}")
    
    return df


# 실행 예시
if __name__ == "__main__":
    # 파일 경로 설정
    input_csv = "data.csv"  # 입력 파일
    output_csv = "result.csv"  # 출력 파일 (선택사항)
    
    # 실행
    result = add_sensing_column(input_csv, output_csv)
    
    # 결과 확인 (처음 30행)
    print("\n=== 결과 샘플 (20-30행) ===")
    print(result[['actual', 'predicted', 'Sensing']].iloc[20:30])
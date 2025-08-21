import pandas as pd
import os

def split_csv_by_date(input_file, output_folder='output_by_date'):
    """
    CSV 파일을 CURRTIME 컬럼의 날짜별로 분리
    CURRTIME 컬럼은 수정하지 않고 그대로 유지
    """
    
    # CSV 파일 읽기
    print(f"📂 파일 읽는 중: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ 총 {len(df)}개 행 읽기 완료")
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 출력 폴더 생성: {output_folder}")
    
    # CURRTIME에서 날짜 부분만 추출 (데이터는 수정하지 않음)
    # CURRTIME 형식에 따라 날짜 추출
    df['temp_date'] = df['CURRTIME'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else 'unknown')
    
    # 고유한 날짜 목록
    unique_dates = df['temp_date'].unique()
    print(f"\n📅 발견된 날짜: {len(unique_dates)}개")
    
    # 날짜별로 파일 분리 및 저장
    for date in unique_dates:
        # 해당 날짜의 데이터만 필터링
        date_df = df[df['temp_date'] == date].copy()
        
        # 임시 컬럼 제거 (원본 데이터 그대로 유지)
        date_df = date_df.drop('temp_date', axis=1)
        
        # 파일명 생성 (특수문자 제거)
        safe_date = str(date).replace('/', '-').replace(':', '-')
        output_file = os.path.join(output_folder, f'data_{safe_date}.csv')
        
        # CSV로 저장
        date_df.to_csv(output_file, index=False)
        print(f"   ✅ {safe_date}: {len(date_df)}개 행 → {output_file}")
    
    print(f"\n🎉 완료! 총 {len(unique_dates)}개 파일로 분리되었습니다.")
    return unique_dates


# 메인 실행 코드
if __name__ == "__main__":
    # 입력 파일 경로
    input_csv = "your_file.csv"  # ← 여기에 실제 CSV 파일 경로 입력
    
    # 날짜별로 분리 실행
    split_csv_by_date(input_csv, output_folder='split_by_date')
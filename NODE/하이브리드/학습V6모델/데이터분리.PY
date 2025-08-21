import pandas as pd
import os

def split_csv_by_date(input_file, output_folder='output_by_date'):
    """
    CURRTIME (202508012053 형식)을 날짜별(20250801)로 CSV 파일 분리
    """
    
    # CSV 파일 읽기
    print(f"📂 파일 읽는 중: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ 총 {len(df)}개 행 읽기 완료")
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 출력 폴더 생성: {output_folder}")
    
    # CURRTIME에서 날짜 부분(YYYYMMDD)만 추출
    # 202508012053 -> 20250801
    df['date_only'] = df['CURRTIME'].astype(str).str[:8]
    
    # 고유한 날짜 목록
    unique_dates = df['date_only'].unique()
    print(f"\n📅 발견된 날짜: {len(unique_dates)}개")
    for date in sorted(unique_dates):
        print(f"   - {date}")
    
    # 날짜별로 파일 분리 및 저장
    print(f"\n💾 날짜별 CSV 파일 생성 중...")
    
    for date in unique_dates:
        # 해당 날짜의 데이터만 필터링
        date_df = df[df['date_only'] == date].copy()
        
        # 임시 컬럼 제거 (원본 CURRTIME은 그대로 유지)
        date_df = date_df.drop('date_only', axis=1)
        
        # 파일명: 20250801.csv
        output_file = os.path.join(output_folder, f'{date}.csv')
        
        # CSV로 저장
        date_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ {date}.csv: {len(date_df)}개 행 저장 완료")
    
    print(f"\n🎉 완료! 총 {len(unique_dates)}개 파일로 분리되었습니다.")
    print(f"📁 저장 위치: {output_folder}/")
    
    return unique_dates


# 간단한 사용 예시
def quick_split(input_file):
    """빠른 실행용 함수"""
    split_csv_by_date(input_file, 'split_by_date')


# 메인 실행 코드
if __name__ == "__main__":
    # 입력 파일 경로
    input_csv = "data.csv"  # ← 여기에 실제 CSV 파일명 입력
    
    # 실행
    split_csv_by_date(input_csv)
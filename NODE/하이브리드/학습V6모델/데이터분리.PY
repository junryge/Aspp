import pandas as pd
import os
from datetime import datetime

def split_csv_by_date(input_file, output_folder='output_by_date'):
    """
    CSV 파일을 CURRTIME 컬럼의 날짜별로 분리하는 함수
    
    Parameters:
    -----------
    input_file : str
        입력 CSV 파일 경로
    output_folder : str
        출력 폴더 경로 (기본값: 'output_by_date')
    """
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(input_file)
        print(f"✅ CSV 파일 읽기 완료: {input_file}")
        print(f"   총 {len(df)}개 행, {len(df.columns)}개 컬럼")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    # CURRTIME 컬럼 확인
    if 'CURRTIME' not in df.columns:
        print("❌ CURRTIME 컬럼이 없습니다.")
        print(f"   사용 가능한 컬럼: {', '.join(df.columns)}")
        return
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 출력 폴더 생성: {output_folder}")
    
    # CURRTIME을 datetime으로 변환
    try:
        # 다양한 날짜 형식을 자동으로 파싱
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
    except Exception as e:
        print(f"⚠️ 날짜 변환 중 오류 발생, 문자열로 처리합니다: {e}")
    
    # 날짜 부분만 추출 (시간 제거)
    if pd.api.types.is_datetime64_any_dtype(df['CURRTIME']):
        df['date_only'] = df['CURRTIME'].dt.date
    else:
        # 문자열인 경우 날짜 부분만 추출 시도
        df['date_only'] = df['CURRTIME'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else 'unknown')
    
    # 고유한 날짜 목록
    unique_dates = df['date_only'].unique()
    print(f"\n📅 발견된 날짜: {len(unique_dates)}개")
    
    # 날짜별로 파일 분리 및 저장
    for date in unique_dates:
        # 해당 날짜의 데이터만 필터링
        date_df = df[df['date_only'] == date].copy()
        
        # 임시 컬럼 제거
        date_df = date_df.drop('date_only', axis=1)
        
        # 파일명 생성 (안전한 파일명을 위해 특수문자 제거)
        safe_date = str(date).replace('/', '-').replace(':', '-')
        output_file = os.path.join(output_folder, f'data_{safe_date}.csv')
        
        # CSV로 저장
        date_df.to_csv(output_file, index=False)
        print(f"   ✅ {safe_date}: {len(date_df)}개 행 → {output_file}")
    
    print(f"\n🎉 완료! 총 {len(unique_dates)}개 파일로 분리되었습니다.")
    print(f"   출력 폴더: {output_folder}")


def analyze_csv_dates(input_file):
    """
    CSV 파일의 CURRTIME 컬럼을 분석하는 함수
    
    Parameters:
    -----------
    input_file : str
        입력 CSV 파일 경로
    """
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(input_file)
        print(f"📊 CSV 파일 분석: {input_file}")
        print(f"   총 행 수: {len(df)}")
        print(f"   컬럼: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    if 'CURRTIME' not in df.columns:
        print("❌ CURRTIME 컬럼이 없습니다.")
        return
    
    # CURRTIME 샘플 출력
    print(f"\n📅 CURRTIME 샘플 (처음 5개):")
    for i, val in enumerate(df['CURRTIME'].head()):
        print(f"   {i+1}. {val}")
    
    # 날짜 형식 분석
    try:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
        
        print(f"\n📈 날짜 범위:")
        print(f"   시작: {df['CURRTIME'].min()}")
        print(f"   종료: {df['CURRTIME'].max()}")
        
        # 날짜별 데이터 개수
        date_counts = df['CURRTIME'].dt.date.value_counts().sort_index()
        print(f"\n📊 날짜별 데이터 개수:")
        for date, count in date_counts.items():
            print(f"   {date}: {count}개")
            
    except Exception as e:
        print(f"\n⚠️ 날짜 형식 자동 파싱 실패: {e}")
        print("   문자열로 처리합니다.")
        
        # 문자열로 날짜 추출 시도
        df['date_only'] = df['CURRTIME'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else 'unknown')
        date_counts = df['date_only'].value_counts().sort_index()
        
        print(f"\n📊 날짜별 데이터 개수 (문자열 기준):")
        for date, count in date_counts.items():
            print(f"   {date}: {count}개")


# 사용 예제
if __name__ == "__main__":
    # 입력 파일 경로 설정
    input_csv = "your_file.csv"  # 여기에 실제 파일 경로를 입력하세요
    
    # 옵션 1: CSV 분석만 수행
    print("=" * 50)
    print("1. CSV 파일 분석")
    print("=" * 50)
    analyze_csv_dates(input_csv)
    
    # 옵션 2: 날짜별로 파일 분리
    print("\n" + "=" * 50)
    print("2. 날짜별 파일 분리")
    print("=" * 50)
    split_csv_by_date(input_csv, output_folder='split_by_date')
    
    # 옵션 3: 특정 날짜 형식이 있는 경우 (예: YYYYMMDD)
    # 필요시 아래 코드를 사용하여 날짜 형식을 명시할 수 있습니다
    """
    df = pd.read_csv(input_csv)
    # 특정 형식으로 파싱 (예: 20240101 형식)
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d')
    """
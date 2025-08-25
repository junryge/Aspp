import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def process_semiconductor_data():
    """OUTPUT_BY_DATE 폴더의 CSV 파일들을 통합하고 전처리"""
    
    print("🚀 반도체 물류 데이터 전처리 시작")
    print("=" * 60)
    
    # 필요한 컬럼들
    required_columns = [
        "CURRTIME", "TOTALCNT", "M14AM10A", "M10AM14A", "M14AM10ASUM",
        "M14AM14B", "M14BM14A", "M14AM14BSUM", "M14AM16", "M16M14A", 
        "M14AM16SUM", "TIME"
    ]
    
    # OUTPUT_BY_DATE 폴더에서 모든 CSV 파일 찾기
    folder_path = "OUTPUT_BY_DATE"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files.sort()  # 날짜순 정렬
    
    print(f"📂 {len(csv_files)}개의 CSV 파일 발견")
    
    # 모든 데이터를 저장할 리스트
    all_data = []
    processed_files = 0
    
    # 각 파일 처리
    for file_path in tqdm(csv_files, desc="파일 읽기"):
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 필요한 컬럼이 있는지 확인
            if all(col in df.columns for col in required_columns):
                # 필요한 컬럼만 선택
                df = df[required_columns]
                
                # 파일명에서 날짜 추출 (예: 20240101.csv → 20240101)
                file_date = os.path.basename(file_path).replace('.csv', '').replace('.CSV', '')
                df['FILE_DATE'] = file_date
                
                all_data.append(df)
                processed_files += 1
            else:
                missing = [col for col in required_columns if col not in df.columns]
                print(f"\n⚠️ {os.path.basename(file_path)}: 누락된 컬럼 {missing}")
                
        except Exception as e:
            print(f"\n❌ {os.path.basename(file_path)} 읽기 오류: {e}")
            continue
    
    print(f"\n✅ {processed_files}개 파일 성공적으로 읽음")
    
    if not all_data:
        print("❌ 처리할 데이터가 없습니다!")
        return
    
    # 모든 데이터 통합
    print("\n📊 데이터 통합 중...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 총 {len(combined_df):,}개 행 통합 완료")
    
    # 중복 제거 (유사 패턴 그룹화)
    print("\n🔍 중복 패턴 제거 중...")
    original_size = len(combined_df)
    
    # 패턴 정의를 위한 반올림
    combined_df['PATTERN_TOTALCNT'] = (combined_df['TOTALCNT'] // 50) * 50
    combined_df['PATTERN_M14AM14B'] = (combined_df['M14AM14B'] // 20) * 20
    combined_df['PATTERN_M14AM10A'] = (combined_df['M14AM10A'] // 10) * 10
    combined_df['PATTERN_M14AM16'] = (combined_df['M14AM16'] // 20) * 20
    combined_df['PATTERN_M14AM14BSUM'] = (combined_df['M14AM14BSUM'] // 50) * 50
    
    # 패턴 기반 중복 제거
    pattern_cols = ['PATTERN_TOTALCNT', 'PATTERN_M14AM14B', 'PATTERN_M14AM10A', 
                   'PATTERN_M14AM16', 'PATTERN_M14AM14BSUM']
    
    # 각 패턴 그룹에서 첫 번째 행만 유지
    combined_df = combined_df.drop_duplicates(subset=pattern_cols, keep='first')
    
    # 패턴 컬럼 제거
    combined_df = combined_df.drop(columns=pattern_cols)
    
    removed_count = original_size - len(combined_df)
    print(f"✅ {removed_count:,}개 중복 제거 ({removed_count/original_size*100:.1f}%)")
    
    # 데이터 정제
    print("\n🧹 데이터 정제 중...")
    
    # 1. 결측값 제거
    before_na = len(combined_df)
    combined_df = combined_df.dropna()
    after_na = len(combined_df)
    if before_na > after_na:
        print(f"   - 결측값 {before_na - after_na:,}개 행 제거")
    
    # 2. 음수값 제거 (센서 데이터는 음수가 될 수 없음)
    numeric_cols = ["TOTALCNT", "M14AM10A", "M14AM14B", "M14AM16", 
                   "M14AM14BSUM", "M14AM10ASUM", "M14AM16SUM"]
    
    for col in numeric_cols:
        if col in combined_df.columns:
            before = len(combined_df)
            combined_df = combined_df[combined_df[col] >= 0]
            after = len(combined_df)
            if before > after:
                print(f"   - {col} 음수값 {before - after:,}개 행 제거")
    
    # 3. 극단적 이상값 제거 (각 컬럼의 상위 0.1%)
    for col in ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM']:
        if col in combined_df.columns:
            q999 = combined_df[col].quantile(0.999)
            before = len(combined_df)
            combined_df = combined_df[combined_df[col] <= q999]
            after = len(combined_df)
            if before > after:
                print(f"   - {col} 이상값(>{q999:.0f}) {before - after:,}개 행 제거")
    
    # FILE_DATE 컬럼 제거 (최종 파일에는 불필요)
    if 'FILE_DATE' in combined_df.columns:
        combined_df = combined_df.drop('FILE_DATE', axis=1)
    
    # 인덱스 재설정
    combined_df = combined_df.reset_index(drop=True)
    
    # 최종 데이터 저장
    output_path = "semiconductor_consolidated_data.csv"
    print(f"\n💾 데이터 저장 중: {output_path}")
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 데이터 전처리 완료!")
    print(f"📁 출력 파일: {output_path}")
    print(f"📏 최종 데이터: {len(combined_df):,}개 행")
    print(f"📐 컬럼 수: {len(combined_df.columns)}개")
    print(f"💾 파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # 주요 통계
    print("\n📈 주요 통계:")
    print(f"- TOTALCNT 평균: {combined_df['TOTALCNT'].mean():.0f}")
    print(f"- TOTALCNT 범위: {combined_df['TOTALCNT'].min():.0f} ~ {combined_df['TOTALCNT'].max():.0f}")
    print(f"- M14AM14B 평균: {combined_df['M14AM14B'].mean():.0f}")
    print(f"- M14AM14B 최대: {combined_df['M14AM14B'].max():.0f}")
    
    # 황금 패턴 비율
    golden_pattern = (combined_df['M14AM14B'] > 300) & (combined_df['M14AM10A'] < 80)
    golden_ratio = golden_pattern.sum() / len(combined_df) * 100
    print(f"- 황금 패턴(M14AM14B>300 & M14AM10A<80) 비율: {golden_ratio:.2f}%")
    
    return combined_df

if __name__ == "__main__":
    # 실행
    process_semiconductor_data()
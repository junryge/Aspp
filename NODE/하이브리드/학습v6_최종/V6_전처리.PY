import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataConsolidator:
    """반도체 물류 데이터 통합 및 중복 제거"""
    
    def __init__(self):
        # 모든 컬럼 유지
        self.all_columns = [
            "CURRTIME", "TOTALCNT", "M14AM10A", "M10AM14A", "M14AM10ASUM",
            "M14AM14B", "M14BM14A", "M14AM14BSUM", "M14AM16", "M16M14A", 
            "M14AM16SUM", "TIME"
        ]
        self.pattern_columns = [
            "TOTALCNT", "M14AM10A", "M14AM14B", "M14AM16", "M14AM14BSUM"
        ]
        self.unique_patterns = set()
        
    def load_all_csv_files(self, folder_path):
        """OUTPUT_BY_DATE 폴더에서 모든 CSV 파일 로드 및 통합"""
        print("📂 데이터 파일 로드 중...")
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        all_files.sort()
        
        all_data = []
        file_count = 0
        
        for file in tqdm(all_files, desc="파일 읽기"):
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                
                # 필요한 컬럼이 있는지 확인
                missing_cols = [col for col in self.all_columns if col not in df.columns]
                if missing_cols:
                    print(f"⚠️ {os.path.basename(file)}: 누락된 컬럼 {missing_cols}")
                    continue
                
                # 파일 정보 추가
                df['SOURCE_FILE'] = os.path.basename(file)
                all_data.append(df)
                file_count += 1
                
            except Exception as e:
                print(f"❌ {os.path.basename(file)} 읽기 실패: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✅ 총 {len(combined_df):,}개 행 로드 완료 ({file_count}개 파일)")
            return combined_df
        else:
            raise ValueError("데이터를 찾을 수 없습니다.")
    
    def create_pattern_hash(self, row):
        """패턴의 고유 해시 생성 (중복 검사용)"""
        # 주요 패턴 컬럼값을 반올림하여 패턴 생성
        pattern_values = []
        
        # TOTALCNT는 50 단위로 반올림
        pattern_values.append(round(row['TOTALCNT'] / 50) * 50)
        
        # M14AM14B는 20 단위로 반올림
        pattern_values.append(round(row['M14AM14B'] / 20) * 20)
        
        # M14AM10A는 10 단위로 반올림
        pattern_values.append(round(row['M14AM10A'] / 10) * 10)
        
        # M14AM16는 20 단위로 반올림
        pattern_values.append(round(row['M14AM16'] / 20) * 20)
        
        # M14AM14BSUM는 50 단위로 반올림
        pattern_values.append(round(row['M14AM14BSUM'] / 50) * 50)
        
        return tuple(pattern_values)
    
    def remove_duplicates_smart(self, df):
        """스마트한 중복 제거 - 유사 패턴 제거"""
        print("\n🔍 중복 및 유사 패턴 제거 중...")
        
        original_size = len(df)
        
        # 결과를 저장할 리스트
        filtered_rows = []
        seen_patterns = set()
        
        # 진행 상황 표시
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="패턴 분석"):
            # 패턴 해시 생성
            pattern_hash = self.create_pattern_hash(row)
            
            # 새로운 패턴인 경우만 추가
            if pattern_hash not in seen_patterns:
                seen_patterns.add(pattern_hash)
                filtered_rows.append(row)
        
        # 필터링된 데이터프레임 생성
        filtered_df = pd.DataFrame(filtered_rows)
        
        # SOURCE_FILE 컬럼 제거 (최종 파일에는 불필요)
        if 'SOURCE_FILE' in filtered_df.columns:
            filtered_df = filtered_df.drop('SOURCE_FILE', axis=1)
        
        removed_count = original_size - len(filtered_df)
        print(f"✅ 중복 제거 완료: {removed_count:,}개 제거 ({removed_count/original_size*100:.1f}%)")
        print(f"   원본: {original_size:,}개 → 필터링: {len(filtered_df):,}개")
        
        return filtered_df
    
    def validate_data(self, df):
        """데이터 유효성 검사"""
        print("\n🔍 데이터 유효성 검사 중...")
        
        # 1. 결측값 확인
        missing_counts = df[self.all_columns].isnull().sum()
        if missing_counts.any():
            print("⚠️ 결측값 발견:")
            print(missing_counts[missing_counts > 0])
            
            # 결측값 제거
            df = df.dropna(subset=self.all_columns)
            print(f"   → 결측값 있는 행 제거 후: {len(df):,}개")
        
        # 2. 이상값 확인 (음수 값)
        numeric_cols = ["TOTALCNT", "M14AM10A", "M14AM14B", "M14AM16", 
                       "M14AM14BSUM", "M14AM10ASUM", "M14AM16SUM"]
        
        for col in numeric_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"⚠️ {col}에 음수값 {negative_count}개 발견 → 제거")
                    df = df[df[col] >= 0]
        
        # 3. 극단적 이상값 제거 (상위 0.1% 제거)
        for col in ['TOTALCNT', 'M14AM14B']:
            if col in df.columns:
                q999 = df[col].quantile(0.999)
                outliers = df[col] > q999
                if outliers.sum() > 0:
                    print(f"⚠️ {col} 극단 이상값 {outliers.sum()}개 발견 (>{q999:.0f}) → 제거")
                    df = df[~outliers]
        
        print(f"✅ 유효성 검사 완료: 최종 {len(df):,}개 행")
        return df
    
    def add_useful_features(self, df):
        """분석에 유용한 추가 특징 생성"""
        print("\n🔧 추가 특징 생성 중...")
        
        # 1. 비율 특징 (안전하게 0으로 나누기 방지)
        df['M14B_M14A_RATIO'] = np.where(
            df['M14AM10A'] > 0, 
            df['M14AM14B'] / df['M14AM10A'], 
            0
        )
        
        df['M16_M14A_RATIO'] = np.where(
            df['M14AM10A'] > 0, 
            df['M14AM16'] / df['M14AM10A'], 
            0
        )
        
        # 2. 변화율 특징 (shift 사용)
        df['TOTALCNT_CHANGE'] = df['TOTALCNT'].diff().fillna(0)
        df['M14AM14B_CHANGE'] = df['M14AM14B'].diff().fillna(0)
        
        # 3. 이동 평균 (10분)
        df['TOTALCNT_MA10'] = df['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        df['M14AM14B_MA10'] = df['M14AM14B'].rolling(window=10, min_periods=1).mean()
        
        print("✅ 추가 특징 생성 완료")
        return df
    
    def save_consolidated_data(self, df, output_path):
        """통합된 데이터 저장"""
        print(f"\n💾 데이터 저장 중: {output_path}")
        
        # 컬럼 순서 정리 (원본 컬럼 먼저, 추가 특징은 뒤에)
        original_cols = [col for col in self.all_columns if col in df.columns]
        additional_cols = [col for col in df.columns if col not in original_cols]
        
        df = df[original_cols + additional_cols]
        
        # CSV 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 요약 정보 출력
        print(f"\n📊 최종 데이터 요약:")
        print(f"- 총 행 수: {len(df):,}개")
        print(f"- 총 컬럼 수: {len(df.columns)}개")
        print(f"- 파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        print(f"\n주요 통계:")
        print(f"- TOTALCNT 평균: {df['TOTALCNT'].mean():.0f}")
        print(f"- TOTALCNT 범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
        print(f"- M14AM14B 평균: {df['M14AM14B'].mean():.0f}")
        print(f"- M14AM14B > 300인 비율: {(df['M14AM14B'] > 300).mean()*100:.1f}%")

def main():
    """메인 실행 함수"""
    print("🚀 반도체 물류 데이터 통합 시작")
    print("=" * 60)
    
    # 설정
    folder_path = "OUTPUT_BY_DATE"  # 데이터 폴더 경로
    output_path = "semiconductor_consolidated_data.csv"  # 출력 파일명
    
    # 데이터 통합기 초기화
    consolidator = DataConsolidator()
    
    try:
        # 1. 모든 CSV 파일 로드
        df = consolidator.load_all_csv_files(folder_path)
        
        # 2. 중복 패턴 제거
        df = consolidator.remove_duplicates_smart(df)
        
        # 3. 데이터 유효성 검사
        df = consolidator.validate_data(df)
        
        # 4. 추가 특징 생성 (선택사항)
        df = consolidator.add_useful_features(df)
        
        # 5. 통합 데이터 저장
        consolidator.save_consolidated_data(df, output_path)
        
        print("\n" + "=" * 60)
        print("🎉 데이터 통합 완료!")
        print(f"📁 저장 위치: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
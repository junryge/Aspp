#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
큐 변화량 CSV 필터링 및 분석 도구
================================================================================
RESULT_2025_with_queue_analysis.csv 파일의 필터링과 통계 분석
"""

import pandas as pd
import numpy as np
import chardet
import os

def read_csv_safely(filepath):
    """한글 CSV 파일 안전하게 읽기"""
    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return None
    
    # 다양한 인코딩 시도
    encodings = ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✅ 파일 읽기 성공 (인코딩: {encoding})")
            return df
        except UnicodeDecodeError:
            continue
    
    # 모든 인코딩 실패시 chardet 사용
    try:
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
            print(f"감지된 인코딩: {encoding}")
            df = pd.read_csv(filepath, encoding=encoding)
            return df
    except:
        print("❌ 파일을 읽을 수 없습니다.")
        return None

def filter_by_range_and_change(df, current_range='200-250', change_range='0~20', time_horizon=5):
    """
    특정 구간과 변화량으로 필터링
    
    Parameters:
    - current_range: 현재값 구간 (예: '200-250', '250-300', '300-350')
    - change_range: 변화량 구간 (예: '0~20', '21~40', '41~60', '61~80', '81~100', '100+')
    - time_horizon: 시간대 (5, 10, 20, 30)
    """
    # 컬럼명
    range_col = '현재값구간'
    change_col = f'변화구간_{time_horizon}분'
    direction_col = f'변화방향_{time_horizon}분'
    
    # 필터링
    filtered = df[(df[range_col] == current_range) & 
                  (df[change_col] == change_range)]
    
    # 상승/하락 분리
    상승 = filtered[filtered[direction_col] == '상승']
    하락 = filtered[filtered[direction_col] == '하락']
    유지 = filtered[filtered[direction_col] == '유지']
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"📊 {current_range} 구간, {time_horizon}분 변화량 {change_range}")
    print(f"{'='*60}")
    print(f"- 전체: {len(filtered)}개")
    
    if len(filtered) > 0:
        print(f"- 상승: {len(상승)}개 ({len(상승)/len(filtered)*100:.1f}%)")
        print(f"- 하락: {len(하락)}개 ({len(하락)/len(filtered)*100:.1f}%)")
        print(f"- 유지: {len(유지)}개 ({len(유지)/len(filtered)*100:.1f}%)")
    
    return filtered, 상승, 하락, 유지

def analyze_all_combinations(df):
    """모든 구간과 변화량 조합 분석"""
    
    # 분석할 구간들
    ranges = ['200-250', '250-300', '300-350']
    change_bins = ['0~20', '21~40', '41~60', '61~80', '81~100']
    time_horizons = [5, 10, 20, 30]
    
    # 결과 저장할 리스트
    all_results = []
    
    for time in time_horizons:
        print(f"\n\n🕐 {time}분 후 변화량 분석")
        print("="*80)
        
        for current_range in ranges:
            for change_range in change_bins:
                filtered, 상승, 하락, 유지 = filter_by_range_and_change(
                    df, current_range, change_range, time
                )
                
                # 결과 저장
                if len(filtered) > 0:
                    all_results.append({
                        '시간대(분)': time,
                        '현재값구간': current_range,
                        '변화량구간': change_range,
                        '전체케이스': len(filtered),
                        '상승': len(상승),
                        '하락': len(하락),
                        '유지': len(유지),
                        '상승비율(%)': round(len(상승)/len(filtered)*100, 1),
                        '하락비율(%)': round(len(하락)/len(filtered)*100, 1)
                    })
    
    # DataFrame으로 변환
    summary_df = pd.DataFrame(all_results)
    
    # 엑셀로 저장
    output_file = 'queue_analysis_summary.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='변화량분석통계', index=False)
        
        # 열 너비 자동 조정
        worksheet = writer.sheets['변화량분석통계']
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    print(f"\n\n💾 분석 결과가 '{output_file}'에 저장되었습니다.")
    return summary_df

def analyze_stable_cases(df):
    """안정 구간 케이스 분석"""
    
    stable_col = '안정구간_과거30분<300_향후10분≤300'
    
    if stable_col in df.columns:
        stable_cases = df[df[stable_col] == 'O']
        print(f"\n\n🏷️ 안정 구간 분석")
        print("="*60)
        print(f"- 전체 데이터: {len(df)}개")
        print(f"- 안정 구간 케이스: {len(stable_cases)}개")
        print(f"- 비율: {len(stable_cases)/len(df)*100:.2f}%")
        
        # 안정 구간 케이스만 별도 저장
        if len(stable_cases) > 0:
            stable_cases.to_csv('stable_cases.csv', index=False, encoding='utf-8-sig')
            print(f"- 안정 케이스가 'stable_cases.csv'에 저장되었습니다.")
    else:
        print(f"⚠️ '{stable_col}' 컬럼을 찾을 수 없습니다.")

def main():
    """메인 실행 함수"""
    
    print("="*80)
    print("📊 큐 변화량 CSV 필터링 및 분석 시작")
    print("="*80)
    
    # 1. CSV 파일 읽기
    filepath = 'RESULT_2025_with_queue_analysis.csv'
    df = read_csv_safely(filepath)
    
    if df is None:
        return
    
    print(f"\n✅ 데이터 로드 완료: {len(df)} 행")
    print(f"컬럼 수: {len(df.columns)}")
    
    # 2. 특정 필터링 예제 (200-250 구간, 5분 변화 0~20)
    print("\n\n[예제] 200-250 구간, 5분 변화량 0~20 필터링")
    filtered, 상승, 하락, 유지 = filter_by_range_and_change(df, '200-250', '0~20', 5)
    
    # 필터링된 데이터 저장
    if len(filtered) > 0:
        filtered.to_csv('filtered_200_250_change_0_20.csv', index=False, encoding='utf-8-sig')
        print(f"💾 필터링 결과가 'filtered_200_250_change_0_20.csv'에 저장되었습니다.")
    
    # 3. 전체 조합 분석
    analyze_all = input("\n전체 구간과 변화량 조합을 분석하시겠습니까? (y/n): ")
    if analyze_all.lower() == 'y':
        summary_df = analyze_all_combinations(df)
        
        # 요약 통계 출력
        print("\n\n📈 요약 통계")
        print("="*60)
        for time in [5, 10, 20, 30]:
            time_data = summary_df[summary_df['시간대(분)'] == time]
            if len(time_data) > 0:
                print(f"\n{time}분 후:")
                print(f"  - 평균 상승 비율: {time_data['상승비율(%)'].mean():.1f}%")
                print(f"  - 평균 하락 비율: {time_data['하락비율(%)'].mean():.1f}%")
    
    # 4. 안정 구간 분석
    analyze_stable_cases(df)
    
    print("\n\n✅ 모든 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()
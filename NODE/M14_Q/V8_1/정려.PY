import pandas as pd
import csv

def fix_dt_columns(input_file, output_file):
    """
    CSV 파일에서 DT 컬럼을 A열 기준으로 맞추고 누락된 데이터는 NULL 처리
    
    Parameters:
    - input_file: 입력 CSV 파일 경로 (예: 'A.csv')
    - output_file: 출력 CSV 파일 경로
    """
    
    print(f"파일 읽기 시작: {input_file}")
    
    # CSV 파일 읽기
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    
    if len(all_rows) < 2:
        print("데이터가 충분하지 않습니다.")
        return
    
    # 헤더 처리
    headers = all_rows[0]
    data_rows = all_rows[1:]
    
    print(f"총 컬럼 수: {len(headers)}")
    print(f"총 데이터 행 수: {len(data_rows)}")
    print(f"헤더 샘플: {headers[:10]}")
    
    # DT 인덱스 찾기 (DT로 시작하는 모든 컬럼)
    dt_indices = [i for i, h in enumerate(headers) if h == 'DT']
    print(f"\nDT 컬럼 위치: {dt_indices}")
    print(f"DT 컬럼 개수: {len(dt_indices)}")
    
    # 수정된 데이터 저장
    fixed_rows = []
    error_count = 0
    
    for row_idx, row in enumerate(data_rows, start=1):
        if not row or len(row) < 1:
            continue
        
        # A열(첫 번째 DT) 기준 날짜
        reference_dt = row[0]
        
        # 새 행 생성 (원본 복사)
        new_row = row.copy()
        
        # 모든 DT 컬럼을 A열 기준으로 수정
        for dt_idx in dt_indices:
            if dt_idx < len(new_row):
                current_dt = new_row[dt_idx]
                
                # DT가 기준과 다르면 수정
                if current_dt != reference_dt:
                    if row_idx <= 10:  # 처음 10개만 로그 출력
                        print(f"행 {row_idx}: 컬럼 {dt_idx} 수정 ({current_dt} -> {reference_dt})")
                    new_row[dt_idx] = reference_dt
                    error_count += 1
                    
                    # DT 다음 값 컬럼이 있으면 NULL 처리
                    if dt_idx + 1 < len(new_row):
                        if row_idx <= 10:
                            print(f"  -> 값 컬럼 {dt_idx+1} NULL 처리")
                        new_row[dt_idx + 1] = 'NULL'
            else:
                # DT 컬럼이 없으면 추가
                while len(new_row) <= dt_idx:
                    new_row.append('')
                new_row[dt_idx] = reference_dt
                
                # 다음 값 컬럼도 NULL로
                if dt_idx + 1 >= len(new_row):
                    new_row.append('NULL')
                else:
                    new_row[dt_idx + 1] = 'NULL'
        
        fixed_rows.append(new_row)
        
        if row_idx % 5000 == 0:
            print(f"처리 중... {row_idx}/{len(data_rows)} 행")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(fixed_rows)
    
    print(f"\n{'='*60}")
    print(f"처리 완료!")
    print(f"총 처리 행 수: {len(fixed_rows)}")
    print(f"DT 불일치 발견 및 수정: {error_count}건")
    print(f"결과 파일: {output_file}")
    print(f"{'='*60}")


def analyze_before_fix(input_file):
    """
    수정 전 데이터 분석
    """
    print(f"\n{'='*60}")
    print("데이터 분석 시작")
    print(f"{'='*60}\n")
    
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    
    headers = all_rows[0]
    data_rows = all_rows[1:]
    
    dt_indices = [i for i, h in enumerate(headers) if h == 'DT']
    
    print(f"파일: {input_file}")
    print(f"총 행 수: {len(data_rows)}")
    print(f"DT 컬럼 개수: {len(dt_indices)}")
    print(f"DT 컬럼 위치: {dt_indices[:10]}..." if len(dt_indices) > 10 else f"DT 컬럼 위치: {dt_indices}")
    
    mismatch_count = 0
    sample_mismatches = []
    
    for row_idx, row in enumerate(data_rows[:1000], start=1):  # 처음 1000개만 체크
        if not row or len(row) < 1:
            continue
        
        reference_dt = row[0]
        row_has_mismatch = False
        
        for dt_idx in dt_indices:
            if dt_idx < len(row):
                if row[dt_idx] != reference_dt:
                    row_has_mismatch = True
                    if len(sample_mismatches) < 5:
                        sample_mismatches.append({
                            'row': row_idx,
                            'reference': reference_dt,
                            'column': dt_idx,
                            'value': row[dt_idx]
                        })
        
        if row_has_mismatch:
            mismatch_count += 1
    
    print(f"\n처음 1000개 행 중 불일치 발견: {mismatch_count}건")
    
    if sample_mismatches:
        print(f"\n불일치 샘플 (처음 5개):")
        for m in sample_mismatches:
            print(f"  행 {m['row']}: 컬럼 {m['column']} - 기준={m['reference']}, 실제={m['value']}")


if __name__ == "__main__":
    # 파일명 설정
    input_file = "A.csv"  # 입력 파일
    output_file = "A_fixed.csv"  # 출력 파일
    
    # 분석 실행
    analyze_before_fix(input_file)
    
    # 데이터 수정
    print(f"\n{'='*60}")
    print("데이터 수정 시작")
    print(f"{'='*60}")
    
    fix_dt_columns(input_file, output_file)
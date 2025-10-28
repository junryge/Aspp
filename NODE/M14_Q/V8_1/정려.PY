import csv

def fix_dt_data_pairs(input_file, output_file):
    """
    DT-데이터 쌍 구조의 CSV 파일에서 A열 기준으로 모든 DT를 맞추고 누락된 데이터는 NULL 처리
    
    구조: DT(A열), 데이터(B열), DT(C열), 데이터(D열), ..., DT(O열), 데이터(P열)
    
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
    print(f"헤더: {headers}")
    
    # DT 인덱스 찾기 (짝수 인덱스: 0, 2, 4, 6, 8...)
    dt_indices = [i for i in range(0, len(headers), 2)]
    data_indices = [i for i in range(1, len(headers), 2)]
    
    print(f"\nDT 컬럼 위치 (짝수): {dt_indices}")
    print(f"데이터 컬럼 위치 (홀수): {data_indices}")
    print(f"DT-데이터 쌍 개수: {len(dt_indices)}")
    
    # 수정된 데이터 저장
    fixed_rows = []
    total_mismatch = 0
    null_count = 0
    
    for row_idx, row in enumerate(data_rows, start=1):
        if not row or len(row) < 1:
            continue
        
        # A열(첫 번째 DT) 기준 날짜
        reference_dt = row[0] if len(row) > 0 else ''
        
        # 새 행 생성 (원본 복사)
        new_row = row.copy()
        
        # 행이 짧으면 필요한 만큼 확장
        while len(new_row) < len(headers):
            new_row.append('')
        
        # 각 DT-데이터 쌍 처리
        for pair_idx, dt_idx in enumerate(dt_indices):
            data_idx = dt_idx + 1  # 데이터는 DT 다음 열
            
            if dt_idx == 0:
                # A열은 기준이므로 건너뜀
                continue
            
            # 현재 DT 값 확인
            current_dt = new_row[dt_idx] if dt_idx < len(new_row) else ''
            
            # DT가 기준과 다르거나 비어있으면 수정
            if current_dt != reference_dt:
                if row_idx <= 10:  # 처음 10개만 로그 출력
                    print(f"행 {row_idx}: 쌍 {pair_idx} - DT 불일치")
                    print(f"  컬럼 {dt_idx}({headers[dt_idx]}): '{current_dt}' -> '{reference_dt}'")
                    if data_idx < len(headers):
                        print(f"  컬럼 {data_idx}({headers[data_idx]}): 데이터 -> NULL")
                
                # DT를 기준값으로 변경
                new_row[dt_idx] = reference_dt
                total_mismatch += 1
                
                # 해당 데이터 값을 NULL로 처리
                if data_idx < len(new_row):
                    new_row[data_idx] = 'NULL'
                    null_count += 1
        
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
    print(f"DT 불일치 발견 및 수정: {total_mismatch}건")
    print(f"NULL 처리된 데이터: {null_count}건")
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
    
    # DT는 짝수 인덱스
    dt_indices = [i for i in range(0, len(headers), 2)]
    
    print(f"파일: {input_file}")
    print(f"총 행 수: {len(data_rows)}")
    print(f"총 컬럼 수: {len(headers)}")
    print(f"DT-데이터 쌍 개수: {len(dt_indices)}")
    
    mismatch_count = 0
    sample_mismatches = []
    
    # 처음 1000개 행만 체크
    check_rows = min(1000, len(data_rows))
    
    for row_idx, row in enumerate(data_rows[:check_rows], start=1):
        if not row or len(row) < 1:
            continue
        
        reference_dt = row[0]
        row_has_mismatch = False
        
        for dt_idx in dt_indices[1:]:  # A열(0번 인덱스) 제외
            if dt_idx < len(row):
                current_dt = row[dt_idx]
                if current_dt != reference_dt:
                    row_has_mismatch = True
                    if len(sample_mismatches) < 5:
                        sample_mismatches.append({
                            'row': row_idx,
                            'column': dt_idx,
                            'reference': reference_dt,
                            'value': current_dt
                        })
        
        if row_has_mismatch:
            mismatch_count += 1
    
    print(f"\n처음 {check_rows}개 행 중 불일치 발견: {mismatch_count}건")
    print(f"불일치 비율: {mismatch_count/check_rows*100:.1f}%")
    
    if sample_mismatches:
        print(f"\n불일치 샘플 (처음 5개):")
        for m in sample_mismatches:
            col_name = headers[m['column']] if m['column'] < len(headers) else f"컬럼{m['column']}"
            print(f"  행 {m['row']}: {col_name} (컬럼 {m['column']})")
            print(f"    기준={m['reference']}, 실제={m['value']}")


if __name__ == "__main__":
    # 파일명 설정
    input_file = "A.csv"  # 입력 파일
    output_file = "A_fixed.csv"  # 출력 파일
    
    # 분석 실행
    analyze_before_fix(input_file)
    
    # 데이터 수정
    print(f"\n{'='*60}")
    print("데이터 수정 시작")
    print(f"{'='*60}\n")
    
    fix_dt_data_pairs(input_file, output_file)
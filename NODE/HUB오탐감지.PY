import pandas as pd

# 파일 읽기
df = pd.read_csv('H7__1_')  # 파일명 수정

# 1. 사전감지OK_NG 컬럼 (실제값>=350 & 현재값<=350)
df['사전감지OK_NG'] = ''
surge_condition = (df['실제값'] >= 350) & (df['현재값'] <= 350)
df.loc[surge_condition, '사전감지OK_NG'] = 'Y'

# 2. 사전감지결과 컬럼 (OK / 사전감지 / NG)
df['사전감지결과'] = ''
ok_condition = surge_condition & (df['예측값_MAX'] >= 350)
df.loc[ok_condition, '사전감지결과'] = 'OK'

ng_condition = surge_condition & (df['예측값_MAX'] < 350)
ng_indices = df[ng_condition].index.tolist()

for idx in ng_indices:
    found = False
    for back in range(1, 6):  # 1~5분전 확인
        prev_idx = idx - back
        if prev_idx >= 0:
            prev_pred = df.loc[prev_idx, '예측값_MAX']
            if prev_pred >= 350:
                df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                found = True
                break
    if not found:
        df.loc[idx, '사전감지결과'] = 'NG'

# 3. 오탐/사전감지 컬럼 (실제값<350 & 예측값_MAX>=350)
df['오탐/사전감지'] = ''
false_positive_condition = (df['실제값'] < 350) & (df['예측값_MAX'] >= 350)
fp_indices = df[false_positive_condition].index.tolist()

for idx in fp_indices:
    found = False
    for offset in range(-5, 6):  # 앞뒤 5분 확인
        if offset == 0:
            continue
        check_idx = idx + offset
        if 0 <= check_idx < len(df):
            if df.loc[check_idx, '실제값'] >= 350:
                df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                found = True
                break
    if not found:
        df.loc[idx, '오탐/사전감지'] = '오탐'

# 컬럼 순서 재배치 - 실제값 바로 뒤에 배치
cols = df.columns.tolist()
cols.remove('사전감지OK_NG')
cols.remove('사전감지결과')
cols.remove('오탐/사전감지')
actual_idx = cols.index('실제값')
new_cols = cols[:actual_idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[actual_idx+1:]
df = df[new_cols]

# 저장
df.to_csv('H7_결과_최종.csv', index=False, encoding='utf-8-sig')

# 집계 출력
total_surge = len(df[surge_condition])
ok_cnt = len(df[df['사전감지결과'] == 'OK'])
pre_cnt = len(df[df['사전감지결과'].str.contains('사전감지', na=False)])
ng_cnt = len(df[df['사전감지결과'] == 'NG'])

fp_total = len(fp_indices)
fp_pre = len(df[df['오탐/사전감지'] == '오탐사전감지'])
fp_real = len(df[df['오탐/사전감지'] == '오탐'])

print("=== 급증 감지 결과 ===")
print(f"급증 케이스: {total_surge}개")
print(f"OK: {ok_cnt}개 ({ok_cnt/total_surge*100:.1f}%)")
print(f"사전감지: {pre_cnt}개 ({pre_cnt/total_surge*100:.1f}%)")
print(f"NG: {ng_cnt}개 ({ng_cnt/total_surge*100:.1f}%)")
print(f"총 감지율: {(ok_cnt+pre_cnt)/total_surge*100:.1f}%")

print(f"\n=== 오탐 분석 ===")
print(f"오탐 후보: {fp_total}개")
print(f"오탐사전감지: {fp_pre}개 ({fp_pre/fp_total*100:.1f}%)")
print(f"오탐: {fp_real}개 ({fp_real/fp_total*100:.1f}%)")
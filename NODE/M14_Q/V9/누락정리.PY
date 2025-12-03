import pandas as pd

# === 파일 로딩 ===
df = pd.read_csv('merged_output.csv', encoding='utf-8-sig')

# === 누락 현황 출력 ===
print("=== 컬럼별 누락 현황 ===")
for col in df.columns:
    if col in ['CURRTIME', 'datetime']:
        continue
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"{col}: {missing}개 누락")
        # 누락된 CURRTIME 목록
        missing_times = df[df[col].isna()]['CURRTIME'].tolist()
        print(f"  → {missing_times[:5]}...")  # 앞에 5개만 미리보기

# === 누락 있는 행만 CSV로 저장 ===
any_missing = df.isna().any(axis=1)
df_missing = df[any_missing]
df_missing.to_csv('missing_check.csv', index=False, encoding='utf-8-sig')

print(f"\n누락 있는 행: {len(df_missing)}개")
print("저장: missing_check.csv")
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('1760913988503_pasted-content-1760913988499.txt', sep='\t')
df.columns = df.columns.str.strip()

# 실제 사전감지 케이스
actual_anomaly_indices = df[df['사전감지'] == '사전감지'].index.tolist()

# 조건으로 예측
actual_values = df['실제값'].values
predicted_anomaly = []

for i in range(30, len(df)):
    seq = actual_values[i-30:i]
    seq_max = seq.max()
    
    if seq_max >= 300:
        continue
    
    has_283_plus = np.any(seq >= 283)
    increase_rate = seq[-1] - seq[0]
    has_15_increase = increase_rate >= 15
    
    if has_283_plus and has_15_increase:
        is_actual_anomaly = i in actual_anomaly_indices
        if is_actual_anomaly:
            predicted_anomaly.append(i)

print("="*80)
print("✅ 정확히 감지된 15개 케이스 - 실제값 vs 예측값12")
print("="*80)
print()

correct_df = df.loc[predicted_anomaly]

for idx, row in correct_df.iterrows():
    실제값 = row['실제값']
    예측값12 = row['예측값12']
    오차 = 실제값 - 예측값12
    
    print(f"📍 {row['예측 수행 시점']}")
    print(f"   실제값:   {실제값:.0f} 🔴")
    print(f"   예측값12: {예측값12:.2f}")
    print(f"   오차:     {오차:.2f} ({'예측 낮음' if 오차 > 0 else '예측 높음'})")
    print()

print("="*80)
print("📊 통계 요약")
print("="*80)

실제값들 = correct_df['실제값'].values
예측값들 = correct_df['예측값12'].values
오차들 = 실제값들 - 예측값들

print(f"\n실제값 범위:   {실제값들.min():.0f} ~ {실제값들.max():.0f}")
print(f"예측값12 범위: {예측값들.min():.2f} ~ {예측값들.max():.2f}")
print(f"\n평균 실제값:   {실제값들.mean():.2f}")
print(f"평균 예측값12: {예측값들.mean():.2f}")
print(f"평균 오차:     {오차들.mean():.2f}")
print(f"평균 절대오차: {np.abs(오차들).mean():.2f}")

print(f"\n예측 낮음(under-prediction): {(오차들 > 0).sum()}개")
print(f"예측 높음(over-prediction):  {(오차들 < 0).sum()}개")

print(f"\n💡 예측값12가 실제값보다 평균 {오차들.mean():.2f}만큼 {'낮습니다' if 오차들.mean() > 0 else '높습니다'}")

# 결과를 CSV로 저장
result_df = correct_df[['예측 수행 시점', '실제시점', '실제값', '예측값12']].copy()
result_df['오차'] = result_df['실제값'] - result_df['예측값12']
result_df['오차율(%)'] = (result_df['오차'] / result_df['실제값'] * 100).round(2)
result_df.to_csv('정확감지_15개_실제값_예측값12.csv', index=False, encoding='utf-8-sig')

print("\n✅ 결과 저장: 정확감지_15개_실제값_예측값12.csv")
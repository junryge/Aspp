# 실제 모델 생성 (15분 예측 - n_estimators=160 + 최대 약화)
if use_gpu:
    model = xgb.XGBRegressor(
        n_estimators=160,           # 200 → 160 (트리 거의 절반!)
        max_depth=4,                # 6 → 4 (깊이 크게 감소)
        learning_rate=0.1,          # 0.05 → 0.1 (학습률 2배)
        subsample=0.6,              # 0.8 → 0.6 (샘플링 대폭 감소)
        colsample_bytree=0.6,       # 0.8 → 0.6 (컬럼 샘플링 대폭 감소)
        min_child_weight=8,         # 리프 노드 제약 강력
        gamma=0.5,                  # 분할 매우 어렵게
        reg_alpha=0.6,              # L1 정규화 강력
        reg_lambda=2.5,             # L2 정규화 강력
        random_state=42,
        tree_method='gpu_hist',
        gpu_id=0,
        predictor='gpu_predictor'
    )
else:
    model = xgb.XGBRegressor(
        n_estimators=160,           # 200 → 160 (트리 거의 절반!)
        max_depth=4,                # 6 → 4 (깊이 크게 감소)
        learning_rate=0.1,          # 0.05 → 0.1 (학습률 2배)
        subsample=0.6,              # 0.8 → 0.6 (샘플링 대폭 감소)
        colsample_bytree=0.6,       # 0.8 → 0.6 (컬럼 샘플링 대폭 감소)
        min_child_weight=8,         # 리프 노드 제약 강력
        gamma=0.5,                  # 분할 매우 어렵게
        reg_alpha=0.6,              # L1 정규화 강력
        reg_lambda=2.5,             # L2 정규화 강력
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
print("모델 학습 중 (15분 예측 - 최대 약화)...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
# 실제 모델 생성 (15분 예측 - n_estimators=100 + 극단적 약화)
if use_gpu:
    model = xgb.XGBRegressor(
        n_estimators=100,           # 160 → 100 (10분의 1/3 수준!)
        max_depth=3,                # 4 → 3 (매우 얕게)
        learning_rate=0.15,         # 0.1 → 0.15 (학습률 3배, 매우 거칠게)
        subsample=0.5,              # 0.6 → 0.5 (샘플 절반만)
        colsample_bytree=0.5,       # 0.6 → 0.5 (피처 절반만)
        min_child_weight=10,        # 8 → 10 (리프 노드 제약 최대)
        gamma=0.7,                  # 0.5 → 0.7 (분할 거의 불가능하게)
        reg_alpha=0.8,              # 0.6 → 0.8 (L1 정규화 최대)
        reg_lambda=3.0,             # 2.5 → 3.0 (L2 정규화 최대)
        random_state=42,
        tree_method='gpu_hist',
        gpu_id=0,
        predictor='gpu_predictor'
    )
else:
    model = xgb.XGBRegressor(
        n_estimators=100,           # 160 → 100 (10분의 1/3 수준!)
        max_depth=3,                # 4 → 3 (매우 얕게)
        learning_rate=0.15,         # 0.1 → 0.15 (학습률 3배, 매우 거칠게)
        subsample=0.5,              # 0.6 → 0.5 (샘플 절반만)
        colsample_bytree=0.5,       # 0.6 → 0.5 (피처 절반만)
        min_child_weight=10,        # 8 → 10 (리프 노드 제약 최대)
        gamma=0.7,                  # 0.5 → 0.7 (분할 거의 불가능하게)
        reg_alpha=0.8,              # 0.6 → 0.8 (L1 정규화 최대)
        reg_lambda=3.0,             # 2.5 → 3.0 (L2 정규화 최대)
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
print("모델 학습 중 (15분 예측 - 극단적 약화)...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
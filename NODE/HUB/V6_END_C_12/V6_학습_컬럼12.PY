# 실제 모델 생성 (15분 예측 - n_estimators=250 + 성능 하향)
if use_gpu:
    model = xgb.XGBRegressor(
        n_estimators=250,           # 300 → 250 (트리 개수 감소)
        max_depth=5,                # 6 → 5 (깊이 감소)
        learning_rate=0.08,         # 0.05 → 0.08 (학습률 증가 = 덜 섬세)
        subsample=0.65,             # 0.8 → 0.65 (샘플링 크게 감소)
        colsample_bytree=0.65,      # 0.8 → 0.65 (컬럼 샘플링 크게 감소)
        min_child_weight=6,         # 추가 (리프 노드 제약 강화)
        gamma=0.4,                  # 추가 (분할 임계값 증가)
        reg_alpha=0.4,              # 추가 (L1 정규화 강화)
        reg_lambda=2.0,             # 추가 (L2 정규화 강화)
        random_state=42,
        tree_method='gpu_hist',
        gpu_id=0,
        predictor='gpu_predictor'
    )
else:
    model = xgb.XGBRegressor(
        n_estimators=250,           # 300 → 250 (트리 개수 감소)
        max_depth=5,                # 6 → 5 (깊이 감소)
        learning_rate=0.08,         # 0.05 → 0.08 (학습률 증가 = 덜 섬세)
        subsample=0.65,             # 0.8 → 0.65 (샘플링 크게 감소)
        colsample_bytree=0.65,      # 0.8 → 0.65 (컬럼 샘플링 크게 감소)
        min_child_weight=6,         # 추가 (리프 노드 제약 강화)
        gamma=0.4,                  # 추가 (분할 임계값 증가)
        reg_alpha=0.4,              # 추가 (L1 정규화 강화)
        reg_lambda=2.0,             # 추가 (L2 정규화 강화)
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
print("모델 학습 중 (15분 예측 - 성능 하향)...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
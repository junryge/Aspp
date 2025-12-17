model = xgb.XGBRegressor(
    objective='reg:squarederror',
    
    # 빠른 학습
    learning_rate=0.1,         # 0.05 → 0.1 (2배)
    n_estimators=300,          # 500 → 300 (줄여도 됨)
    max_depth=6,               # 8 → 6 (과적합 방지)
    
    # 나머지 동일
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
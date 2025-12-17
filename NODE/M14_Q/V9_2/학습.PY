model = xgb.XGBRegressor(
    objective='reg:squarederror',
    
    # 천천히 세밀하게
    learning_rate=0.01,        # 낮게 유지
    n_estimators=1000,         # 트리 많이
    max_depth=6,               # 8 → 6 (얕게!)
    
    # 과적합 방지 강화
    subsample=0.6,             # 0.8 → 0.6
    colsample_bytree=0.6,      # 0.8 → 0.6
    min_child_weight=10,       # 3 → 10 (크게!)
    gamma=0.3,                 # 0.1 → 0.3
    reg_alpha=1.0,             # 0.1 → 1.0
    reg_lambda=3.0,            # 1.0 → 3.0
    
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
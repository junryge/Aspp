def train_model_with_checkpoint(model, model_name, X_train, y_train, X_val, y_val, 
                                epochs, batch_size, checkpoint_manager, 
                                start_epoch=0, initial_lr=0.0005):
    """체크포인트를 지원하는 모델 학습 함수"""
    
    # 옵티마이저 설정 (학습률 조정 가능)
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # 학습 이력 초기화
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    # 기존 이력이 있다면 로드
    state = checkpoint_manager.load_state()
    if state and model_name in state.get('model_histories', {}):
        history = state['model_histories'][model_name]
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"\n{model_name} - Epoch {epoch+1}/{epochs}")
            
            # 에폭별 학습
            epoch_history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=1
            )
            
            # 이력 저장
            for key in history.keys():
                if key in epoch_history.history:
                    history[key].append(epoch_history.history[key][0])
            
            # 현재 검증 손실
            current_val_loss = epoch_history.history['val_loss'][0]
            
            # 최고 성능 모델 저장
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights_path = checkpoint_manager.save_model_weights(model, model_name, epoch)
                patience_counter = 0
                logger.info(f"최고 성능 갱신! Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 조기 종료 확인
            if patience_counter >= patience:
                logger.info(f"조기 종료 - {patience}에폭 동안 개선 없음")
                break
            
            # 매 5에폭마다 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                # 현재 상태 저장
                current_state = checkpoint_manager.load_state() or {}
                current_state['current_model'] = model_name
                current_state['current_epoch'] = epoch + 1
                
                if 'model_histories' not in current_state:
                    current_state['model_histories'] = {}
                current_state['model_histories'][model_name] = history
                
                if 'completed_models' not in current_state:
                    current_state['completed_models'] = []
                    
                checkpoint_manager.save_state(current_state)
                checkpoint_manager.save_model_weights(model, f"{model_name}_checkpoint", epoch)
                
    except KeyboardInterrupt:
        logger.warning(f"\n{model_name} 학습이 사용자에 의해 중단되었습니다.")
        # 중단 시점 상태 저장
        current_state = checkpoint_manager.load_state() or {}
        current_state['current_model'] = model_name
        current_state['current_epoch'] = epoch
        current_state['interrupted'] = True
        
        if 'model_histories' not in current_state:
            current_state['model_histories'] = {}
        current_state['model_histories'][model_name] = history
        
        checkpoint_manager.save_state(current_state)
        checkpoint_manager.save_model_weights(model, f"{model_name}_interrupted", epoch)
        raise
        
    except Exception as e:
        logger.error(f"\n{model_name} 학습 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        # 오류 시점 상태 저장
        current_state = checkpoint_manager.load_state() or {}
        current_state['current_model'] = model_name
        current_state['current_epoch'] = epoch
        current_state['error'] = str(e)
        
        if 'model_histories' not in current_state:
            current_state['model_histories'] = {}
        current_state['model_histories'][model_name] = history
        
        checkpoint_manager.save_state(current_state)
        raise
    
    # 학습 완료 상태 저장
    current_state = checkpoint_manager.load_state() or {}
    if 'completed_models' not in current_state:
        current_state['completed_models'] = []
    if model_name not in current_state['completed_models']:
        current_state['completed_models'].append(model_name)
    
    # 'model_histories' 키가 없으면 생성 (수정된 부분)
    if 'model_histories' not in current_state:
        current_state['model_histories'] = {}
        
    current_state['model_histories'][model_name] = history
    checkpoint_manager.save_state(current_state)
    
    # 최고 성능 가중치 로드
    if 'best_weights_path' in locals():
        checkpoint_manager.load_model_weights(model, best_weights_path)
    
    return history

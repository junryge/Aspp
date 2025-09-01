import tensorflow as tf
import numpy as np
import os
import h5py

def check_saved_model_structure():
    """저장된 모델의 구조를 확인하는 함수"""
    
    print("="*60)
    print("🔍 저장된 모델 구조 확인")
    print("="*60)
    
    # 1. PatchTST 모델 확인
    if os.path.exists('./checkpoints/PatchTST_best.h5'):
        print("\n📊 PatchTST_best.h5 파일 분석:")
        try:
            # H5 파일 구조 확인
            with h5py.File('./checkpoints/PatchTST_best.h5', 'r') as f:
                print("\n파일 구조:")
                def print_structure(name, obj):
                    print(f"  {name}")
                f.visititems(print_structure)
                
                # 레이어 정보 확인
                if 'layer_names' in f.attrs:
                    layer_names = f.attrs['layer_names']
                    print(f"\n레이어 수: {len(layer_names)}")
                    print("레이어 이름들:")
                    for i, name in enumerate(layer_names):
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        print(f"  [{i}] {name}")
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
            
    # 2. PatchTST_PINN 모델 확인
    if os.path.exists('./checkpoints/PatchTST_PINN_best.h5'):
        print("\n📊 PatchTST_PINN_best.h5 파일 분석:")
        try:
            with h5py.File('./checkpoints/PatchTST_PINN_best.h5', 'r') as f:
                print("\n파일 구조:")
                def print_structure(name, obj):
                    print(f"  {name}")
                f.visititems(print_structure)
                
                if 'layer_names' in f.attrs:
                    layer_names = f.attrs['layer_names']
                    print(f"\n레이어 수: {len(layer_names)}")
                    print("레이어 이름들:")
                    for i, name in enumerate(layer_names):
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        print(f"  [{i}] {name}")
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
    
    # 3. 전체 모델로 로드 시도
    print("\n" + "="*60)
    print("🔧 전체 모델 로드 시도")
    print("="*60)
    
    # PatchTST 전체 모델 로드
    if os.path.exists('./checkpoints/PatchTST_best.h5'):
        print("\n📌 PatchTST 전체 모델 로드 시도:")
        try:
            # 커스텀 객체 없이 로드
            model = tf.keras.models.load_model('./checkpoints/PatchTST_best.h5', compile=False)
            print("✅ 성공! 모델 구조:")
            model.summary()
            
            # 입력 형태 확인
            print(f"\n입력 형태: {model.input_shape}")
            print(f"출력 형태: {model.output_shape}")
            
            return model
        except Exception as e:
            print(f"❌ 전체 모델 로드 실패: {e}")
            
            # 가중치만 있는지 확인
            try:
                # 간단한 모델로 가중치만 로드 시도
                print("\n가중치만 로드 시도...")
                # 여기서는 구조를 알아야 함
            except:
                pass
    
    # 4. 파일 정보 확인
    print("\n" + "="*60)
    print("📁 체크포인트 디렉토리 내용")
    print("="*60)
    
    checkpoint_dir = './checkpoints'
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        for file in files:
            filepath = os.path.join(checkpoint_dir, file)
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"  - {file}: {size:.2f} MB")
    
    # 5. 모델 구조 추정
    print("\n💡 모델 구조 힌트:")
    print("- 입력: (batch_size, 20, 39) - 20분 시퀀스, 39개 특성")
    print("- 출력: (batch_size, 1) - 10분 후 예측값")
    print("- PatchTST: 시퀀스 데이터만 사용")
    print("- PatchTST_PINN: 시퀀스 + 물리 데이터 (3개 특성)")

def load_model_with_custom_objects():
    """커스텀 객체를 포함한 모델 로드"""
    
    print("\n" + "="*60)
    print("🔧 커스텀 객체로 모델 로드 시도")
    print("="*60)
    
    # TransformerEncoderLayer 정의
    from tensorflow.keras import layers
    
    class TransformerEncoderLayer(layers.Layer):
        def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model
            self.num_heads = num_heads
            self.dff = dff
            self.dropout_rate = dropout_rate
            
            self.mha = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout_rate
            )
            self.ffn = tf.keras.Sequential([
                layers.Dense(dff, activation='relu'),
                layers.Dense(d_model)
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dropout2 = layers.Dropout(dropout_rate)
        
        def call(self, inputs, training=False):
            attn_output = self.mha(inputs, inputs, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)
            
            return out2
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'dff': self.dff,
                'dropout_rate': self.dropout_rate
            })
            return config
    
    # 커스텀 객체로 로드
    custom_objects = {
        'TransformerEncoderLayer': TransformerEncoderLayer
    }
    
    try:
        if os.path.exists('./checkpoints/PatchTST_best.h5'):
            print("\nPatchTST 모델 로드 중...")
            model = tf.keras.models.load_model(
                './checkpoints/PatchTST_best.h5',
                custom_objects=custom_objects,
                compile=False
            )
            print("✅ PatchTST 모델 로드 성공!")
            return model
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        
        # 다른 방법 시도
        print("\n다른 방법으로 시도...")
        try:
            # SavedModel 형식인지 확인
            model = tf.saved_model.load('./checkpoints/PatchTST_best.h5')
            print("✅ SavedModel 형식으로 로드 성공!")
            return model
        except:
            pass

def main():
    """메인 실행 함수"""
    print("🔍 HUBROOM 모델 구조 확인 도구")
    print("="*60)
    
    # 1. 저장된 모델 구조 확인
    check_saved_model_structure()
    
    # 2. 커스텀 객체로 로드 시도
    model = load_model_with_custom_objects()
    
    if model:
        print("\n✅ 모델 로드 성공!")
        print("이 정보를 바탕으로 평가 스크립트를 수정하세요.")
    else:
        print("\n💡 제안사항:")
        print("1. 학습 시 사용한 정확한 모델 구조가 필요합니다")
        print("2. 또는 학습 스크립트에서 모델을 다시 빌드하고 가중치만 로드해야 합니다")
        print("3. checkpoint 파일이 완전한 모델인지 가중치만인지 확인이 필요합니다")

if __name__ == "__main__":
    main()
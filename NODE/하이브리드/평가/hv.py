import os

model_dir = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\models_v5'

print("모델 디렉토리 파일 목록:")
for file in os.listdir(model_dir):
    filepath = os.path.join(model_dir, file)
    size = os.path.getsize(filepath) / (1024*1024)  # MB 단위
    print(f"  - {file}: {size:.2f} MB")
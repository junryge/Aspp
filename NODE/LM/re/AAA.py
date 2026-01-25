# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 20:21:07 2026
@author: ggg3g
"""
import torch
import pickle
import numpy as np
import whisper
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

# 1. Whisper로 A.wav 대본 자동 추출
print("음성 → 텍스트 변환 중...")
stt_model = whisper.load_model("base")
result = stt_model.transcribe("A.wav", language="ko")
ref_text = result["text"]
print(f"자동 추출된 대본: {ref_text}")

# ★ 여기 추가! 잘못 인식된 단어 수정
ref_text = ref_text.replace("2000", "이천")
ref_text = ref_text.replace("이 중략", "이준력")
print(f"수정된 대본: {ref_text}")

# Whisper 메모리 해제
del stt_model
torch.cuda.empty_cache()

# 2. TTS 모델 로드
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 3. 내 목소리 프롬프트 생성 & 저장
voice_clone_prompt = model.create_voice_clone_prompt(
    ref_audio="A.wav",
    ref_text=ref_text,
)

with open("my_voice_prompt.pkl", "wb") as f:
    pickle.dump(voice_clone_prompt, f)

print("목소리 프롬프트 저장 완료!")

# 4. 음성 생성
wavs, sr = model.generate_voice_clone(
    text="안녕하세요, 테스트 음성입니다. 저는 누굴까요 맞춰보세요 수연이 너무 너무 사랑합니다.많이 많이",
    language="Korean",
    voice_clone_prompt=voice_clone_prompt,
)

# 5. 저장
audio_data = (wavs[0] * 32767).astype(np.int16)
audio = AudioSegment(audio_data.tobytes(), frame_rate=sr, sample_width=2, channels=1)
audio.export("output.wav", format="wav")

print("음성 생성 완료!")
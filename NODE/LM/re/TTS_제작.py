# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 20:35:08 2026

@author: ggg3g
"""

# # -*- coding: utf-8 -*-
# """
# my_voice_prompt.pkl 로 음성 생성
# """
# import torch
# import pickle
# import numpy as np
# from pydub import AudioSegment
# from qwen_tts import Qwen3TTSModel

# # 1. 저장된 내 목소리 로드
# with open("my_voice_prompt.pkl", "rb") as f:
#     voice_clone_prompt = pickle.load(f)

# print("목소리 프롬프트 로드 완료!")

# # 2. TTS 모델 로드
# model = Qwen3TTSModel.from_pretrained(
#     "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
#     device_map="cuda:0",
#     dtype=torch.bfloat16,
# )

# # 3. 음성 생성 (텍스트만 바꾸면 됨!)
# wavs, sr = model.generate_voice_clone(
#     text="안녕하세요, 테스트 음성입니다. 저는 누굴까요 맞춰보세요 수연이 너무 너무 사랑합니다.많이 많이",
#     language="Korean",
#     voice_clone_prompt=voice_clone_prompt,
# )

# # 4. 저장
# audio_data = (wavs[0] * 32767).astype(np.int16)
# audio = AudioSegment(audio_data.tobytes(), frame_rate=sr, sample_width=2, channels=1)
# audio.export("output.wav", format="wav")

# print("음성 생성 완료: output.wav")


# -*- coding: utf-8 -*-
"""
my_voice_prompt.pkl 로 음성 생성 (속도 조절 가능)
"""
import torch
import pickle
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
from qwen_tts import Qwen3TTSModel

# 1. 저장된 내 목소리 로드
with open("my_voice_prompt.pkl", "rb") as f:
    voice_clone_prompt = pickle.load(f)
print("목소리 프롬프트 로드 완료!")

# 2. TTS 모델 로드
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 3. 음성 생성
wavs, sr = model.generate_voice_clone(
    text="안녕하세요, 테스트 음성입니다. 저는 누굴까요 맞춰보세요 수연이 너무 너무 사랑합니다. 많이 많이",
    language="Korean",
    voice_clone_prompt=voice_clone_prompt,
)

# 4. AudioSegment 생성
audio_data = (wavs[0] * 32767).astype(np.int16)
audio = AudioSegment(audio_data.tobytes(), frame_rate=sr, sample_width=2, channels=1)

# ★ 5. 속도 조절 (0.5~2.0, 1.0이 원본)
speed = 1.0  # ← 여기서 조절! (0.7=느리게, 1.0=원본, 1.3=빠르게)

if speed > 1.0:
    audio = speedup(audio, playback_speed=speed)
elif speed < 1.0:
    audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)

# 6. 저장
audio.export("output.wav", format="wav")
print(f"음성 생성 완료: output.wav (속도: {speed}배)")
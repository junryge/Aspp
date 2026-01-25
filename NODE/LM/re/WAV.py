import torch
import numpy as np
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_voice_clone(
    text="안녕하세요, M4A 테스트입니다.",
    ref_audio="A.m4a",                    # ← 여기만 변경!
    ref_text="A.m4a에서 말한 내용 대본",   # ← 실제 대본으로 변경!
    language="Korean",
)

# M4A로 저장
audio_data = (wavs[0] * 32767).astype(np.int16)
audio = AudioSegment(audio_data.tobytes(), frame_rate=sr, sample_width=2, channels=1)
audio.export("output.m4a", format="m4a", bitrate="192k")

print("M4A 저장 완료!")
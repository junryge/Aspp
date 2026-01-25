#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 음성 생성 모듈
- 분석 결과를 자연스러운 음성으로 변환
- my_voice_prompt.pkl 기반 음성 클론
- 속도 조절 지원
"""

import os
import re
import torch
import pickle
import numpy as np
from io import BytesIO
from pydub import AudioSegment
from pydub.effects import speedup
import logging

logger = logging.getLogger(__name__)

# ========================================
# 전역 변수
# ========================================
tts_model = None
voice_clone_prompt = None
TTS_AVAILABLE = False

# TTS 설정
TTS_CONFIG = {
    "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice_prompt_path": "my_voice_prompt.pkl",
    "default_speed": 1.0,
    "language": "Korean",
    "output_dir": "tts_output"
}


# ========================================
# TTS 초기화
# ========================================
def init_tts():
    """TTS 모델 및 음성 프롬프트 로드"""
    global tts_model, voice_clone_prompt, TTS_AVAILABLE
    
    try:
        # 출력 폴더 생성
        if not os.path.exists(TTS_CONFIG["output_dir"]):
            os.makedirs(TTS_CONFIG["output_dir"])
        
        # 음성 프롬프트 로드
        voice_path = TTS_CONFIG["voice_prompt_path"]
        if not os.path.exists(voice_path):
            logger.warning(f"Voice prompt not found: {voice_path}")
            return False
        
        with open(voice_path, "rb") as f:
            voice_clone_prompt = pickle.load(f)
        logger.info("Voice prompt loaded successfully")
        
        # TTS 모델 로드
        from qwen_tts import Qwen3TTSModel
        
        tts_model = Qwen3TTSModel.from_pretrained(
            TTS_CONFIG["model_name"],
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        logger.info("TTS model loaded successfully")
        
        TTS_AVAILABLE = True
        return True
        
    except ImportError as e:
        logger.error(f"TTS module import failed: {e}")
        logger.info("Install: pip install qwen-tts pydub --break-system-packages")
        return False
    except Exception as e:
        logger.error(f"TTS initialization failed: {e}")
        return False


def is_tts_available() -> bool:
    """TTS 사용 가능 여부"""
    return TTS_AVAILABLE and tts_model is not None and voice_clone_prompt is not None


# ========================================
# 텍스트 전처리
# ========================================
def clean_text_for_tts(text: str) -> str:
    """TTS용 텍스트 정제 - 마크다운/특수문자 제거"""
    # 마크다운 제거
    text = re.sub(r'#{1,6}\s*', '', text)  # 헤더
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # bold/italic
    text = re.sub(r'`[^`]+`', '', text)  # 코드
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 링크
    
    # 테이블 제거
    text = re.sub(r'\|[^\n]+\|', '', text)
    text = re.sub(r'-{3,}', '', text)
    
    # 이모지 제거 (일부 유지)
    emoji_map = {
        '✅': '정상',
        '🔴': '지연',
        '⚠️': '경고',
        '🟡': '주의',
        '📦': '',
        '📍': '',
        '⏱️': '',
        '🕒': '',
        '🤖': '',
        '📊': '',
        '🔧': '',
    }
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    # 특수문자 정리
    text = re.sub(r'[=\-]{3,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()


def summarize_for_speech(analysis_result: dict) -> str:
    """분석 결과에서 결론과 권장 조치만 추출하여 음성용 텍스트 생성"""
    
    lines = []
    analysis_text = analysis_result.get('analysis', '')
    
    # 1. 결론 추출 - 정규식으로 직접 찾기 ("결론:" 또는 "결론 :" 형식)
    conclusion_match = re.search(r'결론\s*[:：]\s*(.+?)(?:\n|$)', analysis_text, re.IGNORECASE)
    if conclusion_match:
        conclusion = clean_text_for_tts(conclusion_match.group(1).strip())
        if conclusion:
            lines.append(f"결론입니다. {conclusion}")
    
    # 2. 권장 조치 추출
    recommendation = extract_section(analysis_text, ['권장 조치', '권장사항', '조치 사항', '개선 방안'])
    if recommendation:
        # "결론:" 부분이 섞여있으면 제거
        recommendation = re.sub(r'결론\s*[:：].*', '', recommendation).strip()
        if recommendation:
            lines.append(f"권장 조치입니다. {recommendation}")
    
    # 결론/권장 조치가 없으면 전처리 결과에서 핵심만 추출
    if not lines:
        preprocess = analysis_result.get('preprocess_details', {})
        delays = preprocess.get('delays', []) if preprocess else []
        
        if delays:
            lines.append("결론입니다. 지연이 발생했습니다.")
            cause = delays[0].get('cause', '') if delays else ''
            if 'HCACK' in cause:
                lines.append("권장 조치입니다. 첫째, OHT 차량 가용성을 점검하세요. 둘째, Rail Cut 여부를 확인하세요. 셋째, 동시간대 작업 부하를 분석하세요.")
            else:
                lines.append("권장 조치입니다. 해당 구간의 설비 상태를 점검하세요.")
        else:
            lines.append("결론입니다. 이송이 정상적으로 완료되었습니다. 특별한 조치가 필요하지 않습니다.")
    
    result = ' '.join(lines)
    
    if len(result) > 500:
        result = result[:500]
    
    return result


def extract_section(text: str, keywords: list) -> str:
    """텍스트에서 특정 섹션 추출 (결론, 권장 조치 등)"""
    if not text:
        return ""
    
    lines = text.split('\n')
    capturing = False
    captured = []
    
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        # 키워드로 시작하는 섹션 찾기
        is_section_header = any(kw in line_lower for kw in keywords)
        
        if is_section_header:
            capturing = True
            # 헤더 자체는 스킵 (### 결론, **권장 조치** 등)
            if line_clean.startswith('#') or line_clean.startswith('*'):
                continue
            # 헤더와 내용이 같은 줄인 경우 (결론: 내용...)
            if ':' in line_clean:
                content = line_clean.split(':', 1)[1].strip()
                if content:
                    captured.append(content)
            continue
        
        # 캡처 중일 때
        if capturing:
            # 다른 섹션 시작하면 중단
            if line_clean.startswith('#') or line_clean.startswith('---') or line_clean.startswith('==='):
                break
            # 빈 줄 2개 연속이면 중단
            if not line_clean and captured and not captured[-1]:
                break
            # 내용 추가
            if line_clean:
                # 번호 목록 정리 (1. 2. 3. → 첫째, 둘째, 셋째)
                cleaned = re.sub(r'^(\d+)\.\s*', lambda m: ['첫째, ', '둘째, ', '셋째, ', '넷째, '][int(m.group(1))-1] if int(m.group(1)) <= 4 else '', line_clean)
                # 마크다운/특수문자 제거
                cleaned = clean_text_for_tts(cleaned)
                if cleaned:
                    captured.append(cleaned)
    
    # 최대 4줄만 반환
    return ' '.join(captured[:4])


# ========================================
# 음성 생성
# ========================================
def generate_speech(text: str, speed: float = 1.0, output_filename: str = None) -> dict:
    """텍스트를 음성으로 변환"""
    
    if not is_tts_available():
        return {
            "success": False,
            "error": "TTS 모델이 로드되지 않았습니다."
        }
    
    try:
        # 텍스트 정제
        clean_text = clean_text_for_tts(text)
        
        if len(clean_text) < 5:
            return {
                "success": False,
                "error": "텍스트가 너무 짧습니다."
            }
        
        # 텍스트 길이 제한 (TTS 안정성)
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "... 이하 생략합니다."
        
        logger.info(f"Generating speech for: {clean_text[:100]}...")
        
        # 음성 생성
        wavs, sr = tts_model.generate_voice_clone(
            text=clean_text,
            language=TTS_CONFIG["language"],
            voice_clone_prompt=voice_clone_prompt,
        )
        
        # AudioSegment 생성
        audio_data = (wavs[0] * 32767).astype(np.int16)
        audio = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        
        # 속도 조절
        if speed > 1.0:
            audio = speedup(audio, playback_speed=speed)
        elif speed < 1.0:
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed)
            }).set_frame_rate(audio.frame_rate)
        
        # 파일 저장
        if output_filename is None:
            import time
            output_filename = f"tts_{int(time.time())}.wav"
        
        output_path = os.path.join(TTS_CONFIG["output_dir"], output_filename)
        audio.export(output_path, format="wav")
        
        logger.info(f"Speech generated: {output_path}")
        
        return {
            "success": True,
            "filename": output_filename,
            "filepath": output_path,
            "duration_ms": len(audio),
            "speed": speed,
            "text_length": len(clean_text)
        }
        
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def generate_speech_from_analysis(analysis_result: dict, speed: float = 1.0) -> dict:
    """분석 결과를 음성으로 변환"""
    
    # 분석 결과 요약
    summary_text = summarize_for_speech(analysis_result)
    
    # 파일명 생성
    filename = analysis_result.get('filename', 'unknown')
    safe_filename = re.sub(r'[^\w\-_]', '_', filename)
    output_filename = f"tts_{safe_filename}.wav"
    
    return generate_speech(summary_text, speed, output_filename)


# ========================================
# 테스트
# ========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("TTS 모듈 테스트")
    print("=" * 50)
    
    # 초기화
    if init_tts():
        print("✅ TTS 초기화 성공")
        
        # 테스트 텍스트
        test_text = "안녕하세요. AMHS 로그 분석 결과를 알려드리겠습니다. 총 245건의 레코드가 분석되었습니다."
        
        result = generate_speech(test_text, speed=1.0)
        
        if result["success"]:
            print(f"✅ 음성 생성 성공: {result['filepath']}")
            print(f"   - 길이: {result['duration_ms']}ms")
        else:
            print(f"❌ 음성 생성 실패: {result['error']}")
    else:
        print("❌ TTS 초기화 실패")

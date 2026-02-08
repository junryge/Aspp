#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
앱 설정 관리 - ENV_CONFIG, 토큰 로딩, GGUF 모델 목록
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 앱 기본 경로
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class AppConfig:
    """앱 전체 설정 관리"""

    ENV_CONFIG = {
        "dev": {
            "url": "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "name": "DEV(30B)"
        },
        "prod": {
            "url": "http://summary.llm.skhynix.com/v1/chat/completions",
            "model": "Qwen3-Next-80B-A3B-Instruct",
            "name": "PROD(80B)"
        },
        "common": {
            "url": "http://common.llm.skhynix.com/v1/chat/completions",
            "model": "gpt-oss-20b",
            "name": "COMMON(20B)"
        },
        "local": {
            "url": "",
            "model": "Qwen3-14B-Q4_K_M",
            "name": "LOCAL(14B-GGUF)"
        }
    }

    AVAILABLE_GGUF_MODELS = {
        "qwen3-14b": {
            "path": str(BASE_DIR / "Qwen3-14B-Q4_K_M.gguf"),
            "name": "Qwen3-14B",
            "desc": "14B Q4_K_M - 균형 잡힌 성능",
            "gpu_layers": 35,
            "ctx": 8192
        },
        "qwen3-8b": {
            "path": str(BASE_DIR / "Qwen3-8B-Q6_K.gguf"),
            "name": "Qwen3-8B",
            "desc": "8B Q6_K - 빠른 추론",
            "gpu_layers": 35,
            "ctx": 8192
        },
        "qwen3-1.7b": {
            "path": str(BASE_DIR / "qwen3-1.7b-q8_0.gguf"),
            "name": "Qwen3-1.7B",
            "desc": "1.7B Q8_0 - 경량 모델",
            "gpu_layers": 35,
            "ctx": 4096
        },
    }

    def __init__(self):
        self.api_token = None
        self.llm_mode = "api"       # "api" 또는 "local"
        self.env_mode = "common"    # "dev", "prod", "common", "local"
        self.api_url = self.ENV_CONFIG["common"]["url"]
        self.api_model = self.ENV_CONFIG["common"]["model"]
        self.current_gguf_model = "qwen3-14b"
        self.gguf_model_path = self.AVAILABLE_GGUF_MODELS["qwen3-14b"]["path"]

    def load_token(self) -> bool:
        """token.txt에서 API 토큰 로드 (로컬 → 상위 → 홈 디렉토리)"""
        paths = [
            str(BASE_DIR / "token.txt"),
            str(BASE_DIR.parent / "token.txt"),
            os.path.expanduser("~/token.txt"),
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        token = f.read().strip()
                    if token and "REPLACE" not in token:
                        self.api_token = token
                        logger.info(f"토큰 로드 성공: {p}")
                        return True
                except Exception as e:
                    logger.error(f"토큰 로드 실패: {e}")
        logger.warning("토큰 파일을 찾을 수 없습니다")
        return False

    def set_env(self, env: str) -> bool:
        """LLM 환경 전환"""
        if env not in self.ENV_CONFIG:
            return False

        self.env_mode = env
        if env == "local":
            self.llm_mode = "local"
        else:
            if not self.api_token:
                return False
            self.llm_mode = "api"
            self.api_url = self.ENV_CONFIG[env]["url"]
            self.api_model = self.ENV_CONFIG[env]["model"]
        return True

    def switch_gguf_model(self, model_key: str) -> bool:
        """GGUF 모델 전환"""
        if model_key not in self.AVAILABLE_GGUF_MODELS:
            return False
        cfg = self.AVAILABLE_GGUF_MODELS[model_key]
        if not os.path.exists(cfg["path"]):
            return False
        self.current_gguf_model = model_key
        self.gguf_model_path = cfg["path"]
        self.ENV_CONFIG["local"]["model"] = cfg["name"]
        self.ENV_CONFIG["local"]["name"] = f"LOCAL({cfg['name']})"
        return True

    def get_gguf_config(self) -> dict:
        """현재 GGUF 모델 설정 반환"""
        return self.AVAILABLE_GGUF_MODELS.get(self.current_gguf_model, {})

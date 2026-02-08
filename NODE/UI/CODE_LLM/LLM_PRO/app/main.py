#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nomos LLM Desktop App - 진입점
코드 개발 / 수정 / 데이터 분석 데스크탑 애플리케이션
"""

import sys
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# 앱 경로를 PYTHONPATH에 추가
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def main():
    """앱 메인 함수"""
    logger.info("=" * 50)
    logger.info("Nomos LLM Desktop v1.0 시작")
    logger.info("=" * 50)

    # 1. 설정 로드
    from app.core.config import AppConfig
    config = AppConfig()

    # 2. API 토큰 로드 시도
    if config.load_token():
        config.llm_mode = "api"
        logger.info(f"API 모드: {config.ENV_CONFIG[config.env_mode]['name']}")
    else:
        logger.warning("API 토큰 없음 - 로컬 GGUF 모드로 시작")
        config.llm_mode = "local"
        config.env_mode = "local"

    # 3. LLM Provider 초기화
    from app.core.llm_provider import LLMProvider
    provider = LLMProvider(config)

    # 4. GGUF 서버 시작 (aider가 로컬 모델과 통신할 때 필요)
    from app.core.gguf_server import GGUFServer
    gguf_server = GGUFServer(provider, port=10002)
    gguf_server.start()
    logger.info("GGUF 호환 서버 시작 (port 10002)")

    # 5. 로컬 모드면 모델 자동 로드 (비동기 X - 앱 시작 시 바로 로드)
    if config.llm_mode == "local":
        logger.info("로컬 GGUF 모델 로딩 시작...")
        if provider.load_local_model():
            logger.info("로컬 모델 로드 성공")
        else:
            logger.warning("로컬 모델 로드 실패 - UI에서 수동 로드 필요")

    # 6. Aider Bridge 초기화
    from app.aider.bridge import AiderBridge
    bridge = AiderBridge(config, provider)

    # 7. Qt 앱 실행
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon

    app = QApplication(sys.argv)
    app.setApplicationName("Nomos LLM")
    app.setApplicationVersion("1.0")

    # 아이콘 설정 (있으면)
    icon_path = os.path.join(APP_DIR, "magi.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 8. 메인 윈도우
    from app.main_window import MainWindow
    window = MainWindow(config, provider, bridge)
    window.show()

    logger.info("앱 준비 완료!")
    logger.info(f"모드: {config.llm_mode} | 환경: {config.env_mode}")

    # 9. 이벤트 루프
    exit_code = app.exec()

    # 10. 정리
    gguf_server.stop()
    logger.info("앱 종료")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

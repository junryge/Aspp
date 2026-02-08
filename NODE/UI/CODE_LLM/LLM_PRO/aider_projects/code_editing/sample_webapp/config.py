"""앱 설정"""

import os

# Flask 설정
DEBUG = True
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
PORT = int(os.environ.get("PORT", 5000))

# DB 설정
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///todos.db")

# CORS 설정
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5000",
]

# 페이징 설정
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

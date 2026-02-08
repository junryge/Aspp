# -*- mode: python ; coding: utf-8 -*-
"""
Nomos LLM Desktop - PyInstaller spec 파일
빌드: pyinstaller nomos.spec
"""

import os
import sys

block_cipher = None

# 프로젝트 루트
BASE_DIR = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(BASE_DIR, 'app', 'main.py')],
    pathex=[BASE_DIR],
    binaries=[],
    datas=[],
    hiddenimports=[
        # PySide6
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        # 앱 모듈
        'app',
        'app.main',
        'app.main_window',
        'app.core',
        'app.core.config',
        'app.core.llm_provider',
        'app.core.prompt_builder',
        'app.core.self_correction',
        'app.core.gguf_server',
        'app.aider',
        'app.aider.bridge',
        'app.aider.project_manager',
        'app.aider.git_ops',
        'app.agent',
        'app.agent.sk_provider',
        'app.agent.nanobot_manager',
        'app.agent.tools',
        'app.ui',
        'app.ui.theme',
        'app.ui.sidebar',
        'app.ui.header',
        'app.ui.chat_panel',
        'app.ui.code_editor',
        'app.ui.project_panel',
        'app.ui.diff_viewer',
        'app.ui.dialogs',
        'app.ui.workers',
        # 외부 라이브러리
        'markdown',
        'markdown.extensions',
        'markdown.extensions.fenced_code',
        'markdown.extensions.tables',
        'markdown.extensions.nl2br',
        'pygments',
        'pygments.lexers',
        'pygments.token',
        'requests',
        'httpx',
        # llama-cpp-python
        'llama_cpp',
        # aider
        'aider',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Qt 충돌 방지
        'PyQt5', 'PyQt6',
        'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
        # 대형 ML/DL 프레임워크
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'tf_keras', 'keras',
        'onnxruntime', 'onnx',
        'transformers', 'tokenizers', 'huggingface_hub',
        'diffusers', 'accelerate', 'safetensors',
        # 데이터/과학 라이브러리
        'numpy', 'pandas', 'scipy', 'sklearn', 'scikit_learn',
        'polars', 'pyarrow', 'duckdb',
        'matplotlib', 'plotly', 'seaborn', 'bokeh',
        'h5py', 'tables',
        # NLP
        'spacy', 'thinc', 'nltk', 'langcodes', 'srsly',
        'sympy',
        # 이미지/미디어
        'PIL', 'Pillow', 'cv2', 'opencv',
        'sounddevice', 'soundfile',
        # Google/Cloud
        'google.cloud', 'google.api_core', 'googleapiclient',
        'grpc', 'grpcio',
        # 기타 대형 패키지
        'faiss', 'faiss_cpu',
        'llvmlite', 'numba',
        'lxml',
        'hf_xet',
        # 개발 도구
        'tkinter',
        'jupyter', 'IPython', 'notebook',
        'pytest', 'sphinx',
        'uvicorn', 'fastapi', 'starlette',
        # Pythonwin
        'Pythonwin', 'win32com', 'pythoncom',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NomosLLM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # 콘솔 숨김 (GUI 앱)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NomosLLM',
)

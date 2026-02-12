# -*- mode: python ; coding: utf-8 -*-
# 3D Campus Builder PyInstaller Spec File
# 사용법: pyinstaller build_exe.spec

import os

block_cipher = None

a = Analysis(
    ['3D_Campus_Builder.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 프로젝트 JSON 파일 포함 (있으면)
        ('SK_Hynix_이천캠퍼스.json', '.') if os.path.exists('SK_Hynix_이천캠퍼스.json') else (None, None),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'tkinter.colorchooser',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'numpy', 'pandas', 'scipy', 'PIL',
        'pytest', 'setuptools', 'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# None 항목 필터링
a.datas = [d for d in a.datas if d[0] is not None]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='3D_Campus_Builder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 앱이므로 콘솔 창 없음
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 아이콘 파일이 있으면 여기에 경로 지정
)

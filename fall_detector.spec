# -*- mode: python ; coding: utf-8 -*-

import sys
import os

project_root = os.path.abspath(".")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

a = Analysis(
    ['src/model_ia.py'],
    pathex=['src'],
    datas=[
        ('src/keras/model_cnn.keras', 'keras'),
        ('src/keras/model_lstm.keras', 'keras'),
        ('src/keras/final_model.keras', 'keras'),
        ('src/config/*', 'config'),
        ('src/model/*', 'model'),
        ('src/preprocessing/*', 'preprocessing'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fall_detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)

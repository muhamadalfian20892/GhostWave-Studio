# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# GUI Application Analysis
a_gui = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[('ghostwave.ico', '.'), ('changelog.txt', '.')],
    hiddenimports=[
        'scipy.special',
        'scipy.signal',
        'scipy.io.wavfile',
        'scipy.fft',
        'soundfile',
        'cryptography',
        'cryptography.fernet',
        'wx',
        'requests',
        'urllib.request',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchaudio', 'tensorflow', 'matplotlib', 'tkinter', 'pandas', 'IPython', 'PIL'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name='GhostWaveStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ghostwave.ico',
)

# CLI Tool Analysis
a_cli = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('ghostwave.ico', '.'), ('changelog.txt', '.')],
    hiddenimports=[
        'scipy.special',
        'scipy.signal',
        'scipy.io.wavfile',
        'scipy.fft',
        'soundfile',
        'cryptography',
        'cryptography.fernet',
        'wx',
        'requests',
        'urllib.request',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchaudio', 'tensorflow', 'matplotlib', 'tkinter', 'pandas', 'IPython', 'PIL'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name='ghostwave-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ghostwave.ico',
)

coll = COLLECT(
    exe_gui,
    a_gui.binaries,
    a_gui.zipfiles,
    a_gui.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='GhostWaveStudio',
)

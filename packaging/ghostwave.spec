# -*- mode: python ; coding: utf-8 -*-
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
ICON_PATH = os.path.join(ROOT, 'assets', 'ghostwave.ico')
COMMON_DATAS = [
    (ICON_PATH, '.'),
    (os.path.join(ROOT, 'changelog.txt'), '.'),
    (os.path.join(ROOT, 'docs', 'README.txt'), '.'),
    (os.path.join(ROOT, 'docs', 'README_ID.txt'), '.'),
    (os.path.join(ROOT, 'locales'), 'locales'),
]
COMMON_HIDDEN = [
    'scipy.special',
    'scipy.signal',
    'scipy.io.wavfile',
    'scipy.fft',
    'soundfile',
    'sounddevice',
    'cryptography',
    'cryptography.fernet',
    'wx',
    'wx.lib.scrolledpanel',
    'urllib.request',
    'ghostwave.batch_processor',
]
COMMON_EXCLUDES = [
    'torch', 'torchaudio', 'tensorflow', 'matplotlib', 'tkinter', 'pandas',
    'IPython', 'PIL', 'sphinx', 'docutils', 'babel', 'alabaster', 'jieba',
    'lxml', 'setuptools', 'wheel', 'pkg_resources', 'yaml', 'zope', 'pytz',
    'pytest',
]

block_cipher = None

# GUI Application Analysis
a_gui = Analysis(
    [os.path.join(ROOT, 'ghostwave', 'gui.py')],
    pathex=[ROOT],
    binaries=[],
    datas=COMMON_DATAS,
    hiddenimports=COMMON_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES,
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
    icon=ICON_PATH,
)

# CLI Tool Analysis
a_cli = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=[],
    datas=COMMON_DATAS,
    hiddenimports=COMMON_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES,
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
    icon=ICON_PATH,
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

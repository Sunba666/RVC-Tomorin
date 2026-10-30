"""Build the desktop frontend; external RVC assets are never collected."""
import os
import sys
import shutil
from pathlib import Path
root = Path(__file__).resolve().parent
deps = root / '.build-deps'
if deps.exists():
    sys.path.insert(0, str(deps))
sys.path.insert(0, str(root))
if os.name != 'nt':
    raise SystemExit('Build the Windows executable on Windows.')
from PyInstaller.__main__ import run
licenses = root / 'licenses'
licenses.mkdir(exist_ok=True)
for source, name in [(Path(sys.base_prefix) / 'LICENSE.txt', 'Python.txt'),
                     (Path(sys.base_prefix) / 'tcl/tcl8.6/license.terms', 'Tcl.txt')]:
    if source.is_file(): shutil.copy2(source, licenses / name)
from importlib.metadata import distribution, PackageNotFoundError
try:
    dist = distribution('yt-dlp')
    for file in dist.files or []:
        if Path(str(file)).name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
            source = Path(dist.locate_file(file))
            if source.is_file(): shutil.copy2(source, licenses / ('yt-dlp-' + source.name))
except PackageNotFoundError:
    pass
args = ['--noconfirm', '--clean', '--onefile', '--windowed', '--name', 'VoiceWorkbench',
        '--paths', str(root), '--distpath', str(root / 'dist'), '--workpath', str(root / 'build'),
        '--specpath', str(root / 'build'), '--add-data', str(root / 'workers') + os.pathsep + 'workers',
        '--add-data', str(root / 'LICENSE') + os.pathsep + 'licenses',
        '--add-data', str(licenses) + os.pathsep + 'licenses',
        '--collect-all', 'yt_dlp', '--copy-metadata', 'yt-dlp']
for module in ['torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'gradio', 'fastapi', 'cv2', 'faiss', 'PyQt5', 'PyQt6', 'PySide6', 'IPython', 'sympy', 'sklearn', 'soundfile']:
    args += ['--exclude-module', module]
# The frontend uses yt-dlp's standard-library downloader. Avoid picking up
# unrelated optional dependencies from a large external RVC Python install.
for module in ['mutagen', 'Crypto', 'Cryptodome', 'requests', 'urllib3', 'brotli', 'brotlicffi', 'websockets', 'certifi', 'curl_cffi', 'secretstorage', 'keyring', 'yt_dlp.__pyinstaller', 'setuptools', 'pkg_resources', 'PyInstaller']:
    args += ['--exclude-module', module]
if deps.exists():
    args += ['--paths', str(deps)]
args += [str(root / 'main.py')]
run(args)

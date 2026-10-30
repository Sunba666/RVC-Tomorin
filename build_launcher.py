"""Build only the small release downloader, without collecting the ML runtime."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / '.build-deps'))
from PyInstaller.__main__ import run
args = ['--noconfirm', '--clean', '--onefile', '--windowed', '--name', 'RVC-Tomorin-Setup',
        '--distpath', str(root / 'dist'), '--workpath', str(root / 'build/launcher'),
        '--specpath', str(root / 'build/launcher'), '--paths', str(root),
        '--add-data', str(root / 'release') + ';release',
        '--add-data', str(root / 'licenses') + ';licenses']
for module in ('torch', 'numpy', 'scipy', 'gradio', 'transformers', 'matplotlib', 'pandas',
               'cv2', 'faiss', 'sklearn', 'soundfile', 'PyQt5', 'PyQt6', 'PySide6', 'yt_dlp'):
    args += ['--exclude-module', module]
run(args + [str(root / 'download_launcher.py')])

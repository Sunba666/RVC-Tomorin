"""Assemble an offline Windows runtime from an explicitly supplied RVC engine.

Only allowlisted engine assets are copied. User weights, indices, media, logs,
credentials and application settings never enter the public distribution.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_URL = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th'


def sha256(file):
    digest = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def ignore_generated(folder, names):
    return [n for n in names if n in ('__pycache__', 'Scripts', '.git', 'direct_url.json')
            or n.endswith(('.pyc', '.pyo', '.local.json'))]


def copy_tree(source, target):
    if not target.exists():
        print('Copying ' + source.name, flush=True)
        shutil.copytree(source, target, ignore=ignore_generated)


def prepare(engine, destination):
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'engine'
    target.mkdir(exist_ok=True)
    for name in ('runtime', 'configs', 'infer', 'train', 'tools', 'i18n'):
        copy_tree(engine / name, target / name)
    for name in ('ffmpeg.exe', 'ffprobe.exe', 'LICENSE', 'MIT协议暨相关引用库协议'):
        shutil.copy2(engine / name, target / name)
    for name in ('hubert_base',):
        copy_tree(engine / 'assets' / name, target / 'assets' / name)
    selected = ['rmvpe/rmvpe.pt'] + [f'pretrained_v2/f0{k}{rate}.pth'
        for rate in ('32k', '40k', '48k') for k in ('G', 'D')]
    for relative in selected:
        output = target / 'assets' / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(engine / 'assets' / relative, output)
    for name in ('weights', 'indices'):
        (target / 'assets' / name).mkdir(exist_ok=True)
    (target / 'runtime/python312._pth').write_text(
        'python312.zip\n.\nLib\nLib/site-packages\n..\nimport site\n', encoding='ascii')
    # Install adapter dependencies into the copied runtime, never the user's engine.
    packages = target / 'runtime/Lib/site-packages'
    if not ((packages / 'demucs').is_dir() and (packages / 'cloudpickle').is_dir() and (packages / 'retrying.py').is_file()):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            '--no-deps', '--no-compile', '--target', str(packages), 'demucs==4.0.1',
            'dora-search==0.1.12', 'julius==0.2.7', 'omegaconf==2.3.0',
            'antlr4-python3-runtime==4.9.3', 'treetable==0.2.6', 'lameenc==1.8.1',
            'openunmix==1.3.0', 'submitit==1.5.3', 'cloudpickle==3.1.1', 'retrying==1.3.4'], check=True)
    # Application-local VC runtime avoids a separate Visual C++ setup on a clean PC.
    system = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32'
    for name in ('msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
                 'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
                 'concrt140.dll', 'vcomp140.dll'):
        if (system / name).is_file():
            shutil.copy2(system / name, target / 'runtime' / name)
    # Console launchers are not used and can contain installation-time paths.
    scripts = packages / 'bin'
    if scripts.exists():
        shutil.rmtree(scripts)
    separator = target / 'assets/separator/955717e8-8726e21a.th'
    separator.parent.mkdir(parents=True, exist_ok=True)
    if not separator.exists():
        print('Downloading bundled vocal separator', flush=True)
        pending = separator.with_suffix('.pending')
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, pending.open('wb') as output:
            shutil.copyfileobj(response, output)
        if not sha256(pending).startswith('8726e21a'):
            raise RuntimeError('Demucs checkpoint digest does not match the upstream filename')
        pending.replace(separator)
    (separator.parent / 'htdemucs.yaml').write_text('model: htdemucs\nsources: [drums, bass, other, vocals]\n', encoding='utf-8')
    licenses = destination / 'licenses'
    licenses.mkdir(exist_ok=True)
    for name, url in {
        'Demucs-MIT.txt': 'https://raw.githubusercontent.com/facebookresearch/demucs/main/LICENSE',
        'RVC-MIT.txt': 'https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main/LICENSE',
        'ContentVec-MIT.txt': 'https://raw.githubusercontent.com/auspicious3000/contentvec/main/LICENSE',
        'FFmpeg-GPLv3.txt': 'https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/COPYING.GPLv3',
    }.items():
        file = licenses / name
        if not file.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                file.write_bytes(response.read())
    shutil.copy2(target / 'runtime/LICENSE.txt', licenses / 'CPython.txt')
    # All installed package metadata and license files stay beside their libraries.
    manifest = {'schema': 1, 'version': '0.3.0', 'engine': 'engine',
        'separator_backend': 'demucs', 'separator_model': 'engine/assets/separator/955717e8-8726e21a.th',
        'separator_config': 'engine/assets/separator/htdemucs.yaml',
        'character_models_included': False, 'offline_runtime': True,
        'separator_source': MODEL_URL, 'separator_sha256': sha256(separator)}
    (destination / 'portable.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Offline engine prepared: ' + str(destination), flush=True)


def finish(destination, frontend):
    shutil.copy2(frontend, destination / 'VoiceWorkbench.exe')
    for name in ('LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.md', '使用说明.txt', 'CHANGELOG.md', 'TESTING.md'):
        shutil.copy2(ROOT / name, destination / name)
    (destination / 'Start.bat').write_text('@echo off\r\nstart "" "%~dp0VoiceWorkbench.exe"\r\n', encoding='ascii')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--frontend', type=Path)
    args = parser.parse_args()
    if args.engine:
        prepare(args.engine.resolve(), args.destination.resolve())
    if args.frontend:
        finish(args.destination.resolve(), args.frontend.resolve())

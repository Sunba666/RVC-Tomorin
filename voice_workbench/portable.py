"""Relative-path discovery for the self-contained Windows distribution."""
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path


def bundle_root():
    override = os.environ.get('VOICE_WORKBENCH_BUNDLE')
    root = Path(override) if override else (Path(sys.executable).parent if getattr(sys, 'frozen', False)
                                         else Path(__file__).resolve().parents[1])
    return root.resolve() if (root / 'portable.json').is_file() else None


def bundle_info():
    root = bundle_root()
    if root is None:
        return None, {}
    try:
        info = json.loads((root / 'portable.json').read_text(encoding='utf-8-sig'))
        if info.get('schema') != 1:
            raise ValueError('未知的整合包版本')
        for key in ('engine', 'separator_model', 'separator_config'):
            file = (root / info[key]).resolve()
            if not file.is_relative_to(root) or Path(info[key]).is_absolute():
                raise ValueError('整合包路径超出程序目录')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError('整合包清单损坏，请重新完整解压软件。') from error
    return root, info


def runtime_defaults():
    root, info = bundle_info()
    if root is None:
        return {}
    engine = root / info['engine']
    return {'ffmpeg': str(engine), 'rvc_root': str(engine),
            'python': str(engine / 'runtime/python.exe'),
            'separator_model': str(root / info['separator_model']),
            'separator_config': str(root / info['separator_config']),
            'separator_backend': info.get('separator_backend', 'demucs')}


def _import_file(source, directory):
    source = Path(source)
    before = source.stat()
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / ('.import-' + uuid.uuid4().hex)
    digest = hashlib.sha256()
    try:
        with source.open('rb') as reader, pending.open('xb') as writer:
            for part in iter(lambda: reader.read(1024 * 1024), b''):
                digest.update(part)
                writer.write(part)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('导入时原文件发生变化，请等文件复制完成后重试。')
        if not after.st_size:
            raise ValueError('模型文件是空的。')
        name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', source.stem)[:60] or 'voice'
        target = directory / (name + '-' + digest.hexdigest()[:12] + source.suffix.lower())
        os.replace(pending, target)
        return str(target)
    finally:
        pending.unlink(missing_ok=True)


def import_voice_model(source, home):
    source = Path(source)
    if source.suffix.lower() != '.pth' or not source.is_file():
        raise ValueError('请选择有效的 RVC .pth 音色模型。')
    directory = Path(home) / 'models'
    model = _import_file(source, directory)
    # Prefer a matching name; otherwise accept exactly one neighbouring index.
    same_name = source.with_suffix('.index')
    nearby = sorted(source.parent.glob('*.index'))
    index = same_name if same_name.is_file() else (nearby[0] if len(nearby) == 1 else None)
    return {'model': model, 'index': _import_file(index, directory) if index else ''}

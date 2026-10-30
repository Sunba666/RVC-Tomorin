"""Verified, resumable installation of the public portable release."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import urllib.error
import uuid
import zipfile
from contextlib import contextmanager


class InstallCancelled(Exception):
    pass


@contextmanager
def install_lock(root):
    handle = (root / '.installer.lock').open('a+b')
    if handle.seek(0, 2) == 0: handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError('另一个安装进程正在运行，请等待它完成。') from error
    try: yield
    finally: handle.close()


def checksum(file, cancel=lambda: False):
    result = hashlib.sha256()
    with Path(file).open('rb') as reader:
        for part in iter(lambda: reader.read(1024 * 1024), b''):
            if cancel(): raise InstallCancelled('已取消，可以下次继续。')
            result.update(part)
    return result.hexdigest()


class RuntimeInstaller:
    def __init__(self, manifest, root, cache=None, adjacent=None, notify=None, cancel=None, base_url=None):
        self.info = manifest
        self.root = Path(root).resolve()
        self.cache = Path(cache).resolve() if cache else self.root / 'downloads'
        self.adjacent = Path(adjacent).resolve() if adjacent else self.root
        self.notify = notify or (lambda percent, message: None)
        self.cancel = cancel or (lambda: False)
        self.base_url = base_url or f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}/'
        for name in [manifest['folder'], manifest['archive']] + [p['name'] for p in manifest['parts']]:
            if not name or name in ('.', '..') or Path(name).name != name or any(c in name for c in '\\/:'):
                raise ValueError('发行清单中的文件名无效。')
        self.target = self.root / manifest['folder']

    def check(self):
        if self.cancel(): raise InstallCancelled('已取消，可以下次继续。')

    def ready(self, folder=None):
        folder = Path(folder) if folder else self.target
        required = ['VoiceWorkbench.exe', 'portable.json', 'engine/runtime/python.exe',
                    'engine/ffmpeg.exe', 'engine/ffprobe.exe', 'engine/infer/vc/modules.py',
                    'engine/assets/hubert_base/pytorch_model.bin', 'engine/assets/rmvpe/rmvpe.pt',
                    'engine/assets/separator/955717e8-8726e21a.th']
        required += [f'engine/assets/pretrained_v2/f0{k}{sr}.pth'
                     for k in ('G', 'D') for sr in ('32k', '40k', '48k')]
        try:
            return (all((folder / f).is_file() for f in required) and
                    json.loads((folder / 'portable.json').read_text(encoding='utf-8-sig'))['version'] == self.info['version'])
        except (OSError, ValueError, KeyError):
            return False

    def valid_part(self, file, record):
        return file.is_file() and file.stat().st_size == record['bytes'] and checksum(file, self.cancel) == record['sha256']

    def download(self, record, number, count):
        target = self.cache / record['name']
        for candidate in (self.adjacent / record['name'], target):
            self.check()
            if self.valid_part(candidate, record):
                return candidate
        pending = target.with_name(target.name + '.download')
        last = None
        for attempt in range(3):
            self.check()
            offset = pending.stat().st_size if pending.exists() else 0
            if offset > record['bytes']:
                pending.unlink(); offset = 0
            headers = {'User-Agent': 'RVC-Tomorin-Launcher/0.3.0', 'Accept-Encoding': 'identity'}
            if offset: headers['Range'] = f'bytes={offset}-'
            try:
                request = urllib.request.Request(self.base_url + record['name'], headers=headers)
                with urllib.request.urlopen(request, timeout=45) as response:
                    status = getattr(response, 'status', 200)
                    if status == 206:
                        value = response.headers.get('Content-Range', '')
                        if not value.startswith(f'bytes {offset}-'):
                            raise ValueError('服务器续传位置不一致，请重试。')
                    elif status == 200:
                        offset = 0
                    else:
                        raise OSError(f'下载服务器返回 {status}')
                    with pending.open('ab' if offset else 'wb') as writer:
                        done = offset
                        while True:
                            self.check()
                            data = response.read(1024 * 1024)
                            if not data: break
                            writer.write(data); done += len(data)
                            if done > record['bytes']: raise ValueError('下载大小与发行清单不符。')
                            self.notify(60 * (number + done / record['bytes']) / count,
                                f'正在下载运行环境 {number + 1}/{count} · {done / 2**20:.0f} / {record["bytes"] / 2**20:.0f} MB')
                self.notify(60 * (number + 1) / count, '正在校验下载文件…')
                if not self.valid_part(pending, record):
                    pending.unlink(missing_ok=True)
                    raise ValueError('下载文件不完整或校验失败，正在重试。')
                os.replace(pending, target)
                return target
            except InstallCancelled:
                raise
            except (OSError, ValueError, urllib.error.URLError) as error:
                last = error
                if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                    if self.valid_part(pending, record):
                        os.replace(pending, target); return target
                    pending.unlink(missing_ok=True)
                self.notify(60 * number / count, f'连接中断，正在重试（{attempt + 1}/3）…')
                for _ in range(10):
                    self.check(); time.sleep(.1)
        raise RuntimeError('无法下载运行环境，请检查网络后点击重试。\n' + str(last))

    def archive(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        for candidate in (self.adjacent / self.info['archive'], self.cache / self.info['archive']):
            if candidate.is_file() and candidate.stat().st_size == self.info['bytes']:
                self.notify(1, '正在检查已有整合包…')
                if checksum(candidate, self.cancel) == self.info['sha256']:
                    return candidate
        parts = [self.download(p, i, len(self.info['parts'])) for i, p in enumerate(self.info['parts'])]
        target = self.cache / self.info['archive']
        pending = target.with_name(target.name + '.assembling')
        digest = hashlib.sha256(); done = 0
        try:
            with pending.open('wb') as writer:
                for part in parts:
                    with part.open('rb') as reader:
                        for data in iter(lambda: reader.read(4 * 1024 * 1024), b''):
                            self.check(); writer.write(data); digest.update(data); done += len(data)
                            self.notify(60 + 10 * done / self.info['bytes'], '正在准备安装文件…')
            if done != self.info['bytes'] or digest.hexdigest() != self.info['sha256']:
                raise ValueError('整合包校验失败，请重新下载。')
            os.replace(pending, target)
        finally:
            pending.unlink(missing_ok=True)
        # Only installer-owned cached parts are removed; adjacent user files stay.
        for part in parts:
            if part.parent.resolve() == self.cache: part.unlink(missing_ok=True)
        return target

    def install(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with install_lock(self.root):
            return self._install()

    def _install(self):
        if self.ready():
            self.notify(100, '软件已经就绪，正在打开…')
            return self.target / 'VoiceWorkbench.exe'
        if self.target.exists():
            raise RuntimeError('安装目录已有不完整或不同版本的文件。为保留你的数据，本次没有覆盖。\n' + str(self.target))
        self.root.mkdir(parents=True, exist_ok=True)
        required = self.info.get('min_space_bytes', 20 * 2**30)
        if shutil.disk_usage(self.root).free < required:
            raise RuntimeError(f'安装位置需要至少 {required / 2**30:.0f} GB 空闲空间，请先清理磁盘。')
        archive = self.archive()
        stage = self.root / ('.install-' + uuid.uuid4().hex)
        stage.mkdir()
        try:
            with zipfile.ZipFile(archive) as bundle:
                total = sum(x.file_size for x in bundle.infolist()); done = 0
                for item in bundle.infolist():
                    self.check()
                    path = PurePosixPath(item.filename)
                    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != self.info['folder'] or '\\' in item.filename or ':' in item.filename:
                        raise ValueError('整合包中存在不安全路径。')
                    target = stage.joinpath(*path.parts)
                    if not target.resolve().is_relative_to(stage.resolve()): raise ValueError('解压路径越界。')
                    if item.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(item) as reader, target.open('xb') as writer:
                        for data in iter(lambda: reader.read(1024 * 1024), b''):
                            self.check(); writer.write(data); done += len(data)
                            self.notify(70 + 29 * done / max(1, total), '正在解压并准备软件…')
            extracted = stage / self.info['folder']
            if not self.ready(extracted): raise ValueError('整合包缺少必需文件，请重新下载。')
            self.check()
            os.replace(extracted, self.target)
        finally:
            if stage.exists():
                if stage.resolve().parent != self.root or not stage.name.startswith('.install-') or stage.is_symlink() or stage.is_junction():
                    raise RuntimeError('临时安装目录检查失败，已保留文件。')
                shutil.rmtree(stage)
        if archive.parent.resolve() == self.cache: archive.unlink(missing_ok=True)
        self.notify(100, '安装完成，正在打开软件…')
        return self.target / 'VoiceWorkbench.exe'

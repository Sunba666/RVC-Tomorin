"""把能够自动准备的依赖装到本机数据目录。

只处理可以确定性下载、校验并离线运行的部分（目前是 FFmpeg 与 FFprobe）；
RVC 环境、角色模型和显卡驱动包含数 GB 权重与用户自有文件，仍需手动选择。
"""
from __future__ import annotations

import hashlib
import http.client
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

FFMPEG_URL = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
FFMPEG_LABEL = '约 110 MB'
# 留空跳过可信摘要校验；若想锁定某个确定版本（防止下载途中被替换），填入该压缩包的 SHA256（S01）。
FFMPEG_SHA256 = ''
USER_AGENT = 'VoiceWorkbench/0.2.1 local desktop tool'
CHUNK = 256 * 1024
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
BINARIES = ('ffmpeg.exe', 'ffprobe.exe')


class InstallCancelled(RuntimeError):
    """安装被取消。"""


class InstallBusy(RuntimeError):
    """同一数据目录已经有另一个安装在途。"""


class TruncatedDownload(OSError):
    """服务器断开，或实际收到的字节数与声明不符。"""


def tools_folder(home: Path) -> Path:
    return Path(home) / 'tools'


def ffmpeg_folder(home: Path) -> Path:
    return tools_folder(home) / 'ffmpeg'


def _report(progress, fraction: float, note: str = ''):
    if progress:
        progress(max(0., min(1., fraction)), note)


def _header_int(headers, name: str, default: int = 0) -> int:
    try:
        return int(headers.get(name) or default)
    except (TypeError, ValueError):
        return default


def _range_start(content_range) -> int:
    """从 `bytes 500-999/1000` 里取出起点；解析失败返回 None。"""
    match = re.match(r'bytes\s+(\d+)-', (content_range or '').strip())
    return int(match.group(1)) if match else None


def acquire_tool_lock(home: Path):
    """同一数据目录同一时间只允许一个安装进程；Windows 用字节范围文件锁。"""
    if os.name != 'nt':
        return None
    import msvcrt
    folder = tools_folder(home)
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / 'install.lock').open('a+b')
    if handle.seek(0, 2) == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise InstallBusy('正在安装（可能另一个窗口正在安装），请稍候再试。') from error
    return handle


def _fetch_once(url: str, destination: Path, etag_file: Path, progress, cancel, timeout: int) -> Path:
    start = destination.stat().st_size if destination.exists() else 0
    headers = {'User-Agent': USER_AGENT}
    if start:
        headers['Range'] = f'bytes={start}-'
        try:
            etag = etag_file.read_text(encoding='utf-8').strip()
        except OSError:
            etag = ''
        if etag:
            headers['If-Range'] = etag
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, 'status', 200)
        if status == 206:
            # 断点续传：服务器必须从我们请求的位置开始，否则这份响应不可信。
            cr_start = _range_start(response.headers.get('Content-Range', ''))
            if start and cr_start == start:
                resume_from = start
            else:
                destination.unlink(missing_ok=True)
                raise OSError('服务器返回的断点位置与请求不符，已从头重新下载。')
        else:
            resume_from = 0  # 服务器忽略了 Range（返回 200 全量），从头写。
        etag = response.headers.get('ETag') or response.headers.get('Etag')
        if etag:
            try:
                etag_file.write_text(etag, encoding='utf-8')
            except OSError:
                pass
        remaining = _header_int(response.headers, 'Content-Length', 0)
        total = resume_from + remaining
        with destination.open('ab' if resume_from else 'wb') as file:
            while True:
                if cancel and cancel():
                    raise InstallCancelled('已取消安装。')
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                file.write(chunk)
                done = file.tell()
                _report(progress, done / total if total else 0, f'{done / 1048576:.0f} MB')
        if total and destination.stat().st_size != total:
            raise TruncatedDownload(f'下载不完整：期望 {total} 字节，实际只收到 {destination.stat().st_size}。')
    return destination


def fetch(url: str, destination: Path, progress=None, cancel=None, timeout: int = 30, retries: int = 2) -> Path:
    """下载到 destination，支持断点续传；cancel() 为真时抛出 InstallCancelled。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    etag_file = destination.with_name(destination.name + '.etag')
    last = RuntimeError('下载失败。')
    for attempt in range(retries + 1):
        if cancel and cancel():
            raise InstallCancelled('已取消安装。')
        try:
            return _fetch_once(url, destination, etag_file, progress, cancel, timeout)
        except InstallCancelled:
            raise
        except urllib.error.HTTPError as error:
            if error.code == 416 and destination.exists():
                # 断点已失效或越过文件末尾：丢弃残片，从头重下。
                destination.unlink(missing_ok=True)
                etag_file.unlink(missing_ok=True)
            last = error
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, TruncatedDownload) as error:
            last = error
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'下载失败：{last}') from last


def executable_version(executable: Path) -> str:
    """跑一次 -version，确认装好的程序真的能启动。"""
    try:
        run = subprocess.run([str(executable), '-version'], capture_output=True, timeout=60, creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ''
    text = run.stdout.decode('utf-8', 'replace').strip()
    if run.returncode or not text:
        return ''
    first = text.splitlines()[0].strip()
    return first if 'version' in first.lower() else ''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_binaries(archive: Path, staging: Path, cancel) -> dict:
    """把压缩包里 /bin/ 下的 ffmpeg.exe 与 ffprobe.exe 解到暂存目录，逐个落盘。"""
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(name).name: name for name in bundle.namelist() if '/bin/' in name.replace('\\', '/')}
        wanted = {name: members.get(name) for name in BINARIES}
        if not all(wanted.values()):
            raise RuntimeError('压缩包里没有找到 ffmpeg.exe / ffprobe.exe，可能是下载到的页面而不是压缩包。')
        for name in BINARIES:
            if cancel and cancel():
                raise InstallCancelled('已取消安装。')
            with bundle.open(wanted[name]) as source, (staging / name).open('wb') as output:
                shutil.copyfileobj(source, output)
            if os.name != 'nt':
                os.chmod(staging / name, 0o755)
    return wanted


def install_ffmpeg(home: Path, progress=None, cancel=None, url: str = FFMPEG_URL) -> Path:
    """下载并解压 FFmpeg 到 <home>/tools/ffmpeg，返回该文件夹。

    先在暂存目录解压并校验 ffmpeg.exe 与 ffprobe.exe 都能运行，成功后才替换到正式
    目录——任何一步失败（下载损坏、解压缺文件、二进制无法运行）都不会覆盖可用的旧安装。
    """
    target = ffmpeg_folder(home)
    tools = tools_folder(home)
    tools.mkdir(parents=True, exist_ok=True)
    archive = tools / 'downloads' / 'ffmpeg.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = tools / 'downloads' / ('.ffmpeg-stage-' + uuid.uuid4().hex + '.bin')
    staging.mkdir(parents=True, exist_ok=True)
    lock = acquire_tool_lock(home)
    try:
        _report(progress, 0, f'准备下载 FFmpeg（{FFMPEG_LABEL}）')
        fetch(url, archive, progress=lambda fraction, note: _report(progress, .05 + .78 * fraction, note),
              cancel=cancel, retries=2)
        if FFMPEG_SHA256:
            actual = _sha256(archive)
            if actual.lower() != FFMPEG_SHA256.lower():
                raise RuntimeError(f'下载的 FFmpeg 压缩包摘要不匹配（期望 {FFMPEG_SHA256[:12]}…，实际 {actual[:12]}…），已中止以防安装被篡改的文件。')
        _report(progress, .85, '解压 ffmpeg.exe 与 ffprobe.exe')
        _extract_binaries(archive, staging, cancel)
        if cancel and cancel():
            raise InstallCancelled('已取消安装。')
        _report(progress, .92, '检查能否运行')
        for name in BINARIES:
            version = executable_version(staging / name)
            if not version:
                raise RuntimeError(f'安装后的 {name} 无法运行，可能被安全软件拦截；也可以手动下载后选择文件夹。')
        if cancel and cancel():
            raise InstallCancelled('已取消安装。')
        _report(progress, .97, '替换已有安装')
        target.mkdir(parents=True, exist_ok=True)
        # 提交阶段是事务化的：先把现有二进制移到备份目录（腾出位置），再逐一放入新文件。
        # 若第二个文件替换失败（例如被占用），回滚已提交的部分并把旧文件还原，避免新旧混用。
        backup = tools / 'downloads' / ('.ffmpeg-backup-' + uuid.uuid4().hex + '.bin')
        backup.mkdir(parents=True, exist_ok=True)
        moved = []
        committed = []
        try:
            for name in BINARIES:
                old = target / name
                if old.is_file() and not old.is_symlink():
                    os.replace(old, backup / name)
                    moved.append(name)
            for name in BINARIES:
                os.replace(staging / name, target / name)
                committed.append(name)
        except Exception as error:
            for name in committed:
                try:
                    (target / name).unlink(missing_ok=True)
                except OSError:
                    pass
            unrestored = []
            for name in moved:
                if not (backup / name).is_file():
                    continue
                try:
                    os.replace(backup / name, target / name)
                except OSError:
                    unrestored.append(name)
            if unrestored:
                # 恢复也失败：保留备份目录并给出所在位置，不误报「已恢复」。
                raise RuntimeError(
                    f'安装时替换已有文件失败，且旧文件恢复也失败（未恢复：{"、".join(unrestored)}）。'
                    f'旧文件仍保留在备份目录，请手工把它们复制回 {target}：{backup}（原始错误：{error}）'
                ) from error
            shutil.rmtree(backup, ignore_errors=True)
            raise RuntimeError(f'安装时替换已有文件失败，原安装已恢复（{error}）') from error

        shutil.rmtree(backup, ignore_errors=True)
        _report(progress, 1, executable_version(target / 'ffmpeg.exe'))
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + '.etag').unlink(missing_ok=True)
        downloads = archive.parent
        try:
            downloads.rmdir()
        except OSError:
            pass
        if lock:
            lock.close()
from __future__ import annotations
import hashlib
import json
import math
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, fields
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from .portable import bundle_root, runtime_defaults


def data_home() -> Path:
    explicit = os.environ.get('VOICE_WORKBENCH_HOME')
    if explicit:
        return Path(explicit)
    portable = bundle_root()
    if portable is not None:
        return portable / 'data'
    return Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share')) / 'VoiceWorkbench'


@dataclass
class Settings:
    ffmpeg: str = ''
    rvc_root: str = ''
    python: str = ''
    model: str = ''
    index: str = ''
    separator_model: str = ''
    separator_config: str = ''
    separator_backend: str = 'bs_roformer'
    cookies: str = ''
    output: str = str(Path.home() / 'Music' / 'VoiceWorkbench')
    pitch: int = 2
    index_rate: float = .4
    accompaniment: float = .35

    @classmethod
    def load(cls, file: Path) -> 'Settings':
        values = json.loads(file.read_text(encoding='utf-8-sig')) if file.exists() else {}
        portable = runtime_defaults()
        if portable:
            values.update(portable)
            values.setdefault('output', str(file.parent / 'outputs'))
            for key in ('model', 'index', 'output'):
                if values.get(key) and not Path(values[key]).is_absolute():
                    values[key] = str((file.parent / values[key]).resolve())
        return cls(**{k: v for k, v in values.items() if k in {f.name for f in fields(cls)}})

    def save(self, file: Path):
        file.parent.mkdir(parents=True, exist_ok=True)
        pending = file.with_suffix('.pending')
        values = asdict(self)
        if bundle_root() is not None:
            for key in runtime_defaults():
                values.pop(key, None)
            for key in ('model', 'index', 'output'):
                if values.get(key):
                    try:
                        values[key] = Path(values[key]).resolve().relative_to(file.parent.resolve()).as_posix()
                    except ValueError:
                        pass
        pending.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(pending, file)

    def executable(self, name: str) -> str:
        extension = '.exe' if os.name == 'nt' else ''
        candidates = []
        if self.ffmpeg:
            supplied = Path(self.ffmpeg).expanduser()
            folder = supplied if supplied.is_dir() else supplied.parent
            candidates.append(folder / (name + extension))
        if self.rvc_root:
            candidates.append(Path(self.rvc_root) / (name + extension))
        candidates.append(data_home() / 'tools' / 'ffmpeg' / (name + extension))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        found = shutil.which(name)
        if found:
            return found
        raise ValueError(f'找不到 {name}，可以在「设置 → 环境检测」里一键安装 FFmpeg，或手动选择 FFmpeg 所在文件夹。')

    def engine_python(self) -> str:
        if self.python and Path(self.python).is_file():
            if getattr(sys, 'frozen', False) and Path(self.python).resolve() == Path(sys.executable).resolve():
                raise ValueError('请选择 RVC 环境中的 Python，不要选择本软件的 EXE。')
            return str(Path(self.python).resolve())
        bundled = Path(self.rvc_root) / 'runtime/python.exe'
        if bundled.is_file():
            return str(bundled.resolve())
        raise ValueError('请选择安装了 RVC 依赖的 Python 解释器。')

    def validate(self, convert: bool):
        self.executable('ffmpeg'); self.executable('ffprobe')
        if not str(self.output).strip():
            raise ValueError('请选择输出文件夹。')
        if self.cookies and not Path(self.cookies).is_file():
            raise ValueError('所选 Cookie 文件不存在。')
        if not convert:
            return
        root = Path(self.rvc_root)
        if not (root / 'infer/vc/modules.py').is_file():
            raise ValueError('请选择兼容的 RVC WebUI 根文件夹。')
        self.engine_python()
        if not self.model or not Path(self.model).is_file():
            raise ValueError('请选择角色的 .pth 模型。')
        if self.index and not Path(self.index).is_file():
            raise ValueError('检索索引文件不存在。')
        if not -24 <= int(self.pitch) <= 24 or not 0 <= float(self.index_rate) <= 1:
            raise ValueError('升调范围为 -24～24，检索比例范围为 0～1。')
        if not 0 <= float(self.accompaniment) <= 1:
            raise ValueError('伴奏比例应在 0～1 之间。')
        legacy_hubert = root / 'assets/hubert/hubert_base.pt'
        transformer_hubert = root / 'assets/hubert_base'
        if not legacy_hubert.is_file() and not (transformer_hubert / 'config.json').is_file():
            raise ValueError('RVC HuBERT 基础模型缺失，请先完成 RVC 环境配置。')
        for file in [root / 'assets/rmvpe/rmvpe.pt']:
            if not file.is_file():
                raise ValueError(f'RVC 基础文件缺失：{file.name}')
        self.separator_paths()

    def separator_paths(self):
        if self.separator_backend not in ('bs_roformer', 'demucs'):
            raise ValueError('不支持的人声分离引擎。')
        folder = Path(self.rvc_root) / 'assets/pymss_weights'
        stem = 'model_bs_roformer_ep_317_sdr_12.9755'
        model = Path(self.separator_model) if self.separator_model else folder / (stem + '.ckpt')
        config = Path(self.separator_config) if self.separator_config else folder / (stem + '.yaml')
        if not model.is_file() or not config.is_file():
            raise ValueError('找不到人声分离模型，请在设置中选择 .ckpt 与 .yaml。')
        return str(model.resolve()), str(config.resolve())


@dataclass
class Request:
    source: str
    mode: str = 'convert'
    name: str = ''
    start: float = 0
    end: float | None = None

    def validate(self):
        if self.mode not in ('original', 'convert'):
            raise ValueError('未知处理模式。')
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError('起点必须是大于或等于 0 的秒数。')
        if self.end is not None and (not math.isfinite(self.end) or self.end <= self.start):
            raise ValueError('终点必须大于起点。')
        if not self.source.strip():
            raise ValueError('请粘贴视频链接或选择本地音视频文件。')
        if is_url(self.source):
            parsed = urlsplit(self.source)
            if not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('视频链接格式不正确。')
        elif not Path(self.source).is_file():
            raise ValueError('本地文件不存在，或链接不是 http/https 地址。若是继续之前的任务，原文件可能已被移动、改名或删除，'
                             '请把它放回原位置，或重新添加该文件开始一个新任务。')


def is_url(text: str) -> bool:
    return urlsplit(text).scheme.lower() in ('http', 'https')


def safe_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text).strip(' .')[:80] or '音频'
    if text.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        text = '_' + text
    return text


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


STAGE_FILE = '阶段状态.json'
STAGES = ('download', 'extract', 'separate', 'convert', 'mix')
AUDIO_SETTINGS = ('model', 'index', 'pitch', 'index_rate', 'accompaniment', 'separator_model', 'separator_config', 'separator_backend')
SEPARATION_CACHE_BYTES = 6 * 2 ** 30
PIPELINE_VERSION = '2'  # 混音/分离/转换算法或 worker 行为一变就 +1，让相关阶段与缓存自动失效（Q05）
MAX_SEPARATION_SECONDS = 1800  # 分离把整轨读入内存，超过 30 分钟提示先裁剪，避免内存吃满（S03）


def pipeline_parts(settings, *workers) -> list:
    """引擎与 worker 脚本的版本标识：升级 RVC 环境或 worker 时，相关阶段自动失效（Q05）。"""
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    try:
        engine = file_stamp(settings.engine_python())
    except Exception:
        engine = []
    return [PIPELINE_VERSION, engine] + [file_stamp(root / name) for name in workers]


def acquire_task_lock(folder):
    """阻止两个进程同时操作同一任务目录（Q06）。Windows 用 msvcrt 文件锁；其余平台返回 None。"""
    if os.name != 'nt':
        return None
    import msvcrt
    Path(folder).mkdir(parents=True, exist_ok=True)
    handle = (Path(folder) / '.vwb-task.lock').open('a+b')
    if handle.seek(0, 2) == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise RuntimeError('这个任务文件夹正在被另一进程处理，请等它结束或换一个位置。') from error
    return handle


def fingerprint(*parts) -> str:
    """阶段指纹：参数或输入一变就失效，用于判断某个阶段能否直接复用。"""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]


def file_stamp(path) -> list:
    """便宜的文件标识（路径、大小、修改时间），避免为几百 MB 的模型反复做哈希。"""
    if not str(path or '').strip():
        return []
    file = Path(path)
    try:
        info = file.stat()
    except OSError:
        return []
    return [str(file.resolve()), info.st_size, info.st_mtime_ns]


def tempo_chain(tempo: float) -> str:
    """把任意变速比拆成若干 atempo= 片段，每个都落在 FFmpeg 合法的 [0.5, 2.0] 区间（A01）。"""
    if 0.5 <= tempo <= 2.0:
        return f'atempo={tempo}'
    if tempo < 0.5:
        k = max(2, math.ceil(math.log(tempo) / math.log(0.5)))
    else:
        k = max(2, math.ceil(math.log(tempo) / math.log(2.0)))
    step = tempo ** (1 / k)
    return ','.join(f'atempo={step}' for _ in range(k))


def pretty_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GB'


def load_request(folder: Path) -> 'Request':
    file = Path(folder) / STAGE_FILE
    try:
        state = json.loads(file.read_text(encoding='utf-8'))
        return Request(**state['request'])
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError('这个文件夹里没有可继续的任务记录。') from error


def task_settings(folder: Path, current: Settings) -> Settings:
    """Restore sound choices while keeping current runtime paths and credentials."""
    try:
        state = json.loads((Path(folder) / STAGE_FILE).read_text(encoding='utf-8'))
        snapshot = state.get('settings', {})
        if not isinstance(snapshot, dict): raise ValueError('任务参数记录无效。')
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError('无法读取这次任务的参数记录。') from error
    values = asdict(current)
    values.update({key: snapshot[key] for key in AUDIO_SETTINGS if key in snapshot})
    try:
        values['pitch'] = int(values['pitch'])
        values['index_rate'] = float(values['index_rate'])
        values['accompaniment'] = float(values['accompaniment'])
        if not (-24 <= values['pitch'] <= 24 and 0 <= values['index_rate'] <= 1 and 0 <= values['accompaniment'] <= 1):
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('保存的音色参数无效。') from error
    return Settings(**values)


class SeparationCache:
    """人声与伴奏按原音内容寻址；只改音高、检索比例或伴奏比重时不必重新分离。"""
    FILES = ('人声.wav', '伴奏.wav', '分离记录.json')
    MANIFEST = '缓存摘要.json'

    def __init__(self, home: Path):
        self.root = Path(home) / 'cache' / 'separation'

    @staticmethod
    def _wav_ok(path: Path) -> bool:
        """Q03：缓存命中的 WAV 必须是可打开、非空、帧数与文件长度一致的有效波形。"""
        import wave
        try:
            if path.stat().st_size < 44:
                return False
            with wave.open(str(path), 'rb') as wav:
                frames = wav.getnframes()
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                if frames <= 0 or not (1 <= channels <= 8) or not (1 <= width <= 4) or not (8000 <= rate <= 192000):
                    return False
                return path.stat().st_size >= 44 + frames * channels * width
        except (OSError, EOFError, ValueError, wave.Error):
            return False

    def key(self, original: Path, model: str, config: str, parts=()) -> str:
        return fingerprint('separation-v1', digest(original), file_stamp(model), file_stamp(config), *parts)

    def entry(self, key: str) -> Path:
        if not re.fullmatch(r'[0-9a-f]{16}', key):
            raise ValueError('缓存标识无效。')
        return self.root / key

    def has(self, key: str) -> bool:
        folder = self.entry(key)
        if not all((folder / name).is_file() for name in self.FILES):
            return False
        # Q03：空文件/损坏 WAV 不得命中缓存。
        if not self._wav_ok(folder / '人声.wav') or not self._wav_ok(folder / '伴奏.wav'):
            return False
        try:
            json.loads((folder / '分离记录.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        manifest = folder / self.MANIFEST
        if manifest.is_file():
            try:
                expected = json.loads(manifest.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                return False
            for name in self.FILES:
                if digest(folder / name) != expected.get(name, {}).get('sha256'):
                    return False
        return True

    def restore(self, key: str, folder: Path) -> bool:
        if not self.has(key):
            return False
        for name in self.FILES:
            shutil.copy2(self.entry(key) / name, Path(folder) / name)
        os.utime(self.entry(key), None)
        return True

    def store(self, key: str, folder: Path) -> bool:
        source = Path(folder)
        if self.has(key) or not all((source / name).is_file() for name in self.FILES):
            return self.has(key)
        target = self.entry(key)
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.pending')
        temporary.mkdir(parents=True)
        try:
            for name in self.FILES:
                shutil.copy2(source / name, temporary / name)
            manifest = {name: {'sha256': digest(temporary / name), 'size': (temporary / name).stat().st_size} for name in self.FILES}
            (temporary / self.MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            if target.is_symlink() or target.resolve().parent != self.root.resolve():
                raise ValueError('缓存路径不在缓存目录内。')
            if self.has(key): return True  # Another process published while copying.
            shutil.rmtree(target, ignore_errors=True)
            try:
                os.rename(temporary, target)
            except FileExistsError:
                if not self.has(key): raise
        finally:
            if temporary.resolve().parent == self.root.resolve():
                shutil.rmtree(temporary, ignore_errors=True)
        self.prune()
        return True

    def entries(self) -> list[tuple[Path, int]]:
        if not self.root.is_dir():
            return []
        found = []
        for folder in self.root.iterdir():
            if not re.fullmatch(r'[0-9a-f]{16}', folder.name) or not folder.is_dir() or folder.is_symlink():
                continue
            try:
                found.append((folder, sum(file.stat().st_size for file in folder.glob('*') if file.is_file())))
            except OSError:
                continue
        return found

    def size(self) -> int:
        return sum(size for _, size in self.entries())

    def count(self) -> int:
        return len(self.entries())

    def clear(self) -> int:
        removed = 0
        for folder, size in self.entries():
            shutil.rmtree(folder, ignore_errors=True)
            if not folder.exists():  # Q04：按真实删除结果统计，删除失败不计入成功。
                removed += size
        return removed

    def prune(self):
        if not self.root.is_dir():
            return
        entries, total = [], 0
        for folder in self.root.iterdir():
            if (not re.fullmatch(r'[0-9a-f]{16}', folder.name) or not folder.is_dir()
                    or folder.is_symlink() or folder.resolve().parent != self.root.resolve()):
                continue
            size = sum(file.stat().st_size for file in folder.glob('*') if file.is_file())
            entries.append((folder.stat().st_mtime, folder, size))
            total += size
        for _, folder, size in sorted(entries):
            if total <= SEPARATION_CACHE_BYTES:
                break
            shutil.rmtree(folder, ignore_errors=True)
            total -= size


def executable_runs(executable: str, args: tuple = ('-version',)) -> bool:
    """真正运行一次版本检查，判断程序能否执行（不只查文件存在）。E01。"""
    if not executable:
        return False
    try:
        result = subprocess.run([str(executable), *args], capture_output=True, timeout=10,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _same_drive(a: Path, b: Path) -> bool:
    """判断两个路径是否落在同一磁盘分区，用于区分数据盘与成品盘的空间检查（S02）。"""
    return os.path.splitdrive(os.path.abspath(str(a)))[0].lower() == os.path.splitdrive(os.path.abspath(str(b)))[0].lower()


def environment_report(settings: 'Settings', home: Path | None = None) -> list[dict]:
    """界面用的文件检查与临时写入探测，不加载深度学习模型。"""
    home = Path(home) if home is not None else data_home()
    items = []

    def add(key: str, label: str, state: str, detail: str, auto: bool = False):
        items.append({'key': key, 'label': label, 'state': state, 'detail': detail, 'auto': auto})

    found = {}
    for name in ('ffmpeg', 'ffprobe'):
        try:
            found[name] = settings.executable(name)
        except (ValueError, OSError):
            found[name] = ''
    if found['ffmpeg'] and found['ffprobe']:
        if executable_runs(found['ffmpeg']) and executable_runs(found['ffprobe']):
            add('ffmpeg', 'FFmpeg 与 FFprobe', 'ok', f'已找到且可运行：{Path(found["ffmpeg"]).parent}', auto=True)
        else:
            add('ffmpeg', 'FFmpeg 与 FFprobe', 'warn', f'文件存在但无法运行：{Path(found["ffmpeg"]).parent}，请更换可用版本或重新一键安装。', auto=True)
    else:
        missing = '、'.join(name for name in ('ffmpeg', 'ffprobe') if not found[name])
        add('ffmpeg', 'FFmpeg 与 FFprobe', 'missing', f'缺少 {missing}；提取原音、人声分离与合成都要用到它，可以一键安装（约 110 MB）。', auto=True)

    root = Path(settings.rvc_root).expanduser() if str(settings.rvc_root).strip() else None
    if root is not None and (root / 'infer/vc/modules.py').is_file():
        hubert = (root / 'assets/hubert/hubert_base.pt').is_file() or (root / 'assets/hubert_base/config.json').is_file()
        rmvpe = (root / 'assets/rmvpe/rmvpe.pt').is_file()
        missing_bits = []
        if not hubert:
            missing_bits.append('HuBERT 基础模型')
        if not rmvpe:
            missing_bits.append('RMVPE 音高模型')
        if missing_bits:
            add('rvc', 'RVC 环境', 'warn', f'目录存在，但缺少：{"、".join(missing_bits)}；转换音色会失败。')
        else:
            add('rvc', 'RVC 环境', 'ok', str(root))
    elif root is not None:
        add('rvc', 'RVC 环境', 'missing', '这个文件夹不是兼容的 RVC WebUI 根目录（缺 infer/vc/modules.py）。')
    else:
        add('rvc', 'RVC 环境', 'missing', '尚未选择；「转换音色」需要本机的 RVC WebUI 环境。')

    try:
        engine = settings.engine_python()
        runs = executable_runs(engine, ('--version',))
    except (ValueError, OSError) as error:
        add('python', 'RVC 的 Python', 'missing', str(error))
    else:
        if runs:
            add('python', 'RVC 的 Python', 'ok', engine)
        else:
            add('python', 'RVC 的 Python', 'warn', f'文件存在但无法运行：{engine}')

    if settings.model and Path(settings.model).is_file():
        add('model', '角色模型', 'ok', Path(settings.model).name)
    else:
        add('model', '角色模型', 'warn' if bundle_root() else 'missing',
            '还未导入音色。原音导出与训练可直接使用；换声前点击「导入音色」。')

    if settings.index and Path(settings.index).is_file():
        add('index', '检索索引', 'ok', Path(settings.index).name)
    elif settings.index:
        add('index', '检索索引', 'missing', '所选 .index 文件不存在。')
    else:
        add('index', '检索索引', 'warn', '未设置，音色贴合度会低一些；有 .index 时可以补上。')

    try:
        model, config = settings.separator_paths()
    except (ValueError, OSError) as error:
        add('separation', '人声分离模型', 'missing', str(error))
    else:
        add('separation', '人声分离模型', 'ok', Path(model).name)

    output = Path(settings.output).expanduser() if str(settings.output).strip() else None
    if output is None:
        add('output', '成品保存位置', 'missing', '尚未选择保存位置。')
    else:
        try:
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=output, prefix='.voice-workbench-') as probe_file:
                probe_file.write(b'probe'); probe_file.flush()
        except OSError as error:
            add('output', '成品保存位置', 'missing', f'无法写入 {output}：{error}')
        else:
            add('output', '成品保存位置', 'ok', str(output))

    try:
        import yt_dlp
    except Exception:
        add('downloader', '链接下载组件', 'warn', '未检测到 yt-dlp；贴链接下载会失败，本地文件仍可处理。')
    else:
        version = getattr(getattr(yt_dlp, 'version', None), '__version__', '已内置')
        add('downloader', '链接下载组件', 'ok', f'yt-dlp {version}；只处理单个公开视频链接。')

    def disk_free(folder: Path):
        if folder is None:
            return None
        try:
            while not folder.exists() and folder != folder.parent:
                folder = folder.parent
            return shutil.disk_usage(folder).free
        except OSError:
            return None

    # S02：数据盘（缓存/任务/试听）与成品盘（输出）分开检查，任一分区紧张都提示。
    home_free = disk_free(home)
    out_free = disk_free(output) if (output is not None and not _same_drive(home, output)) else None
    parts, low = [], []
    if home_free is not None:
        parts.append(f'数据盘 {pretty_bytes(home_free)} 可用')
        if home_free < 2 * 2 ** 30:
            low.append('数据盘')
    if out_free is not None:
        parts.append(f'成品盘 {pretty_bytes(out_free)} 可用')
        if out_free < 2 * 2 ** 30:
            low.append('成品盘')
    if low:
        add('disk', '磁盘剩余空间', 'warn', '、'.join(parts) + f'；{"、".join(low)}不足 2 GB，建议清理后继续。')
    elif parts:
        add('disk', '磁盘剩余空间', 'ok', '、'.join(parts))
    else:
        add('disk', '磁盘剩余空间', 'warn', '无法检测磁盘剩余空间。')

    cache = SeparationCache(home)
    try:
        size, count = cache.size(), cache.count()
    except OSError:
        size, count = 0, 0
    add('cache', '分离缓存', 'ok' if count else 'warn',
        f'{count} 份，占用 {pretty_bytes(size)}（上限 {pretty_bytes(SEPARATION_CACHE_BYTES)}）'
        if count else '还没有缓存；第一次转换音色后会在这里生成。')
    return items


class Store:
    def __init__(self, home: Path):
        home.mkdir(parents=True, exist_ok=True)
        self.file = home / 'tasks.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT, mode TEXT, state TEXT, stage TEXT, created TEXT, folder TEXT, error TEXT, result TEXT)')
            db.execute("UPDATE jobs SET state='interrupted', stage='上次运行中断，可重新提交' WHERE state IN ('queued','running','cancelling')")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.file, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, job: str, request: Request, folder: Path):
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)', (job, request.name or Path(request.source).stem[:60], request.mode, 'queued', '等待处理', time.strftime('%Y-%m-%d %H:%M:%S'), str(folder), '', ''))

    def update(self, job: str, **values):
        if not set(values) <= {'state', 'stage', 'error', 'result', 'name'}:
            raise ValueError('Invalid task fields')
        with self.connect() as db:
            db.execute('UPDATE jobs SET ' + ','.join(k + '=?' for k in values) + ' WHERE id=?', [*values.values(), job])

    def list(self):
        # Q07：活动任务（等待/处理/取消中）永远显示，即便它比最近 100 条历史更早，
        # 也不会被新任务挤出列表；历史任务照旧只保留最近 100 条。
        with self.connect() as db:
            active = db.execute("SELECT * FROM jobs WHERE state IN ('queued','running','cancelling') ORDER BY created DESC, rowid DESC").fetchall()
            recent = db.execute("SELECT * FROM jobs WHERE state NOT IN ('queued','running','cancelling') ORDER BY created DESC, rowid DESC LIMIT 100").fetchall()
        return [dict(row) for row in active] + [dict(row) for row in recent]

    def get(self, job: str):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job,)).fetchone()
        return dict(row) if row else None


class Cancelled(Exception):
    pass


class Task:
    def __init__(self, request: Request, settings: Settings, folder: Path, event=None, cancel=None, home: Path | None = None):
        self.request, self.settings, self.folder = request, settings, folder
        self.event = event or (lambda stage, percent=None: None)
        self.cancel_event = cancel or threading.Event()
        self.home = Path(home) if home else data_home()
        self.cache = SeparationCache(self.home)
        self.state = self.load_state()
        # 先把请求与有效参数落盘：任务哪怕在第一步就失败，也还能从详情里「继续制作」。
        self.state['request'] = asdict(self.request)
        self.state['settings'] = {key: getattr(self.settings, key) for key in AUDIO_SETTINGS}
        # TrainingTask owns creation of a fresh directory in run(). It has no
        # stage-resume route, so it must not create that directory here.
        if self.request.mode != 'train': self.save_state()
        self.process = None
        self.process_tree = None
        self.lock = threading.Lock()

    def load_state(self) -> dict:
        try:
            state = json.loads((self.folder / STAGE_FILE).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        state.setdefault('request', {})
        state.setdefault('stages', {})
        if not isinstance(state['stages'], dict):
            state['stages'] = {}
        return state

    def save_state(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        file = self.folder / STAGE_FILE
        pending = file.with_name(file.name + '.' + uuid.uuid4().hex + '.pending')  # Q06：多个进程共用 .pending 会双写崩溃
        pending.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(pending, file)

    def reuse(self, stage: str, fp: str):
        record = self.state['stages'].get(stage)
        if not isinstance(record, dict) or record.get('fingerprint') != fp:
            return None
        if record.get('cache_key'):
            artifacts = record.get('artifacts')
            if not artifacts or any(file_stamp(path) != stamp or not Path(path).is_file()
                                    for path, stamp in artifacts.items()):
                return None
            return record
        file = record.get('file')
        if file and (not Path(file).is_file() or record.get('file_stamp') != file_stamp(file)):
            return None
        return record

    def done(self, stage: str, fp: str, **values):
        # Rebuilding an upstream artifact invalidates its descendants, even if
        # a worker happens to preserve the same output size and timestamp.
        for later in STAGES[STAGES.index(stage) + 1:]:
            self.state['stages'].pop(later, None)
        if values.get('file'): values['file_stamp'] = file_stamp(values['file'])
        self.state['stages'][stage] = dict(fingerprint=fp, **values)
        self.save_state()

    def clear(self, *paths):
        for path in paths:
            if not path:
                continue
            file = Path(path)
            if file.is_file():
                file.unlink()

    def check(self):
        if self.cancel_event.is_set():
            raise Cancelled('已取消')

    def stop(self):
        self.cancel_event.set()
        with self.lock:
            if self.process and self.process.poll() is None:
                if self.process_tree: self.process_tree.stop()
                else: self.process.terminate()

    def run_command(self, args, cwd=None):
        self.check()
        from .processes import ProcessTree
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        if bundle_root() is not None:
            env.pop('PYTHONHOME', None)
            env.pop('PYTHONPATH', None)
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        with self.lock:
            self.check()
            self.process = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', creationflags=flags, start_new_session=os.name != 'nt')
            process = self.process
            try: self.process_tree = ProcessTree(process)
            except Exception:
                process.terminate(); process.wait(); process.stdout.close(); self.process = None
                raise RuntimeError('无法建立工作进程保护，未继续执行任务。')
        tree = self.process_tree
        def finish_children():
            process.wait()
            tree.stop()  # RVC exits with os._exit; loader children may still own stdout.
        watcher = threading.Thread(target=finish_children, daemon=True)
        watcher.start()
        tail = []
        try:
            with (self.folder / '处理日志.txt').open('a', encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line); log.flush()
                    tail.append(line.rstrip()); tail = tail[-12:]
                    if line.startswith('PROGRESS_JSON '):
                        data = json.loads(line[14:]); self.event(data['stage'], data.get('percent'))
                    elif line.startswith('PROGRESS '):
                        self.event(line.strip()[9:], None)
            code = process.wait()
        finally:
            tree.stop(); process.wait(); watcher.join(timeout=2)
            process.stdout.close()
            with self.lock:
                tree.close(); self.process_tree = None; self.process = None
        self.check()
        if code:
            raise RuntimeError('处理失败：' + '\n'.join(tail)[-2000:])

    def run_captured(self, args, timeout=None):
        """run_command 的静默兄弟：捕获输出、支持超时，并把进程纳入取消保护（Q08）。"""
        self.check()
        from .processes import ProcessTree
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        with self.lock:
            self.check()
            self.process = subprocess.Popen(list(map(str, args)), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', creationflags=flags, start_new_session=os.name != 'nt')
            process = self.process
            try:
                self.process_tree = ProcessTree(process)
            except Exception:
                process.terminate(); process.wait(); process.stdout.close(); process.stderr.close(); self.process = None
                raise RuntimeError('无法建立工作进程保护，未继续执行任务。')
        tree = self.process_tree
        timed_out = []
        watcher = None
        if timeout:
            def kill_on_timeout():
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out.append(True)
                    tree.stop()
            watcher = threading.Thread(target=kill_on_timeout, daemon=True)
            watcher.start()
        try:
            out, err = process.communicate()
        finally:
            if watcher:
                tree.stop(); process.wait(); watcher.join(timeout=2)
            process.stdout.close(); process.stderr.close()
            with self.lock:
                tree.close(); self.process_tree = None; self.process = None
        self.check()
        if timed_out:
            raise RuntimeError('处理超时，已中断。')
        return process.returncode, out, err

    def probe(self, path: Path) -> float:
        code, out, err = self.run_captured([self.settings.executable('ffprobe'), '-v', 'error', '-show_entries', 'format=duration:stream=codec_type', '-of', 'json', str(path)], timeout=120)
        if code:
            raise ValueError('无法读取音轨，请确认文件可正常播放。\n' + err[-500:])
        info = json.loads(out)
        if not any(s['codec_type'] == 'audio' for s in info.get('streams', [])):
            raise ValueError('此文件没有可用音轨。')
        duration = float(info['format']['duration'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('无法读取音频时长。')
        return duration

    def download(self):
        if not is_url(self.request.source):
            return Path(self.request.source).resolve(), Path(self.request.source).stem
        try:
            import yt_dlp
        except ImportError as error:
            raise ValueError('缺少下载模块：请安装 requirements.txt，或使用 Windows 打包版。') from error
        self.event('解析并下载音轨', 5)
        last_progress = [0.]
        def progress(data):
            self.check()
            total = data.get('total_bytes') or data.get('total_bytes_estimate')
            if total and time.monotonic() - last_progress[0] >= .25:
                last_progress[0] = time.monotonic()
                self.event('下载音轨', min(20, 5 + 15 * data.get('downloaded_bytes', 0) / total))
        class Logger:
            def debug(self, text): pass
            def warning(self, text): pass
            def error(self, text): pass
        opts = dict(format='bestaudio/best', outtmpl=str(self.folder / '下载音轨.%(ext)s'), noplaylist=True, quiet=True, no_warnings=True, retries=2, socket_timeout=15, progress_hooks=[progress], logger=Logger(), ffmpeg_location=str(Path(self.settings.executable('ffmpeg')).parent))
        if self.settings.cookies:
            opts['cookiefile'] = self.settings.cookies
        try:
            with yt_dlp.YoutubeDL(opts) as downloader:
                info = downloader.extract_info(self.request.source, download=True)
                if not info or 'entries' in info:
                    raise ValueError('请使用单个视频链接。')
                candidates = [Path(item['filepath']) for item in info.get('requested_downloads', []) if item.get('filepath')]
                candidates.append(Path(downloader.prepare_filename(info)))
                file = next((p for p in candidates if p.is_file() and p.parent.resolve() == self.folder.resolve()), None)
                if file is None:
                    raise ValueError('没有找到下载的音轨。')
                return file, info.get('title') or '视频音频'
        except Exception as error:
            self.check()
            raise RuntimeError('链接解析或下载失败。请检查链接、网络和视频访问权限，也可下载后选择本地文件。\n' + str(error)[:800]) from error

    def run(self):
        """四段流水线：提取 → 分离 → 转换 → 混音，每段独立可复用。"""
        self.request.validate(); self.settings.validate(self.request.mode == 'convert')
        self.folder.mkdir(parents=True, exist_ok=True)
        lock = acquire_task_lock(self.folder)
        try:
            self.state['request'] = asdict(self.request); self.save_state()
            source, title = self.stage_download(); self.check()
            title = safe_name(self.request.name or title)
            total = self.probe(source)
            if self.request.start >= total or (self.request.end is not None and self.request.end > total + .1):
                raise ValueError(f'裁剪范围超出音频时长（{total:.2f} 秒）。')
            length = (self.request.end if self.request.end is not None else total) - self.request.start
            if self.request.mode == 'convert' and length > MAX_SEPARATION_SECONDS:
                raise ValueError(f'这段音频长达 {length / 60:.1f} 分钟，人声分离会把整轨读入内存，容易耗尽内存。'
                                 f'请在「命名与裁剪」里把片段缩到 {MAX_SEPARATION_SECONDS // 60} 分钟以内，或分段制作。')
            original = self.stage_extract(source, title, length)
            output = original
            if self.request.mode == 'convert':
                vocals, accompaniment = self.stage_separate(original)
                vocals = self.stage_convert(vocals)
                output = self.stage_mix(title, length, vocals, accompaniment)
            self.stage_verify(output, length, title)
            self.check(); self.event('完成', 100)
            return output, title
        finally:
            if lock is not None:
                lock.close()

    def stage_download(self):
        fp = fingerprint('download', self.request.source)
        record = self.reuse('download', fp)
        if record:
            return Path(record['file']), record.get('title') or ''
        file, title = self.download()
        self.done('download', fp, file=str(file), title=title)
        return file, title

    def stage_extract(self, source: Path, title: str, length: float) -> Path:
        original = self.folder / (title + '_原音.wav')
        fp = fingerprint('extract', file_stamp(source), self.request.start, self.request.end, title)
        record = self.reuse('extract', fp)
        if record:
            return Path(record['file'])
        ff = self.settings.executable('ffmpeg')
        self.event('提取 WAV', 25)
        args = [ff, '-nostdin', '-v', 'error', '-n', '-i', source, '-map', '0:a:0', '-vn', '-ss', str(self.request.start)]
        if self.request.end is not None:
            args += ['-t', str(length)]
        self.clear(original)
        self.run_command(args + ['-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', original])
        self.done('extract', fp, file=str(original))
        return original

    def stage_separate(self, original: Path):
        model, yaml = self.settings.separator_paths()
        key = self.cache.key(original, model, yaml, [self.settings.separator_backend, *pipeline_parts(self.settings, 'workers/separate_worker.py')])
        self.event('分离人声与伴奏（多轮处理）', 35)
        record = self.reuse('separate', key)
        if record is None:
            try:
                restored = self.cache.restore(key, self.folder)
            except OSError:
                restored = False
            if restored:
                self.done('separate', key, cache_key=key,
                          artifacts={str(self.folder / name): file_stamp(self.folder / name) for name in SeparationCache.FILES})
                record = {'cache_key': key}
        if record:
            self.event('复用已分离的人声与伴奏', 35)
            return self.folder / '人声.wav', self.folder / '伴奏.wav'
        config = asdict(self.settings)
        config.update(input=str(original), folder=str(self.folder), separation_model=model, separation_yaml=yaml)
        config_file = self.folder / '运行参数.local.json'
        config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        python = self.settings.engine_python()
        root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
        self.clear(*[self.folder / name for name in SeparationCache.FILES])
        self.run_command([python, '-u', root / 'workers/separate_worker.py', config_file])
        try:
            self.cache.store(key, self.folder)
        except OSError:
            self.event('共享缓存未保存，继续使用本次分离结果', 35)
        self.done('separate', key, cache_key=key,
                  artifacts={str(self.folder / name): file_stamp(self.folder / name) for name in SeparationCache.FILES})
        return self.folder / '人声.wav', self.folder / '伴奏.wav'

    def stage_convert(self, vocals: Path) -> Path:
        output = self.folder / '转换后人声.wav'
        fp = fingerprint('convert', file_stamp(vocals), file_stamp(self.settings.model), file_stamp(self.settings.index),
                         int(self.settings.pitch), round(float(self.settings.index_rate), 4),
                         *pipeline_parts(self.settings, 'workers/rvc_worker.py'))
        record = self.reuse('convert', fp)
        if record:
            self.event('复用已转换的人声', 65)
            return Path(record['file'])
        self.event('转换角色音色', 65)
        config = asdict(self.settings)
        config.update(input=str(vocals), folder=str(self.folder))
        config_file = self.folder / '运行参数.local.json'
        config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        python = self.settings.engine_python()
        root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
        self.clear(output, self.folder / '变声记录.json')
        self.run_command([python, '-u', root / 'workers/rvc_worker.py', config_file], cwd=self.settings.rvc_root)
        self.done('convert', fp, file=str(output))
        return output

    def stage_mix(self, title: str, length: float, vocals: Path, accompaniment: Path) -> Path:
        output = self.folder / (title + '_角色翻唱.wav')
        fp = fingerprint('mix', title, file_stamp(vocals), file_stamp(accompaniment), int(self.settings.pitch),
                         round(float(self.settings.accompaniment), 4), round(length, 3), PIPELINE_VERSION)
        record = self.reuse('mix', fp)
        if record:
            self.event('复用已合成的成品', 85)
            return Path(record['file'])
        self.event('合成完整音频', 85)
        ff = self.settings.executable('ffmpeg')
        shifted = self.folder / '变调伴奏.wav'
        if self.settings.pitch:
            factor = 2 ** (self.settings.pitch / 12)
            self.clear(shifted)
            self.run_command([ff, '-nostdin', '-v', 'error', '-n', '-i', accompaniment, '-af', f'aresample=40000,asetrate=40000*{factor},aresample=44100,{tempo_chain(1 / factor)},apad,atrim=duration={length}', '-c:a', 'pcm_s16le', shifted])
            accompaniment = shifted
        def loud(file):
            _code, _out, err = self.run_captured([ff, '-nostdin', '-hide_banner', '-nostats', '-i', str(file), '-af', 'ebur128', '-f', 'null', '-'], timeout=600)
            values = re.findall(r'I:\s*(-?\d+\.\d+) LUFS', err)
            return float(values[-1]) if values else -100
        lv, la = loud(vocals), loud(accompaniment)
        gain = 0 if la < -80 else min(1 if lv - la > 35 else 4, 10 ** ((lv - la) / 20) * self.settings.accompaniment)
        fmt = 'aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo'
        fc = f'[0:a]{fmt},highpass=f=65,equalizer=f=250:t=q:w=0.8:g=-1.3,equalizer=f=3600:t=q:w=0.9:g=1.2,asplit=2[v][key];[1:a]{fmt},volume={gain}[a];[a][key]sidechaincompress=threshold=0.03:ratio=4:attack=8:release=240[m];[v][m]amix=inputs=2:duration=first,loudnorm=I=-16:TP=-1.5:LRA=11[out]'
        self.clear(output)
        self.run_command([ff, '-nostdin', '-v', 'error', '-n', '-i', vocals, '-i', accompaniment, '-filter_complex', fc, '-map', '[out]', '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', output])
        self.done('mix', fp, file=str(output))
        return output

    def stage_verify(self, output: Path, length: float, title: str):
        ff = self.settings.executable('ffmpeg')
        self.event('检查成品', 95)
        self.run_command([ff, '-nostdin', '-v', 'error', '-xerror', '-i', output, '-f', 'null', '-'])
        actual = self.probe(output)
        if abs(actual - length) > .35:
            raise RuntimeError('成品与原素材时长不一致，任务没有标记完成。')
        if self.request.mode == 'convert':
            self._check_signal(output)
        manifest = dict(name=title, mode=self.request.mode, source=self.request.source, duration=actual, start=self.request.start, end=self.request.end, file=str(output), sha256=digest(output), created=time.strftime('%Y-%m-%d %H:%M:%S'))
        if self.request.mode == 'convert':
            manifest.update(model_sha256=digest(Path(self.settings.model)), index_sha256=digest(Path(self.settings.index)) if self.settings.index else None, pitch=self.settings.pitch)
        (self.folder / '成品记录.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    def _check_signal(self, output: Path):
        """A02：角色转换增加响度与削波检查，全静音或削波的成品不能标记完成。"""
        ff = self.settings.executable('ffmpeg')
        _code, _out, err = self.run_captured([ff, '-nostdin', '-hide_banner', '-nostats', '-i', str(output), '-af', 'ebur128', '-f', 'null', '-'], timeout=600)
        loudness = re.findall(r'I:\s*(-?\d+\.\d+) LUFS', err)
        integrated = float(loudness[-1]) if loudness else -100
        if integrated < -60:
            raise RuntimeError('成品几乎全是静音，已判定为失败（可能转换失败或素材无声）。')
        _code, _out, err = self.run_captured([ff, '-nostdin', '-hide_banner', '-nostats', '-i', str(output), '-af', 'volumedetect', '-f', 'null', '-'], timeout=600)
        peaks = re.findall(r'max_volume:\s*(-?\d+\.\d+) dB', err)
        if peaks and float(peaks[-1]) >= -0.1:
            raise RuntimeError('成品出现削波（音量触顶），已判定为失败。')


class Manager:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.store = Store(self.home); self.events = queue.Queue(); self.queue = queue.Queue()
        self.tasks = {}
        self.pending = {}
        self._store_lock = threading.Lock()
        self.retry_lock = threading.Lock()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, request: Request, settings: Settings):
        request.validate()
        task_class = Task
        if request.mode == 'train':
            from .training import TrainingTask, validate_training_environment
            validate_training_environment(settings, request); task_class = TrainingTask
        else: settings.validate(request.mode == 'convert')
        job = uuid.uuid4().hex[:12]
        folder = Path(settings.output).expanduser().resolve() / (time.strftime('%Y%m%d_%H%M%S') + '_' + job)
        self.store.add(job, request, folder)
        def event(stage, percent=None):
            self.store.update(job, stage=stage); self.events.put((job, stage, percent))
        try:
            task = task_class(request, settings, folder, event=event, home=self.home)
        except Exception as error:
            # Q01：任务构造失败会留下没人接手的「等待中」记录，这里直接落成失败。
            self._safe_update(job, state='failed', stage='提交失败', error=f'任务未能加入队列：{error}')
            self.events.put((job, 'refresh', None))
            raise
        self.tasks[job] = task; self.queue.put((job, task)); return job

    def _safe_update(self, job: str, **values):
        """写任务状态失败时保留待补写，数据库恢复后再重试；同时向界面报告存储异常。

        既不能让唯一工作线程崩溃（Q02），也不能悄悄吞掉写入失败——否则任务会永久停在
        「处理中」且无法重试（复核缺口 2）。写失败时记录最终状态并发出 storage_error 事件。
        """
        try:
            self.store.update(job, **values)
            with self._store_lock:
                self.pending.pop(job, None)
        except Exception as error:
            with self._store_lock:
                self.pending[job] = {**self.pending.get(job, {}), **values}
            self.events.put((job, 'storage_error', str(error)))

    def flush_pending(self):
        """数据库恢复后的状态补写：把之前没写进去的任务最终状态补上。"""
        with self._store_lock:
            items = list(self.pending.items())
        for job, values in items:
            try:
                self.store.update(job, **values)
                with self._store_lock:
                    self.pending.pop(job, None)
                self.events.put((job, 'refresh', None))
            except Exception:
                pass

    def retry(self, job: str):
        """在同一任务文件夹里重跑：已完成的阶段直接复用，只执行失效与后续阶段。"""
        with self.retry_lock:
            return self._retry(job)

    def _retry(self, job: str):
        # 先把之前没写进去的最终状态补上（数据库若已恢复，这里会把陈旧 running 更正掉），
        # 否则任务会被误判为「正在处理」而拒绝重试。
        self.flush_pending()
        row = self.store.get(job)
        if row is None:
            raise ValueError('任务不存在。')
        if job in self.tasks or row['state'] in ('queued', 'running', 'cancelling'):
            raise ValueError('任务正在处理或等待中，请结束后再重试。')
        if row['mode'] == 'train':
            raise ValueError('训练任务请重新提交。')
        folder = Path(row['folder'])
        request = load_request(folder)
        settings = task_settings(folder, Settings.load(self.home / 'settings.json'))
        if request.mode == 'train':
            raise ValueError('训练任务请重新提交。')
        settings.validate(request.mode == 'convert')
        def event(stage, percent=None):
            self.store.update(job, stage=stage); self.events.put((job, stage, percent))
        task = Task(request, settings, folder, event=event, home=self.home)
        self.store.update(job, state='queued', stage='等待重跑', error='', result='')
        self.tasks[job] = task; self.queue.put((job, task)); return job

    def cancel(self, job):
        task = self.tasks.get(job)
        if task:
            task.stop(); self.store.update(job, state='cancelling', stage='正在取消')

    def shutdown(self):
        for task in list(self.tasks.values()): task.stop()
        self.queue.put(None)
        self.thread.join(timeout=2)

    def _work(self):
        while True:
            self.flush_pending()
            item = self.queue.get()
            if item is None:
                self.queue.task_done(); return
            job, task = item
            try:
                task.check(); self.store.update(job, state='running')
                result, title = task.run()
                self._safe_update(job, state='completed', stage='完成', result=str(result), name=title)
            except Cancelled:
                self._safe_update(job, state='cancelled', stage='已取消')
            except Exception as error:
                self._safe_update(job, state='failed', stage='处理失败', error=str(error))
            finally:
                self.tasks.pop(job, None); self.events.put((job, 'refresh', None)); self.queue.task_done()

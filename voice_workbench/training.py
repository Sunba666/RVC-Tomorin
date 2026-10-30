"""Desktop-facing training orchestration; ML dependencies stay in external workers."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import shutil
import sys

from .core import Settings, Task, digest, safe_name

MEDIA_SUFFIXES = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.aiff', '.wma', '.mp4', '.mkv', '.webm', '.mov', '.avi', '.m4v', '.ts'}


def collect_materials(folder):
    import os
    result = []
    for current, directories, files in os.walk(folder, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.'))
        result += [str(Path(current, name).resolve()) for name in sorted(files) if Path(name).suffix.lower() in MEDIA_SUFFIXES]
        if len(result) > 2000: raise ValueError('素材超过 2000 个，请分批导入。')
    return result


@dataclass
class TrainingRequest:
    sources: list[str]
    name: str = '我的音色'
    epochs: int = 200
    batch_size: int = 4
    sample_rate: str = '40k'
    save_every: int = 10
    learning_rate: float = .0001
    extract_vocals: bool = True
    quality: str = 'balanced'
    keep_checkpoints: bool = False
    mode: str = field(default='train', init=False)

    @property
    def source(self):
        return self.sources[0] if self.sources else ''

    def validate(self):
        if not self.sources: raise ValueError('请添加同一人的人声素材或音视频。')
        if len(self.sources) > 2000: raise ValueError('单个训练任务最多使用 2000 个素材。')
        for value in self.sources:
            path = Path(value)
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                raise ValueError('素材不存在或格式不受支持：' + path.name)
        self.validate_parameters()

    def validate_parameters(self):
        if not self.name.strip(): raise ValueError('请为新模型填写名称。')
        for value, low, high, label in [(self.epochs, 1, 1200, '训练轮次'), (self.batch_size, 1, 32, '批大小'), (self.save_every, 1, 1200, '保存间隔')]:
            if type(value) is not int or not low <= value <= high: raise ValueError(f'{label}范围为 {low}～{high}。')
        if self.sample_rate not in ('32k', '40k', '48k'): raise ValueError('采样率应为 32k、40k 或 48k。')
        if not math.isfinite(self.learning_rate) or not .000001 <= self.learning_rate <= .001:
            raise ValueError('学习率范围为 0.000001～0.001。')
        if self.quality not in ('balanced', 'strict'): raise ValueError('未知素材筛选强度。')


def validate_training_environment(settings: Settings, request: TrainingRequest):
    settings.validate(False)
    settings.engine_python()
    root = Path(settings.rvc_root)
    required = ['train/train.py', 'train/utils.py', 'train/preprocess.py', 'train/dataset/extract_f0.py',
                'train/dataset/extract_hubert_feature.py', 'infer/hubert.py', 'assets/hubert_base/config.json',
                'assets/rmvpe/rmvpe.pt', f'assets/pretrained_v2/f0G{request.sample_rate}.pth',
                f'assets/pretrained_v2/f0D{request.sample_rate}.pth',
                'configs/' + ('v1/' if request.sample_rate == '40k' else 'v2/') + request.sample_rate + '.json']
    for file in required:
        if not (root / file).is_file(): raise ValueError('训练环境缺少 ' + file + '。当前自动训练适配带 train/ 与 Transformers HuBERT 的 RVC 环境。')
    if request.extract_vocals: settings.separator_paths()
    folder = Path(settings.output).expanduser().resolve()
    while not folder.exists(): folder = folder.parent
    if shutil.disk_usage(folder).free < 2 * 2**30: raise ValueError('训练所在磁盘至少需要 2 GB 空闲空间，请更换保存位置。')


class TrainingTask(Task):
    def run(self):
        request, settings = self.request, self.settings
        request.validate(); validate_training_environment(settings, request)
        self.folder.mkdir(parents=True, exist_ok=False)
        assets = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1])) / 'workers'
        worker = assets / 'train_worker.py'
        config = asdict(settings)
        config.update(training=asdict(request), folder=str(self.folder), experiment=str(self.folder / 'workspace'), export_name=safe_name(request.name), candidates=[])
        config_file = self.folder / 'training.local.json'
        source_reports = []
        ff = settings.executable('ffmpeg')
        for number, value in enumerate(dict.fromkeys(request.sources)):
            self.check()
            path = Path(value).resolve()
            duration = self.probe(path)
            location = self.folder / 'materials' / f'{number:04d}'
            location.mkdir(parents=True)
            self.event(f'提取素材 {number + 1}/{len(request.sources)}：{path.name}', 2 + 8 * number / len(request.sources))
            self.run_command([ff, '-nostdin', '-v', 'error', '-n', '-i', path, '-map', '0:a:0', '-vn',
                              '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', '-f', 'segment', '-segment_time', '120',
                              '-reset_timestamps', '1', location / 'audio_%04d.wav'])
            source_reports.append({'file': str(path), 'duration': duration})
            for piece in sorted(location.glob('audio_*.wav')):
                self.check()
                item = {'input': str(piece), 'source': str(path)}
                if request.extract_vocals:
                    separated = location / piece.stem
                    separated.mkdir()
                    model, yaml = settings.separator_paths()
                    separation = dict(config, input=str(piece), folder=str(separated), separation_model=model, separation_yaml=yaml)
                    separate_config = separated / 'separation.local.json'
                    separate_config.write_text(json.dumps(separation, ensure_ascii=False), encoding='utf-8')
                    self.event(f'自动分离人声：{path.name}', 10 + 20 * number / len(request.sources))
                    self.run_command([settings.engine_python(), assets / 'separate_worker.py', separate_config])
                    item.update(vocals=str(separated / '人声.wav'), accompaniment=str(separated / '伴奏.wav'))
                else:
                    item['vocals'] = str(piece)
                config['candidates'].append(item)
        config['sources'] = source_reports
        config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        for action, label, percent in [('prepare', '筛选人声与清理静音', 32), ('preprocess', '切分训练片段', 42),
                                       ('f0', '提取音高', 48), ('features', '提取音色特征', 55),
                                       ('manifest', '检查训练数据', 65), ('train', '训练模型', 70), ('export', '生成检索索引并导出模型', 93)]:
            self.check(); self.event(label, percent)
            self.run_command([settings.engine_python(), worker, config_file, action], cwd=settings.rvc_root)
        report_file = self.folder / '训练记录.json'
        report = json.loads(report_file.read_text(encoding='utf-8'))
        if report.get('epochs_completed') != request.epochs: raise RuntimeError('实际训练轮次与请求不符，未标记完成。')
        for path_key, sha_key in [('model', 'model_sha256'), ('index', 'index_sha256')]:
            file = Path(report[path_key])
            if not file.is_file() or digest(file) != report[sha_key]: raise RuntimeError('导出模型校验失败。')
        self.check()
        if not request.keep_checkpoints:
            workspace = (self.folder / 'workspace').resolve()
            for pattern in ('G_*.pth', 'D_*.pth'):
                for path in workspace.glob(pattern):
                    if path.resolve().parent != workspace: raise RuntimeError('检查点路径不在训练目录内。')
                    path.unlink()
        self.event('训练完成，可以选择使用新模型', 100)
        return report_file, safe_name(request.name)


def trained_settings(settings: Settings, report_file: Path):
    report = json.loads(report_file.read_text(encoding='utf-8'))
    for key in ('model', 'index'):
        path = Path(report[key])
        if not path.is_file() or digest(path) != report[key + '_sha256']:
            raise ValueError('模型文件缺失或校验不符，请重新导出。')
    # 兼容性预检：旧记录没有这些字段时退回仅 sha256；新记录则严格核对。
    if report.get('model_version') and report['model_version'] != 'v2':
        raise ValueError('模型版本不受支持，请用当前训练流程重新导出。')
    if report.get('model_f0') is not None and report['model_f0'] != 1:
        raise ValueError('模型音高方法不支持，请重新导出。')
    if report.get('model_sr') and report['model_sr'] != report.get('training', {}).get('sample_rate'):
        raise ValueError('模型采样率与训练配置不一致，请重新导出。')
    if report.get('index_dim') and report['index_dim'] != 768:
        raise ValueError('检索索引维度无效，请重新导出。')
    if report.get('index_vectors') is not None and int(report['index_vectors']) <= 0:
        raise ValueError('检索索引为空，请重新训练并导出。')
    values = asdict(settings)
    values.update(model=report['model'], index=report['index'])
    return Settings(**values)

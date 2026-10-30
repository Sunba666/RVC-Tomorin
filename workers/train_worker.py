"""RVC v2 training adapter. The user's engine stays unchanged.

RVC checkpoint layout follows RVC-Project (MIT), used as an external dependency.
This module also initializes spawn children before their training functions load.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import runpy
import shutil
import sys


def configure(file):
    config = json.loads(Path(file).read_text(encoding='utf-8-sig'))
    root = Path(config['rvc_root']).resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(Path(__file__).resolve().parent)]
    sys.dont_write_bytecode = True
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', RVC_CUDA_GRAPH='0',
                      CUDA_VISIBLE_DEVICES='-1' if os.environ.get('VOICE_WORKBENCH_CPU') == '1' else '0',
                      RVC_AUDIO_FORCE_CPU='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ['PATH'] = str(root) + os.pathsep + os.environ.get('PATH', '')
    import torch
    torch.set_num_threads(4)
    return config


def engine_stage(config, stage):
    root, exp = Path(config['rvc_root']), Path(config['experiment'])
    request = config['training']
    import torch
    cuda = torch.cuda.is_available()
    if stage == 'preprocess':
        exp.mkdir(parents=True, exist_ok=True)
        args = [str(Path(config['folder']) / 'clean_vocals'), str(int(request['sample_rate'][:-1]) * 1000), '1', str(exp), 'True', '3.7']
        script = 'train/preprocess.py'
    elif stage == 'f0':
        script = 'train/dataset/extract_f0.py'
        args = ['cuda', '1', '0', '0', str(exp), 'false'] if cuda else ['cpu', str(exp), '1', 'rmvpe']
    else:
        script = 'train/dataset/extract_hubert_feature.py'
        args = ['cuda:0', '1', '0', '0', str(exp), 'v2', 'false'] if cuda else ['cpu', '1', '0', str(exp), 'v2', 'false']
    sys.argv = [str(root / script), *args]
    runpy.run_path(str(root / script), run_name='__main__')


def manifest(config):
    import numpy as np
    import soundfile as sf
    folder, exp = Path(config['folder']), Path(config['experiment'])
    request = config['training']
    accepted = []
    rejected = []
    for wav in sorted((exp / '0_gt_wavs').glob('*.wav')):
        paths = [wav, exp / '3_feature768' / (wav.stem + '.npy'), exp / '2a_f0' / (wav.name + '.npy'), exp / '2b-f0nsf' / (wav.name + '.npy')]
        try:
            feature, f0, f0f = [np.load(p, allow_pickle=False) for p in paths[1:]]
            if feature.ndim != 2 or feature.shape[1] != 768 or len(feature) < 25:
                raise ValueError('特征无效')
            if f0.ndim != 1 or f0f.ndim != 1 or len(f0) == 0 or len(f0f) == 0:
                raise ValueError('音高数组为空或维度错误')
            if not all(np.isfinite(x).all() for x in (feature, f0, f0f)):
                raise ValueError('特征无效')
            frames = len(feature)
            if not (0.05 <= len(f0) / frames <= 20.0 and 0.05 <= len(f0f) / frames <= 20.0):
                raise ValueError('音高与特征帧长严重不匹配')
            for array in (f0, f0f):
                if float(np.nanmax(array)) > 4000 or float(np.nanmin(array)) < -1:
                    raise ValueError('音高超出合理范围')
            accepted.append(paths)
        except Exception: rejected.append(wav.name)
    if len(accepted) < 2: raise ValueError('完整的音频/音高/特征少于 2 段，请检查训练日志或补充素材。')
    duration = sum(sf.info(str(paths[0])).duration for paths in accepted)
    if duration < 30: raise ValueError('音高和特征提取后有效片段不足 30 秒，请补充清晰素材。')
    (exp / 'filelist.txt').write_text('\n'.join('|'.join(p.as_posix() for p in paths) + '|0' for paths in accepted), encoding='utf-8')
    config_path = Path(config['rvc_root']) / 'configs' / ('v1' if request['sample_rate'] == '40k' else 'v2') / (request['sample_rate'] + '.json')
    training = json.loads(config_path.read_text(encoding='utf-8'))
    training['train'].update(epochs=request['epochs'], batch_size=request['batch_size'], learning_rate=request['learning_rate'], log_interval=10)
    (exp / 'config.json').write_text(json.dumps(training, indent=2), encoding='utf-8')
    (folder / '训练数据检查.json').write_text(json.dumps({'accepted': len(accepted), 'rejected': rejected, 'duration_seconds': duration}, ensure_ascii=False, indent=2), encoding='utf-8')


def install_training_adapter(config):
    """Patch only this process, including spawn children; no engine files are edited."""
    import torch
    from train import utils, process_ckpt
    original_loader = torch.utils.data.DataLoader
    class LocalLoader(original_loader):
        def __init__(self, *args, **kwargs):
            # A single local worker avoids four full Windows copies of the ML
            # runtime. GPU training remains in the dedicated external process.
            kwargs.update(num_workers=0, persistent_workers=False, prefetch_factor=None)
            super().__init__(*args, **kwargs)
    torch.utils.data.DataLoader = LocalLoader
    request = config['training']
    exp = Path(config['experiment'])
    def hparams():
        values = utils.HParams(**json.loads((exp / 'config.json').read_text(encoding='utf-8')))
        values.model_dir = values.experiment_dir = str(exp)
        values.data.training_files = str(exp / 'filelist.txt')
        values.name = 'voice_workbench'
        values.save_every_epoch = min(request['save_every'], request['epochs'])
        values.total_epoch = request['epochs']
        values.pretrainG = str(Path(config['rvc_root']) / f"assets/pretrained_v2/f0G{request['sample_rate']}.pth")
        values.pretrainD = str(Path(config['rvc_root']) / f"assets/pretrained_v2/f0D{request['sample_rate']}.pth")
        values.version, values.gpus, values.sample_rate = 'v2', '0', request['sample_rate']
        values.if_f0, values.if_latest, values.save_every_weights, values.if_cache_data_in_gpu = 1, 1, '0', 0
        return values
    utils.get_hparams = hparams
    original_logger = utils.get_logger
    class Progress(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if '====>' in message:
                number = re.search(r'\d+', message)
                if number:
                    epoch = int(number.group())
                    print('PROGRESS_JSON ' + json.dumps({'stage': f"训练轮次 {epoch}/{request['epochs']}", 'percent': 70 + 21 * min(epoch / request['epochs'], 1)}, ensure_ascii=False), flush=True)
    def logger(*args, **kwargs):
        result = original_logger(*args, **kwargs)
        result.addHandler(Progress()); return result
    utils.get_logger = logger

    def save(ckpt, sr, f0, name, epoch, version, hps):
        # Inference-ready RVC checkpoint schema, compatible with the external engine.
        values = dict(weight={key: value.detach().cpu().half() for key, value in ckpt.items() if 'enc_q' not in key},
                      config=[hps.data.filter_length // 2 + 1, 32, hps.model.inter_channels, hps.model.hidden_channels,
                              hps.model.filter_channels, hps.model.n_heads, hps.model.n_layers, hps.model.kernel_size,
                              hps.model.p_dropout, hps.model.resblock, hps.model.resblock_kernel_sizes,
                              hps.model.resblock_dilation_sizes, hps.model.upsample_rates, hps.model.upsample_initial_channel,
                              hps.model.upsample_kernel_sizes, hps.model.spk_embed_dim, hps.model.gin_channels, hps.data.sampling_rate],
                      info=f'{epoch}epoch', sr=sr, f0=f0, version=version, vwb_epochs=epoch)
        destination = Path(config['folder']) / 'trained.pth'
        pending = destination.with_suffix('.pending')
        torch.save(values, pending); os.replace(pending, destination)
        return 'success'
    process_ckpt.savee = save


def export(config):
    import faiss
    import numpy as np
    import torch
    folder, exp = Path(config['folder']), Path(config['experiment'])
    request = config['training']
    trained = folder / 'trained.pth'
    weights = torch.load(trained, map_location='cpu', weights_only=True)
    if weights.get('version') != 'v2' or weights.get('f0') != 1 or weights.get('sr') != request['sample_rate'] or weights.get('vwb_epochs') != request['epochs']:
        raise ValueError('导出的模型版本或轮次不符。')
    if not weights.get('weight') or not all(torch.isfinite(tensor).all().item() for tensor in weights['weight'].values()):
        raise ValueError('模型参数包含无效值，不能导出。请降低学习率后重试。')
    rng = np.random.default_rng(1337)
    features = []
    total_vectors = 0
    # Bound index construction to 100k vectors, sampling throughout the dataset.
    paths = [line.split('|')[1] for line in (exp / 'filelist.txt').read_text(encoding='utf-8').splitlines()]
    sizes = [np.load(path, mmap_mode='r', allow_pickle=False).shape[0] for path in paths]
    total_vectors = sum(sizes)
    budget = min(100000, total_vectors)
    wanted = np.sort(rng.choice(total_vectors, budget, replace=False))
    offset = 0
    for path, count in zip(paths, sizes):
        take = wanted[(wanted >= offset) & (wanted < offset + count)] - offset
        if len(take): features.append(np.load(path, allow_pickle=False)[take])
        offset += count
    data = np.ascontiguousarray(np.concatenate(features), dtype='float32')
    faiss.omp_set_num_threads(4)
    cells = min(512, max(1, len(data) // 39), int(16 * np.sqrt(len(data))))
    index = faiss.IndexFlatL2(768) if len(data) < 512 else faiss.index_factory(768, f'IVF{cells},Flat')
    if not index.is_trained: index.train(data)
    if hasattr(index, 'nprobe'): index.nprobe = min(8, cells)
    index.add(data)
    name = config['export_name']
    if Path(name).name != name or '/' in name or '\\' in name: raise ValueError('模型名称无效。')
    model, index_file = folder / (name + '.pth'), folder / (name + '.index')
    index_file.write_bytes(faiss.serialize_index(index).tobytes())
    checked = faiss.deserialize_index(np.frombuffer(index_file.read_bytes(), dtype='uint8'))
    if checked.ntotal != len(data) or checked.d != 768: raise ValueError('索引校验失败。')
    os.replace(trained, model)
    dataset = json.loads((folder / '素材筛选报告.json').read_text(encoding='utf-8'))
    report = dict(name=request['name'], model=str(model), index=str(index_file),
                  model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), index_sha256=hashlib.sha256(index_file.read_bytes()).hexdigest(),
                  epochs_completed=weights['vwb_epochs'], training=request, dataset_seconds=dataset['duration_seconds'],
                  accepted_segments=len(dataset['accepted']), rejected_segments=len(dataset['rejected']), index_vectors=len(data),
                  available_vectors=total_vectors, model_version='v2', model_f0=1, model_sr=request['sample_rate'],
                  index_dim=768, index_ntotal=len(data), sources=config['sources'], limitations=dataset['limitations'])
    (folder / '训练记录.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PROGRESS 模型和检索索引已导出并验证', flush=True)


# Windows multiprocessing re-runs this file as __mp_main__; install the same
# process-local paths/checkpoint adapter before unpickling RVC's loader/runner.
if __name__ in ('__main__', '__mp_main__'):
    settings = configure(sys.argv[1])
    action = sys.argv[2]
    if action == 'train':
        install_training_adapter(settings)
        from train import train as trainer
        if __name__ == '__main__':
            # One selected device needs no distributed rendezvous, including CPU.
            trainer.run(0, 1, trainer.hps, trainer.utils.get_logger(settings['experiment']), False)
    elif __name__ == '__main__':
        if action == 'prepare':
            from training_data import prepare
            prepare(settings)
        elif action in ('preprocess', 'f0', 'features'): engine_stage(settings, action)
        elif action == 'manifest': manifest(settings)
        elif action == 'export': export(settings)
        else: raise ValueError('未知训练阶段。')

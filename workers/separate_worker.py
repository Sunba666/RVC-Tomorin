"""Run in the bundled or explicitly selected audio runtime."""
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
root = Path(config['rvc_root']).resolve()
folder = Path(config['folder']).resolve()
sys.dont_write_bytecode = True
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
if os.environ.get('VOICE_WORKBENCH_CPU') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PATH'] = str(root) + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, str(root / 'tools'))

import numpy as np
import soundfile as sf
import torch

torch.set_num_threads(4)
audio, rate = sf.read(config['input'], dtype='float32', always_2d=True)
if rate != 44100 or audio.shape[1] != 2:
    raise ValueError('分离输入必须是 44100 Hz 双声道音频。')
last = [-1]


def progress(done, total, message):
    step = min(10, int(done / total * 10)) if total else 0
    if step > last[0]:
        last[0] = step
        print(f'PROGRESS 人声分离 {step * 10}%（多轮处理）', flush=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
backend = config.get('separator_backend', 'bs_roformer')
if backend == 'demucs':
    from demucs.apply import apply_model
    from demucs.states import load_model
    # The checkpoint is bundled and its digest checked by the release builder.
    # Demucs 4 predates PyTorch's weights_only default change.
    original_load = torch.load
    def trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = trusted_checkpoint
    try:
        model = load_model(Path(config['separation_model']))
    finally:
        torch.load = original_load
    model.eval()
    mix = torch.from_numpy(audio.T.copy())
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std().clamp_min(1e-8)
    normalized = (mix - mean) / std
    print('PROGRESS 人声分离 0%（内置模型）', flush=True)
    with torch.inference_mode():
        separated = apply_model(model, normalized[None], device=device, shifts=1,
                                split=True, overlap=.25, progress=True, num_workers=0)[0]
    vocals = (separated[model.sources.index('vocals')].cpu() * std + mean).numpy()
    print('PROGRESS 人声分离 100%', flush=True)
elif backend == 'bs_roformer':
    from pymss import MSSeparator
    with MSSeparator(model_type='bs_roformer', model_path=config['separation_model'],
                     config_path=config['separation_yaml'], device=device, use_tta=True,
                     store_dirs=str(folder), progress_callback=progress,
                     inference_params={'batch_size': 1, 'overlap_size': 176400,
                                       'use_amp': device == 'cuda', 'normalize': False}) as separator:
        vocals = np.asarray(separator.separate(audio.T, pbar=False)['vocals'], dtype=np.float32)
else:
    raise ValueError('不支持的人声分离引擎。')
if vocals.shape[0] == 2:
    vocals = vocals.T
if vocals.shape != audio.shape or not np.isfinite(vocals).all():
    raise ValueError('分离结果无效。')
accompaniment = audio - vocals
peak = max(float(np.abs(vocals).max()), float(np.abs(accompaniment).max()))
gain = min(1., .999 / peak) if peak else 1.
for name, data in [('人声.wav', vocals), ('伴奏.wav', accompaniment)]:
    destination = folder / name
    if destination.exists():
        raise FileExistsError(destination)
    sf.write(destination, data * gain, rate, subtype='PCM_24')
report = {'device': device, 'backend': backend, 'frames': len(audio), 'duration': len(audio) / rate,
          'shared_gain': gain, 'vocal_rms': float(np.sqrt(np.mean(vocals ** 2))),
          'accompaniment_rms': float(np.sqrt(np.mean(accompaniment ** 2)))}
(folder / '分离记录.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('PROGRESS 人声和伴奏分离完成', flush=True)

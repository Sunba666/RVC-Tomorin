"""RVC adapter for bundled and custom engines; character models are imported."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

settings = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
root = Path(settings['rvc_root']).resolve()
folder = Path(settings['folder']).resolve()
model = Path(settings['model']).resolve()
index_path = str(Path(settings['index']).resolve()) if settings.get('index') else ''
sys.dont_write_bytecode = True
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
if os.environ.get('VOICE_WORKBENCH_CPU') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(root)
sys.path.insert(0, str(root))
sys.argv = [sys.argv[0]]
os.environ.update(weight_root=str(model.parent), rmvpe_root=str(root / 'assets/rmvpe'),
                  index_root=str(Path(index_path).parent) if index_path else str(root / 'assets/indices'),
                  outside_index_root=str(root / 'assets/indices'), RVC_CUDA_GRAPH='0',
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
os.environ['PATH'] = str(root) + os.pathsep + os.environ.get('PATH', '')

import numpy as np
import torch
import soundfile as sf
import faiss

torch.set_num_threads(4)
faiss.omp_set_num_threads(4)
if index_path:
    # Windows FAISS does not consistently support Unicode paths.
    index = faiss.deserialize_index(np.frombuffer(Path(index_path).read_bytes(), dtype='uint8'))
    if index.ntotal < 1:
        raise ValueError('检索索引没有特征。')
    original_read = faiss.read_index

    def read_index(path, *extra):
        return index if str(path) == index_path and not extra else original_read(path, *extra)
    faiss.read_index = read_index

from configs.config import Config
from infer.vc.modules import VC
from infer.vc.utils import load_hubert
from infer.rmvpe import RMVPE

config = Config()
config.x_query = 6
config.x_center = 30
config.x_max = 41
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(3 * 2**30, max(1.7 * 2**30, free - .7 * 2**30)) / total)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print('PROGRESS 加载角色音色', flush=True)
vc = VC(config)
vc.get_vc(model.name)
vc.hubert_model = load_hubert(config)
vc.pipeline.model_rmvpe = RMVPE(str(root / 'assets/rmvpe/rmvpe.pt'),
                              is_half=config.is_half, device=config.device)
status, result = vc.vc_single(0, str(folder / '人声.wav'), int(settings['pitch']), 'rmvpe',
                             index_path, float(settings['index_rate']) if index_path else 0.,
                             0, .65, 0.)
if result is None or result[1] is None:
    raise RuntimeError(str(status))
rate, samples = result
if len(samples) == 0 or not np.isfinite(samples).all():
    raise ValueError('转换后音频无效。')
destination = folder / '转换后人声.wav'
if destination.exists():
    raise FileExistsError(destination)
sf.write(destination, samples, rate, subtype='PCM_16')
(folder / '变声记录.json').write_text(json.dumps({
    'device': str(config.device),
    'sample_rate': rate, 'duration': len(samples) / rate, 'pitch': int(settings['pitch']),
    'index_rate': float(settings['index_rate']) if index_path else 0.,
    'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
print('PROGRESS 角色音色转换完成', flush=True)

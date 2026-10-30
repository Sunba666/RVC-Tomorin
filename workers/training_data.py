"""Conservative segment screening, run only in the external ML environment."""
import hashlib
import json
import math
from pathlib import Path


def quality_reason(audio, pitch_ratio, residual_db=None, strict=False):
    import numpy as np
    if len(audio) == 0 or not np.isfinite(audio).all(): return '无效采样'
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 10 ** (-42 / 20): return '音量过低或静音'
    if float(np.mean(np.abs(audio) >= .999)) > .01: return '严重削波'
    if abs(float(audio.mean())) > rms * .4: return '直流或异常信号'
    if pitch_ratio < (.2 if strict else .1): return '有效音高过少'
    if residual_db is not None and residual_db < (-15 if strict else -22): return '人声相对伴奏过弱'
    return None


def prepare(config):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from infer.rmvpe import RMVPE

    folder = Path(config['folder'])
    target = folder / 'clean_vocals'
    target.mkdir()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = RMVPE(str(Path(config['rvc_root']) / 'assets/rmvpe/rmvpe.pt'), is_half=False, device=device)
    report = {'accepted': [], 'rejected': [], 'duration_seconds': 0., 'device': device,
              'screening': config['training']['quality'],
              'limitations': '自动分离和筛选不保证完全无伴奏、无混响或单一说话人；请使用同一个人的素材。'}
    seen = set()
    for number, item in enumerate(config['candidates']):
        print('PROGRESS ' + f"筛选人声 {number + 1}/{len(config['candidates'])}", flush=True)
        samples, rate = sf.read(item['vocals'], dtype='float32', always_2d=True)
        mono = samples.mean(axis=1)
        if len(mono) < rate * 1.5 or not np.isfinite(mono).all():
            report['rejected'].append({'source': item['source'], 'reason': '素材过短或采样无效'}); continue
        other = None
        if item.get('accompaniment'):
            other, other_rate = sf.read(item['accompaniment'], dtype='float32', always_2d=True)
            if other_rate != rate or len(other) != len(mono): raise ValueError('分离轨道时长不一致。')
            other = other.mean(axis=1)
        audio16 = librosa.resample(mono, orig_sr=rate, target_sr=16000)
        pitches = np.asarray(detector.infer_from_audio(audio16, thred=.08))
        intervals = librosa.effects.split(mono, top_db=32, frame_length=2048, hop_length=256)
        spans = []
        for begin, end in intervals:
            begin, end = max(0, int(begin) - int(rate * .08)), min(len(mono), int(end) + int(rate * .08))
            # Pad consonants with context; screen whole segments instead of deleting unvoiced phonemes.
            if spans and begin <= spans[-1][1]: spans[-1][1] = max(spans[-1][1], end)
            else: spans.append([begin, end])
        for begin, end in spans:
            cursor = begin
            while cursor < end:
                stop = min(end, cursor + 8 * rate)
                if end - stop < 1.5 * rate: stop = end
                clip = mono[cursor:stop]
                seconds = len(clip) / rate
                metadata = {'source': item['source'], 'piece': Path(item['input']).name, 'start': cursor / rate, 'duration': seconds}
                if seconds < 1.5:
                    report['rejected'].append(dict(metadata, reason='有效片段短于 1.5 秒')); cursor = stop; continue
                pitch = pitches[int(cursor / rate * 100):int(stop / rate * 100)]
                ratio = float(np.mean((pitch >= 50) & (pitch <= 1100))) if len(pitch) else 0.
                rms = float(np.sqrt(np.mean(clip ** 2)))
                residual_db = None
                if other is not None:
                    ar = float(np.sqrt(np.mean(other[cursor:stop] ** 2)))
                    residual_db = min(120., 20 * math.log10(max(rms, 1e-9) / max(ar, 1e-9)))
                reason = quality_reason(clip, ratio, residual_db, config['training']['quality'] == 'strict')
                fingerprint = hashlib.sha256(np.clip(clip * 32767, -32768, 32767).astype('<i2').tobytes()).hexdigest()
                if fingerprint in seen: reason = '重复片段'
                if reason: report['rejected'].append(dict(metadata, reason=reason))
                else:
                    seen.add(fingerprint)
                    path = target / f'voice_{len(report["accepted"]):06d}.wav'
                    sf.write(path, clip / max(1., float(np.abs(clip).max()) / .98), rate, subtype='PCM_16')
                    report['accepted'].append(dict(metadata, file=str(path), voiced_ratio=ratio, vocal_to_accompaniment_db=residual_db))
                    report['duration_seconds'] += seconds
                cursor = stop
    (folder / '素材筛选报告.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if report['duration_seconds'] < 30:
        raise ValueError(f"筛选后只有 {report['duration_seconds']:.1f} 秒有效人声，至少需要 30 秒。请补充更清晰、单一人的素材，详情见素材筛选报告。")
    print(f"PROGRESS 有效人声 {report['duration_seconds']:.1f} 秒，保留 {len(report['accepted'])} 段", flush=True)

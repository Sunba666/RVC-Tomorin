import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_workbench.core import Cancelled, Manager, Request, Settings, Task
from voice_workbench.training import TrainingRequest, collect_materials, trained_settings


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-training-')
        self.folder = Path(self.temp.name).resolve()
        self.assertEqual(self.folder.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temp.cleanup)
        self.source = self.folder / '素材.wav'; self.source.write_bytes(b'fixture')

    def test_invalid_training_parameters_rejected_before_queue(self):
        for options in [dict(epochs=0), dict(epochs=1.5), dict(batch_size=True), dict(batch_size=33),
                        dict(learning_rate=float('nan')), dict(learning_rate=.2), dict(sample_rate='44k'),
                        dict(quality='unknown'), dict(save_every=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingRequest([str(self.source)], **options).validate()
        with self.assertRaises(ValueError): TrainingRequest([]).validate()

    def test_recursive_material_import_ignores_non_media(self):
        nested = self.folder / '子目录'; nested.mkdir()
        (nested / '素材.MP4').write_bytes(b'fixture')
        (nested / 'weights.pth').write_bytes(b'fixture')
        sources = collect_materials(self.folder)
        self.assertEqual({Path(p).name for p in sources}, {'素材.wav', '素材.MP4'})

    def test_model_activation_verifies_both_files_before_changing_settings(self):
        model, index = self.folder / '角色.pth', self.folder / '角色.index'
        model.write_bytes(b'model'); index.write_bytes(b'index')
        report = self.folder / '训练记录.json'
        report.write_text(json.dumps({'model': str(model), 'index': str(index),
            'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest()}), encoding='utf-8')
        settings = Settings(model='original.pth', pitch=4)
        activated = trained_settings(settings, report)
        self.assertEqual(activated.model, str(model)); self.assertEqual(activated.pitch, 4)
        self.assertEqual(settings.model, 'original.pth')
        index.write_bytes(b'changed')
        with self.assertRaises(ValueError): trained_settings(settings, report)
        self.assertEqual(settings.model, 'original.pth')

    def test_model_activation_rejects_incompatible_model_metadata(self):
        model, index = self.folder / '角色.pth', self.folder / '角色.index'
        model.write_bytes(b'model'); index.write_bytes(b'index')
        base = {'model': str(model), 'index': str(index),
                'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
                'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
                'index_vectors': 100, 'training': {'sample_rate': '40k'}}
        for bad in [{'model_version': 'v1'}, {'model_f0': 0}, {'model_sr': '32k'}, {'index_dim': 256}, {'index_vectors': 0}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                report = self.folder / '训练记录.json'
                report.write_text(json.dumps({**base, **bad}), encoding='utf-8')
                trained_settings(Settings(), report)
        report = self.folder / '训练记录.json'
        report.write_text(json.dumps({**base, 'model_version': 'v2', 'model_f0': 1, 'model_sr': '40k', 'index_dim': 768}), encoding='utf-8')
        self.assertEqual(trained_settings(Settings(), report).model, str(model))

    def test_training_preferences_tolerate_corrupt_file(self):
        from voice_workbench.training_ui import TrainingPanel
        panel = TrainingPanel.__new__(TrainingPanel)  # 跳过 __init__，只测偏好读取
        host = self.folder / 'training-options.json'
        panel.preferences = host

        # 内容为 []（非对象）：旧代码在 saved.items() 处抛 AttributeError，现应备份并回退默认值。
        host.write_text('[]', encoding='utf-8')
        defaults = panel._load_preferences()
        self.assertEqual(defaults['epochs'], 200)
        self.assertEqual(defaults['sample_rate'], '40k')
        self.assertTrue(host.with_name('training-options.json.corrupt').exists(), '坏文件应被备份')

        # 合法字典但个别键类型损坏：丢弃坏键、保留好键。
        host.write_text(json.dumps({'epochs': [], 'batch_size': 16, 'quality': 'strict'}), encoding='utf-8')
        defaults = panel._load_preferences()
        self.assertEqual(defaults['epochs'], 200, 'list 类型 epochs 应被丢弃并回默认')
        self.assertEqual(defaults['batch_size'], 16, '合法键应保留')
        self.assertEqual(defaults['quality'], 'strict')

        # 合法文件完整加载，且布尔不被 int 兼容性破坏。
        host.write_text(json.dumps({'epochs': 300, 'learning_rate': 0.0002, 'extract_vocals': False}), encoding='utf-8')
        defaults = panel._load_preferences()
        self.assertEqual(defaults['epochs'], 300)
        self.assertEqual(defaults['learning_rate'], 0.0002)
        self.assertIs(defaults['extract_vocals'], False)

    def test_failed_training_does_not_appear_as_completed_model(self):
        manager = Manager(self.folder / 'history'); self.addCleanup(manager.shutdown)
        with patch('voice_workbench.training.validate_training_environment'), patch('voice_workbench.training.TrainingTask.run', side_effect=ValueError('有效人声不足')):
            job = manager.submit(TrainingRequest([str(self.source)]), Settings(output=str(self.folder / 'output')))
            manager.queue.join()
        row = next(row for row in manager.store.list() if row['id'] == job)
        self.assertEqual((row['mode'], row['state'], row['result']), ('train', 'failed', ''))
        self.assertIn('有效人声不足', row['error'])

    @unittest.skipUnless(os.name == 'nt', 'Windows descendant cancellation')
    def test_cancellation_stops_descendant_that_inherits_stdout(self):
        folder = self.folder / 'worker'; folder.mkdir()
        pid_file = folder / 'child.pid'
        task = Task(Request(str(self.source), 'original'), Settings(), folder)
        code = ('import subprocess,sys,time;from pathlib import Path;'
                'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
                f'Path({str(pid_file)!r}).write_text(str(child.pid));time.sleep(60)')
        errors = []
        def run():
            try: task.run_command([sys.executable, '-c', code])
            except Exception as error: errors.append(error)
        thread = threading.Thread(target=run); thread.start()
        handle = None
        try:
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline: time.sleep(.02)
            self.assertTrue(pid_file.exists())
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x100000, False, int(pid_file.read_text()))
            self.assertTrue(handle)
            task.stop(); thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(errors[0], Cancelled)
            self.assertEqual(kernel.WaitForSingleObject(handle, 3000), 0)
        finally:
            task.stop(); thread.join(10)
            if handle: kernel.CloseHandle(handle)


class ManifestValidationTests(unittest.TestCase):
    @staticmethod
    def _run(wavs):
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workers'))
        from train_worker import manifest
        with tempfile.TemporaryDirectory(prefix='vwb-manifest-') as td:
            root = Path(td).resolve()
            exp = root / 'exp'
            for sub in ('0_gt_wavs', '3_feature768', '2a_f0', '2b-f0nsf'):
                (exp / sub).mkdir(parents=True)
            rvc = root / 'rvc'
            (rvc / 'configs' / 'v2').mkdir(parents=True)
            (rvc / 'configs' / 'v2' / '48k.json').write_text(json.dumps({'train': {}}), encoding='utf-8')
            for stem, f0, f0f, feature in wavs:
                (exp / '0_gt_wavs' / f'{stem}.wav').write_bytes(b'fixture')
                feature = np.zeros((50, 768), dtype='float32') if feature is None else feature
                np.save(exp / '3_feature768' / f'{stem}.npy', feature)
                np.save(exp / '2a_f0' / f'{stem}.wav.npy', np.asarray(f0, dtype='float32'))
                np.save(exp / '2b-f0nsf' / f'{stem}.wav.npy', np.asarray(f0f, dtype='float32'))
            config = {'folder': str(root), 'experiment': str(exp), 'rvc_root': str(rvc),
                      'training': {'sample_rate': '48k', 'epochs': 1, 'batch_size': 1, 'learning_rate': .0001}}
            info = type('Info', (), {'duration': 20.0})
            with patch('soundfile.info', return_value=info()):
                try:
                    manifest(config)
                    check = json.loads((root / '训练数据检查.json').read_text(encoding='utf-8'))
                    return {'raised': None, 'lines': len((exp / 'filelist.txt').read_text(encoding='utf-8').splitlines()), 'rejected': check['rejected']}
                except ValueError as error:
                    return {'raised': str(error), 'lines': None, 'rejected': None}

    def test_manifest_accepts_valid_pitch_arrays(self):
        good = [300.0] * 60
        result = self._run([('a', good, good, None), ('b', good, good, None)])
        self.assertIsNone(result['raised'], result['raised'])
        self.assertEqual(result['lines'], 2)
        self.assertEqual(result['rejected'], [])

    def test_manifest_rejects_bad_pitch_arrays(self):
        good = [300.0] * 60
        cases = {
            'empty_f0': ([], good),
            'empty_f0f': (good, []),
            'mismatched_length': ([300.0], good),
            'two_dimensional': ([[300.0] * 30] * 2, good),
            'out_of_range_pitch': ([1e6] * 60, good),
        }
        for label, (f0, f0f) in cases.items():
            with self.subTest(label=label):
                result = self._run([('a', good, good, None), ('b', good, good, None), ('c', f0, f0f, None)])
                self.assertIsNone(result['raised'], result['raised'])
                self.assertEqual(result['lines'], 2)
                self.assertEqual(result['rejected'], ['c.wav'])

    def test_manifest_requires_at_least_two_valid_segments(self):
        result = self._run([('a', [], [300.0] * 60, None), ('b', [], [300.0] * 60, None)])
        self.assertIsNotNone(result['raised'])
        self.assertIn('少于 2 段', result['raised'])


if __name__ == '__main__': unittest.main(verbosity=2)

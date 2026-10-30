import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
import struct
import zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_workbench.core import (STAGE_FILE, Cancelled, Manager, Request, SeparationCache, Settings,
                                  Store, Task, environment_report, load_request, pretty_bytes, safe_name, tempo_chain)
from voice_workbench.gui import source_from_text, preview_clip, prune_preview_dir

FFMPEG = os.environ.get('VOICE_TEST_FFMPEG') or shutil.which('ffmpeg')


class SetupReportTests(unittest.TestCase):
    def test_setup_report_covers_version_installer_and_items(self):
        from voice_workbench.gui import setup_report
        with tempfile.TemporaryDirectory(prefix='vwb-setup-') as temporary:
            home = Path(temporary).resolve() / 'home'
            report = setup_report(home)
            from voice_workbench import __version__
            self.assertEqual(report['version'], __version__)
            self.assertEqual(report['home'], str(home))
            self.assertFalse(report['settings_exists'])
            self.assertTrue(report['installer'], '打包版必须带上安装模块')
            states = {item['key']: item['state'] for item in report['items']}
            self.assertEqual(len(states), 10)
            self.assertEqual(states['ffmpeg'], 'missing')


class ValidationTests(unittest.TestCase):
    def test_share_text_and_bv(self):
        self.assertEqual(source_from_text('【音乐分享】 https://b23.tv/abc123。'), 'https://b23.tv/abc123')
        self.assertEqual(source_from_text('BV13x41117TL'), 'https://www.bilibili.com/video/BV13x41117TL')

    def test_reject_unsafe_or_invalid_requests(self):
        for request in [Request('ftp://host/a'), Request('https://example.com', start=float('nan')),
                        Request('https://example.com', start=3, end=2), Request('https://user:secret@example.com')]:
            with self.subTest(request=request), self.assertRaises(ValueError): request.validate()

    def test_export_names_cannot_escape_folder(self):
        for name in ['../a', r'..\b', 'NUL', 'CON.txt', 'a|b', '']:
            result = safe_name(name)
            self.assertNotIn('/', result); self.assertNotIn('\\', result)
            self.assertNotIn('|', result); self.assertNotIn(result.upper(), ['NUL', 'CON.TXT'])


class TempoChainTests(unittest.TestCase):
    def test_tempo_chain_factors_stay_in_legal_range(self):
        self.assertEqual(tempo_chain(1.0), 'atempo=1.0')
        self.assertEqual(tempo_chain(0.5), 'atempo=0.5')
        self.assertEqual(tempo_chain(2.0), 'atempo=2.0')
        self.assertEqual(tempo_chain(0.25), 'atempo=0.5,atempo=0.5')
        self.assertEqual(tempo_chain(4.0), 'atempo=2.0,atempo=2.0')
        for pitch in range(-24, 25):
            tempo = 1 / (2 ** (pitch / 12))
            factors = [float(part[len('atempo='):]) for part in tempo_chain(tempo).split(',')]
            self.assertTrue(all(0.5 <= factor <= 2.0001 for factor in factors), (pitch, factors))
            product = 1.0
            for factor in factors:
                product *= factor
            self.assertAlmostEqual(product, tempo, places=3, msg=(pitch, factors))


@unittest.skipUnless(FFMPEG, 'Set VOICE_TEST_FFMPEG to run real audio tests')
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-tests-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('vwb-tests-'))
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / '中文素材.wav'
        samples = array.array('h', (int(5000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(96000)))
        with wave.open(str(self.source), 'wb') as file:
            file.setparams((1, 2, 16000, 0, 'NONE', 'not compressed')); file.writeframes(samples.tobytes())
        self.settings = Settings(ffmpeg=str(Path(FFMPEG).parent), output=str(self.root / '输出'))

    def test_real_original_trim_format_and_manifest(self):
        folder = self.root / '任务'
        original_hash = self.source.read_bytes()
        result, title = Task(Request(str(self.source), 'original', '裁剪', 1, 3), self.settings, folder).run()
        with wave.open(str(result), 'rb') as file:
            self.assertEqual((file.getframerate(), file.getnchannels(), file.getsampwidth(), file.getnframes()), (44100, 2, 2, 88200))
        self.assertEqual(self.source.read_bytes(), original_hash)
        self.assertEqual(json.loads((folder / '成品记录.json').read_text(encoding='utf-8'))['mode'], 'original')
        self.assertFalse((folder / '转换后人声.wav').exists())

    def test_out_of_duration_range_fails_without_completed_output(self):
        folder = self.root / '越界'
        with self.assertRaises(ValueError): Task(Request(str(self.source), 'original', end=20), self.settings, folder).run()
        self.assertFalse((folder / '成品记录.json').exists())

    def test_real_media_with_no_audio_is_rejected(self):
        video = self.root / '没有音轨.png'
        def chunk(kind, data): return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        video.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 1, 1, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\0\xff\0\0')) + chunk(b'IEND', b''))
        with self.assertRaisesRegex(ValueError, '音轨'):
            Task(Request(str(video), 'original'), self.settings, self.root / '无音轨').run()

    def test_worker_can_be_cancelled_while_silent(self):
        folder = self.root / '取消'; folder.mkdir()
        task = Task(Request(str(self.source), 'original'), self.settings, folder)
        errors = []
        def work():
            try: task.run_command([sys.executable, '-c', 'import time;time.sleep(30)'])
            except Exception as error: errors.append(error)
        thread = threading.Thread(target=work); thread.start()
        limit = time.monotonic() + 5
        while task.process is None and time.monotonic() < limit: time.sleep(.02)
        self.assertIsNotNone(task.process)
        task.stop(); thread.join(5)
        self.assertFalse(thread.is_alive()); self.assertIsInstance(errors[0], Cancelled)

    def test_manager_reports_failure_and_does_not_claim_result(self):
        bad = self.root / '坏文件.mp4'; bad.write_bytes(b'not a video')
        manager = Manager(self.root / '状态')
        self.addCleanup(manager.shutdown)
        job = manager.submit(Request(str(bad), 'original'), self.settings)
        manager.queue.join()
        row = next(row for row in manager.store.list() if row['id'] == job)
        self.assertEqual(row['state'], 'failed'); self.assertTrue(row['error']); self.assertFalse(row['result'])

    def test_interrupted_history_survives_restart(self):
        store = Store(self.root / '恢复')
        store.add('job', Request(str(self.source)), self.root / 'output')
        store.update('job', state='running')
        restarted = Store(self.root / '恢复')
        self.assertEqual(restarted.list()[0]['state'], 'interrupted')

    def test_run_captured_times_out(self):
        folder = self.root / '超时'; folder.mkdir()
        task = Task(Request(str(self.source), 'original'), self.settings, folder)
        with self.assertRaisesRegex(RuntimeError, '超时'):
            task.run_captured([sys.executable, '-c', 'import time;time.sleep(30)'], timeout=1)

    def test_run_captured_is_cancellable(self):
        folder = self.root / '取消'; folder.mkdir()
        task = Task(Request(str(self.source), 'original'), self.settings, folder)
        errors = []
        def work():
            try:
                task.run_captured([sys.executable, '-c', 'import time;time.sleep(30)'])
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=work); thread.start()
        limit = time.monotonic() + 5
        while task.process is None and time.monotonic() < limit:
            time.sleep(.02)
        self.assertIsNotNone(task.process)
        task.stop(); thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], Cancelled)

    def test_silent_convert_output_is_rejected(self):
        folder = self.root / '静音'; folder.mkdir()
        silent = folder / '静音成品.wav'
        with wave.open(str(silent), 'wb') as file:
            file.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
            file.writeframes(b'\x00' * (44100 * 4))
        task = Task(Request(str(self.source), 'convert'), self.settings, folder)
        with self.assertRaisesRegex(RuntimeError, '静音'):
            task._check_signal(silent)

    def test_missing_local_source_reports_clear_error(self):
        folder = self.root / '续跑丢失'
        missing = self.root / '已经移走的素材.wav'
        task = Task(Request(str(missing), 'original'), self.settings, folder)
        with self.assertRaisesRegex(ValueError, '本地文件不存在'):
            task.run()

    def test_long_conversion_is_rejected_before_separation(self):
        from unittest.mock import patch
        source, settings, _calls = self.conversion_fixture()
        folder = self.root / '太长'
        task = Task(Request(str(source), 'convert'), settings, folder)
        with patch.object(Task, 'probe', return_value=4000.0):
            with self.assertRaisesRegex(ValueError, '分钟'):
                task.run()
        self.assertFalse((folder / '人声.wav').exists(), '超长素材不应进入人声分离')

    def conversion_fixture(self):
        """伪造一套 RVC 环境与 worker，只验证流水线的阶段复用，不加载模型。"""
        root = self.root / 'rvc'
        for relative in ('infer/vc/modules.py', 'assets/rmvpe/rmvpe.pt', 'assets/hubert_base/config.json',
                         'assets/pymss_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt',
                         'assets/pymss_weights/model_bs_roformer_ep_317_sdr_12.9755.yaml', '角色.pth'):
            file = root / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'vwb-test')
        source = self.root / '转换素材.wav'
        self.write_wav(source, 2.0)
        settings = Settings(ffmpeg=str(Path(FFMPEG).parent), rvc_root=str(root), python=sys.executable,
                            model=str(root / '角色.pth'), output=str(self.root / '输出'),
                            pitch=2, index_rate=.4, accompaniment=.35)
        calls = []
        def fake(task, args, cwd=None):
            args = [str(item) for item in args]
            line = ' '.join(args)
            calls.append(line)
            if 'separate_worker.py' in line:
                folder = Path(json.loads(Path(args[-1]).read_text(encoding='utf-8'))['folder'])
                self.write_wav(folder / '人声.wav'); self.write_wav(folder / '伴奏.wav')
                (folder / '分离记录.json').write_text('{}', encoding='utf-8')
            elif 'rvc_worker.py' in line:
                folder = Path(json.loads(Path(args[-1]).read_text(encoding='utf-8'))['folder'])
                self.write_wav(folder / '转换后人声.wav')
            elif '-xerror' in args:
                return
            elif '-i' in args and args[-1].endswith('.wav'):
                shutil.copy(args[args.index('-i') + 1], args[-1])
            else:
                raise AssertionError('unexpected command: ' + line[:200])
        original = Task.run_command
        Task.run_command = fake
        self.addCleanup(setattr, Task, 'run_command', original)
        return source, settings, calls

    @staticmethod
    def write_wav(path, seconds=2.0, rate=44100):
        frames = int(seconds * rate)
        samples = array.array('h', (int(800 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(frames)))
        stereo = array.array('h')
        for value in samples:
            stereo.extend((value, value))
        with wave.open(str(path), 'wb') as file:
            file.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
            file.writeframes(stereo.tobytes())

    def test_conversion_reuses_separation_across_runs(self):
        source, settings, calls = self.conversion_fixture()
        home = self.root / '缓存'
        Task(Request(str(source), 'convert', '测试'), settings, self.root / '任务一', home=home).run()
        self.assertEqual(sum('separate_worker.py' in line for line in calls), 1)
        settings.pitch = 5
        folder = self.root / '任务二'
        result, _ = Task(Request(str(source), 'convert', '测试'), settings, folder, home=home).run()
        self.assertEqual(sum('separate_worker.py' in line for line in calls), 1, '改音高不应重新分离')
        self.assertEqual(sum('rvc_worker.py' in line for line in calls), 2)
        self.assertTrue(Path(result).is_file())
        self.assertEqual(len(list((home / 'cache' / 'separation').iterdir())), 1)

    def test_resume_skips_completed_stages_in_same_folder(self):
        source, settings, calls = self.conversion_fixture()
        home = self.root / '缓存'
        folder = self.root / '任务三'
        Task(Request(str(source), 'convert', '测试'), settings, folder, home=home).run()
        (folder / '转换后人声.wav').unlink(); (folder / '测试_角色翻唱.wav').unlink()
        Task(Request(str(source), 'convert', '测试'), settings, folder, home=home).run()
        self.assertEqual(sum('separate_worker.py' in line for line in calls), 1)
        self.assertEqual(sum('rvc_worker.py' in line for line in calls), 2)
        self.assertTrue((folder / '成品记录.json').is_file())

    def test_manager_retry_resumes_from_failed_stage(self):
        source, settings, calls = self.conversion_fixture()
        home = self.root / '重试'
        home.mkdir()
        settings.save(home / 'settings.json')
        failing = {'armed': True}
        normal = Task.run_command
        def fake(task, args, cwd=None):
            if 'rvc_worker.py' in ' '.join(str(item) for item in args) and failing['armed']:
                failing['armed'] = False
                raise RuntimeError('模拟转换失败')
            return normal(task, args, cwd)
        Task.run_command = fake
        manager = Manager(home)
        self.addCleanup(manager.shutdown)
        job = manager.submit(Request(str(source), 'convert', '重试'), settings)
        manager.queue.join()
        row = next(item for item in manager.store.list() if item['id'] == job)
        self.assertEqual(row['state'], 'failed')
        separations = sum('separate_worker.py' in line for line in calls)
        manager.retry(job)
        manager.queue.join()
        row = next(item for item in manager.store.list() if item['id'] == job)
        self.assertEqual(row['state'], 'completed')
        self.assertTrue(row['result'])
        self.assertEqual(sum('separate_worker.py' in line for line in calls), separations, '重试不应重新分离')

    def test_preview_clip_skips_leading_silence(self):
        rate = 44100
        silence = array.array('h', [0] * int(14 * rate))
        tone = array.array('h', (int(3000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(int(2 * rate))))
        stereo = array.array('h')
        for value in silence + tone:
            stereo.extend((value, value))
        source = self.root / '带前奏.wav'
        with wave.open(str(source), 'wb') as file:
            file.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
            file.writeframes(stereo.tobytes())
        clip = Path(preview_clip(self.settings, source, self.root / 'preview-skip.wav'))
        with wave.open(str(clip), 'rb') as file:
            data = array.array('h', file.readframes(file.getnframes()))
        self.assertTrue(data, '应能生成试听片段')
        mean_abs = sum(abs(v) for v in data) / len(data)
        self.assertGreater(mean_abs, 500, '试听应跳过开头 14 秒静音，落在有声段落')

    def test_deferred_state_is_flushed_after_database_recovery(self):
        # 复核缺口 2：最终状态写入失败时不能悄悄吞掉，要保留待补写、发 storage_error 事件，
        # 数据库恢复后 flush_pending 把陈旧 running 更正掉，使任务可以被重试。
        import queue as queue_mod
        store = Store(self.root / '状态')
        store.add('job', Request(str(self.source), 'original'), self.root / '输出')
        store.update('job', state='running')
        manager = Manager.__new__(Manager)
        manager.store = store
        manager.events = queue_mod.Queue()
        manager.pending = {}
        manager._store_lock = threading.Lock()
        real_update = store.update
        state = {'hit': False}

        def flaky_update(job, **values):
            if values.get('state') == 'completed' and not state['hit']:
                state['hit'] = True
                raise OSError('database is locked')
            return real_update(job, **values)

        store.update = flaky_update
        # 写失败：不抛异常，保留待补写并上报存储异常
        manager._safe_update('job', state='completed', stage='完成', result='x', name='n')
        self.assertTrue(state['hit'], '应当模拟到完成状态写入失败')
        self.assertEqual(manager.pending['job']['state'], 'completed', '失败的状态应保留待补写')
        self.assertEqual(store.get('job')['state'], 'running', '数据库里仍应是陈旧的 running')
        event = manager.events.get_nowait()
        self.assertEqual(event[1], 'storage_error', '应向界面报告存储异常')
        # 数据库恢复后补写
        manager.flush_pending()
        self.assertNotIn('job', manager.pending, '补写成功后应清除待补写')
        self.assertEqual(store.get('job')['state'], 'completed', '补写后应落到最终状态')


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-env-')
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.home = self.root / 'home'
        previous_home = os.environ.get('VOICE_WORKBENCH_HOME')
        os.environ['VOICE_WORKBENCH_HOME'] = str(self.home)
        def restore_home():
            if previous_home is None:
                os.environ.pop('VOICE_WORKBENCH_HOME', None)
            else:
                os.environ['VOICE_WORKBENCH_HOME'] = previous_home
        self.addCleanup(restore_home)

    def test_report_marks_missing_and_ready_items(self):
        empty = Settings(ffmpeg=str(self.root / 'nothing'), output=str(self.root / '输出'))
        states = {item['key']: item['state'] for item in environment_report(empty, self.home)}
        for key in ('ffmpeg', 'rvc', 'python', 'model', 'separation'):
            self.assertEqual(states[key], 'missing', key)
        self.assertEqual(states['index'], 'warn')
        self.assertEqual(states['output'], 'ok', '可以自动创建的输出目录应判为就绪')
        self.assertEqual(states['cache'], 'warn', '没有缓存时只是提示')

        root = self.root / 'rvc'
        for relative in ('infer/vc/modules.py', 'assets/rmvpe/rmvpe.pt', 'assets/hubert_base/config.json',
                         'assets/pymss_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt',
                         'assets/pymss_weights/model_bs_roformer_ep_317_sdr_12.9755.yaml',
                         'ffmpeg.exe', 'ffprobe.exe', '角色.pth', '检索.index'):
            file = root / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'vwb-test')
        ready = Settings(ffmpeg=str(Path(FFMPEG).parent if FFMPEG else root), rvc_root=str(root), python=sys.executable,
                         model=str(root / '角色.pth'), index=str(root / '检索.index'), output=str(self.root / '输出'))
        states = {item['key']: item['state'] for item in environment_report(ready, self.home)}
        for key in ('rvc', 'python', 'model', 'index', 'separation', 'output'):
            self.assertEqual(states[key], 'ok', key)
        if FFMPEG:
            self.assertEqual(states['ffmpeg'], 'ok', '真实可运行的 ffmpeg 应判为就绪')
        else:
            self.assertEqual(states['ffmpeg'], 'warn', '无法运行的 ffmpeg 应判为注意而非就绪')

    def test_rvc_environment_warns_when_conversion_deps_missing(self):
        root = self.root / 'rvc-partial'
        (root / 'infer/vc/modules.py').parent.mkdir(parents=True, exist_ok=True)
        (root / 'infer/vc/modules.py').write_bytes(b'vwb-test')
        settings = Settings(rvc_root=str(root), python=sys.executable, output=str(self.root / '输出'))
        items = {item['key']: item for item in environment_report(settings, self.home)}
        self.assertEqual(items['rvc']['state'], 'warn', '缺 HuBERT/RMVPE 时不得显示就绪')

    def test_only_ffmpeg_is_offered_for_automatic_install(self):
        items = {item['key']: item for item in environment_report(Settings(output=str(self.root / '输出')), self.home)}
        self.assertTrue(items['ffmpeg']['auto'], 'FFmpeg 可以一键下载安装')
        for key in ('rvc', 'python', 'model', 'index', 'separation'):
            self.assertFalse(items[key]['auto'], f'{key} 需要用户自己的环境或模型，不应自动下载')

    def test_cache_size_and_clear(self):
        cache = SeparationCache(self.home)
        entry = cache.entry('a' * 16)
        entry.mkdir(parents=True)
        for name in cache.FILES:
            (entry / name).write_bytes(b'x' * 1024)
        self.assertEqual((cache.count(), cache.size()), (1, 3 * 1024))
        self.assertEqual(cache.clear(), 3 * 1024)
        self.assertEqual((cache.count(), cache.size()), (0, 0))

    def test_clear_reports_only_actually_deleted(self):
        cache = SeparationCache(self.home)
        entry = cache.entry('b' * 16)
        entry.mkdir(parents=True)
        for name in cache.FILES:
            (entry / name).write_bytes(b'x' * 1024)
        import shutil as _shutil
        original = _shutil.rmtree
        _shutil.rmtree = lambda *a, **k: None  # 模拟删除失败但不抛异常（ignore_errors 吞掉）
        try:
            self.assertEqual(cache.clear(), 0, '删除失败不得虚报已清理')
        finally:
            _shutil.rmtree = original
        self.assertEqual(cache.clear(), 3 * 1024)

    def test_wav_validation_rejects_empty_and_accepts_valid(self):
        cache = SeparationCache(self.home)
        empty = self.root / '空.wav'; empty.write_bytes(b'')
        self.assertFalse(cache._wav_ok(empty))
        valid = self.root / '有效.wav'
        with wave.open(str(valid), 'wb') as file:
            file.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
            file.writeframes(array.array('h', [1000] * 44100 * 2).tobytes())
        self.assertTrue(cache._wav_ok(valid))

    def test_corrupt_cache_is_not_hit(self):
        cache = SeparationCache(self.home)
        key = 'c' * 16
        entry = cache.entry(key); entry.mkdir(parents=True)
        (entry / '人声.wav').write_bytes(b'')  # 空文件，不应命中
        with wave.open(str(entry / '伴奏.wav'), 'wb') as file:
            file.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
            file.writeframes(array.array('h', [1000] * 44100 * 2).tobytes())
        (entry / '分离记录.json').write_text('{}', encoding='utf-8')
        self.assertFalse(cache.has(key))

    def test_cache_key_changes_with_pipeline_stamp(self):
        cache = SeparationCache(self.home)
        original = self.root / '原音.wav'; original.write_bytes(b'x' * 100)
        self.assertNotEqual(cache.key(original, 'm', 'c'), cache.key(original, 'm', 'c', ['v2', ['p']]))

    def test_preview_dir_prunes_oldest_first(self):
        root = self.root / 'preview'
        root.mkdir(parents=True)
        files = []
        for index, size in ((0, 4000), (1, 1000), (2, 1000)):
            file = root / f'{index:016x}.wav'
            file.write_bytes(b'x' * size)
            files.append(file)
        orphan = root / '片段.wav'
        orphan.write_bytes(b'z' * 9000)
        os.utime(files[0], (1010.0, 1010.0))
        os.utime(files[1], (2020.0, 2020.0))
        os.utime(files[2], (3030.0, 3030.0))
        self.assertEqual(prune_preview_dir(root, limit=2500), 4000)
        self.assertFalse(files[0].exists(), '最旧的十六进制命名条目应先淘汰')
        self.assertTrue(files[1].exists())
        self.assertTrue(files[2].exists())
        self.assertTrue(orphan.exists(), '非缓存命名的文件不应被清理')

    @unittest.skipUnless(os.name == 'nt', 'Windows 文件锁')
    def test_task_lock_blocks_second_process(self):
        from voice_workbench.core import acquire_task_lock
        folder = self.root / '锁'; folder.mkdir(parents=True, exist_ok=True)
        first = acquire_task_lock(folder)
        self.assertIsNotNone(first)
        try:
            with self.assertRaisesRegex(RuntimeError, '另一进程'):
                acquire_task_lock(folder)
        finally:
            first.close()
        second = acquire_task_lock(folder)
        self.assertIsNotNone(second); second.close()

    def test_save_state_leaves_no_pending_file(self):
        folder = self.root / '任务'
        Task(Request('未下载.wav', 'convert', '命名'), Settings(output=str(self.root / '输出')), folder, home=self.home)
        self.assertTrue((folder / STAGE_FILE).is_file())
        self.assertFalse(any(path.name.endswith('.pending') for path in folder.iterdir()))

    def test_task_records_request_and_parameters_before_running(self):
        settings = Settings(output=str(self.root / '输出'), pitch=7, index_rate=.2, accompaniment=.5)
        folder = self.root / '任务'
        Task(Request('还没有下载的音源.wav', 'convert', '命名中的任务'), settings, folder, home=self.home)
        state = json.loads((folder / STAGE_FILE).read_text(encoding='utf-8'))
        self.assertEqual({key: state['settings'][key] for key in ('pitch', 'index_rate', 'accompaniment')},
                         {'pitch': 7, 'index_rate': .2, 'accompaniment': .5})
        self.assertNotIn('cookies', state['settings'])
        self.assertEqual(state['request']['mode'], 'convert')
        self.assertEqual(load_request(folder).name, '命名中的任务')

    def test_pretty_bytes_units(self):
        self.assertEqual(pretty_bytes(0), '0 B')
        self.assertEqual(pretty_bytes(1536), '1.5 KB')
        self.assertEqual(pretty_bytes(6 * 2 ** 30), '6.0 GB')


class CliResumeTests(unittest.TestCase):
    def test_resume_rejects_conflicting_parameters(self):
        from unittest.mock import patch
        from voice_workbench import cli
        folder = str((Path('.').resolve() / '任务'))
        cases = [
            (['--resume', folder, '--name', '新名'], '--name'),
            (['--resume', folder, '--start', '5'], '--start'),
            (['--resume', folder, '--end', '10'], '--end'),
            (['--resume', folder, '--mode', 'convert'], '--mode'),
            (['--resume', folder, '一个来源'], '来源'),
        ]
        for argv, expect in cases:
            with self.subTest(expect=expect):
                with patch('voice_workbench.cli.Settings.load', return_value=Settings()), patch.object(sys, 'argv', ['cli'] + argv):
                    with self.assertRaises(SystemExit) as caught:
                        cli.main()
                    self.assertEqual(caught.exception.code, 2)

    def test_resume_without_conflicts_runs_task(self):
        from unittest.mock import patch
        from voice_workbench import cli
        folder = Path('.').resolve() / '任务'
        with tempfile.TemporaryDirectory(prefix='vwb-cli-') as td:
            target = Path(td) / 'out.wav'
            with patch('voice_workbench.cli.Settings.load', return_value=Settings()), \
                 patch('voice_workbench.cli.load_request', return_value=Request('x.wav', 'convert', '命名')), \
                 patch('voice_workbench.cli.task_settings', return_value=Settings()), \
                 patch('voice_workbench.cli.Task') as FakeTask:
                FakeTask.return_value.run.return_value = (target, '命名')
                with patch.object(sys, 'argv', ['cli', '--resume', str(folder)]):
                    cli.main()
                FakeTask.assert_called_once()


if __name__ == '__main__': unittest.main(verbosity=2)

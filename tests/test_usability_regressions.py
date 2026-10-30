import json
import hashlib
import os
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_workflow as helpers
from voice_workbench.core import Manager, Request, Settings, Task, environment_report
from voice_workbench.gui import App
from voice_workbench.training import TrainingRequest, TrainingTask


class TrainingEntryTests(unittest.TestCase):
    def test_training_constructor_does_not_claim_new_output_directory(self):
        with tempfile.TemporaryDirectory(prefix='vwb-entry-') as temporary:
            folder = Path(temporary)
            source = folder / 'voice.wav'; source.write_bytes(b'fixture')
            task = TrainingTask(TrainingRequest([str(source)]), Settings(), folder / 'training')
            # Reach the real TrainingTask.run directory creation, without ML.
            with patch('voice_workbench.training.validate_training_environment'), \
                    patch.object(Settings, 'executable', side_effect=RuntimeError('reached extraction')):
                with self.assertRaisesRegex(RuntimeError, 'reached extraction'):
                    task.run()

    def test_environment_probe_preserves_existing_file(self):
        with tempfile.TemporaryDirectory(prefix='vwb-probe-') as temporary:
            folder = Path(temporary)
            existing = folder / '.voice-workbench-write-test'
            existing.write_bytes(b'user-owned')
            environment_report(Settings(output=str(folder)), folder)
            self.assertEqual(existing.read_bytes(), b'user-owned')


@unittest.skipUnless(helpers.FFMPEG, 'FFmpeg required')
class UsabilityRecoveryTests(unittest.TestCase):
    setUp = helpers.WorkflowTests.setUp
    conversion_fixture = helpers.WorkflowTests.conversion_fixture
    write_wav = staticmethod(helpers.WorkflowTests.write_wav)

    def test_retry_keeps_original_voice_and_parameters_after_global_change(self):
        source, settings, calls = self.conversion_fixture()
        home = self.root / 'profile'
        settings.save(home / 'settings.json')
        manager = Manager(home)
        self.addCleanup(manager.shutdown)
        job = manager.submit(Request(str(source), 'convert'), settings)
        manager.queue.join()
        before = dict(model=settings.model, index=settings.index, pitch=settings.pitch,
                      index_rate=settings.index_rate, accompaniment=settings.accompaniment)
        settings.model = str(self.root / 'other.pth')
        Path(settings.model).write_bytes(b'another')
        settings.pitch = 8; settings.index_rate = .9; settings.accompaniment = .8
        settings.save(home / 'settings.json')
        captured = []
        def inspect(task):
            captured.append({key: getattr(task.settings, key) for key in before})
            return source, 'retried'
        with patch.object(Task, 'run', inspect):
            manager.retry(job); manager.queue.join()
        self.assertEqual(captured, [before])

    def test_preview_does_not_reuse_different_song_with_same_name_and_mtime(self):
        first, second = self.root / 'song-a/人声.wav', self.root / 'song-b/人声.wav'
        first.parent.mkdir(); second.parent.mkdir()
        self.write_wav(first, seconds=1); self.write_wav(second, seconds=2)
        stamp = 1700000000
        os.utime(first, (stamp, stamp)); os.utime(second, (stamp, stamp))
        # Exercise App.preview without audible playback or a live Tk window.
        class Value:
            def set(self, text): pass
        class Player:
            def play(self, files): pass
            def stop(self): pass
        class Fixture:
            home = self.root / 'profile'
            settings = self.settings
            player = Player()
            message = Value()
            root = None
            _preview_busy = False
            _preview_token = 0
            preview_queue = queue.Queue()
        app = Fixture()

        def run_and_collect(sources):
            app._preview_busy = False  # 模拟 drain_preview 已消费上一次结果
            App.preview(app, sources)
            _, clips, error, _ = app.preview_queue.get(timeout=60)
            self.assertIsNone(error, '试听片段生成不应报错')
            return Path(clips[0]).read_bytes()

        one = run_and_collect([first])
        two = run_and_collect([second])
        self.assertNotEqual(hashlib.sha256(one).hexdigest(), hashlib.sha256(two).hexdigest())


if __name__ == '__main__': unittest.main()

"""Recovery regressions for interrupted or edited audio tasks."""
import os
from pathlib import Path
import shutil
import threading
import unittest
from unittest.mock import patch

import test_workflow as helpers
from voice_workbench.core import Manager, Request, SeparationCache, Task, file_stamp


@unittest.skipUnless(helpers.FFMPEG, 'FFmpeg required')
class PipelineRecoveryTests(unittest.TestCase):
    setUp = helpers.WorkflowTests.setUp
    conversion_fixture = helpers.WorkflowTests.conversion_fixture
    write_wav = staticmethod(helpers.WorkflowTests.write_wav)

    def initial_task(self):
        source, settings, calls = self.conversion_fixture()
        home, folder = self.root / 'profile', self.root / 'job'
        request = Request(str(source), 'convert', '恢复测试')
        Task(request, settings, folder, home=home).run()
        return request, settings, calls, home, folder

    def test_missing_local_stem_restored_without_separation(self):
        request, settings, calls, home, folder = self.initial_task()
        (folder / '人声.wav').unlink()
        Task(request, settings, folder, home=home).run()
        self.assertTrue((folder / '人声.wav').is_file())
        self.assertEqual(sum('separate_worker.py' in call for call in calls), 1)

    def test_evicted_shared_cache_keeps_valid_local_stems(self):
        request, settings, calls, home, folder = self.initial_task()
        # This is an isolated fixture underneath TemporaryDirectory.
        cache = (home / 'cache').resolve()
        self.assertTrue(cache.is_relative_to(self.root))
        shutil.rmtree(cache)
        Task(request, settings, folder, home=home).run()
        self.assertEqual(sum('separate_worker.py' in call for call in calls), 1)

    def test_rebuilt_conversion_invalidates_old_mix_even_with_same_timestamp(self):
        request, settings, calls, home, folder = self.initial_task()
        output = folder / '转换后人声.wav'
        stamp = output.stat()
        output.unlink()
        normal = Task.run_command
        def rebuilt(task, args, cwd=None):
            normal(task, args, cwd)
            if 'rvc_worker.py' in ' '.join(map(str, args)):
                os.utime(output, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        before = sum('-filter_complex' in call for call in calls)
        with patch.object(Task, 'run_command', rebuilt):
            Task(request, settings, folder, home=home).run()
        self.assertEqual(sum('-filter_complex' in call for call in calls), before + 1)

    def test_cache_write_failure_does_not_fail_conversion(self):
        source, settings, calls = self.conversion_fixture()
        with patch.object(SeparationCache, 'store', side_effect=OSError('disk full')):
            result, _ = Task(Request(str(source), 'convert'), settings, self.root / 'job', home=self.root / 'profile').run()
        self.assertTrue(result.is_file())

    def test_subsecond_edit_changes_file_stamp(self):
        file = self.root / 'model.pth'
        file.write_bytes(b'first')
        whole = 1700000000 * 10**9
        os.utime(file, ns=(whole, whole + 100000000))
        first = file_stamp(file)
        file.write_bytes(b'other')
        os.utime(file, ns=(whole, whole + 200000000))
        self.assertNotEqual(first, file_stamp(file))

    def test_running_task_cannot_be_retried(self):
        home = self.root / 'profile'
        self.settings.save(home / 'settings.json')
        manager = Manager(home)
        entered, release = threading.Event(), threading.Event()
        def blocked(task):
            task.folder.mkdir(parents=True, exist_ok=True)
            task.state['request'] = dict(source=str(self.source), mode='original', name='', start=0, end=None)
            task.save_state(); entered.set()
            if not release.wait(5): raise RuntimeError('fixture timeout')
            return self.source, 'fixture'
        try:
            with patch.object(Task, 'run', blocked):
                job = manager.submit(Request(str(self.source), 'original'), self.settings)
                self.assertTrue(entered.wait(3))
                original = manager.tasks[job]
                try:
                    with self.assertRaisesRegex(ValueError, '运行|处理|等待'):
                        manager.retry(job)
                    self.assertIs(manager.tasks[job], original)
                finally:
                    release.set(); manager.queue.join()
        finally:
            release.set(); manager.shutdown()


if __name__ == '__main__': unittest.main()

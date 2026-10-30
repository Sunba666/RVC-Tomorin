import array
import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import wave
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_workbench.core import STAGE_FILE, Request, Settings
from voice_workbench.gui import App, acquire_instance_lock, preview_clip

FFMPEG = os.environ.get('VOICE_TEST_FFMPEG') or shutil.which('ffmpeg')


@unittest.skipUnless(FFMPEG and (os.name == 'nt' or os.environ.get('DISPLAY')), 'A desktop and FFmpeg are required')
class DesktopFlowTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-gui-')
        self.folder = Path(self.temp.name).resolve()
        self.assertEqual(self.folder.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temp.cleanup)
        self.environment = patch.dict(os.environ, {'VOICE_WORKBENCH_HOME': str(self.folder / 'data')})
        self.environment.start(); self.addCleanup(self.environment.stop)
        Settings(ffmpeg=str(Path(FFMPEG).parent), output=str(self.folder / 'out')).save(self.folder / 'data/settings.json')
        self.source = self.folder / '用户素材.wav'
        with wave.open(str(self.source), 'wb') as file:
            file.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            file.writeframes(array.array('h', [2000, -2000] * 16000).tobytes())
        self.root = tk.Tk(); self.root.withdraw(); self.app = App(self.root)
        self.addCleanup(self.cleanup_gui)

    def cleanup_gui(self):
        self.app.close()
        self.app = None
        self.root = None
        gc.collect()

    def test_submit_from_desktop_and_openable_completed_result(self):
        self.app.source.set(str(self.source)); self.app.mode.set('original')
        self.app.name.set('桌面真实流程'); self.app.end.set('1')
        self.app.submit()
        deadline = time.monotonic() + 10
        while self.app.manager.tasks and time.monotonic() < deadline:
            self.root.update(); time.sleep(.03)
        self.root.update(); self.app.refresh()
        selected = self.app.selected()
        self.assertEqual(selected['state'], 'completed')
        self.assertTrue(Path(selected['result']).is_file())
        with patch('voice_workbench.gui.open_file') as opened:
            self.app.open_selected(); opened.assert_called_once_with(Path(selected['result']))

    def test_bad_trim_shows_error_and_does_not_enqueue(self):
        self.app.source.set(str(self.source)); self.app.start.set('8'); self.app.end.set('2')
        with patch('voice_workbench.gui.messagebox.showerror') as error:
            self.app.submit(); error.assert_called_once()
        self.assertEqual(self.app.manager.store.list(), [])

    def test_import_voice_from_main_window_copies_model_and_matching_index(self):
        model = self.folder / '我的音色.pth'
        index = model.with_suffix('.index')
        model.write_bytes(b'test-model')
        index.write_bytes(b'test-index')
        with patch('voice_workbench.gui.filedialog.askopenfilename', return_value=str(model)):
            self.app.import_model_button.invoke()
        deadline = time.monotonic() + 5
        while not self.app.settings.model and time.monotonic() < deadline:
            self.root.update(); time.sleep(.02)
        self.assertEqual(Path(self.app.settings.model).read_bytes(), b'test-model')
        self.assertEqual(Path(self.app.settings.index).read_bytes(), b'test-index')
        self.assertEqual(self.app.mode.get(), 'convert')
        saved = Settings.load(self.app.settings_file)
        self.assertEqual(saved.model, self.app.settings.model)
        self.assertEqual(model.read_bytes(), b'test-model')

    def test_training_controls_submit_selected_materials_and_parameters(self):
        panel = self.app.training_panel
        self.app.show_workspace('training')
        panel.append_sources([str(self.source), str(self.source)])
        panel.values['name'].set('桌面训练')
        panel.values['epochs'].set('25'); panel.values['batch_size'].set('2')
        captured = []
        def enqueue(request, settings):
            captured.append(request)
            self.app.manager.store.add('training-test', request, self.folder / 'trained')
            return 'training-test'
        with patch.object(self.app.manager, 'submit', side_effect=enqueue): panel.start_button.invoke()
        self.assertEqual(len(captured), 1)
        self.assertEqual((captured[0].epochs, captured[0].batch_size, len(captured[0].sources)), (25, 2, 1))
        self.assertTrue(captured[0].extract_vocals)
        self.assertEqual(self.app.selected()['mode'], 'train')
        self.assertEqual(self.app.use_model_button.winfo_manager(), '')

    def test_training_invalid_parameters_show_error(self):
        panel = self.app.training_panel
        panel.append_sources([str(self.source)]); panel.values['epochs'].set('0')
        with patch('voice_workbench.training_ui.messagebox.showerror') as error:
            panel.start_button.invoke(); error.assert_called_once()
        self.assertEqual(self.app.manager.store.list(), [])

    def walk(self, widget):
        for child in widget.winfo_children():
            yield child
            yield from self.walk(child)

    def test_health_hint_and_environment_page(self):
        self.app.update_health()
        self.assertTrue(self.app.health_label.winfo_manager(), '缺少依赖时应显示提示')
        self.assertIn('未就绪', self.app.health_label.cget('text'))
        dialog = self.app.settings_dialog('environment')
        buttons = [item.cget('text') for item in self.walk(dialog) if item.winfo_class() == 'TButton']
        self.assertIn('重新检测', buttons)
        tabs = self.app.settings_tabs
        self.assertIn('环境检测', [tabs.tab(item, 'text') for item in tabs.tabs()])

    def test_failed_task_can_resume_and_reuse_parameters(self):
        request = Request(str(self.source), 'convert', '失败任务')
        folder = self.folder / 'failed-task'
        folder.mkdir(parents=True)
        self.app.manager.store.add('failed-job', request, folder)
        self.app.manager.store.update('failed-job', state='failed')
        (folder / STAGE_FILE).write_text(json.dumps({'request': asdict(request), 'settings': {'pitch': 5, 'index_rate': .7, 'accompaniment': .2}}), encoding='utf-8')
        self.app.refresh()
        self.app.tree.selection_set('failed-job')
        self.app.update_actions()
        self.assertFalse(self.app.resume_button.instate(['disabled']))
        self.assertEqual(self.app.resume_button.cget('text'), '继续制作')
        self.app.reuse_parameters(self.app.selected())
        self.assertEqual((self.app.settings.pitch, self.app.settings.index_rate, self.app.settings.accompaniment), (5, .7, .2))
        self.assertEqual(self.app.source.get(), str(self.source))
        self.assertEqual(self.app.mode.get(), 'convert')

    def test_preview_clip_trims_and_is_reused(self):
        destination = self.folder / 'preview' / '片段.wav'
        clip = Path(preview_clip(self.app.settings, self.source, destination))
        self.assertTrue(clip.is_file())
        with wave.open(str(clip), 'rb') as file:
            self.assertEqual((file.getframerate(), file.getnchannels(), file.getsampwidth()), (44100, 2, 2))
            self.assertLessEqual(file.getnframes(), 12 * 44100)
        stamp = clip.stat().st_mtime_ns
        preview_clip(self.app.settings, self.source, destination)
        self.assertEqual(clip.stat().st_mtime_ns, stamp, '同一片段的试听文件应复用')

    def test_completed_comparison_controls_fit_minimum_details_window(self):
        folder = self.folder / 'completed-conversion'
        folder.mkdir()
        for name in ('人声.wav', '转换后人声.wav'): shutil.copy2(self.source, folder / name)
        self.app.manager.store.add('layout-result', Request(str(self.source), 'convert'), folder)
        self.app.manager.store.update('layout-result', state='completed', result=str(folder / '转换后人声.wav'))
        self.app.refresh(); self.app.tree.selection_set('layout-result')
        self.root.deiconify()
        dialog = self.app.details(); dialog.geometry('580x360'); self.root.update()
        for widget in self.walk(dialog):
            if widget.winfo_class() == 'TButton':
                self.assertTrue(widget.winfo_ismapped(), widget.cget('text'))
                self.assertGreaterEqual(widget.winfo_width(), widget.winfo_reqwidth(), widget.cget('text'))
        dialog.destroy()

    @unittest.skipUnless(os.name == 'nt', 'Windows instance guard')
    def test_install_state_shared_and_double_start_guarded(self):
        release = threading.Event()
        started = threading.Event()

        def fake_install(home, progress=None, cancel=None, url=None):
            started.set()
            release.wait(5)
            if progress:
                progress(1.0, '完成')
            return Path(home) / 'tools' / 'ffmpeg'

        self.assertFalse(self.app.install['running'])
        with patch('voice_workbench.install.install_ffmpeg', side_effect=fake_install) as installer:
            self.assertIsNone(self.app.start_install())
            self.assertTrue(started.wait(2), '第一次安装应启动工作线程')
            self.assertTrue(self.app.install['running'])
            self.assertIsNotNone(self.app.start_install(), '安装中重复点击应被拒绝')
            release.set()
            deadline = time.monotonic() + 5
            while self.app.install['running'] and time.monotonic() < deadline:
                self.root.update(); time.sleep(.01)
            self.app.drain_install()
        self.assertFalse(self.app.install['running'])
        self.assertTrue(self.app.install['finished'])
        self.assertIsNone(self.app.install['error'])
        installer.assert_called_once()

    def test_second_window_cannot_reset_running_task_history(self):
        folder = self.folder / 'instance'
        handle = acquire_instance_lock(folder)
        try:
            with self.assertRaises(RuntimeError): acquire_instance_lock(folder)
        finally:
            handle.close()
        released = acquire_instance_lock(folder); released.close()


if __name__ == '__main__': unittest.main(verbosity=2)

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_workbench.core import Settings, data_home
from voice_workbench.portable import bundle_info, import_voice_model


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-portable-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '中文 整合包'
        self.root.mkdir()
        (self.root / 'portable.json').write_text(json.dumps({'schema': 1, 'engine': 'engine',
            'separator_backend': 'demucs', 'separator_model': 'engine/model.th',
            'separator_config': 'engine/model.yaml'}), encoding='utf-8')
        self.environ = patch.dict(os.environ, {'VOICE_WORKBENCH_BUNDLE': str(self.root)})
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.previous_home = os.environ.pop('VOICE_WORKBENCH_HOME', None)
        self.addCleanup(self.restore_home)

    def restore_home(self):
        if self.previous_home is not None:
            os.environ['VOICE_WORKBENCH_HOME'] = self.previous_home

    def test_fresh_profile_uses_only_bundled_engine_without_configuration(self):
        settings = Settings.load(data_home() / 'settings.json')
        self.assertEqual(data_home(), self.root / 'data')
        self.assertEqual(Path(settings.rvc_root), self.root / 'engine')
        self.assertEqual(Path(settings.python), self.root / 'engine/runtime/python.exe')
        self.assertEqual(settings.separator_backend, 'demucs')
        self.assertFalse(settings.model, '公开整合包不能自动夹带开发者角色模型')

    def test_old_external_engine_settings_cannot_override_bundle(self):
        file = data_home() / 'settings.json'
        file.parent.mkdir()
        file.write_text(json.dumps({'rvc_root': 'X:/old-rvc', 'python': 'X:/old/python.exe',
                                   'ffmpeg': 'X:/old-ffmpeg', 'separator_model': 'X:/old.ckpt'}))
        settings = Settings.load(file)
        self.assertTrue(Path(settings.python).is_relative_to(self.root))
        self.assertTrue(Path(settings.separator_model).is_relative_to(self.root))

    def test_imported_model_and_index_survive_bundle_relocation(self):
        original = Path(self.temp.name) / 'original'
        original.mkdir()
        (original / 'voice.pth').write_bytes(b'voice-data')
        (original / 'voice.index').write_bytes(b'index-data')
        imported = import_voice_model(original / 'voice.pth', data_home())
        settings = Settings.load(data_home() / 'settings.json')
        settings.model, settings.index = imported['model'], imported['index']
        settings.save(data_home() / 'settings.json')
        serialized = (data_home() / 'settings.json').read_text(encoding='utf-8')
        self.assertNotIn(str(self.root).replace('\\', '\\\\'), serialized)
        moved = Path(self.temp.name) / 'relocated'
        shutil.move(self.root, moved)
        os.environ['VOICE_WORKBENCH_BUNDLE'] = str(moved)
        restored = Settings.load(data_home() / 'settings.json')
        self.assertEqual(Path(restored.model).read_bytes(), b'voice-data')
        self.assertEqual(Path(restored.index).read_bytes(), b'index-data')
        self.assertEqual(Path(restored.rvc_root), moved / 'engine')
        self.assertEqual((original / 'voice.pth').read_bytes(), b'voice-data')

    def test_manifest_path_escape_is_rejected(self):
        file = self.root / 'portable.json'
        data = json.loads(file.read_text())
        data['engine'] = '../external'
        file.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, '整合包清单损坏'):
            bundle_info()

    def test_repeated_import_is_deduplicated_and_ambiguous_index_not_selected(self):
        folder = Path(self.temp.name) / 'voices'
        folder.mkdir()
        source = folder / 'a.pth'
        source.write_bytes(b'voice')
        (folder / 'b.index').write_bytes(b'b')
        (folder / 'c.index').write_bytes(b'c')
        first = import_voice_model(source, data_home())
        second = import_voice_model(source, data_home())
        self.assertEqual(first, second)
        self.assertFalse(first['index'])
        self.assertEqual(len(list((data_home() / 'models').iterdir())), 1)

    def test_invalid_or_empty_import_does_not_leave_temporary_files(self):
        source = Path(self.temp.name) / 'empty.pth'
        source.touch()
        with self.assertRaises(ValueError):
            import_voice_model(source, data_home())
        self.assertFalse(list((data_home() / 'models').iterdir()))
        with self.assertRaises(ValueError):
            import_voice_model(source.with_suffix('.txt'), data_home())


if __name__ == '__main__':
    unittest.main()

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import TestCase, main
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_runtime import RuntimeInstaller, InstallCancelled, install_lock


class DownloadTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rvc-download-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.adjacent = self.base / 'adjacent'; self.adjacent.mkdir()
        self.archive = self.adjacent / 'runtime.zip'
        self.required = ['VoiceWorkbench.exe', 'engine/runtime/python.exe', 'engine/ffmpeg.exe',
                         'engine/ffprobe.exe', 'engine/infer/vc/modules.py',
                         'engine/assets/hubert_base/pytorch_model.bin', 'engine/assets/rmvpe/rmvpe.pt',
                         'engine/assets/separator/955717e8-8726e21a.th']
        self.required += [f'engine/assets/pretrained_v2/f0{k}{s}.pth'
                          for k in ('G', 'D') for s in ('32k', '40k', '48k')]
        self.create_archive()

    def create_archive(self, unsafe=False, chunk_size=1500):
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name in self.required: z.writestr('app/' + name, b'test foundation content')
            z.writestr('app/portable.json', json.dumps({'version': '0.3.0'}))
            if unsafe: z.writestr('app/../../escaped.txt', b'no')
        data = self.archive.read_bytes()
        self.info = {'folder': 'app', 'archive': 'runtime.zip', 'version': '0.3.0',
            'repository': 'owner/repo', 'tag': 'v0.3.0', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'parts': [], 'min_space_bytes': 1}
        for number, offset in enumerate(range(0, len(data), chunk_size)):
            chunk = data[offset:offset+chunk_size]
            name = 'runtime.zip.' + str(number+1).zfill(3)
            (self.adjacent/name).write_bytes(chunk)
            self.info['parts'].append({'name': name, 'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})

    def installer(self, **kwargs):
        return RuntimeInstaller(self.info, self.base / 'installed', adjacent=self.adjacent, **kwargs)

    def test_verified_offline_parts_install_and_preserve_original_parts(self):
        self.archive.unlink()
        installer = self.installer()
        exe = installer.install()
        self.assertTrue(exe.is_file())
        self.assertTrue(installer.ready())
        self.assertTrue(all((self.adjacent/p['name']).exists() for p in self.info['parts']))
        self.assertFalse(list(installer.root.glob('.install-*')))
        self.assertFalse((installer.cache/self.info['archive']).exists())

    def test_completed_install_keeps_personal_data_and_needs_no_download(self):
        installer = self.installer(); installer.install()
        personal = installer.target/'data/settings.json'
        personal.parent.mkdir(); personal.write_bytes(b'personal')
        with patch.object(installer, 'archive', side_effect=AssertionError('must not download')):
            with patch('download_runtime.shutil.disk_usage', side_effect=AssertionError('must not require fresh space')):
                installer.install()
        self.assertEqual(personal.read_bytes(), b'personal')

    def test_more_than_nine_parts_are_all_assembled_in_manifest_order(self):
        self.create_archive(chunk_size=200)
        self.assertGreater(len(self.info['parts']), 9)
        self.archive.unlink()
        installer=self.installer()
        self.assertTrue(installer.install().is_file())

    def test_incomplete_existing_install_is_not_overwritten(self):
        installer = self.installer()
        installer.target.mkdir(parents=True)
        personal = installer.target/'my-model.pth'; personal.write_bytes(b'keep')
        with self.assertRaisesRegex(RuntimeError, '没有覆盖'): installer.install()
        self.assertEqual(personal.read_bytes(), b'keep')

    def test_zip_path_traversal_is_rejected_and_own_stage_cleaned(self):
        self.create_archive(unsafe=True)
        installer = self.installer()
        with self.assertRaisesRegex(ValueError, '不安全路径'): installer.install()
        self.assertFalse((self.base/'escaped.txt').exists())
        self.assertFalse(installer.target.exists())
        self.assertFalse(list(installer.root.glob('.install-*')))

    def test_cancel_during_extract_preserves_archive_and_cleans_stage(self):
        event = threading.Event()
        installer = self.installer(cancel=event.is_set, notify=lambda p, s: event.set() if p > 70 else None)
        with self.assertRaises(InstallCancelled): installer.install()
        self.assertTrue(self.archive.exists())
        self.assertFalse(installer.target.exists())
        self.assertFalse(list(installer.root.glob('.install-*')))

    def test_concurrent_installer_is_rejected(self):
        root=self.base/'lock'; root.mkdir()
        with install_lock(root):
            with self.assertRaisesRegex(RuntimeError, '另一个安装'): self.lock_again(root)

    def lock_again(self, root):
        with install_lock(root): self.fail('second lock must fail')

    def test_manifest_rejects_outside_file_names(self):
        self.info['parts'][0]['name']='../private.txt'
        with self.assertRaises(ValueError): self.installer()

    def serve(self, supports_range):
        data = (self.adjacent/self.info['parts'][0]['name']).read_bytes()
        ranges=[]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                value=self.headers.get('Range');ranges.append(value)
                offset=int(value[6:-1]) if value and supports_range else 0
                self.send_response(206 if value and supports_range else 200)
                if value and supports_range:self.send_header('Content-Range',f'bytes {offset}-{len(data)-1}/{len(data)}')
                self.send_header('Content-Length',str(len(data)-offset));self.end_headers();self.wfile.write(data[offset:])
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def close():server.shutdown();server.server_close();thread.join()
        self.addCleanup(close)
        return f'http://127.0.0.1:{server.server_port}/',ranges,data

    def download_test(self, supports_range):
        base,ranges,data=self.serve(supports_range)
        installer=RuntimeInstaller(self.info,self.base/'network',adjacent=self.base/'empty',base_url=base)
        installer.cache.mkdir(parents=True)
        record=self.info['parts'][0]
        (installer.cache/(record['name']+'.download')).write_bytes(data[:123])
        result=installer.download(record,0,len(self.info['parts']))
        self.assertEqual(result.read_bytes(),data)
        self.assertEqual(ranges,['bytes=123-'])

    def test_http_download_resumes_from_valid_offset(self): self.download_test(True)
    def test_server_ignoring_range_restarts_instead_of_appending(self): self.download_test(False)


if __name__=='__main__':main()

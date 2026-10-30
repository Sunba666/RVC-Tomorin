import os
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_workbench.core import Settings
from voice_workbench.install import (InstallCancelled, fetch, ffmpeg_folder, install_ffmpeg,
                                    tools_folder)

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class Handler(BaseHTTPRequestHandler):
    payload = b''
    honour_range = True
    ranges = []
    truncate_at = None       # 声明完整长度却只写前少许字节后断开
    reply_416 = False        # 收到 Range 请求就返回 416
    misdeclare_range = False  # 返回 206 但 Content-Range 从 0 开始（模拟忽略断点）

    def do_GET(self):
        header = self.headers.get('Range', '')
        if header:
            Handler.ranges.append(header)
        if self.reply_416 and header:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{len(self.payload)}')
            self.end_headers()
            return
        if not self.honour_range:
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            if self.truncate_at is not None:
                self.wfile.write(self.payload[:self.truncate_at])
            else:
                self.wfile.write(self.payload)
            return
        start = 0
        if header.startswith('bytes='):
            start = int(header[6:].split('-')[0])
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        if start:
            declared = 0 if self.misdeclare_range else start
            self.send_header('Content-Range', f'bytes {declared}-{len(self.payload) - 1}/{len(self.payload)}')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-install-')
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        Handler.payload = PAYLOAD
        Handler.honour_range = True
        Handler.ranges = []
        Handler.truncate_at = None
        Handler.reply_416 = False
        Handler.misdeclare_range = False
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/file.bin'

    def progress_recorder(self):
        seen = []

        def record(fraction, note):
            seen.append((fraction, note))

        return seen, record


class FetchTests(ServerCase):
    def test_downloads_and_reports_progress(self):
        destination = self.root / 'download' / 'payload.bin'
        seen, progress = self.progress_recorder()
        fetch(self.url, destination, progress=progress)
        self.assertEqual(destination.read_bytes(), PAYLOAD)
        self.assertTrue(seen)
        fractions = [fraction for fraction, _ in seen]
        self.assertEqual(fractions, sorted(fractions), '进度不应回退')
        self.assertGreaterEqual(max(fractions), .99)
        self.assertIn('MB', seen[-1][1])

    def test_resumes_partial_file(self):
        destination = self.root / 'payload.bin'
        destination.write_bytes(PAYLOAD[:400_000])
        fetch(self.url, destination)
        self.assertEqual(destination.read_bytes(), PAYLOAD)
        self.assertEqual(Handler.ranges[-1], f'bytes={400_000}-')

    def test_restarts_when_server_ignores_range(self):
        destination = self.root / 'payload.bin'
        destination.write_bytes(b'broken')
        Handler.honour_range = False
        fetch(self.url, destination)
        self.assertEqual(destination.read_bytes(), PAYLOAD)

    def test_cancel_stops_download(self):
        destination = self.root / 'payload.bin'
        cancel = lambda: destination.exists() and destination.stat().st_size > 0
        with self.assertRaises(InstallCancelled):
            fetch(self.url, destination, cancel=cancel)
        self.assertLess(destination.stat().st_size, len(PAYLOAD))

    def test_reports_network_failure(self):
        with self.assertRaises(RuntimeError):
            fetch('http://127.0.0.1:1/nothing.bin', self.root / 'x.bin', retries=0)

    def test_detects_truncated_download(self):
        Handler.honour_range = False
        Handler.truncate_at = 3
        with self.assertRaises(RuntimeError):
            fetch(self.url, self.root / 'payload.bin', retries=0)
        self.assertNotEqual((self.root / 'payload.bin').read_bytes(), PAYLOAD)

    def test_restarts_when_content_range_mismatches(self):
        destination = self.root / 'payload.bin'
        destination.write_bytes(PAYLOAD[:400_000])
        Handler.misdeclare_range = True
        fetch(self.url, destination)
        self.assertEqual(destination.read_bytes(), PAYLOAD)
        self.assertEqual([header for header in Handler.ranges if header], ['bytes=400000-'])

    def test_recovers_from_416(self):
        destination = self.root / 'payload.bin'
        destination.write_bytes(b'x' * 100)
        Handler.reply_416 = True
        fetch(self.url, destination)
        self.assertEqual(destination.read_bytes(), PAYLOAD)
        range_requests = [header for header in Handler.ranges if header]
        self.assertEqual(len(range_requests), 1, '416 之后应从零重下，不再带 Range')


def build_zip(members):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return buffer.getvalue()


class InstallTests(ServerCase):
    def serve(self, payload):
        Handler.payload = payload
        Handler.honour_range = False
        return self.url

    def test_installs_ffmpeg_into_tools_folder(self):
        archive = build_zip({'ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe': b'MZ'+b'a' * 100,
                             'ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe': b'MZ'+b'b' * 100,
                             'ffmpeg-9.0.2-essentials_build/README.txt': b'readme'})
        url = self.serve(archive)
        seen, progress = self.progress_recorder()
        with mock.patch('voice_workbench.install.executable_version', return_value='ffmpeg version 9.0.2-test'):
            folder = install_ffmpeg(self.root / 'home', progress=progress, url=url)
        self.assertEqual(folder, ffmpeg_folder(self.root / 'home'))
        self.assertEqual((folder / 'ffmpeg.exe').read_bytes(), b'MZ' + b'a' * 100)
        self.assertEqual((folder / 'ffprobe.exe').read_bytes(), b'MZ' + b'b' * 100)
        self.assertFalse((tools_folder(self.root / 'home') / 'downloads' / 'ffmpeg.zip').exists(), '安装完成后应删除压缩包')
        self.assertEqual(seen[-1], (1.0, 'ffmpeg version 9.0.2-test'))
        self.assertFalse(list(folder.glob('*.part')))

    def test_rejects_archive_without_binaries(self):
        url = self.serve(build_zip({'ffmpeg/README.txt': b'no binaries'}))
        with mock.patch('voice_workbench.install.executable_version', return_value='ffmpeg version test'):
            with self.assertRaises(RuntimeError) as caught:
                install_ffmpeg(self.root / 'home', url=url)
        self.assertIn('ffmpeg.exe', str(caught.exception))
        self.assertFalse((ffmpeg_folder(self.root / 'home') / 'ffmpeg.exe').exists())

    def test_fails_when_binary_cannot_run(self):
        archive = build_zip({'ffmpeg/bin/ffmpeg.exe': b'MZ', 'ffmpeg/bin/ffprobe.exe': b'MZ'})
        url = self.serve(archive)
        with mock.patch('voice_workbench.install.executable_version', return_value=''):
            with self.assertRaises(RuntimeError):
                install_ffmpeg(self.root / 'home', url=url)

    def test_preserves_working_install_when_new_binary_fails(self):
        target = ffmpeg_folder(self.root / 'home')
        target.mkdir(parents=True)
        (target / 'ffmpeg.exe').write_bytes(b'OLD-ffmpeg')
        (target / 'ffprobe.exe').write_bytes(b'OLD-ffprobe')
        archive = build_zip({'ffmpeg/bin/ffmpeg.exe': b'MZ' + b'x' * 50,
                             'ffmpeg/bin/ffprobe.exe': b'MZ' + b'y' * 50})
        url = self.serve(archive)
        with mock.patch('voice_workbench.install.executable_version', return_value=''):
            with self.assertRaises(RuntimeError):
                install_ffmpeg(self.root / 'home', url=url)
        self.assertEqual((target / 'ffmpeg.exe').read_bytes(), b'OLD-ffmpeg')
        self.assertEqual((target / 'ffprobe.exe').read_bytes(), b'OLD-ffprobe')

    def test_checks_ffprobe_not_just_ffmpeg(self):
        archive = build_zip({'ffmpeg/bin/ffmpeg.exe': b'MZ', 'ffmpeg/bin/ffprobe.exe': b'MZ'})
        url = self.serve(archive)

        def fake_version(path):
            return 'ffmpeg version test' if 'ffmpeg.exe' in str(path) else ''

        with mock.patch('voice_workbench.install.executable_version', side_effect=fake_version):
            with self.assertRaises(RuntimeError) as caught:
                install_ffmpeg(self.root / 'home', url=url)
        self.assertIn('ffprobe.exe', str(caught.exception))
        self.assertFalse((ffmpeg_folder(self.root / 'home') / 'ffmpeg.exe').exists(),
                         'ffprobe 校验失败时不应提交已经就绪的 ffmpeg')

    def test_rejects_archive_when_sha256_mismatches(self):
        archive = build_zip({'ffmpeg/bin/ffmpeg.exe': b'MZ' + b'x' * 50, 'ffmpeg/bin/ffprobe.exe': b'MZ' + b'y' * 50})
        url = self.serve(archive)
        with mock.patch('voice_workbench.install.FFMPEG_SHA256', '0' * 64):
            with mock.patch('voice_workbench.install.executable_version', return_value='ffmpeg version test'):
                with self.assertRaises(RuntimeError) as caught:
                    install_ffmpeg(self.root / 'home', url=url)
        self.assertIn('摘要不匹配', str(caught.exception))
        self.assertFalse((ffmpeg_folder(self.root / 'home') / 'ffmpeg.exe').exists(),
                         '摘要不符时应中止，不提交任何二进制')

    def test_commit_rolls_back_when_second_file_is_busy(self):
        # 复用既有安装：提交阶段第一个二进制（ffmpeg.exe）成功后，第二个（ffprobe.exe）
        # 被占用导致替换失败，应回滚已提交的部分并把两个旧文件都还原，不能新旧混用。
        target = ffmpeg_folder(self.root / 'home')
        target.mkdir(parents=True)
        (target / 'ffmpeg.exe').write_bytes(b'OLD-ffmpeg')
        (target / 'ffprobe.exe').write_bytes(b'OLD-ffprobe')
        archive = build_zip({'ffmpeg/bin/ffmpeg.exe': b'MZ' + b'x' * 60,
                             'ffmpeg/bin/ffprobe.exe': b'MZ' + b'y' * 60})
        url = self.serve(archive)
        real_replace = os.replace
        state = {'hit': False}

        def flaky_replace(src, dst):
            src_s = str(src).replace('\\', '/')
            dst_s = str(dst).replace('\\', '/')
            # 提交阶段：staging 下的 ffprobe.exe → 正式目录，模拟第二个文件被占用。
            if '.ffmpeg-stage' in src_s and src_s.endswith('/ffprobe.exe') and dst_s.endswith('/ffprobe.exe'):
                state['hit'] = True
                raise OSError('另一个程序正在使用这个文件')
            return real_replace(src, dst)

        with mock.patch('voice_workbench.install.os.replace', side_effect=flaky_replace):
            with mock.patch('voice_workbench.install.executable_version', return_value='ffmpeg version test'):
                with self.assertRaises(RuntimeError) as caught:
                    install_ffmpeg(self.root / 'home', url=url)
        self.assertTrue(state['hit'], '应当模拟到第二个文件的替换失败')
        self.assertIn('原安装已恢复', str(caught.exception))
        self.assertEqual((target / 'ffmpeg.exe').read_bytes(), b'OLD-ffmpeg', 'ffmpeg.exe 应回滚到旧版')
        self.assertEqual((target / 'ffprobe.exe').read_bytes(), b'OLD-ffprobe', 'ffprobe.exe 应保持旧版')

    def test_commit_and_restore_both_fail_keeps_backup(self):
        # 提交失败、且恢复旧文件也失败时：不得删除备份，错误提示要给出备份位置，让用户能手动找回旧文件。
        target = ffmpeg_folder(self.root / 'home')
        target.mkdir(parents=True)
        (target / 'ffmpeg.exe').write_bytes(b'OLD-ffmpeg')
        (target / 'ffprobe.exe').write_bytes(b'OLD-ffprobe')
        archive = build_zip({'ffmpeg/bin/ffmpeg.exe': b'MZ' + b'x' * 60,
                             'ffmpeg/bin/ffprobe.exe': b'MZ' + b'y' * 60})
        url = self.serve(archive)
        real_replace = os.replace
        state = {'commit_hit': False, 'restore_hit': False}

        def flaky_replace(src, dst):
            src_s = str(src).replace('\\', '/')
            dst_s = str(dst).replace('\\', '/')
            if '.ffmpeg-stage' in src_s and src_s.endswith('/ffprobe.exe') and dst_s.endswith('/ffprobe.exe'):
                state['commit_hit'] = True
                raise OSError('另一个程序正在使用这个文件（提交失败）')
            if '.ffmpeg-backup' in src_s and src_s.endswith('/ffprobe.exe') and dst_s.endswith('/ffprobe.exe'):
                state['restore_hit'] = True
                raise OSError('另一个程序持续占用这个文件（恢复失败）')
            return real_replace(src, dst)

        with mock.patch('voice_workbench.install.os.replace', side_effect=flaky_replace):
            with mock.patch('voice_workbench.install.executable_version', return_value='ffmpeg version test'):
                with self.assertRaises(RuntimeError) as caught:
                    install_ffmpeg(self.root / 'home', url=url)
        self.assertTrue(state['commit_hit'], '应当模拟到提交的 ffprobe.exe 失败')
        self.assertTrue(state['restore_hit'], '应当模拟到恢复的 ffprobe.exe 也失败')
        message = str(caught.exception)
        self.assertIn('恢复也失败', message, '错误提示应明确指出恢复也失败了')
        self.assertIn('备份目录', message, '错误提示应给出备份目录位置')
        backups = list((tools_folder(self.root / 'home') / 'downloads').glob('.ffmpeg-backup-*.bin'))
        self.assertTrue(backups, '恢复失败时备份目录不应被删除')
        self.assertEqual((backups[0] / 'ffprobe.exe').read_bytes(), b'OLD-ffprobe', '旧 ffprobe.exe 应保留在备份里')
        self.assertEqual((target / 'ffmpeg.exe').read_bytes(), b'OLD-ffmpeg', '恢复成功的 ffmpeg.exe 应回到正式目录')


class ExecutableLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vwb-lookup-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve() / 'home'
        self.previous = os.environ.get('VOICE_WORKBENCH_HOME')
        os.environ['VOICE_WORKBENCH_HOME'] = str(self.home)
        self.addCleanup(self.restore_home)

    def restore_home(self):
        if self.previous is None:
            os.environ.pop('VOICE_WORKBENCH_HOME', None)
        else:
            os.environ['VOICE_WORKBENCH_HOME'] = self.previous

    def test_settings_find_binaries_from_tools_folder(self):
        tools = ffmpeg_folder(self.home)
        tools.mkdir(parents=True)
        (tools / 'ffmpeg.exe').write_bytes(b'MZ')
        settings = Settings(ffmpeg='', output=str(self.home / '输出'))
        self.assertEqual(Path(settings.executable('ffmpeg')), tools / 'ffmpeg.exe')


if __name__ == '__main__':
    unittest.main()
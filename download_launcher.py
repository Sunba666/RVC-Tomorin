"""Small Windows launcher: no Python or RVC installation required by the user."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from download_runtime import RuntimeInstaller, InstallCancelled


def default_manifest():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)) / 'release/portable-0.3.0.json'


def default_root():
    return Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Programs/RVC-Tomorin'


def shortcut(executable):
    if os.name != 'nt': return
    quoted = str(executable).replace("'", "''")
    script = "$d=[Environment]::GetFolderPath('DesktopDirectory'); $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut((Join-Path $d 'RVC-Tomorin.lnk')); "
    script += f"$s.TargetPath='{quoted}'; $s.WorkingDirectory='{str(executable.parent).replace(chr(39), chr(39)*2)}'; $s.Save()"
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=default_manifest())
    parser.add_argument('--destination', type=Path, default=default_root())
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--offline-folder', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--smoke-ui', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    adjacent = args.offline_folder or (Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent)
    if args.headless:
        exe = RuntimeInstaller(manifest, args.destination, cache=args.cache, adjacent=adjacent,
            notify=lambda p, s: print(f'{p:.0f}% {s}', flush=True)).install()
        if args.report:
            args.report.write_text(json.dumps({'installed': str(exe), 'version': manifest['version']}, ensure_ascii=False), encoding='utf-8')
        print(str(exe)); return
    root = tk.Tk()
    root.title('RVC-Tomorin · 安装与启动')
    root.geometry('560x320'); root.resizable(False, False); root.configure(bg='#f5f6f3')
    style = ttk.Style(root); style.theme_use('clam')
    style.configure('TFrame', background='#f5f6f3')
    style.configure('TLabel', background='#f5f6f3', foreground='#243b36', font=('Microsoft YaHei UI', 10))
    style.configure('TProgressbar', background='#367565', troughcolor='#e1e7e1')
    body = ttk.Frame(root, padding=24); body.pack(fill='both', expand=True)
    footer = ttk.Frame(body); footer.pack(side='bottom', fill='x', pady=(10, 0))
    button = ttk.Button(footer, text='取消'); button.pack(side='right')
    ttk.Label(body, text='RVC-Tomorin · 角色音频工坊', font=('Microsoft YaHei UI', 16, 'bold')).pack(anchor='w')
    ttk.Label(body, text='首次自动下载约 6 GB。环境会一起准备好，无需填写路径。', wraplength=490).pack(anchor='w', pady=(12, 16))
    progress = ttk.Progressbar(body, maximum=100); progress.pack(fill='x')
    message = tk.StringVar(value='正在检查软件…')
    ttk.Label(body, textvariable=message, wraplength=490).pack(anchor='w', pady=10)
    events = queue.Queue(); stop = threading.Event(); active = [False]
    def close():
        if active[0]: stop.set(); message.set('正在取消，已下载内容可下次继续…')
        else: root.destroy()
    button.configure(command=close); root.protocol('WM_DELETE_WINDOW', close)
    def start():
        if active[0]: return
        stop.clear(); active[0] = True; button.configure(text='取消', command=close)
        def work():
            try:
                installer = RuntimeInstaller(manifest, args.destination, cache=args.cache,
                    adjacent=adjacent, cancel=stop.is_set, notify=lambda p, s: events.put(('progress', (p, s))))
                exe = installer.install()
                try: shortcut(exe)
                except Exception: pass
                events.put(('done', exe))
            except InstallCancelled as error: events.put(('cancelled', str(error)))
            except Exception as error: events.put(('error', str(error)))
        threading.Thread(target=work, daemon=True).start()
    def poll():
        try:
            # Coalesce download/extraction updates instead of building a huge UI queue.
            pending = None
            while True:
                kind, value = events.get_nowait()
                if kind == 'progress': pending = value; continue
                pending = None
                active[0] = False
                if kind == 'done':
                    subprocess.Popen([str(value)], cwd=value.parent)
                    root.destroy(); return
                if kind == 'cancelled': root.destroy(); return
                message.set(value[:150]); button.configure(text='重试', command=start)
        except queue.Empty: pass
        if pending is not None:
            progress['value'], text = pending; message.set(text)
        root.after(100, poll)
    if args.smoke_ui:
        if args.report:
            args.report.write_text(json.dumps({'title': root.title(), 'version': manifest['version'],
                'widgets': len(body.winfo_children())}, ensure_ascii=False), encoding='utf-8')
        root.after(200, root.destroy)
    else:
        root.after(150, start); root.after(100, poll)
    root.mainloop()


if __name__ == '__main__':
    main()

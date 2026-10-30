from __future__ import annotations
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tempfile
import wave
import tkinter as tk
import webbrowser
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from . import __version__
from .portable import bundle_root, import_voice_model
from .core import (STAGE_FILE, Manager, Request, SeparationCache, Settings, data_home, environment_report,
                   pretty_bytes, file_stamp, fingerprint, task_settings)

BG, PANEL, INK, MUTED = '#f5f6f3', '#ffffff', '#243b36', '#74817b'
ACCENT, SOFT, LINE = '#367565', '#edf4ef', '#e1e7e1'
FONT = 'Microsoft YaHei UI'
STATES = {'queued': '等待中', 'running': '处理中', 'completed': '已完成',
          'failed': '失败', 'cancelled': '已取消', 'cancelling': '正在取消', 'interrupted': '已中断'}


def open_file(path):
    import subprocess
    if os.name == 'nt':
        os.startfile(str(path))
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', str(path)])
    else:
        subprocess.Popen(['xdg-open', str(path)])


def source_from_text(text):
    text = text.strip().strip('"')
    if re.fullmatch(r'BV[0-9A-Za-z]{10}', text):
        return 'https://www.bilibili.com/video/' + text
    match = re.search(r'https?://[^\s<>"“”]+', text)
    return match.group(0).rstrip('，。；）)】]') if match else text


PREVIEW_SECONDS = 12
PREVIEW_CACHE_BYTES = 2 * 2 ** 30  # 试听缓存上限，超出按最旧优先淘汰（S02）
FFMPEG_LINK = 'https://ffmpeg.org/download.html'
RVC_LINK = 'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI'


def _first_sound(settings, source: Path) -> float:
    """用 silencedetect 找首个有声点，试听跳过开头静音（S05）。失败或本就开头有声则回 0。"""
    try:
        command = [settings.executable('ffmpeg'), '-nostdin', '-hide_banner', '-i', str(source),
                   '-af', 'silencedetect=noise=-45dB:d=0.6', '-f', 'null', '-']
        run = subprocess.run(command, capture_output=True, timeout=90,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        text = run.stderr.decode('utf-8', 'replace')
        starts = [float(m) for m in re.findall(r'silence_start:\s*(-?\d+(?:\.\d+)?)', text)]
        ends = [float(m) for m in re.findall(r'silence_end:\s*(-?\d+(?:\.\d+)?)', text)]
        for begin, end in zip(starts, ends):
            if begin <= 0.1:
                return max(0.0, end - 0.4)  # 留 0.4 秒前摇，避免从有声瞬间硬切
    except Exception:
        pass
    return 0.0


def prune_preview_dir(root: Path, limit: int = PREVIEW_CACHE_BYTES) -> int:
    """试听缓存按最旧优先淘汰到上限以内（S02），只动 16 位十六进制命名的 wav/json，返回实际删除字节数。"""
    root = Path(root)
    if not root.is_dir():
        return 0
    entries = []
    for file in root.iterdir():
        if not re.fullmatch(r'[0-9a-f]{16}\.(wav|json)', file.name):
            continue
        try:
            entries.append((file.stat().st_mtime, file.stat().st_size, file))
        except OSError:
            continue
    total = sum(size for _, size, _ in entries)
    removed = 0
    for _, size, file in sorted(entries):
        if total <= limit:
            break
        try:
            file.unlink(missing_ok=True)
            removed += size
            total -= size
        except OSError:
            pass
    return removed


def preview_clip(settings, source: Path, destination: Path) -> Path:
    """截一段短音频做试听对照，避免整首歌来回播放。跳过开头静音（S05）。"""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f'找不到 {source.name}')
    marker = destination.with_suffix('.json')
    signature = fingerprint('preview-v3', file_stamp(source), PREVIEW_SECONDS)
    def valid(file):
        try:
            with wave.open(str(file), 'rb') as audio:
                count = audio.getnframes()
                return (audio.getnchannels() == 2 and audio.getsampwidth() == 2 and audio.getframerate() == 44100
                        and 0 < count <= PREVIEW_SECONDS * 44100 and len(audio.readframes(count)) == count * 4)
        except (OSError, EOFError, wave.Error): return False
    try:
        if marker.read_text(encoding='utf-8') == signature and valid(destination): return destination
    except OSError: pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = _first_sound(settings, source)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.wav', delete=False) as pending:
        temporary = Path(pending.name)

    def generate(offset: float):
        command = [settings.executable('ffmpeg'), '-nostdin', '-y', '-loglevel', 'error',
                   '-ss', f'{offset:.3f}', '-t', str(PREVIEW_SECONDS),
                   '-i', str(source), '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', str(temporary)]
        return subprocess.run(command, capture_output=True, timeout=30,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        run = generate(start)
        # 检测到的起点若落在结尾导致空片段，退回从头截取一次（S05 兜底）。
        if (run.returncode or not valid(temporary)) and start > 0:
            run = generate(0.0)
        if run.returncode or not valid(temporary):
            raise RuntimeError(run.stderr.decode('utf-8', 'replace').strip() or '无法生成有效试听片段。')
        os.replace(temporary, destination)
        marker.write_text(signature, encoding='utf-8')
    finally:
        temporary.unlink(missing_ok=True)
    prune_preview_dir(destination.parent)
    return destination


def scrollable(parent, style='Card.TFrame'):
    """返回 (外框, 内容框)：内容比可见区高时出现滚动条。"""
    holder = ttk.Frame(parent, style=style)
    canvas = tk.Canvas(holder, bg=PANEL, highlightthickness=0, bd=0)
    bar = ttk.Scrollbar(holder, orient='vertical', command=canvas.yview)
    body = ttk.Frame(canvas, style=style)
    window = canvas.create_window((0, 0), window=body, anchor='nw')
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side='left', fill='both', expand=True)
    bar.pack(side='right', fill='y')
    body.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))

    def wheel(event):
        if canvas.yview() != (0.0, 1.0):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')

    for widget in (canvas, body):
        widget.bind('<MouseWheel>', wheel)
    body.wheel = wheel
    return holder, body


def bind_wheel(widget, handler):
    """把滚轮事件挂到整棵子树，行内的标签和按钮也能滚动。"""
    if handler is None:
        return
    widget.bind('<MouseWheel>', handler)
    for child in widget.winfo_children():
        bind_wheel(child, handler)


class Player:
    """试听对照：Windows 用 winsound 顺序播放，其他系统退回默认播放器。"""

    def __init__(self):
        self.token = 0

    def play(self, files):
        self.stop()
        self.token += 1
        token = self.token

        def run():
            for file in files:
                if token != self.token:
                    return
                try:
                    if os.name == 'nt':
                        import winsound
                        winsound.PlaySound(str(file), winsound.SND_FILENAME)
                    else:
                        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(file)], check=False)
                except Exception:
                    return

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.token += 1
        if os.name == 'nt':
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass


class App:
    def __init__(self, root):
        self.root = root
        self.home = data_home()
        self.settings_file = self.home / 'settings.json'
        self.first_run = not self.settings_file.exists()
        self.portable = bundle_root() is not None
        self.import_token = None
        self.import_queue = queue.Queue()
        try:
            self.settings = Settings.load(self.settings_file)
        except Exception:
            self.settings = Settings()
        self.manager = Manager(self.home)
        self.player = Player()
        self.rows = {}
        self.source = tk.StringVar()
        self.name = tk.StringVar()
        self.mode = tk.StringVar(value='original')
        self.start = tk.StringVar(value='0')
        self.end = tk.StringVar()
        self.message = tk.StringVar(value='准备就绪，添加一份素材开始吧。')
        self.history_count = tk.StringVar(value='暂无任务')
        self.options_open = False
        self.mode_cards = {}
        self.active_job = None
        self.closed = False
        # 安装在 App 层共享状态：关闭设置页再打开不会丢失「正在安装」标记（I07）。
        self.install = {'running': False, 'fraction': 0., 'note': '', 'error': None, 'finished': False}
        self.install_queue = queue.Queue()
        self.install_views = []
        # 试听片段的生成移到后台线程，主线程只经 preview_queue 收敛结果（P02）。
        self.preview_queue = queue.Queue()
        self._preview_busy = False
        self._preview_token = 0
        self.build()
        self.refresh()
        self.update_health()
        self.welcome_token = self.root.after(600, self.welcome)
        self.poll_token = self.root.after(250, self.poll)

    def build(self):
        root = self.root
        root.title('角色音频工坊 · Voice Workbench')
        root.configure(bg=BG)
        width, height = min(1080, root.winfo_screenwidth() - 80), min(820, root.winfo_screenheight() - 80)
        root.geometry(f'{width}x{height}')
        root.minsize(min(900, width), min(740, height))
        root.option_add('*Font', (FONT, 10))
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('Card.TFrame', background=PANEL)
        style.configure('Card.TLabel', background=PANEL, foreground=INK)
        style.configure('Hint.TLabel', background=PANEL, foreground=MUTED, font=(FONT, 9))
        style.configure('Title.TLabel', background=PANEL, foreground=INK, font=(FONT, 11, 'bold'))
        style.configure('Badge.TLabel', background=SOFT, foreground=ACCENT, padding=(11, 6), font=(FONT, 9))
        style.configure('TButton', padding=(15, 9), background=PANEL, foreground=INK,
                        bordercolor=LINE, lightcolor=PANEL, darkcolor=PANEL, borderwidth=1, focusthickness=1, focuscolor=ACCENT)
        style.map('TButton', background=[('disabled', '#f4f5f2'), ('active', '#eef2ed')],
                  foreground=[('disabled', '#a5ada7')], bordercolor=[('focus', ACCENT)])
        style.configure('Accent.TButton', background=ACCENT, foreground='white', bordercolor=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT, padding=(25, 10), font=(FONT, 10, 'bold'))
        style.map('Accent.TButton', background=[('pressed', '#255b4e'), ('active', '#2d6759')],
                  foreground=[('!disabled', 'white')], bordercolor=[('focus', '#1c4f43')])
        style.configure('Quiet.TButton', background=PANEL, foreground=MUTED, borderwidth=0,
                        padding=(8, 6), focuscolor=ACCENT)
        style.map('Quiet.TButton', background=[('active', '#f0f4ef')],
                  foreground=[('disabled', '#b1b8b2'), ('active', ACCENT)])
        style.configure('Nav.TButton', background=BG, foreground=MUTED, borderwidth=0, padding=(14, 9))
        style.map('Nav.TButton', background=[('selected', SOFT), ('active', '#edf1eb')], foreground=[('selected', ACCENT)])
        style.configure('TCheckbutton', background=PANEL, foreground=INK, padding=(0, 5))
        style.map('TCheckbutton', background=[('active', PANEL)])
        style.configure('TCombobox', padding=(8, 8), foreground=INK, fieldbackground='#fafbf8', bordercolor=LINE)
        style.map('TCombobox', fieldbackground=[('readonly', '#fafbf8')], foreground=[('readonly', INK)])
        style.configure('TEntry', padding=(10, 9), foreground=INK, fieldbackground='#fafbf8',
                        bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, insertcolor=INK)
        style.map('TEntry', bordercolor=[('focus', ACCENT)], lightcolor=[('focus', ACCENT)], darkcolor=[('focus', ACCENT)])
        style.configure('Treeview', rowheight=36, borderwidth=0, relief='flat', background=PANEL,
                        fieldbackground=PANEL, foreground=INK, font=(FONT, 9))
        style.layout('Treeview', [('Treeview.treearea', {'sticky': 'nswe'})])
        style.map('Treeview', background=[('selected', '#e4efe8')], foreground=[('selected', '#245c4c')])
        style.configure('Treeview.Heading', font=(FONT, 9), foreground=MUTED, background='#f7f9f5',
                        padding=(12, 9), borderwidth=0, relief='flat')
        style.map('Treeview.Heading', background=[('active', '#f0f4ef')])
        style.configure('Vertical.TScrollbar', background='#d7e0d8', troughcolor=PANEL,
                        borderwidth=0, arrowsize=12, lightcolor=PANEL, darkcolor=PANEL)
        style.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor='#eaf0e9',
                        borderwidth=0, bordercolor='#eaf0e9', lightcolor=ACCENT, darkcolor=ACCENT, arrowsize=4)
        style.configure('TNotebook', background=PANEL, borderwidth=0, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
        style.configure('TNotebook.Tab', background=PANEL, foreground=MUTED, padding=(22, 10),
                        borderwidth=0, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
        style.map('TNotebook.Tab', background=[('selected', SOFT), ('active', '#f5f7f2')], foreground=[('selected', ACCENT)])

        shell = ttk.Frame(root, padding=(30, 23, 30, 14))
        shell.pack(fill='both', expand=True)
        header = ttk.Frame(shell)
        header.pack(fill='x', pady=(0, 23))
        mark = tk.Canvas(header, width=44, height=44, bg=BG, highlightthickness=0)
        mark.pack(side='left', padx=(0, 13))
        mark.create_oval(0, 0, 43, 43, fill='#e5ede5', outline='')
        for x, height in [(12, 9), (17, 19), (22, 27), (27, 16), (32, 8)]:
            mark.create_line(x, 22 - height / 2, x, 22 + height / 2, fill=ACCENT, width=3, capstyle='round')
        brand = ttk.Frame(header)
        brand.pack(side='left')
        ttk.Label(brand, text='角色音频工坊', font=(FONT, 20, 'bold')).pack(anchor='w')
        ttk.Label(brand, text='提取原音，或换一种声音。', style='Muted.TLabel', font=(FONT, 9)).pack(anchor='w', pady=(3, 0))
        navigation = ttk.Frame(header)
        navigation.pack(side='left', padx=(28, 0))
        self.nav_buttons = {}
        for value, label in [('audio', '音频制作'), ('training', '训练模型')]:
            button = ttk.Button(navigation, text=label, style='Nav.TButton', command=lambda page=value: self.show_workspace(page))
            button.pack(side='left', padx=3)
            self.nav_buttons[value] = button
        self.nav_buttons['audio'].state(['selected'])
        self.settings_button = ttk.Button(header, text='设置与音色' if self.portable else '环境与模型', command=self.settings_dialog)
        self.settings_button.pack(side='right')

        card = tk.Frame(shell, bg=PANEL, highlightthickness=1, highlightbackground=LINE, highlightcolor=LINE)
        card.pack(fill='x')
        form = ttk.Frame(card, style='Card.TFrame', padding=(22, 18, 22, 18))
        self.audio_form = form
        form.pack(fill='x')
        form.columnconfigure(0, weight=1)
        ttk.Label(form, text='添加素材', style='Title.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 9))
        source_row = ttk.Frame(form, style='Card.TFrame')
        source_row.grid(row=1, column=0, sticky='ew')
        source_row.columnconfigure(0, weight=1)
        self.source_entry = ttk.Entry(source_row, textvariable=self.source)
        self.source_entry.grid(row=0, column=0, sticky='ew', padx=(0, 10))
        ttk.Button(source_row, text='导入文件', command=self.choose_source).grid(row=0, column=1)
        ttk.Label(form, text='粘贴视频链接、B 站分享文案或 BV 号，也可以导入本地音视频。', style='Hint.TLabel').grid(row=2, column=0, sticky='w', pady=(7, 17))

        modes = ttk.Frame(form, style='Card.TFrame')
        modes.grid(row=3, column=0, sticky='ew')
        modes.columnconfigure((0, 1), weight=1, uniform='modes')
        for column, (value, title, subtitle) in enumerate([
            ('original', '提取原音', '保留原声音色，导出 WAV'),
            ('convert', '转换音色', '分离人声，再与伴奏合成')
        ]):
            box = tk.Frame(modes, bg=PANEL, highlightthickness=1, highlightbackground=LINE, highlightcolor=LINE, cursor='hand2')
            box.grid(row=0, column=column, sticky='nsew', padx=(0, 6) if column == 0 else (6, 0))
            radio_style = value + '.TRadiobutton'
            style.configure(radio_style, background=PANEL, foreground=INK, font=(FONT, 10, 'bold'),
                            padding=0, focuscolor=ACCENT, indicatormargin=(0, 0, 9, 0), indicatorcolor=PANEL)
            style.map(radio_style, background=[('active', SOFT)], indicatorbackground=[('selected', ACCENT)],
                      indicatorforeground=[('selected', 'white')])
            radio = ttk.Radiobutton(box, text=title, variable=self.mode, value=value, style=radio_style, cursor='hand2')
            radio.pack(anchor='w', padx=15, pady=(12, 3))
            subtitle_label = tk.Label(box, text=subtitle, bg=PANEL, fg=MUTED, font=(FONT, 9), cursor='hand2')
            subtitle_label.pack(anchor='w', padx=39, pady=(0, 12))
            for widget in (box, subtitle_label):
                widget.bind('<Button-1>', lambda event, v=value, r=radio: (self.mode.set(v), r.focus_set()))
            self.mode_cards[value] = (box, subtitle_label, radio_style)
        model_row = ttk.Frame(form, style='Card.TFrame')
        model_row.grid(row=4, column=0, sticky='ew', pady=(10, 3))
        self.import_model_button = ttk.Button(model_row, text='导入音色', command=self.import_model)
        self.import_model_button.pack(side='right', padx=(10, 0))
        self.model_label = ttk.Label(model_row, style='Hint.TLabel')
        self.model_label.pack(side='left', fill='x', expand=True)
        self.health_label = tk.Label(form, text='', bg=PANEL, fg='#a15d4e', font=(FONT, 9), cursor='hand2',
                                     anchor='w', justify='left')
        self.health_label.grid(row=8, column=0, sticky='ew', pady=(9, 0))
        self.health_label.bind('<Button-1>', lambda event: self.settings_dialog('environment'))

        def wrap_form(event):
            self.model_label.configure(wraplength=max(400, event.width - 44))
            self.health_label.configure(wraplength=max(400, event.width - 44))

        form.bind('<Configure>', wrap_form)

        self.options_button = ttk.Button(form, text='命名与裁剪  ＋', style='Quiet.TButton', command=self.toggle_options)
        self.options_button.grid(row=5, column=0, sticky='w', pady=(3, 0))
        self.options_frame = ttk.Frame(form, style='Card.TFrame')
        self.options_frame.grid(row=6, column=0, sticky='ew', pady=(7, 3))
        self.options_frame.columnconfigure(0, weight=1)
        for column, (label, variable, width) in enumerate([
            ('成品名称 · 留空自动命名', self.name, 25), ('起点 / 秒', self.start, 10), ('终点 / 秒 · 留空至结尾', self.end, 18)
        ]):
            field = ttk.Frame(self.options_frame, style='Card.TFrame')
            field.grid(row=0, column=column, sticky='ew', padx=(0, 14) if column < 2 else 0)
            ttk.Label(field, text=label, style='Hint.TLabel').pack(anchor='w', pady=(0, 5))
            ttk.Entry(field, textvariable=variable, width=width).pack(fill='x')
        self.options_frame.grid_remove()
        actions = ttk.Frame(form, style='Card.TFrame')
        actions.grid(row=7, column=0, sticky='ew', pady=(15, 0))
        self.start_button = ttk.Button(actions, text='开始制作', style='Accent.TButton', command=self.submit)
        self.start_button.pack(side='left')
        status = ttk.Frame(actions, style='Card.TFrame', padding=(20, 0, 0, 0))
        status.pack(side='left', fill='x', expand=True)
        status_label = ttk.Label(status, textvariable=self.message, style='Hint.TLabel')
        status_label.pack(anchor='w', fill='x', pady=(0, 7))
        status.bind('<Configure>', lambda event: status_label.configure(wraplength=max(200, event.width - 25)))
        self.progress = ttk.Progressbar(status, maximum=100)
        self.progress.pack(fill='x')

        from .training_ui import TrainingPanel
        self.training_panel = TrainingPanel(self, card)

        history = tk.Frame(shell, bg=PANEL, highlightthickness=1, highlightbackground=LINE, highlightcolor=LINE)
        history.pack(fill='both', expand=True, pady=(18, 0))
        history_header = ttk.Frame(history, style='Card.TFrame', padding=(18, 12))
        history_header.pack(fill='x')
        ttk.Label(history_header, text='任务记录', style='Title.TLabel').pack(side='left')
        ttk.Label(history_header, textvariable=self.history_count, style='Hint.TLabel').pack(side='left', padx=12)
        self.open_button = ttk.Button(history_header, text='打开成品', style='Quiet.TButton', command=self.open_selected)
        self.open_button.pack(side='right')
        ttk.Button(history_header, text='输出文件夹', style='Quiet.TButton', command=self.open_output).pack(side='right', padx=(3, 8))
        self.cancel_button = ttk.Button(history_header, text='取消任务', style='Quiet.TButton', command=self.cancel)
        self.cancel_button.pack(side='right', padx=(0, 10))
        self.resume_button = ttk.Button(history_header, text='继续制作', style='Quiet.TButton', command=self.resume_selected)
        self.resume_button.pack(side='right', padx=(0, 10))
        self.use_model_button = ttk.Button(history_header, text='使用此模型', style='Quiet.TButton', command=self.use_trained_model)
        table = ttk.Frame(history, style='Card.TFrame', padding=(8, 0, 8, 8))
        table.pack(fill='both', expand=True)
        columns = ('name', 'mode', 'state', 'stage', 'created')
        self.tree = ttk.Treeview(table, columns=columns, show='headings', selectmode='browse', height=4)
        for key, text, width in zip(columns, ['名称', '模式', '状态', '处理步骤', '时间'], [220, 85, 85, 240, 140]):
            self.tree.heading(key, text=text, anchor='w')
            self.tree.column(key, width=width, minwidth=70, anchor='w', stretch=key in ('name', 'stage'))
        self.tree.tag_configure('completed', foreground=ACCENT)
        self.tree.tag_configure('failed', foreground='#a15d4e')
        for state in ('cancelled', 'interrupted'): self.tree.tag_configure(state, foreground=MUTED)
        scroll = ttk.Scrollbar(table, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.bind('<Double-1>', lambda event: self.details())
        self.tree.bind('<<TreeviewSelect>>', lambda event: self.update_actions())
        self.tree.bind('<Return>', lambda event: self.details())
        self.empty_state = ttk.Frame(self.tree, style='Card.TFrame')
        ttk.Label(self.empty_state, text='还没有制作记录', style='Card.TLabel', font=(FONT, 11)).pack()
        ttk.Label(self.empty_state, text='添加素材，第一份作品就从这里开始。', style='Hint.TLabel').pack(pady=(6, 0))
        footer = ttk.Frame(shell)
        footer.pack(fill='x', pady=(11, 0))
        ttk.Label(footer, text='双击记录查看详情', style='Muted.TLabel', font=(FONT, 9)).pack(side='left')
        ttk.Label(footer, text='Voice Workbench  /  ' + __version__, style='Muted.TLabel', font=(FONT, 9)).pack(side='right')
        self.mode.trace_add('write', lambda *args: self.update_mode())
        for variable in (self.name, self.start, self.end):
            variable.trace_add('write', lambda *args: self.update_options_label())
        self.update_mode()
        root.protocol('WM_DELETE_WINDOW', self.close)

    def show_workspace(self, page):
        self.audio_form.pack_forget()
        self.training_panel.frame.pack_forget()
        (self.training_panel.frame if page == 'training' else self.audio_form).pack(fill='x')
        for value, button in self.nav_buttons.items():
            button.state(['selected'] if value == page else ['!selected'])

    def toggle_options(self):
        self.options_open = not self.options_open
        if self.options_open: self.options_frame.grid()
        else: self.options_frame.grid_remove()
        self.update_options_label()

    def update_options_label(self):
        customized = bool(self.name.get().strip() or self.end.get().strip() or self.start.get().strip() not in ('', '0'))
        self.options_button.configure(text='命名与裁剪' + (' · 已设置' if customized else '') + ('  －' if self.options_open else '  ＋'))

    def update_mode(self):
        style = ttk.Style(self.root)
        for value, (box, subtitle, radio_style) in self.mode_cards.items():
            selected = self.mode.get() == value
            color = SOFT if selected else PANEL
            box.configure(bg=color, highlightbackground=ACCENT if selected else LINE, highlightcolor=ACCENT if selected else LINE)
            subtitle.configure(bg=color, fg=ACCENT if selected else MUTED)
            style.configure(radio_style, background=color)
            style.map(radio_style, background=[('active', color)])
        self.model_label.configure(text=self.model_description() if self.mode.get() == 'convert' else '原声音色  ·  WAV  /  44.1 kHz  /  双声道')

    def update_actions(self):
        row = self.selected()
        self.cancel_button.state(['!disabled'] if row and row['state'] in ('queued', 'running', 'cancelling') else ['disabled'])
        self.open_button.state(['!disabled'] if row and row['state'] == 'completed' else ['disabled'])
        training = row and row['mode'] == 'train'
        self.open_button.configure(text='打开模型目录' if training else '打开成品')
        resumable = bool(row) and row['mode'] != 'train' and row['state'] in ('failed', 'interrupted', 'cancelled', 'completed')
        self.resume_button.state(['!disabled'] if resumable else ['disabled'])
        self.resume_button.configure(text='用相同参数重跑' if row and row['state'] == 'completed' else '继续制作')
        if training and row['state'] == 'completed': self.use_model_button.pack(side='right', padx=5)
        else: self.use_model_button.pack_forget()

    def update_health(self):
        try:
            items = environment_report(self.settings, self.home)
        except Exception:
            items = []
        missing = [item for item in items if item['state'] == 'missing'
                   and not (self.portable and item['key'] == 'model')]
        if not missing:
            self.health_label.grid_remove()
            return
        names = '、'.join(item['label'] for item in missing[:3])
        self.health_label.configure(text=f'环境还有 {len(missing)} 项未就绪：{names} · 点击查看与安装入口')
        self.health_label.grid()

    def welcome(self):
        if self.closed or not getattr(self, 'first_run', False):
            return
        if self.portable:
            self.message.set('运行环境已内置。添加素材即可导出原音；导入音色即可换声，也可以先训练自己的音色。')
            return
        self.message.set('第一次使用：请先在「环境检测」里补齐 FFmpeg 与 RVC 环境，之后就能贴链接直接制作。')
        self.settings_dialog('environment')

    def import_model(self):
        selected = filedialog.askopenfilename(parent=self.root, title='导入 RVC 音色',
            filetypes=[('RVC 音色模型', '*.pth')])
        if not selected:
            return
        self.import_model_button.state(['disabled'])
        self.message.set('正在导入音色，同文件夹内的匹配索引会一起导入…')
        def work():
            try:
                self.import_queue.put((import_voice_model(selected, self.home), None))
            except Exception as error:
                self.import_queue.put((None, str(error)))
        threading.Thread(target=work, daemon=True).start()
        def poll():
            self.import_token = None
            if self.closed:
                return
            try:
                result, error = self.import_queue.get_nowait()
            except queue.Empty:
                self.import_token = self.root.after(100, poll)
                return
            self.import_model_button.state(['!disabled'])
            if error:
                self.message.set('音色导入失败。')
                messagebox.showerror('无法导入音色', error, parent=self.root)
                return
            self.settings.model, self.settings.index = result['model'], result['index']
            try:
                self.settings.save(self.settings_file)
            except OSError as error:
                messagebox.showerror('无法保存音色选择', str(error), parent=self.root)
                return
            self.mode.set('convert')
            self.update_health()
            self.message.set('音色已导入，可以添加素材开始制作。' + (' 已同时导入检索索引。' if result['index'] else ''))
        self.import_token = self.root.after(100, poll)

    def resume_selected(self):
        row = self.selected()
        if row:
            self.resume_job(row['id'])

    def resume_job(self, job):
        try:
            self.manager.retry(job)
        except Exception as error:
            messagebox.showerror('无法继续', str(error), parent=self.root)
            return False
        self.active_job = job
        self.refresh()
        if self.tree.exists(job):
            self.tree.selection_set(job)
            self.tree.see(job)
        self.update_actions()
        self.message.set('已从上次停下的阶段继续，做过的步骤会直接复用。')
        return True

    def reuse_parameters(self, row):
        try:
            state = json.loads((Path(row['folder']) / STAGE_FILE).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            state = {}
        request = state.get('request') or {}
        if request.get('source'):
            self.source.set(str(request['source']))
        self.mode.set(request.get('mode') if request.get('mode') in ('original', 'convert') else 'convert')
        self.name.set(str(request.get('name') or ''))
        self.start.set(str(request.get('start', 0)))
        self.end.set('' if request.get('end') in (None, '') else str(request['end']))
        try:
            self.settings = task_settings(Path(row['folder']), self.settings)
        except ValueError as error:
            messagebox.showerror('无法沿用参数', str(error), parent=self.root)
            return
        self.update_options_label()
        self.update_mode()
        self.show_workspace('audio')
        self.message.set('已载入这次任务的来源与参数，可以再制作一版。')

    def preview(self, sources):
        if getattr(self, '_preview_busy', False):
            self.message.set('正在准备上一次试听，请稍候。')
            return
        self._preview_busy = True
        self._preview_token += 1
        token = self._preview_token
        self.player.stop()
        self.message.set('正在准备试听片段…')
        sources = [Path(source) for source in sources]

        def work():
            clips = []
            error = None
            try:
                for source in sources:
                    if token != self._preview_token:
                        return  # 已被新试听或停止动作覆盖
                    key = fingerprint('preview-v3', file_stamp(source), PREVIEW_SECONDS)
                    destination = self.home / 'cache' / 'preview' / f'{key}.wav'
                    clips.append(preview_clip(self.settings, source, destination))
            except Exception as caught:
                error = str(caught)
            self.preview_queue.put((token, clips, error, [str(source) for source in sources]))

        threading.Thread(target=work, daemon=True).start()

    def drain_preview(self):
        queue_ = getattr(self, 'preview_queue', None)
        if queue_ is None:
            return
        while True:
            try:
                token, clips, error, source_names = queue_.get_nowait()
            except queue.Empty:
                break
            if token != getattr(self, '_preview_token', 0):
                continue  # 过期结果直接丢弃
            self._preview_busy = False
            if error:
                messagebox.showerror('无法试听', error, parent=self.root)
            else:
                self.player.play(clips)
                self.message.set('正在试听：' + ' → '.join(Path(name).name for name in source_names) + '（每个片段约 12 秒）')

    def stop_playback(self):
        self._preview_token = getattr(self, '_preview_token', 0) + 1
        self._preview_busy = False
        self.player.stop()

    def use_trained_model(self):
        row = self.selected()
        if not row or row['mode'] != 'train' or row['state'] != 'completed': return
        try:
            from .training import trained_settings
            settings = trained_settings(self.settings, Path(row['result']))
            settings.save(self.settings_file)
            self.settings = settings
            self.mode.set('convert')
            self.show_workspace('audio')
            self.update_health()
            self.message.set('已选择新模型，添加音频即可试听转换效果。')
        except Exception as error:
            messagebox.showerror('无法使用模型', str(error), parent=self.root)

    def model_description(self):
        if not self.settings.model:
            return '点击「导入音色」添加 .pth 模型，或在「训练音色」中制作自己的模型。'
        model = Path(self.settings.model).name
        if len(model) > 64: model = model[:30] + '…' + model[-28:]
        return f'模型：{model}  ·  升调 {self.settings.pitch:+d} 半音'

    def place_dialog(self, dialog, width, height):
        self.root.update_idletasks()
        width = min(width, self.root.winfo_screenwidth() - 50)
        height = min(height, self.root.winfo_screenheight() - 70)
        x = max(0, min(self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2, self.root.winfo_screenwidth() - width))
        y = max(0, min(self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2, self.root.winfo_screenheight() - height - 40))
        dialog.geometry(f'{width}x{height}+{x}+{y}')

    def choose_source(self):
        file = filedialog.askopenfilename(title='选择本地音视频', filetypes=[('音视频', '*.mp4 *.mkv *.webm *.mov *.wav *.mp3 *.m4a *.flac *.ogg'), ('所有文件', '*.*')])
        if file:
            self.source.set(file)

    def submit(self):
        try:
            request = Request(source_from_text(self.source.get()), self.mode.get(), self.name.get().strip(), float(self.start.get() or 0), float(self.end.get()) if self.end.get().strip() else None)
            self.enqueue(request)
        except Exception as error:
            messagebox.showerror('还不能开始', str(error), parent=self.root)

    def enqueue(self, request):
        settings = Settings(**asdict(self.settings))
        job = self.manager.submit(request, settings)
        self.active_job = job
        self.progress['value'] = self.training_panel.progress['value'] = 0
        self.message.set('训练已加入队列，将自动处理素材并生成模型。' if request.mode == 'train' else '已加入制作队列，成品将保存在输出文件夹。')
        self.refresh()
        self.tree.selection_set(job)
        self.tree.see(job)
        self.update_actions()
        return job

    def refresh(self):
        selected = self.tree.selection()
        self.rows = {row['id']: row for row in self.manager.store.list()}
        for old in self.tree.get_children():
            if old not in self.rows:
                self.tree.delete(old)
        for position, (job, row) in enumerate(self.rows.items()):
            values = (row['name'], {'original': '原音', 'convert': '音色转换', 'train': '训练模型'}.get(row['mode'], row['mode']), STATES.get(row['state'], row['state']), row['stage'], row['created'][5:16])
            if self.tree.exists(job): self.tree.item(job, values=values, tags=(row['state'],))
            else: self.tree.insert('', 0, iid=job, values=values, tags=(row['state'],))
            self.tree.move(job, '', position)
        if selected and self.tree.exists(selected[0]): self.tree.selection_set(selected[0])
        self.history_count.set(f'{len(self.rows)} 个任务' if self.rows else '暂无任务')
        if self.rows: self.empty_state.place_forget()
        else: self.empty_state.place(relx=.5, rely=.59, anchor='center')
        self.update_actions()

    def poll(self):
        if self.closed:
            return
        self.drain_install()
        self.drain_preview()
        changed = False
        while True:
            try:
                job, stage, percent = self.manager.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if stage == 'storage_error':
                detail = percent if isinstance(percent, str) else ''
                text = '任务状态写入失败，将在数据库恢复后补写。' + (f'（{detail}）' if detail else '')
                self.message.set(text[:100])
            elif stage != 'refresh':
                row = self.rows.get(job) or self.manager.store.get(job) or {}
                name = row.get('name') or ''
                text = f'{name} · {stage}' if name else stage
                self.message.set(text if len(text) <= 100 else text[:97] + '…')
                if percent is not None: self.progress['value'] = self.training_panel.progress['value'] = percent
        if changed:
            self.refresh()
            row = self.rows.get(self.active_job)
            if row and row['state'] in ('failed', 'cancelled', 'interrupted'):
                summary = row['stage'] + ('：' + row['error'].splitlines()[0] if row['error'] else '')
                self.message.set(summary[:100] + ('… 双击查看详情' if len(summary) > 100 else ''))
        self.poll_token = self.root.after(250, self.poll)

    def selected(self):
        values = self.tree.selection()
        return self.rows.get(values[0]) if values else None

    def cancel(self):
        row = self.selected()
        if row and row['state'] in ('queued', 'running', 'cancelling'):
            self.manager.cancel(row['id']); self.refresh()

    def open_selected(self):
        row = self.selected()
        if not row or row['state'] != 'completed':
            messagebox.showinfo('打开成品', '请先选择一条已经完成的任务。', parent=self.root); return
        try: open_file(Path(row['result']).parent if row['mode'] == 'train' else Path(row['result']))
        except OSError as error: messagebox.showerror('打开失败', str(error), parent=self.root)

    def open_output(self):
        folder = Path(self.settings.output).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        open_file(folder)

    def details(self):
        row = self.selected()
        if not row:
            return
        dialog = tk.Toplevel(self.root, bg=PANEL)
        dialog.title('任务详情'); self.place_dialog(dialog, 740, 460); dialog.minsize(580, 360); dialog.transient(self.root)
        header = ttk.Frame(dialog, style='Card.TFrame', padding=(24, 22, 24, 14))
        header.pack(fill='x')
        ttk.Label(header, text=row['name'], style='Title.TLabel', wraplength=650).pack(anchor='w')
        ttk.Label(header, text=STATES.get(row['state'], row['state']) + '  ·  ' + row['created'], style='Hint.TLabel').pack(anchor='w', pady=(7, 0))
        content = ttk.Frame(dialog, style='Card.TFrame', padding=(24, 0, 24, 0))
        content.pack(fill='both', expand=True)
        box = tk.Text(content, wrap='word', padx=14, pady=12, font=(FONT, 10), bg='#f7f9f5', fg=INK,
                      relief='flat', highlightthickness=1, highlightbackground=LINE, spacing3=8)
        scroll = ttk.Scrollbar(content, command=box.yview)
        box.configure(yscrollcommand=scroll.set)
        box.pack(side='left', fill='both', expand=True); scroll.pack(side='right', fill='y')
        extra = ''
        if row['mode'] == 'train':
            report_file = Path(row['folder']) / '素材筛选报告.json'
            if report_file.is_file():
                try:
                    report = json.loads(report_file.read_text(encoding='utf-8'))
                    extra = f"有效人声：{report['duration_seconds']:.1f} 秒\n保留 {len(report['accepted'])} 段，排除 {len(report['rejected'])} 段。\n"
                    reasons = sorted({item['reason'] for item in report['rejected']})
                    if reasons: extra += '排除原因：' + '、'.join(reasons) + '\n'
                    extra += '\n'
                except (OSError, ValueError, KeyError): pass
        box.insert('1.0', f"处理步骤\n{row['stage']}\n\n" + extra + f"输出目录\n{row['folder']}\n\n成品文件\n{row['result'] or '尚未生成'}\n\n" + ('错误详情\n' + row['error'] if row['error'] else '没有错误。'))
        box.configure(state='disabled')
        buttons = ttk.Frame(dialog, style='Card.TFrame', padding=(24, 14))
        buttons.pack(side='bottom', fill='x', before=content)
        actions = ttk.Frame(buttons, style='Card.TFrame')
        actions.pack(side='bottom', fill='x')
        ttk.Button(actions, text='关闭', command=dialog.destroy).pack(side='right')
        if row['mode'] == 'train':
            vocals = Path(row['folder']) / 'clean_vocals'
            if vocals.is_dir():
                ttk.Button(actions, text='人声素材', command=lambda: open_file(vocals)).pack(side='left')
                first = next(vocals.glob('*.wav'), None)
                if first: ttk.Button(actions, text='试听片段', command=lambda: open_file(first)).pack(side='left', padx=10)
        else:
            folder = Path(row['folder'])
            if row['state'] in ('failed', 'interrupted', 'cancelled'):
                ttk.Button(actions, text='从失败处继续', style='Accent.TButton',
                           command=lambda: self.resume_job(row['id']) and dialog.destroy()).pack(side='left')
            human, converted = folder / '人声.wav', folder / '转换后人声.wav'
            if row['state'] == 'completed' and human.is_file() and converted.is_file():
                playback = ttk.Frame(buttons, style='Card.TFrame')
                playback.pack(fill='x', pady=(0, 8))
                ttk.Button(playback, text='试听原人声', command=lambda: self.preview([human])).pack(side='left', padx=(0, 8))
                ttk.Button(playback, text='试听转换后', command=lambda: self.preview([converted])).pack(side='left', padx=(0, 8))
                ttk.Button(playback, text='顺序对照', command=lambda: self.preview([human, converted])).pack(side='left', padx=(0, 8))
                ttk.Button(playback, text='停止', style='Quiet.TButton', command=self.stop_playback).pack(side='left')
            elif row['state'] == 'completed' and row['result'] and Path(row['result']).is_file():
                ttk.Button(actions, text='试听成品', command=lambda: self.preview([Path(row['result'])])).pack(side='left')
            ttk.Button(actions, text='沿用这些参数', style='Quiet.TButton',
                       command=lambda: (self.reuse_parameters(row), dialog.destroy())).pack(side='left', padx=(12, 0))
        dialog.bind('<Escape>', lambda event: dialog.destroy())
        return dialog

    def settings_dialog(self, page_name=None):
        current = getattr(self, 'settings_window', None)
        if current is not None and current.winfo_exists():
            current.lift()
            pages = getattr(self, 'settings_pages', {})
            if page_name in pages:
                self.settings_tabs.select(pages[page_name])
            return current
        dialog = tk.Toplevel(self.root, bg=PANEL)
        self.settings_window = dialog
        dialog.title('设置与音色' if self.portable else '环境与模型')
        width, height = 820, 780
        height = min(height, max(560, self.root.winfo_screenheight() - 100))
        self.place_dialog(dialog, width, height); dialog.minsize(min(700, width), min(640, height)); dialog.transient(self.root)
        holder = ttk.Frame(dialog, style='Card.TFrame', padding=(26, 23, 26, 0))
        holder.pack(fill='both', expand=True)
        ttk.Label(holder, text='设置与音色' if self.portable else '环境与模型', style='Title.TLabel', font=(FONT, 17, 'bold')).pack(anchor='w')
        ttk.Label(holder, text='运行环境随软件提供。你只需导入音色，或训练自己的音色。' if self.portable else
                  '配置一次，之后专心制作。所有设置仅保存在本机。', style='Hint.TLabel').pack(anchor='w', pady=(7, 18))
        tabs = ttk.Notebook(holder)
        tabs.pack(fill='both', expand=True)
        self.settings_tabs = tabs
        pages = self.settings_pages = {}
        variables = {key: tk.StringVar(value=str(value)) for key, value in asdict(self.settings).items()}

        def draft():
            data = {key: value.get().strip() for key, value in variables.items()}
            try:
                data.update(pitch=int(data['pitch'] or 0), index_rate=float(data['index_rate'] or 0),
                            accompaniment=float(data['accompaniment'] or 0))
                return Settings(**data)
            except Exception:
                return self.settings

        def browse(key, kind):
            selected = filedialog.askdirectory(parent=dialog) if kind == 'dir' else filedialog.askopenfilename(parent=dialog)
            if selected:
                variables[key].set(selected)
                if key == 'rvc_root':
                    folder = Path(selected)
                    if (folder / 'runtime/python.exe').exists(): variables['python'].set(str(folder / 'runtime/python.exe'))
                    if (folder / 'ffmpeg.exe').exists(): variables['ffmpeg'].set(str(folder))

        def field(parent, key, label, kind='file'):
            area = ttk.Frame(parent, style='Card.TFrame')
            area.pack(fill='x', pady=(0, 13))
            ttk.Label(area, text=label, style='Card.TLabel').pack(anchor='w', pady=(0, 5))
            line = ttk.Frame(area, style='Card.TFrame')
            line.pack(fill='x')
            ttk.Entry(line, textvariable=variables[key]).pack(side='left', fill='x', expand=True)
            if kind:
                ttk.Button(line, text='浏览', command=lambda: browse(key, kind)).pack(side='right', padx=(9, 0))

        environment = ttk.Frame(tabs, style='Card.TFrame', padding=(1, 16, 1, 0))
        tabs.add(environment, text='运行状态' if self.portable else '环境检测')
        pages['environment'] = environment
        head = ttk.Frame(environment, style='Card.TFrame')
        head.pack(fill='x', pady=(0, 12))
        ttk.Label(head, text='内置运行环境自动就绪，下面显示文件完整性与使用状态。' if self.portable else
                  '按顺序补齐：FFmpeg → RVC 环境 → 角色模型。', style='Hint.TLabel').pack(side='left')
        rows_holder, rows = scrollable(environment)
        rows_holder.pack(fill='both', expand=True)
        summary = tk.StringVar()
        install_buttons = []
        report = {'items': environment_report(draft(), self.home)}

        def set_install_state():
            state = 'disabled' if self.install['running'] else 'normal'
            for button in install_buttons:
                try:
                    button.configure(state=state)
                except tk.TclError:
                    pass

        def install_tools():
            self.start_install()

        def render_install(info):
            # I07：安装状态在 App 层，关闭设置页再打开也不会重复启动。
            try:
                if not dialog.winfo_exists():
                    self.unregister_install_view(render_install)
                    return
            except tk.TclError:
                self.unregister_install_view(render_install)
                return
            if info.get('finished'):
                info['finished'] = False
                if info['error']:
                    summary.set(f"安装失败：{info['error']}")
                else:
                    variables['ffmpeg'].set(str(self.home / 'tools' / 'ffmpeg'))
                    summary.set('FFmpeg 已安装到数据目录，路径已填好；点「保存」后即可使用。')
                    refresh_environment()
                    set_install_state()
                    self.update_health()
                    return
            if info['running']:
                summary.set(f"正在安装 FFmpeg：{info['note']}（{info['fraction']:.0%}）")
            set_install_state()

        self.register_install_view(render_install)
        dialog.protocol('WM_DELETE_WINDOW', lambda: (self.unregister_install_view(render_install), dialog.destroy()))

        def install_missing():
            if not any(item['state'] != 'ok' and item.get('auto') for item in report['items']):
                summary.set('可自动安装的依赖都已就绪；RVC 环境与角色模型仍需手动选择。')
                return
            install_tools()

        def open_folder(path):
            try:
                open_file(Path(path))
            except OSError as error:
                messagebox.showerror('打不开文件夹', str(error), parent=dialog)

        def clear_cache():
            if not messagebox.askyesno('清理缓存', '清理后，下次「转换音色」需要重新分离人声。确定清理吗？', parent=dialog):
                return
            cache = SeparationCache(self.home)
            removed = cache.clear()
            leftover = cache.size()
            if leftover:
                summary.set(f'已清理 {pretty_bytes(removed)}；仍有 {pretty_bytes(leftover)} 因被占用未能删除，请关闭占用后重试。')
            else:
                summary.set(f'已清理 {pretty_bytes(removed)}。')
            refresh_environment()

        def env_actions(item):
            key = item['key']
            if key == 'ffmpeg':
                if item['state'] == 'ok':
                    return [('更换', lambda: (browse('ffmpeg', 'dir'), refresh_environment()))]
                return [('一键安装', install_tools), ('下载页', lambda: webbrowser.open(FFMPEG_LINK)),
                        ('选择文件夹', lambda: (browse('ffmpeg', 'dir'), refresh_environment()))]
            if key == 'rvc':
                return [('说明', lambda: webbrowser.open(RVC_LINK)),
                        ('选择文件夹', lambda: (browse('rvc_root', 'dir'), refresh_environment()))]
            if key == 'python':
                return [('选择文件', lambda: (browse('python', 'file'), refresh_environment()))]
            if key == 'model':
                return [('选择模型', lambda: (browse('model', 'file'), refresh_environment()))]
            if key == 'index':
                return [('选择索引', lambda: (browse('index', 'file'), refresh_environment()))]
            if key in ('separation', 'separator_model'):
                return [('选择 .ckpt', lambda: (browse('separator_model', 'file'), refresh_environment()))]
            if key == 'output':
                return [('选择文件夹', lambda: (browse('output', 'dir'), refresh_environment()))]
            if key == 'cache':
                return [('清理', clear_cache), ('打开', lambda: open_folder(self.home / 'cache' / 'separation'))]
            if key == 'disk':
                return [('打开数据目录', lambda: open_folder(self.home))]
            return []

        def refresh_environment():
            report['items'] = environment_report(draft(), self.home)
            install_buttons.clear()
            for child in rows.winfo_children():
                child.destroy()
            for item in report['items']:
                row = ttk.Frame(rows, style='Card.TFrame')
                row.pack(fill='x', pady=(0, 8))
                color = {'ok': ACCENT, 'warn': '#a9793f'}.get(item['state'], '#a15d4e')
                word = {'ok': '就绪', 'warn': '注意'}.get(item['state'], '缺少')
                tk.Label(row, text='●', fg=color, bg=PANEL, font=(FONT, 11)).pack(side='left', padx=(0, 8))
                text = ttk.Frame(row, style='Card.TFrame')
                text.pack(side='left', fill='x', expand=True)
                ttk.Label(text, text=f"{item['label']}  ·  {word}", style='Card.TLabel', font=(FONT, 10, 'bold')).pack(anchor='w')
                ttk.Label(text, text=item['detail'], style='Hint.TLabel', wraplength=390, justify='left').pack(anchor='w')
                actions = ttk.Frame(row, style='Card.TFrame')
                actions.pack(side='right')
                for label, command in env_actions(item):
                    button = ttk.Button(actions, text=label, style='Quiet.TButton', command=command)
                    if label == '一键安装':
                        install_buttons.append(button)
                    button.pack(side='left', padx=(6, 0))
                bind_wheel(row, getattr(rows, 'wheel', None))

        footer = ttk.Frame(environment, style='Card.TFrame')
        footer.pack(fill='x', pady=(2, 0))
        ttk.Button(footer, text='补齐依赖', style='Quiet.TButton', command=install_missing).pack(side='left')
        ttk.Button(footer, text='重新检测', style='Quiet.TButton', command=refresh_environment).pack(side='left', padx=(8, 0))
        ttk.Label(footer, textvariable=summary, style='Hint.TLabel').pack(side='left', padx=(10, 0))
        refresh_environment()

        for title, hint, specs in [
            ('通用', '原音提取只需配置 FFmpeg 和保存位置。', [
                ('output', '成品保存位置', 'dir'), ('ffmpeg', 'FFmpeg 所在文件夹', 'dir'),
                ('cookies', 'Cookie 文件 · 可选', 'file')]),
            ('角色模型', '选择兼容的本机 RVC 环境，以及你自己的角色模型。', [
                ('rvc_root', 'RVC 根文件夹', 'dir'), ('python', 'RVC 环境的 Python', 'file'),
                ('model', '角色模型 · .pth', 'file'), ('index', '检索索引 · 可选', 'file')]),
            ('高级参数', '使用默认参数即可开始，必要时再调整。', [
                ('separator_model', '人声分离模型 · 留空自动识别', 'file'),
                ('separator_config', '人声分离配置 · 留空自动识别', 'file')])
        ]:
            page = ttk.Frame(tabs, style='Card.TFrame', padding=(1, 18, 1, 0))
            tabs.add(page, text=title)
            pages[title] = page
            ttk.Label(page, text=hint, style='Hint.TLabel').pack(anchor='w', pady=(0, 17))
            for key, label, kind in specs: field(page, key, label, kind)
            if title == '通用':
                ttk.Label(page, text='仅使用你主动选择的 Cookie 文件，留空可尝试公开链接。', style='Hint.TLabel', wraplength=650).pack(anchor='w')
                cache_line = ttk.Frame(page, style='Card.TFrame')
                cache_line.pack(fill='x', pady=(18, 0))
                cache_text = tk.StringVar()

                def read_cache(cache_text=cache_text):
                    cache = SeparationCache(self.home)
                    try:
                        size, count = cache.size(), cache.count()
                    except OSError:
                        size, count = 0, 0
                    cache_text.set(f'分离缓存：{count} 份，占用 {pretty_bytes(size)}（上限 {pretty_bytes(6 * 2 ** 30)}）')

                def clear_here(cache_text=cache_text):
                    if not messagebox.askyesno('清理缓存', '清理后，下次「转换音色」需要重新分离人声。确定清理吗？', parent=dialog):
                        return
                    cache = SeparationCache(self.home)
                    removed = cache.clear()
                    leftover = cache.size()
                    read_cache()
                    if leftover:
                        self.message.set(f'已清理 {pretty_bytes(removed)}，仍有 {pretty_bytes(leftover)} 未能删除（可能被其他程序占用）。')
                    else:
                        self.message.set(f'已清理分离缓存 {pretty_bytes(removed)}。')

                read_cache()
                ttk.Label(cache_line, textvariable=cache_text, style='Hint.TLabel', wraplength=440, justify='left').pack(side='left', fill='x', expand=True)
                ttk.Button(cache_line, text='清理', style='Quiet.TButton', command=clear_here).pack(side='right')
                ttk.Button(cache_line, text='打开', style='Quiet.TButton',
                           command=lambda: open_folder(self.home / 'cache' / 'separation')).pack(side='right', padx=(0, 6))
            if title == '高级参数':
                numbers = ttk.Frame(page, style='Card.TFrame')
                numbers.pack(fill='x', pady=(6, 0))
                for column, (key, label, hint) in enumerate([
                    ('pitch', '升调 / 半音', '−24 ～ 24'), ('index_rate', '检索比例', '0 ～ 1'), ('accompaniment', '伴奏比例', '0 ～ 1')
                ]):
                    numbers.columnconfigure(column, weight=1, uniform='parameters')
                    group = ttk.Frame(numbers, style='Card.TFrame')
                    group.grid(row=0, column=column, sticky='ew', padx=(0, 18) if column < 2 else 0)
                    field(group, key, label, '')
                    ttk.Label(group, text=hint, style='Hint.TLabel').pack(anchor='w')
        buttons = ttk.Frame(dialog, style='Card.TFrame', padding=(26, 16, 26, 20))
        buttons.pack(side='bottom', fill='x', before=holder)

        def import_profile():
            file = filedialog.askopenfilename(parent=dialog, title='选择模型或软件配置 JSON', filetypes=[('JSON', '*.json')])
            if not file: return
            try:
                data = json.loads(Path(file).read_text(encoding='utf-8-sig'))
                for key in variables:
                    if key in data: variables[key].set(str(data[key]))
            except Exception as error: messagebox.showerror('配置导入失败', str(error), parent=dialog)
        def save():
            try:
                data = {key: value.get().strip() for key, value in variables.items()}
                data.update(pitch=int(data['pitch']), index_rate=float(data['index_rate']), accompaniment=float(data['accompaniment']))
                settings = Settings(**data)
                if not -24 <= settings.pitch <= 24 or not 0 <= settings.index_rate <= 1 or not 0 <= settings.accompaniment <= 1: raise ValueError('数字参数超出范围。')
                settings.save(self.settings_file); self.settings = settings
                self.update_mode(); self.update_health(); refresh_environment(); dialog.destroy()
                self.message.set('设置已保存，可以开始制作了。')
            except Exception as error: messagebox.showerror('设置未保存', str(error), parent=dialog)
        ttk.Button(buttons, text='导入配置', style='Quiet.TButton', command=import_profile).pack(side='left')
        ttk.Button(buttons, text='保存设置', style='Accent.TButton', command=save).pack(side='right')
        ttk.Button(buttons, text='取消', command=dialog.destroy).pack(side='right', padx=(0, 10))
        if page_name in pages:
            tabs.select(pages[page_name])
        dialog.bind('<Escape>', lambda event: dialog.destroy())
        return dialog

    def start_install(self):
        """在 App 层发起 FFmpeg 安装。worker 线程只往队列投递事件，绝不触碰 Tk（I08）。"""
        if self.install['running']:
            return 'FFmpeg 正在安装，请稍候。'
        self.install.update(running=True, fraction=0., note='准备下载', error=None, finished=False)

        def work():
            try:
                from .install import install_ffmpeg
                install_ffmpeg(self.home,
                               progress=lambda fraction, note: self.install_queue.put(('progress', fraction, note)),
                               cancel=(lambda: self.closed))
            except Exception as error:
                self.install_queue.put(('error', str(error)))
            else:
                self.install_queue.put(('done', None))

        threading.Thread(target=work, daemon=True).start()
        return None

    def register_install_view(self, view):
        self.install_views.append(view)

    def unregister_install_view(self, view):
        if view in self.install_views:
            self.install_views.remove(view)

    def drain_install(self):
        """主线程收敛安装事件，更新共享状态并回调各视图（I08）。"""
        try:
            while True:
                kind, *payload = self.install_queue.get_nowait()
                if kind == 'progress':
                    self.install['fraction'], self.install['note'] = payload[0], payload[1]
                elif kind == 'done':
                    self.install.update(running=False, error=None, finished=True)
                elif kind == 'error':
                    self.install.update(running=False, error=payload[0], finished=True)
        except queue.Empty:
            pass
        for view in list(self.install_views):
            try:
                view(self.install)
            except tk.TclError:
                self.unregister_install_view(view)

    def close(self):
        if self.manager.tasks and not messagebox.askyesno('退出', '还有任务未完成。取消任务并退出吗？', parent=self.root):
            return
        self.manager.shutdown()
        self._preview_token = getattr(self, '_preview_token', 0) + 1
        self.player.stop()
        self.closed = True
        if self.import_token is not None:
            try: self.root.after_cancel(self.import_token)
            except tk.TclError: pass
        try: self.root.after_cancel(self.poll_token)
        except tk.TclError: pass
        try: self.root.after_cancel(self.welcome_token)
        except (tk.TclError, AttributeError): pass
        self.root.destroy()


def acquire_instance_lock(home):
    if os.name != 'nt': return None
    import msvcrt
    home.mkdir(parents=True, exist_ok=True)
    handle = (home / 'app.lock').open('a+b')
    if handle.seek(0, 2) == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise RuntimeError('角色音频工坊已经打开，请使用已有窗口。') from error
    return handle


def setup_report(home: Path | None = None) -> dict:
    """无窗口的环境快照，供发布包验收与排错使用。"""
    home = Path(home) if home is not None else data_home()
    settings_file = home / 'settings.json'
    return {'version': __version__, 'portable': bundle_root() is not None, 'home': str(home), 'settings': str(settings_file),
            'settings_exists': settings_file.exists(),
            'installer': find_spec('voice_workbench.install') is not None,
            'items': environment_report(Settings.load(settings_file), home)}


def emit(text: str):
    stream = getattr(sys, 'stdout', None)
    if stream is None:
        return
    try:
        stream.write(text + '\n'); stream.flush()
    except (OSError, ValueError):
        pass


def setup_main() -> int:
    """`--environment` 输出环境快照；`--install ffmpeg` 下载并校验 FFmpeg。"""
    home = data_home()
    report = setup_report(home)
    if '--install' in sys.argv:
        following = sys.argv[sys.argv.index('--install') + 1:]
        target = next((value for value in following if not value.startswith('-')), 'ffmpeg')
        if target != 'ffmpeg':
            raise SystemExit(f'未知的安装项：{target}（目前只支持 ffmpeg）。')
        from .install import install_ffmpeg
        report['installed'] = str(install_ffmpeg(home, progress=lambda fraction, note: emit(f'{fraction:6.1%}  {note}')))
        report['items'] = environment_report(Settings.load(home / 'settings.json'), home)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    emit(text)
    if '--setup-report' in sys.argv:
        destination = Path(sys.argv[sys.argv.index('--setup-report') + 1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding='utf-8')
    return 0


def main():
    if '--environment' in sys.argv or '--install' in sys.argv:
        raise SystemExit(setup_main())
    root = tk.Tk()
    try:
        instance_lock = acquire_instance_lock(data_home())
    except RuntimeError as error:
        root.withdraw(); messagebox.showinfo('软件已运行', str(error), parent=root); root.destroy(); return
    try:
        app = App(root)
        if '--training' in sys.argv: app.show_workspace('training')
        if '--smoke-ui' in sys.argv:
            root.withdraw()
            report = {'title': root.title(), 'version': __version__, 'widgets': len(root.winfo_children()), 'mode': app.mode.get(), 'training_available': hasattr(app, 'training_panel'), 'task_count': len(app.rows), 'tk': root.tk.call('info', 'patchlevel'), 'health': app.health_label.cget('text') if app.health_label.winfo_manager() else '', 'resume': app.resume_button.cget('text')}
            if '--smoke-report' in sys.argv:
                destination = Path(sys.argv[sys.argv.index('--smoke-report') + 1])
                destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            root.after(1500, app.close)
        root.mainloop()
    except Exception as error:
        messagebox.showerror('启动失败', str(error), parent=root)
        raise
    finally:
        if instance_lock: instance_lock.close()

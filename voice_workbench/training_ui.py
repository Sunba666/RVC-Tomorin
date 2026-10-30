"""Training controls for the native desktop, with no ML imports."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .training import TrainingRequest, collect_materials, MEDIA_SUFFIXES


class TrainingPanel:
    def __init__(self, app, parent):
        from .gui import PANEL, INK, MUTED, LINE, FONT
        self.app, self.sources = app, []
        self.preferences = app.home / 'training-options.json'
        defaults = self._load_preferences()
        self.values = {key: (tk.BooleanVar(value=value) if type(value) is bool else tk.StringVar(value=str(value)))
                       for key, value in defaults.items() if key not in ('sources', 'mode')}
        self.count = tk.StringVar(value='尚未添加素材')
        self.summary = tk.StringVar()
        self.frame = ttk.Frame(parent, style='Card.TFrame', padding=(22, 15, 22, 17))
        header = ttk.Frame(self.frame, style='Card.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text='训练自己的音色', style='Title.TLabel').pack(side='left')
        ttk.Button(header, text='移除所选', style='Quiet.TButton', command=self.remove_sources).pack(side='right')
        ttk.Button(header, text='添加文件夹', command=self.add_folder).pack(side='right', padx=8)
        ttk.Button(header, text='添加文件', command=self.add_files).pack(side='right')
        files = ttk.Frame(self.frame, style='Card.TFrame')
        files.pack(fill='x', pady=(10, 5))
        self.listbox = tk.Listbox(files, height=3, bg='#fafbf8', fg=INK, font=(FONT, 9),
                                 relief='flat', highlightthickness=1, highlightbackground=LINE,
                                 selectmode='extended', selectbackground='#e4efe8', selectforeground=INK, exportselection=False)
        scroll = ttk.Scrollbar(files, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side='left', fill='x', expand=True); scroll.pack(side='right', fill='y')
        ttk.Label(self.frame, textvariable=self.count, style='Hint.TLabel').pack(anchor='w')
        parameters = ttk.Frame(self.frame, style='Card.TFrame')
        parameters.pack(fill='x', pady=(13, 10))
        parameters.columnconfigure(0, weight=1)
        for column, (key, label, width) in enumerate([('name', '新模型名称', 24), ('epochs', '训练轮次', 9), ('batch_size', '批次大小', 9)]):
            field = ttk.Frame(parameters, style='Card.TFrame')
            field.grid(row=0, column=column, sticky='ew', padx=(0, 14))
            ttk.Label(field, text=label, style='Hint.TLabel').pack(anchor='w', pady=(0, 4))
            ttk.Entry(field, textvariable=self.values[key], width=width).pack(fill='x')
        ttk.Button(parameters, text='更多参数', style='Quiet.TButton', command=self.advanced).grid(row=0, column=3, sticky='se', pady=(0, 2))
        ttk.Label(self.frame, textvariable=self.summary, style='Hint.TLabel').pack(anchor='w')
        ttk.Label(self.frame, text='使用同一个人的清晰素材，建议累计至少 10 分钟。自动分离可能仍有少量伴奏或混响。',
                  style='Hint.TLabel', wraplength=820).pack(anchor='w', pady=(7, 0))
        ttk.Label(self.frame, text='筛选只依据音量、削波、音高等信号特征，无法保证纯人声或单一说话人；'
                  '明显含伴奏、和声或他人声音的素材，请先在列表里选中并用「移除所选」排除。',
                  style='Hint.TLabel', wraplength=820).pack(anchor='w', pady=(3, 0))
        actions = ttk.Frame(self.frame, style='Card.TFrame')
        actions.pack(fill='x', pady=(15, 0))
        self.start_button = ttk.Button(actions, text='开始训练', style='Accent.TButton', command=self.submit)
        self.start_button.pack(side='left')
        status = ttk.Frame(actions, style='Card.TFrame', padding=(20, 0, 0, 0))
        status.pack(side='left', fill='x', expand=True)
        label = ttk.Label(status, textvariable=app.message, style='Hint.TLabel', width=1)
        label.pack(fill='x', pady=(0, 7))
        status.bind('<Configure>', lambda event: label.configure(wraplength=max(200, event.width - 25)))
        self.progress = ttk.Progressbar(status, maximum=100)
        self.progress.pack(fill='x')
        self.refresh_summary()

    def _load_preferences(self):
        """读入训练参数偏好，容忍结构/类型损坏，坏文件自动备份后回退默认值（P01）。"""
        defaults = asdict(TrainingRequest([]))
        try:
            raw = self.preferences.read_text(encoding='utf-8')
        except OSError:
            return defaults
        try:
            saved = json.loads(raw)
        except ValueError:
            self._quarantine_preferences()
            return defaults
        if not isinstance(saved, dict):
            # 内容被手改成列表/数字等非对象（例如 []），旧代码会在 saved.items() 抛 AttributeError。
            self._quarantine_preferences()
            return defaults
        for key, value in list(saved.items()):
            if key not in defaults or key in ('sources', 'mode'):
                continue
            default = defaults[key]
            if type(default) is bool: accepted = type(value) is bool
            elif type(default) is int: accepted = type(value) is int and not isinstance(value, bool)
            elif type(default) is float: accepted = type(value) in (int, float) and not isinstance(value, bool)
            elif type(default) is str: accepted = type(value) is str
            else: accepted = False
            if accepted:
                defaults[key] = value
        return defaults

    def _quarantine_preferences(self):
        """把无法解析的训练参数偏好改名备份（保留内容供手动恢复），下次启动回退默认值。"""
        try:
            broken = self.preferences.with_name(self.preferences.name + '.corrupt')
            if broken.exists():
                broken.unlink()
            self.preferences.replace(broken)
        except OSError:
            pass

    def refresh_summary(self):
        rate = self.values['sample_rate'].get().replace('k', ' kHz')
        separation = '自动分离人声' if self.values['extract_vocals'].get() else '使用已清理的人声'
        quality = '严格筛选' if self.values['quality'].get() == 'strict' else '均衡筛选'
        self.summary.set(f'{separation}  →  {quality}  →  RVC v2  /  {rate}  →  模型与检索索引')

    def append_sources(self, sources):
        seen = {os.path.normcase(str(Path(p).resolve())) for p in self.sources}
        pending = []
        for source in sources:
            path = Path(source).resolve()
            key = os.path.normcase(str(path))
            if key not in seen and path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
                pending.append(str(path)); seen.add(key)
        if len(self.sources) + len(pending) > 2000: raise ValueError('单个任务最多添加 2000 个素材。')
        self.sources.extend(pending)
        self.refresh_sources()

    def refresh_sources(self):
        self.listbox.delete(0, 'end')
        for source in self.sources: self.listbox.insert('end', Path(source).name)
        self.count.set(f'已添加 {len(self.sources)} 个素材 · 原文件会保留' if self.sources else '尚未添加素材')

    def add_files(self):
        sources = filedialog.askopenfilenames(parent=self.app.root, title='添加同一人的音视频素材',
                    filetypes=[('音视频', ' '.join('*' + suffix for suffix in sorted(MEDIA_SUFFIXES))), ('所有文件', '*.*')])
        if sources:
            try: self.append_sources(sources)
            except Exception as error: messagebox.showerror('无法添加', str(error), parent=self.app.root)

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self.app.root, title='选择训练素材文件夹')
        if folder:
            try:
                sources = collect_materials(folder)
                if not sources: raise ValueError('文件夹中没有找到音视频素材。')
                self.append_sources(sources)
            except Exception as error: messagebox.showerror('无法添加', str(error), parent=self.app.root)

    def remove_sources(self):
        selected = set(self.listbox.curselection())
        self.sources = [source for index, source in enumerate(self.sources) if index not in selected]
        self.refresh_sources()

    def request(self):
        values = {key: variable.get() for key, variable in self.values.items()}
        for key in ('epochs', 'batch_size', 'save_every'): values[key] = int(values[key])
        values['learning_rate'] = float(values['learning_rate'])
        values['name'] = values['name'].strip()
        return TrainingRequest(list(self.sources), **values)

    def submit(self):
        try:
            request = self.request()
            self.app.enqueue(request)
            values = asdict(request)
            for key in ('sources', 'mode'): values.pop(key)
            try: self.preferences.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
            except OSError: pass  # A preference write must not undo an enqueued job.
        except Exception as error:
            messagebox.showerror('暂时不能训练', str(error), parent=self.app.root)

    def advanced(self):
        from .gui import PANEL, FONT
        current = getattr(self, 'dialog', None)
        if current is not None and current.winfo_exists(): current.lift(); return current
        dialog = tk.Toplevel(self.app.root, bg=PANEL); self.dialog = dialog
        dialog.title('训练参数'); dialog.transient(self.app.root)
        self.app.place_dialog(dialog, 690, 535); dialog.minsize(640, 500)
        holder = ttk.Frame(dialog, style='Card.TFrame', padding=24); holder.pack(fill='both', expand=True)
        ttk.Label(holder, text='训练参数', style='Title.TLabel', font=(FONT, 16, 'bold')).pack(anchor='w')
        ttk.Label(holder, text='轮次越多，耗时越长；批次越大，占用的显存越多。', style='Hint.TLabel').pack(anchor='w', pady=(6, 18))
        draft = {key: (tk.BooleanVar(value=var.get()) if isinstance(var, tk.BooleanVar) else tk.StringVar(value=var.get()))
                 for key, var in self.values.items()}
        form = ttk.Frame(holder, style='Card.TFrame'); form.pack(fill='x'); form.columnconfigure(1, weight=1)
        for row, (key, label, choices) in enumerate([
            ('sample_rate', '模型采样率', ('32k', '40k', '48k')), ('save_every', '每隔多少轮保存检查点', None),
            ('learning_rate', '学习率', None), ('quality', '素材筛选强度', ('均衡', '严格'))
        ]):
            ttk.Label(form, text=label, style='Card.TLabel').grid(row=row, column=0, sticky='w', padx=(0, 20), pady=9)
            if key == 'quality': draft[key].set('严格' if draft[key].get() == 'strict' else '均衡')
            widget = ttk.Combobox(form, textvariable=draft[key], values=choices, state='readonly') if choices else ttk.Entry(form, textvariable=draft[key])
            widget.grid(row=row, column=1, sticky='ew', pady=7)
        ttk.Checkbutton(holder, text='自动分离人声（视频或含伴奏素材建议开启）', variable=draft['extract_vocals']).pack(anchor='w', pady=(16, 0))
        ttk.Checkbutton(holder, text='完成后保留训练检查点（额外占用磁盘）', variable=draft['keep_checkpoints']).pack(anchor='w')
        ttk.Label(holder, text='均衡筛选保留更多片段；严格筛选会排除音高较少或人声较弱的片段。', style='Hint.TLabel', wraplength=610).pack(anchor='w', pady=8)
        buttons = ttk.Frame(holder, style='Card.TFrame'); buttons.pack(side='bottom', fill='x')
        def save():
            try:
                values = {key: var.get() for key, var in draft.items()}
                values['quality'] = 'strict' if values['quality'] == '严格' else 'balanced'
                for key in ('epochs', 'batch_size', 'save_every'): values[key] = int(values[key])
                values['learning_rate'] = float(values['learning_rate'])
                # Materials are validated separately when training is submitted.
                request = TrainingRequest([], **values)
                request.validate_parameters()
                for key, value in values.items(): self.values[key].set(value)
                self.refresh_summary(); dialog.destroy()
            except Exception as error: messagebox.showerror('参数无效', str(error), parent=dialog)
        ttk.Button(buttons, text='应用参数', style='Accent.TButton', command=save).pack(side='right')
        ttk.Button(buttons, text='取消', command=dialog.destroy).pack(side='right', padx=10)
        dialog.bind('<Escape>', lambda event: dialog.destroy())
        return dialog

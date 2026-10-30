import argparse
import json
import sys
import uuid
from pathlib import Path
from .core import Request, Settings, Task, data_home, load_request, task_settings


def main():
    parser = argparse.ArgumentParser(description='角色音频工坊：制作音频或训练 RVC 模型')
    parser.add_argument('source', nargs='*')
    parser.add_argument('--resume', type=Path, help='继续一个已有任务文件夹，复用已完成的阶段')
    parser.add_argument('--mode', choices=['original', 'convert', 'train'], default=None)
    parser.add_argument('--settings', type=Path, default=data_home() / 'settings.json')
    parser.add_argument('--name', default=None)
    parser.add_argument('--start', type=float, default=None)
    parser.add_argument('--end', type=float, default=None)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sample-rate', choices=['32k', '40k', '48k'], default='40k')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=.0001)
    parser.add_argument('--quality', choices=['balanced', 'strict'], default='balanced')
    parser.add_argument('--no-separation', action='store_true')
    parser.add_argument('--keep-checkpoints', action='store_true')
    args = parser.parse_args()
    settings = Settings.load(args.settings)
    if args.output:
        settings.output = str(args.output.resolve())
    if args.resume:
        conflicts = []
        if args.source: conflicts.append('来源')
        if args.mode is not None: conflicts.append('--mode')
        if args.name is not None: conflicts.append('--name')
        if args.start is not None: conflicts.append('--start')
        if args.end is not None: conflicts.append('--end')
        if conflicts:
            parser.error('继续任务时请勿指定' + '、'.join(conflicts) + '；它们会与已经保存的任务参数冲突。'
                         '若要改参数，请在界面里「沿用这些参数」后重新制作，或只给 --resume。')
        folder = args.resume.expanduser().resolve()
        request = load_request(folder)
        settings = task_settings(folder, settings)
        task = Task(request, settings, folder, event=lambda stage, percent=None: print(stage, flush=True))
        result, title = task.run()
        print(json.dumps({'file': str(result), 'name': title}, ensure_ascii=False))
        return
    mode = args.mode or 'original'
    task_class = Task
    if mode == 'train':
        from .training import TrainingRequest, TrainingTask, collect_materials
        if args.start or args.end is not None: parser.error('训练模式请先裁剪素材，再导入训练。')
        sources = []
        for source in args.source: sources.extend(collect_materials(source) if Path(source).is_dir() else [source])
        request = TrainingRequest(list(dict.fromkeys(sources)), args.name or '我的音色', args.epochs, args.batch_size,
                                  args.sample_rate, args.save_every, args.learning_rate, not args.no_separation,
                                  args.quality, args.keep_checkpoints)
        task_class = TrainingTask
    else:
        if len(args.source) != 1: parser.error('音频制作每个任务只接受一个来源。')
        request = Request(args.source[0], mode, args.name or '', args.start or 0, args.end)
    folder = Path(settings.output) / ('cli_' + uuid.uuid4().hex[:12])
    task = task_class(request, settings, folder, event=lambda stage, percent=None: print(stage, flush=True))
    result, title = task.run()
    print(json.dumps({'file': str(result), 'name': title}, ensure_ascii=False))


if __name__ == '__main__':
    main()
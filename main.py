import sys
from pathlib import Path

# Development-only dependencies live here; release builds bundle yt-dlp.
deps = Path(__file__).resolve().parent / '.build-deps'
sys.path.insert(0, str(Path(__file__).resolve().parent))
if deps.exists():
    sys.path.insert(0, str(deps))

if __name__ == '__main__':
    if '--process' in sys.argv:
        sys.argv.remove('--process')
        from voice_workbench.cli import main
    else:
        from voice_workbench.gui import main
    main()

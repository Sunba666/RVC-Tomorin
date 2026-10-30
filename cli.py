import sys
from pathlib import Path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
if (root / '.build-deps').exists():
    sys.path.insert(0, str(root / '.build-deps'))
from voice_workbench.cli import main
if __name__ == '__main__':
    main()

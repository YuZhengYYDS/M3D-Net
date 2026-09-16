"""Run the four matched models sequentially, with the fixed 50-epoch protocol."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/breast_full_clinical.json')
    parser.add_argument('--manifest', type=Path, default=Path('data/breast/manifest.csv'))
    parser.add_argument('--weights', type=Path, default=Path('data/pretrained'))
    parser.add_argument('--output', type=Path, default=Path('outputs/breast_full_clinical'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-cuda-graphs', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for model in config['models']:
        destination = args.output.resolve() / model
        command = [sys.executable, '-u', str(ROOT / 'train.py'), '--model', model,
                   '--config', str(args.config.resolve()), '--manifest', str(args.manifest.resolve()),
                   '--weights', str(args.weights.resolve()), '--output', str(destination)]
        if args.resume and (destination / 'last.pt').exists():
            command.append('--resume')
        if args.no_cuda_graphs:
            command.append('--no-cuda-graphs')
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

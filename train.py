"""Train a matched image-plus-clinical model for at least 50 epochs."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from m3d.data import ClinicalImageDataset, read_manifest
from m3d.engine import capture_training_graph, evaluate, make_optimizer, set_cosine_after_step, set_seed
from m3d.fusion import MODEL_NAMES, MultiModalModel
from m3d.metadata import FULL_CLINICAL_FIELDS, fit_clinical_normalization
from m3d.pretrained import initialize_image_backbone, sha256_file

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/breast_full_clinical.json'


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def validate_config(config):
    if config['epochs'] < 50:
        raise ValueError('Research runs require at least 50 epochs; use tests/smoke.py for a short check')
    if config['batch_size'] < 2 or config['resample_factor'] < 1 or config['num_workers'] != 0:
        raise ValueError('Require batch_size >= 2, positive resample_factor, and num_workers=0 for exact resume')
    if config['precision'] != 'bf16' or config['metadata_profile'] != 'breast_full_clinical':
        raise ValueError('This release implements BF16 training with the full clinical profile')
    if config['input_size'] != 256 or not np.isfinite(config['lr']) or config['lr'] <= 0:
        raise ValueError('Expected 256-pixel inputs and a positive finite learning rate')


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    frame = read_manifest(args.manifest)
    normalization = fit_clinical_normalization(frame)
    config.update(backbone=args.model, metadata_fields=FULL_CLINICAL_FIELDS,
                  age_normalization=normalization, num_classes=2,
                  manifest_sha256=sha256_file(args.manifest),
                  da_layout_corrected=args.model == 'm3d', projection_residual=args.model == 'm3d')
    if args.no_cuda_graphs:
        config['cuda_graphs'] = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The reference training protocol requires a CUDA GPU with BF16 support')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'RUNNING.lock'
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        execute(args, config, frame, normalization, output)
    finally:
        lock_path.unlink(missing_ok=True)


def execute(args, config, frame, normalization, output):
    if (output / 'config.json').exists() and not args.resume:
        raise FileExistsError('Run already exists; choose a new output directory or pass --resume')
    if args.resume and not (output / 'last.pt').exists():
        raise FileNotFoundError('No last.pt checkpoint to resume')
    source_files = [*ROOT.glob('*.py'), *(ROOT / 'm3d').rglob('*.py')]
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(source_files)}
    if args.resume:
        original_hashes = json.loads((output / 'source_hashes.json').read_text(encoding='utf-8'))
        if hashes != original_hashes:
            raise ValueError('Source code changed since the run started; use a separate output directory')
    torch.set_num_threads(4)
    set_seed(config['seed'])
    datasets = {split: ClinicalImageDataset(frame, split, normalization, config['resample_factor'])
                for split in ('train', 'val')}
    loaders = {split: DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
                               num_workers=0, pin_memory=True, drop_last=split == 'train')
               for split, dataset in datasets.items()}
    if not len(loaders['train']):
        raise ValueError('Training partition is too small for one full batch')
    model = MultiModalModel(backbone=args.model).cuda()
    if config['pretrained']:
        config['pretraining'] = initialize_image_backbone(model, args.model, args.weights)
    config.update(torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
                  train_rows=len(datasets['train']), validation_rows=len(datasets['val']),
                  parameters=sum(parameter.numel() for parameter in model.parameters()))
    optimizer = make_optimizer(model, config)
    # Retain the experiment's disabled scaler and unscale -> clip -> step order.
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    history, first_epoch, elapsed_before = [], 0, 0.0
    if args.resume:
        # Resume files contain Python/NumPy RNG objects: only load your own run.
        checkpoint = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        previous = checkpoint['config']
        mutable_environment = {'gpu', 'torch_version'}
        for key in set(previous) | set(config):
            if key not in mutable_environment and previous.get(key) != config.get(key):
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, first_epoch = checkpoint['history'], checkpoint['epoch']
        if len(history) != first_epoch:
            raise ValueError('Checkpoint history does not match the completed epoch count')
        elapsed_before = history[-1]['elapsed_seconds'] if history else 0.0
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        del checkpoint
    if config['cuda_graphs'] and first_epoch < config['epochs']:
        model = capture_training_graph(model, config['batch_size'])
    write_json(output / 'config.json', config)
    write_json(output / ('resume_source_hashes.json' if args.resume else 'source_hashes.json'), hashes)
    start = time.time() - elapsed_before
    for epoch in range(first_epoch, config['epochs']):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, clinical, labels in loaders['train']:
            images, clinical, labels = images.cuda(), clinical.cuda(), labels.cuda()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=not config['cuda_graphs']):
                logits = model(images, clinical)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient norm at epoch {epoch + 1}')
            scaler.step(optimizer)
            scaler.update()
            lr = set_cosine_after_step(optimizer, epoch, config['epochs'], config['lr'])
            total_loss += loss.item()
            seen += len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        val_loss, val_acc, _ = evaluate(model, loaders['val'], 'cuda')
        row = dict(epoch=epoch + 1, train_acc=correct / seen, train_loss=total_loss / len(loaders['train']),
                   val_acc=val_acc, val_loss=val_loss, lr=lr, train_seen=seen,
                   optimizer_batches=len(loaders['train']), elapsed_seconds=time.time() - start)
        history.append(row)
        # Embed history in the atomic checkpoint so resume never trusts a partial CSV.
        save_checkpoint(output / 'last.pt', dict(
            epoch=epoch + 1, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), config=config, history=history,
        ))
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
    # A weights-only evaluation artifact excludes optimizer and RNG pickle objects.
    save_checkpoint(output / 'model.pt', dict(epoch=config['epochs'], state_dict=model.state_dict(), config=config))
    write_json(output / 'summary.json', dict(
        split='validation', epochs_completed=len(history),
        acc_percent=100 * history[-1]['val_acc'], tail_percent=100 * float(np.mean([row['val_acc'] for row in history[-10:]])),
        val_loss=history[-1]['val_loss'], checkpoint='fixed final epoch',
    ))
    print(f'Completed {config["epochs"]} epochs: {output}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, default='m3d')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--manifest', type=Path, default=Path('data/breast/manifest.csv'))
    parser.add_argument('--weights', type=Path, default=Path('data/pretrained'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-cuda-graphs', action='store_true', help='Eager BF16 execution; record as a changed execution setting')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())

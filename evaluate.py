"""Evaluate a fixed checkpoint on validation or an explicitly selected test split."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from m3d.data import ClinicalImageDataset, read_manifest
from m3d.engine import evaluate
from m3d.fusion import MultiModalModel
from m3d.pretrained import sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='model.pt produced by train.py')
    parser.add_argument('--manifest', type=Path, default=Path('data/breast/manifest.csv'))
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if sha256_file(args.manifest) != config['manifest_sha256']:
        raise ValueError('The manifest must match the frozen training manifest exactly')
    frame = read_manifest(args.manifest)
    # Match historical validation loss batching; test patients appear exactly once.
    repeats = config['resample_factor'] if args.split == 'val' else 1
    dataset = ClinicalImageDataset(frame, args.split, config['age_normalization'], repeats)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    model = MultiModalModel(backbone=config['backbone']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    loss, accuracy, probabilities = evaluate(model, loader, args.device)
    unique = np.arange(0, len(dataset), repeats)
    labels = np.asarray([row.label for row in dataset.rows])[unique]
    probabilities = probabilities[unique]
    prediction = probabilities.argmax(1)
    result = dict(split=args.split, epoch=checkpoint['epoch'], cases=len(unique),
                  acc_percent=100 * accuracy, loss=loss,
                  macro_f1=float(f1_score(labels, prediction, average='macro', zero_division=0)),
                  malignant_auc=float(roc_auc_score(labels, probabilities[:, 1])),
                  confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
                  checkpoint_sha256=sha256_file(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {args.output}')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

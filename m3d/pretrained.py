"""Verified official ImageNet initialization; M3D loads compatible tensors only."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath

import numpy as np
import torch

SOURCES = {
    'transxnet': {
        'filename': 'transx-t.pth.tar',
        'url': 'https://api.github.com/repos/LMMMEng/TransXNet/releases/assets/136349236',
        'sha256': 'c8ef6f4834014e691c0e5a5c2f027b2a23172b16988e35742dda571f6b8e8704',
    },
    'repvit': {
        'filename': 'repvit_m0_9_distill_300e.pth',
        'url': 'https://api.github.com/repos/THU-MIG/RepViT/releases/assets/128092165',
        'sha256': '857eb0e6a992591a16ed96b22a73135270032d2b97c2c1f74dd3d0050e0c83e8',
    },
    'edgenext': {
        'filename': 'edgenext_xx_small.pth',
        'url': 'https://api.github.com/repos/mmaaz60/EdgeNeXt/releases/assets/69154358',
        'sha256': 'd8b309b2ecec97a8544422553dee9409e90022badcba23127eb946e6002ad981',
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize_image_backbone(model, backbone, directory):
    source = SOURCES['transxnet' if backbone.startswith('m3d') else backbone]
    path = Path(directory) / source['filename']
    if not path.is_file():
        raise FileNotFoundError(f'{path}; first run python download.py weights')
    digest = sha256_file(path)
    if digest != source['sha256']:
        raise ValueError(f'Pretrained checkpoint SHA256 mismatch: {path.name}')
    # Official EdgeNeXt metadata includes NumPy scalars and a POSIX path. Map
    # the path to a pure path so Linux-authored weights also load on Windows.
    allowed = [argparse.Namespace, np.core.multiarray.scalar, np.dtype,
               type(np.dtype('float64')), type(np.dtype('float32')),
               (PurePosixPath, 'pathlib.PosixPath')]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    target = model.image_backbone.state_dict()
    matched = {key: value for key, value in state.items() if key in target and value.shape == target[key].shape}
    if not matched or (not backbone.startswith('m3d') and len(matched) != len(target)):
        raise ValueError('Unexpected pretrained architecture; refusing a mismatched warm start')
    model.image_backbone.load_state_dict(matched, strict=False)
    return dict(filename=path.name, sha256=digest, source_url=source['url'],
                matched_tensors=len(matched), total_tensors=len(target),
                matched_elements=sum(value.numel() for value in matched.values()),
                total_elements=sum(value.numel() for value in target.values()),
                missing=sorted(set(target) - set(matched)), ignored=sorted(set(state) - set(matched)))

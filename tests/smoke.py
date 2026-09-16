"""Synthetic forward/backward and checkpoint round-trip checks; no patient data."""
import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from m3d.engine import capture_training_graph, set_seed
from m3d.fusion import MultiModalModel


def smoke(backbone, device, graphs=False):
    set_seed(42)
    model = MultiModalModel(backbone=backbone).to(device)
    images = torch.randn(2, 3, 256, 256, device=device)
    clinical = torch.randn(2, 22, device=device)
    labels = torch.tensor([0, 1], device=device)
    if graphs:
        model = capture_training_graph(model, 2)
    model.train()
    with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda', cache_enabled=not graphs):
        logits = model(images, clinical)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    if logits.shape != (2, 2) or not torch.isfinite(loss):
        raise AssertionError('Invalid logits or loss')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise AssertionError(f'Nonfinite gradient: {name}')
    model.eval()
    with torch.no_grad():
        expected = model(images, clinical)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'weights.pt'
        torch.save(model.state_dict(), path)
        loaded = MultiModalModel(backbone=backbone).to(device).eval()
        loaded.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            actual = loaded(images, clinical)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    print(f'PASS {backbone}: {sum(p.numel() for p in model.parameters())} parameters, forward/backward/reload', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--model', choices=['all', 'm3d', 'm3d_original', 'edgenext', 'repvit', 'transxnet'], default='all')
    parser.add_argument('--cuda-graphs', action='store_true')
    args = parser.parse_args()
    if args.cuda_graphs and args.device != 'cuda':
        parser.error('--cuda-graphs requires --device cuda')
    torch.set_num_threads(4)
    models = ['edgenext', 'repvit', 'transxnet', 'm3d', 'm3d_original'] if args.model == 'all' else [args.model]
    for backbone in models:
        smoke(backbone, args.device, args.cuda_graphs)

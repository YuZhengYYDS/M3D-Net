"""Training primitives preserving the matched comparison's update order."""
import random

import numpy as np
import torch


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model, config):
    group = dict(params=list(model.parameters()), lr=config['lr'], reference_lr=config['lr'])
    return torch.optim.AdamW([group], weight_decay=config['weight_decay'], betas=tuple(config['betas']))


def set_cosine_after_step(optimizer, epoch, epochs, base_lr):
    # Retain the experiment's order: optimizer.step(), then assign epoch cosine.
    factor = .5 * (1 + np.cos(np.pi * epoch / epochs))
    for group in optimizer.param_groups:
        group['lr'] = group.get('reference_lr', base_lr) * factor
    return base_lr * factor


def capture_training_graph(model, batch_size, features=22, input_size=256):
    """Capture BF16 training without advancing RNG or BatchNorm buffers."""
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.train()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        model = torch.cuda.make_graphed_callables(
            model, (torch.zeros(batch_size, 3, input_size, input_size, device='cuda'),
                    torch.zeros(batch_size, features, device='cuda')),
            num_warmup_iters=3, allow_unused_input=True,
        )
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """FP32 inference; validation loss is an unweighted mean of batch means."""
    model.eval()
    losses, labels, probabilities = [], [], []
    for images, clinical, target in loader:
        logits = model(images.to(device), clinical.to(device))
        losses.append(torch.nn.functional.cross_entropy(logits, target.to(device)).item())
        labels.extend(target.tolist())
        probabilities.extend(logits.softmax(1).float().cpu().numpy())
    if not labels:
        raise ValueError('Cannot evaluate an empty loader')
    probabilities = np.asarray(probabilities)
    return float(np.mean(losses)), float(np.mean(probabilities.argmax(1) == labels)), probabilities

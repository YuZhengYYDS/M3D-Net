# Contributing

Thank you for helping improve M3D-Net. Use [GitHub Issues](https://github.com/YuZhengYYDS/M3D-Net/issues) for questions or bug reports and a pull request for a proposed change.

## Report a reproducible issue

Include the repository commit, operating system, GPU, Python/PyTorch versions, exact command, and a short traceback. Prefer a synthetic example. Do not attach patient records, dataset images, credentials, or private checkpoints.

## Validate a change

```bash
pixi run test
pixi run smoke-cuda
```

Use `pixi run smoke` when no CUDA GPU is available, and state which checks you ran. A documentation-only change needs link and rendering checks rather than another training run.

Keep architecture changes and experiment settings explicit. When changing a model, preserve the documented legacy switches or explain checkpoint incompatibilities. Fit preprocessing statistics on training patients only and keep test evaluation separate from model selection.

The repository is source-only. Keep data, weights, run outputs, and performance reports outside commits. Retain upstream attribution and licensing notices when modifying borrowed components.

# Reproduction guide

[Back to the project homepage](../README.md)

This guide documents the released BrEaST image-clinical protocol. The base mammography encoder is available separately; the original AISSLab run recipe is not reconstructed by this entry point.

## Installation

Use Python 3.11 and an NVIDIA CUDA GPU supporting BF16 for the reference training protocol. The environment pins PyTorch 2.7.1 with CUDA 12.8, torchvision 0.22.1, timm 1.0.15, and MMCV Lite 2.2.0. The Pixi lock file covers Windows and Linux; runtime validation for this release was performed on Windows with an RTX 5060. A compatible NVIDIA driver is required; a separate CUDA toolkit is not required for these PyTorch wheels.

With [Pixi](https://pixi.sh/):

```bash
pixi install --locked
pixi run test
pixi run smoke-cuda
```

Alternatively, use Conda:

```bash
conda env create -f environment.yml
conda activate m3d-net
python -m unittest discover -s tests -p "test_*.py"
python tests/smoke.py --device cuda
```

Use the commands below from the repository root. Prefix each `python` command with `pixi run` when using Pixi. The smoke checks use synthetic inputs; they do not train on patient data or produce research results. `python tests/smoke.py --device cpu` is also available for a CPU-only functional check.

## Data preparation

BrEaST is distributed as [TCIA Breast-Lesions-USG](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/), under CC BY 4.0. See the [dataset publication](https://doi.org/10.1038/s41597-024-02984-z) and [authors' repository](https://github.com/best-ippt-pan-pl/BrEaST).

The preparation script takes the complete image/clinical ZIP, checks the clinical workbook's SHA256, and deterministically regenerates the patient split. It retains all 256 cases: 154 benign and 4 normal cases form the nonmalignant class; 98 malignant cases form the positive class. No lesion mask or annotation-derived crop is used.

Download the verified public mirror, then prepare the data:

```bash
python download.py breast
python prepare_data.py --archive data/downloads/breast_lesions_usg.zip
```

Alternatively, download the archive from the source links and provide its local path:

```bash
python prepare_data.py --archive /path/to/breast_lesions_usg.zip --output data/breast
```

The downloader verifies the exact mirror archive hash. The preparation script validates the clinical workbook version independently, allowing a differently packaged archive with the same workbook and images. If an upstream download changes, obtain the matching version rather than bypassing the hash check.

The script creates `data/breast/manifest.csv` and padded images. Seed 42 and the two-stage stratified patient split produce 153 training, 51 validation, and 52 test cases. Age imputation and normalization are fitted only on observed ages from unique training patients. Inputs include age, history/symptoms, physical signs, breast tissue composition, and missing-value indicators. The BI-RADS risk category and diagnostic target are excluded from model inputs. See [the data protocol](data.md) for all 22 fields and the manifest schema.

## Pretrained initialization

```bash
python download.py weights
```

This downloads official ImageNet weights for the three backbone families into `data/pretrained/`, with SHA256 verification. EdgeNeXt, RepViT, and TransXNet load their complete matching image-backbone state. M3D-Net loads only matching TransXNet tensors; its additional modules, clinical encoder, and fusion head are initialized separately. The exact tensor coverage is written into each run's configuration. Every parameter remains trainable.

## Reproduce the matched comparison

```bash
python reproduce.py
```

This runs EdgeNeXt, RepViT, TransXNet, and M3D-Net sequentially using [the shared configuration](../configs/breast_full_clinical.json). Each run trains for **50 epochs**. The trainer rejects research configurations shorter than 50 epochs.

To inspect the commands first:

```bash
python reproduce.py --dry-run
```

To train M3D-Net alone:

```bash
python train.py --model m3d --output outputs/breast_full_clinical/m3d
```

To run the original implementation switches as a separate experiment:

```bash
python train.py --model m3d_original --output outputs/breast_full_clinical/m3d_original
```

The reference configuration uses:

- AdamW, learning rate `2e-5`, weight decay `1e-4`, betas `(0.9, 0.999)`, and gradient norm clipping at `3.0`.
- Batch size 4, two copies per case, and `drop_last=True` for training. Extra training copies receive Gaussian clinical noise with standard deviation `0.01`.
- Full-frame RGB images, aspect-preserving bilinear resizing and black padding to 256 pixels, followed by crop/flip/rotation/color augmentation.
- BF16 training, a disabled gradient scaler, FP32 validation, and CUDA graph capture with RNG and BatchNorm buffers restored after capture.
- The original cosine assignment order: set the epoch's learning rate **after** each optimizer update. The first update of a new epoch uses the preceding learning rate.
- Fixed final-epoch evaluation. `Acc.` is the final validation accuracy, `Tail` averages the final ten validation accuracies, and validation loss is the unweighted mean of batch mean cross-entropies, including the last short batch.

Validation repeats each case twice to preserve the experimental batching and loss calculation. This does not create additional independent patients. With the standard split, each epoch contains 76 optimizer updates and 304 sampled training rows. There is no early stopping or automatic best-checkpoint selection in the reproduction entry point.

CUDA graph capture can be disabled with `--no-cuda-graphs` if required for another GPU. This change is recorded in the configuration. Floating-point results can differ across devices, library versions, and execution modes; the fixed seed does not guarantee identical results on every platform.

## Resume and evaluation

Resume a single run using the same command plus `--resume`, or resume the complete sequence:

```bash
python reproduce.py --resume
```

`last.pt` saves the model, optimizer, RNG states, configuration, and history atomically after every completed epoch. Resume restores these states and validates the configuration. Only load resume checkpoints generated by a trusted run. A `RUNNING.lock` prevents concurrent writes; after an abrupt process termination, remove that run's lock only after confirming no training process still owns it.

After training, `model.pt` contains the final model and configuration for weights-only loading:

```bash
python evaluate.py --checkpoint outputs/breast_full_clinical/m3d/model.pt --split val --output outputs/breast_full_clinical/m3d/validation.json
```

The trainer never loads test images. Test evaluation is an explicit, separate command, using the fixed checkpoint and the same frozen manifest:

```bash
python evaluate.py --checkpoint outputs/breast_full_clinical/m3d/model.pt --split test --output outputs/breast_full_clinical/m3d/test.json
```

Keep validation and test metrics labeled separately. Select the model and protocol before test evaluation. Generated outputs stay under `outputs/` and are ignored by Git.


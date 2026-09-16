# Third-party code and data

Project-authored code, documentation, and architecture schematics are licensed under the [Apache License 2.0](LICENSE). The model implementations retain contributions from the projects below, whose original terms remain applicable. Their upstream license texts are included verbatim. See also [NOTICE](NOTICE).

## TransXNet

- Source: https://github.com/LMMMEng/TransXNet
- License: Apache-2.0; see [licenses/TransXNet.txt](licenses/TransXNet.txt).
- Relevant files: `m3d/models/transxnet.py` and the TransXNet-derived parts of `m3d/models/m3d_net.py`.
- Local changes include multimodal integration, MCA/MUDD/DA modules in M3D, explicit DA/STE switches, MMCV compatibility handling, and packaging/documentation cleanup.

## RepViT

- Source: https://github.com/THU-MIG/RepViT
- License: Apache-2.0; see [licenses/RepViT.txt](licenses/RepViT.txt).
- Relevant file: `m3d/models/repvit.py`.
- This release uses the experimental local implementation, removes global factory registration, and updates the timm utility import path. The M0.9 factory is used by the matched comparison. The upstream attribution for the MobileNet channel-divisibility helper is retained in the source.

## EdgeNeXt

- Source: https://github.com/mmaaz60/EdgeNeXt
- License: MIT; see [licenses/EdgeNeXt.txt](licenses/EdgeNeXt.txt).
- Copyright (c) 2022 Muhammad Maaz; Copyright (c) Meta Platforms, Inc. and affiliates.
- Relevant files: `m3d/models/edgenext.py`, `conv_encoder.py`, `sdta_encoder.py`, and `layers.py`.
- Local changes include the experimental classifier/pooling integration, English documentation cleanup, removal of global factory registration, and the updated timm utility import path.

## BrEaST and pretrained weights

BrEaST is provided under CC BY 4.0 by its original authors and TCIA. Its images and clinical records are downloaded separately. Dataset documentation and attribution are linked in the README.

Official pretrained weights are downloaded from the respective model authors' releases and verified against fixed SHA256 hashes. They are not redistributed in this repository. Their upstream terms remain applicable.

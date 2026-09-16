# Implementation details

## Core modules

`m3d/models/m3d_net.py` preserves the tensor operations of the experimental implementation. The release moves modules into a package, normalizes documentation to English, removes constructor diagnostics and global timm factory registration, and exposes the clinical/fusion code independently of the original training script.

- `MultiScaleCoordinateAttention`: processes channel groups at multiple spatial scales, embeds coordinate information, and fuses the resulting features with a residual connection.
- `MUDDConnection`: dynamically aggregates paths from the current feature tensor and recent features within a stage.
- `DifferentialAttention`: splits query/key/value channels into two groups and uses a learned differential combination of attention maps.
- `HybridTokenMixer`: combines local dynamic convolution and a global attention branch, with optional MCA and a projection residual.

The clinical encoder and the final fusion classifier are shared by every backbone. The image branch returns a 1000-dimensional representation rather than a binary prediction; the final fusion classifier produces the two logits.

## DA layout

Each attention output has shape `[batch, heads, spatial_positions, half_head_channels]`.

The original layout is retained under `m3d_original`:

```python
x = torch.cat([output1, output2], dim=2)
x = x.reshape(batch, channels, height, width)
```

The additional-dataset configuration uses:

```python
x = torch.cat([output1, output2], dim=-1).transpose(-2, -1)
x = x.reshape(batch, channels, height, width)
```

This changes how head channels and spatial positions are restored, without changing parameter shapes or the attention formula.

## STE projection residual

Within `HybridTokenMixer`, `x` is the concatenated local/global feature tensor after optional MCA. The switch changes `self.proj(x)` to `self.proj(x) + x`. It is a structural forward-pass option, independent of the outer block residual. It does not add trainable parameters. The original TransXNet baseline already includes its own projection residual.

For the released tiny M3D image backbone, the configuration affects 12 DA modules and 18 token mixers. Both M3D variants have identical state-dict names and shapes and 17,904,268 total trainable parameters with 22 clinical inputs and two output classes.

## Initialization and update parity

Initialization order follows the experiment: image backbone, clinical encoder, then fusion classifier. Official ImageNet weights are loaded by matching names and tensor shapes after construction. M3D initialization is a partial warm start from TransXNet, not a fully pretrained M3D model. All parameters use the same learning rate and remain trainable.

The release preserves class-ordered sample construction, repeated validation rows, augmentation order, the mean-of-batch-means validation loss, gradient clipping order, and cosine assignment after optimizer updates. Checkpoint writing is simplified to an atomic save after every epoch; it does not alter optimizer updates.

The source-only release has no dependence on the original workstation directory, private spreadsheets, historical search queues, candidate selection records, or frozen experimental reports. The BrEaST recipe reproduces the additional-dataset protocol. It does not claim to reconstruct every historical experiment on an unavailable dataset.

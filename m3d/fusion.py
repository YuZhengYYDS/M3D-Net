"""Shared image/clinical fusion for M3D-Net and matched baselines."""
import torch
from torch import nn

from .clinical import iTransformer
from .models.edgenext import edgenext_xx_small
from .models.repvit import repvit_m0_9
from .models.transxnet import transxnet_t

MODEL_NAMES = ('m3d', 'm3d_original', 'edgenext', 'repvit', 'transxnet')


class MultiModalModel(nn.Module):
    """Fuse a 1000-D image representation with a 1000-D clinical representation.

    ``m3d`` uses the DA layout correction and STE projection residual from the
    additional-dataset experiment. ``m3d_original`` retains both legacy switches.
    State-dict names and parameter initialization match the experiment code.
    """

    def __init__(self, num_classes=2, tabular_features=22, backbone='m3d', input_size=256):
        super().__init__()
        if backbone == 'repvit':
            self.image_backbone = repvit_m0_9()
        elif backbone in ('m3d', 'm3d_original'):
            from .models.m3d_net import transxnet_t as m3d_net_t
            self.image_backbone = m3d_net_t(num_classes=1000, img_size=input_size)
            for module in self.image_backbone.modules():
                if hasattr(module, 'correct_output_layout'):
                    module.correct_output_layout = backbone == 'm3d'
                if hasattr(module, 'projection_residual'):
                    module.projection_residual = backbone == 'm3d'
        elif backbone == 'transxnet':
            self.image_backbone = transxnet_t(num_classes=1000, img_size=input_size)
        elif backbone == 'edgenext':
            self.image_backbone = edgenext_xx_small(classifier_dropout=0.0)
        else:
            raise ValueError(f'Unknown backbone: {backbone}; choose from {MODEL_NAMES}')
        self.tabular_model = iTransformer(
            input_features=tabular_features, seq_length=1, hidden_size=128,
            num_layers=4, dropout=0.3,
        )
        self.image_norm = nn.Identity()
        self.tabular_norm = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2000, 1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, tabular_data):
        image_features = self.image_backbone(image)
        tabular_features = self.tabular_model(tabular_data)
        features = torch.cat((self.image_norm(image_features), self.tabular_norm(tabular_features)), dim=1)
        return self.classifier(features)

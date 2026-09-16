"""Patient-disjoint manifests, fixed clinical encoding, and shared augmentation."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .metadata import FULL_CLINICAL_FIELDS, encode_row, fit_clinical_normalization


def read_manifest(path):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype={'patient_id': str, 'group_id': str, 'image_id': str})
    required = {'image_id', 'patient_id', 'group_id', 'image_path', 'label', 'class_name',
                'split', 'pixel_sha256', 'age', 'symptoms', 'signs', 'tissue_composition'}
    if not required <= set(frame.columns):
        raise ValueError(f'Missing manifest columns: {sorted(required - set(frame.columns))}')
    if set(frame.split) != {'train', 'val', 'test'} or set(frame.label) != {0, 1}:
        raise ValueError('Expected train/val/test partitions with binary labels 0 and 1')
    if frame.image_id.duplicated().any():
        raise ValueError('Each image_id must occur once in the manifest')
    if frame.groupby('label').class_name.nunique().max() != 1:
        raise ValueError('Class names must agree with numeric labels')
    if frame.groupby('label').class_name.first().to_dict() != {0: 'NonMalignant', 1: 'Malignant'}:
        raise ValueError('Expected label 0 = NonMalignant and label 1 = Malignant')
    for column in ('patient_id', 'group_id', 'pixel_sha256', 'image_path'):
        if frame[column].isna().any() or frame.groupby(column).split.nunique().max() != 1:
            raise ValueError(f'Missing or overlapping partition identifiers in {column}')
    for split in ('train', 'val', 'test'):
        if set(frame.loc[frame.split == split, 'label']) != {0, 1}:
            raise ValueError(f'Both classes are required in {split}')
    frame['image_path'] = frame.image_path.map(lambda value: str((path.parent / value).resolve()))
    return frame


def image_transform(training, input_size=256):
    operations = [transforms.Resize(input_size)]
    if training:
        operations.extend([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
        ])
    else:
        operations.append(transforms.CenterCrop(input_size))
    operations.extend([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    return transforms.Compose(operations)


class ClinicalImageDataset(Dataset):
    """Class-ordered samples; only extra training copies receive clinical noise."""

    def __init__(self, frame, split, normalization, repeats=2, input_size=256):
        if repeats < 1:
            raise ValueError('repeats must be positive')
        self.training = split == 'train'
        self.repeats = repeats
        self.transform = image_transform(self.training, input_size)
        config = dict(metadata_profile='breast_full_clinical', age_normalization=normalization)
        self.rows, self.metadata = [], []
        for label in sorted(frame.label.unique()):
            for row in frame[(frame.split == split) & (frame.label == label)].itertuples(index=False):
                values = encode_row(row, config)
                if len(values) != len(FULL_CLINICAL_FIELDS) or not np.isfinite(values).all():
                    raise ValueError('Invalid clinical feature vector')
                for _ in range(repeats):
                    self.rows.append(row)
                    self.metadata.append(values)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row.image_path) as image:
            image = self.transform(image.convert('RGB'))
        clinical = torch.tensor(self.metadata[index], dtype=torch.float32)
        if self.training and index % self.repeats != 0:
            clinical = clinical + torch.randn_like(clinical) * .01
        return image, clinical, int(row.label)

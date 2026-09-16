"""Check leakage boundaries and the historical schedule using synthetic inputs."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from m3d.data import read_manifest
from m3d.engine import set_cosine_after_step
from m3d.metadata import encode_full_clinical, fit_clinical_normalization
from train import validate_config


class ProtocolTests(unittest.TestCase):
    def test_age_fit_ignores_holdout_values(self):
        frame = pd.DataFrame({'patient_id': ['a', 'b', 'c', 'd'], 'split': ['train', 'train', 'val', 'test'], 'age': [30., 50., 70., 80.]})
        initial = fit_clinical_normalization(frame)
        frame.loc[frame.split != 'train', 'age'] = -999
        self.assertEqual(initial, fit_clinical_normalization(frame))
        self.assertEqual(initial['mean'], 40.)
        values = encode_full_clinical(float('nan'), 'no', 'not available', 'homogeneous: fat', initial)
        self.assertEqual(len(values), 22)
        self.assertEqual(values[:2], [0., 1.])

    def test_patient_overlap_is_rejected(self):
        rows = []
        for split in ['train', 'val', 'test']:
            for label in [0, 1]:
                identifier = f'{split}_{label}'
                rows.append(dict(image_id=identifier, patient_id=identifier, group_id=identifier,
                    image_path=f'{identifier}.png', label=label, class_name='Malignant' if label else 'NonMalignant', split=split,
                    pixel_sha256=identifier, age=40, symptoms='no', signs='no', tissue_composition='homogeneous: fat'))
        frame = pd.DataFrame(rows)
        frame.loc[2, 'patient_id'] = frame.loc[0, 'patient_id']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.csv'
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'patient_id'):
                read_manifest(path)

    def test_schedule_is_assigned_after_update(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([{'params': [parameter], 'lr': 2e-5, 'reference_lr': 2e-5}])
        self.assertEqual(optimizer.param_groups[0]['lr'], 2e-5)
        self.assertEqual(set_cosine_after_step(optimizer, 0, 50, 2e-5), 2e-5)
        self.assertAlmostEqual(set_cosine_after_step(optimizer, 25, 50, 2e-5), 1e-5)

    def test_short_research_runs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least 50'):
            validate_config({'epochs': 49})


if __name__ == '__main__':
    unittest.main()

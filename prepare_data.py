"""Prepare the fixed BrEaST image-plus-clinical protocol from a local ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from m3d.metadata import FULL_CLINICAL_FIELDS, encode_row, fit_clinical_normalization

CLINICAL_SHA256 = '89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf'


def prepare(archive, output):
    archive, output = Path(archive), Path(output).resolve()
    manifest = output / 'manifest.csv'
    if manifest.exists():
        raise FileExistsError(f'Refusing to replace a frozen split: {manifest}')
    rows = []
    with zipfile.ZipFile(archive) as bundle:
        sheets = [name for name in bundle.namelist() if name.lower().endswith('.xlsx') and not name.startswith('__MACOSX/')]
        if len(sheets) != 1:
            raise ValueError('Expected exactly one BrEaST clinical XLSX file')
        clinical_bytes = bundle.read(sheets[0])
        if hashlib.sha256(clinical_bytes).hexdigest() != CLINICAL_SHA256:
            raise ValueError('Clinical workbook differs from the version used by this protocol')
        clinical = pd.read_excel(io.BytesIO(clinical_bytes))
        if len(clinical) != 256 or clinical.CaseID.nunique() != 256:
            raise ValueError('Expected all 256 unique BrEaST cases')
        members = {}
        for name in bundle.namelist():
            if name.lower().endswith('.png') and not name.startswith('__MACOSX/'):
                if Path(name).name in members:
                    raise ValueError(f'Ambiguous image filename: {Path(name).name}')
                members[Path(name).name] = name
        (output / 'images').mkdir(parents=True, exist_ok=True)
        for record in clinical.itertuples():
            name = record.Image_filename
            if Path(name).name != name or '_tumor' in name or '_other' in name:
                raise ValueError(f'Expected an unannotated primary ultrasound image: {name}')
            with Image.open(io.BytesIO(bundle.read(members[name]))) as opened:
                image = opened.convert('RGB')
            pixel_hash = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            fitted = ImageOps.contain(image, (256, 256), method=Image.Resampling.BILINEAR)
            padded = Image.new('RGB', (256, 256), 'black')
            padded.paste(fitted, ((256 - fitted.width) // 2, (256 - fitted.height) // 2))
            padded.save(output / 'images' / name)
            label = {'benign': 0, 'normal': 0, 'malignant': 1}[record.Classification]
            case = f'breast_{record.CaseID:03d}'
            rows.append(dict(
                image_id=case, patient_id=case, group_id=case, image_path=f'images/{name}',
                label=label, class_name='Malignant' if label else 'NonMalignant',
                age=pd.to_numeric(record.Age, errors='coerce'), symptoms=record.Symptoms,
                signs=record.Signs, tissue_composition=record.Tissue_composition,
                pixel_sha256=pixel_hash,
                input_sha256=hashlib.sha256(np.asarray(padded).tobytes()).hexdigest(),
            ))
    frame = pd.DataFrame(rows)
    if frame.pixel_sha256.nunique() != 256 or frame.input_sha256.nunique() != 256:
        raise ValueError('Duplicate images detected')
    # Preserve workbook order and the exact two-stage stratified split.
    train_val, test = train_test_split(frame.index, test_size=.2, stratify=frame.label, random_state=42)
    train, val = train_test_split(train_val, test_size=.25, stratify=frame.loc[train_val, 'label'], random_state=42)
    frame['split'] = ''
    for name, indices in [('train', train), ('val', val), ('test', test)]:
        frame.loc[indices, 'split'] = name
    normalization = fit_clinical_normalization(frame)
    config = dict(metadata_profile='breast_full_clinical', age_normalization=normalization)
    values = np.asarray([encode_row(row, config) for row in frame.itertuples()])
    if values.shape != (256, 22) or not np.isfinite(values).all():
        raise ValueError('Invalid clinical encoding')
    frame.to_csv(manifest, index=False)
    provenance = dict(
        dataset='BrEaST / TCIA Breast-Lesions-USG', doi='10.7937/9WKK-Q141',
        paper='https://doi.org/10.1038/s41597-024-02984-z', license='CC BY 4.0',
        clinical_sha256=CLINICAL_SHA256, seed=42,
        preprocessing='Full RGB ultrasound frame, aspect-preserving bilinear resize, black padding to 256',
        metadata_fields=FULL_CLINICAL_FIELDS, age_normalization=normalization,
        splits=frame.split.value_counts().to_dict(),
    )
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Prepared {len(frame)} cases: {manifest}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/breast'))
    args = parser.parse_args()
    prepare(args.archive, args.output)

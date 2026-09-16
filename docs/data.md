# BrEaST data protocol

## Source and labels

Use the BrEaST image/clinical collection distributed as [TCIA Breast-Lesions-USG](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) (DOI: `10.7937/9WKK-Q141`; CC BY 4.0). Dataset documentation is available in the [Scientific Data article](https://doi.org/10.1038/s41597-024-02984-z).

The binary target is `0 = NonMalignant` and `1 = Malignant`. All 256 cases are retained, including the four normal cases in class 0. The primary unannotated ultrasound image is selected using the workbook's `Image_filename`; tumor masks and other annotation images are excluded.

The full frame is converted to RGB, resized with aspect ratio preserved, and centered on a black 256-by-256 canvas. No segmentation mask or diagnosis-derived crop is used.

## Clinical features

The feature order is fixed in `m3d/metadata.py`:

```text
Age, AgeMissing, FamilyHistory, HormonalTherapy, NippleDischarge,
PersonalHistory, BreastInjury, SymptomsMissing, Palpable, SkinRetraction,
BreastScar, NippleRetraction, Redness, Warmth, PeauDOrange, SignsMissing,
TissueHomogeneousFat, TissueHomogeneousFibroglandular,
TissueHeterogeneousFat, TissueHeterogeneousFibroglandular, Lactating,
TissueMissing
```

Age is standardized using the population mean and standard deviation of observed ages from unique training patients. Missing age becomes zero after mean imputation and is marked by `AgeMissing`. Clinical categories use a predefined multi-hot vocabulary and explicit missing flags. Vocabulary is not learned from validation or test records.

The BI-RADS risk category, lesion morphology, interpretation, verification method, and diagnostic classification are not model inputs. Breast tissue composition uses the dataset's published terminology, which includes BI-RADS-style tissue descriptors; excluding the risk category does not mean excluding every term from that lexicon. This is a retrospective dataset protocol, not a claim of prospectively blinded feature collection.

## Partition and batching

The workbook's case order is preserved. `train_test_split` first holds out 20% for testing, then holds out 25% of the remaining cases for validation, stratifying by the binary target at both stages with random state 42. The resulting partitions contain 153 training, 51 validation, and 52 test patients.

The manifest checks patient IDs, group IDs, image paths, and pixel hashes for overlap across partitions. Training and validation cases are each repeated twice, in consecutive copies within each class. Only an extra training copy receives clinical Gaussian noise. Test evaluation uses one copy per case.

## Manifest schema

`prepare_data.py` generates the manifest locally; it is not distributed in the repository. Required columns are:

- Identity: `image_id`, `patient_id`, `group_id`.
- Image: `image_path`, relative to the manifest directory; `pixel_sha256`, computed from the source RGB pixels.
- Target and partition: `label`, `class_name`, `split` (`train`, `val`, or `test`).
- Raw clinical fields: `age`, `symptoms`, `signs`, `tissue_composition`.

The preparation script also writes `input_sha256` for the padded image pixels. The clinical workbook SHA256 is `89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf`.

For another dataset, create an independently frozen patient-disjoint manifest using the same feature schema and binary target convention, and pass `--manifest`. Such a run is a new dataset experiment; it is not automatically the BrEaST reproduction recipe. Do not derive clinical inputs from the target label.

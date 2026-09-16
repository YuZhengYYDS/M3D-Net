"""Predefined BrEaST clinical features with training-only age normalization."""
import numpy as np

CLINICAL_FIELDS=['Age','AgeMissing','FamilyHistory','HormonalTherapy','NippleDischarge','PersonalHistory','BreastInjury','SymptomsMissing']
SYMPTOMS=['family history of breast/ovarian cancer','HRT/hormonal contraception','nipple discharge',
          'personal history of breast cancer','breast injury']
SIGNS=['palpable','skin retraction','breast scar','nipple retraction','redness','warmth','peau d`orange']
TISSUE=['homogeneous: fat','homogeneous: fibroglandular','heterogeneous: predominantly fat',
        'heterogeneous: predominantly fibroglandular','lactating']
FULL_CLINICAL_FIELDS=CLINICAL_FIELDS+['Palpable','SkinRetraction','BreastScar','NippleRetraction',
    'Redness','Warmth','PeauDOrange','SignsMissing','TissueHomogeneousFat','TissueHomogeneousFibroglandular',
    'TissueHeterogeneousFat','TissueHeterogeneousFibroglandular','Lactating','TissueMissing']


def encode_full_clinical(age,symptoms,signs,tissue,normalization):
    values=encode_clinical(age,symptoms,normalization)
    for raw,vocabulary,allow_no in [(signs,SIGNS,True),(tissue,TISSUE,False)]:
        tokens=set(str(raw).split('&'))
        allowed=set(vocabulary+['not available']+(['no'] if allow_no else []))
        assert tokens<=allowed,tokens
        assert not (tokens & {'no','not available'} and len(tokens)>1),tokens
        values.extend(float(s in tokens) for s in vocabulary)
        values.append(float('not available' in tokens))
    assert len(values)==len(FULL_CLINICAL_FIELDS)==22
    return values


def fit_clinical_normalization(frame):
    patients=frame[frame.split=='train'].drop_duplicates('patient_id')
    ages=patients.age.to_numpy(dtype=float);observed=ages[np.isfinite(ages)]
    assert len(observed)>1 and observed.std(ddof=0)>0
    return dict(mean=float(observed.mean()),std=float(observed.std(ddof=0)),
                fit_scope='observed ages of unique training patients',patients=len(patients),
                observed_patients=len(observed),imputation='training observed-age mean')


def encode_clinical(age,symptoms,normalization):
    age=float(age);missing=not np.isfinite(age)
    age_value=0. if missing else (age-normalization['mean'])/normalization['std']
    tokens=set(str(symptoms).split('&'))
    assert tokens <= set(SYMPTOMS+['no','not available']),tokens
    assert not ('no' in tokens and len(tokens)>1)
    assert not ('not available' in tokens and len(tokens)>1)
    return [age_value,float(missing)]+[float(s in tokens) for s in SYMPTOMS]+[float('not available' in tokens)]


def encode_row(row,config):
    # Pandas Series has a .view method: always use indexed fields for Series.
    get=(lambda key,default=None:row.get(key,default)) if hasattr(row,'get') else (lambda key,default=None:getattr(row,key,default))
    if config.get('metadata_profile')=='breast_clinical':
        return encode_clinical(get('age'),get('symptoms'),config['age_normalization'])
    if config.get('metadata_profile')=='breast_full_clinical':
        return encode_full_clinical(get('age'),get('symptoms'),get('signs'),get('tissue_composition'),config['age_normalization'])
    raise ValueError('Expected breast_clinical or breast_full_clinical metadata profile')


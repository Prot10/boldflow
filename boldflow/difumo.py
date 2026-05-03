"""DiFuMo parcellation helpers.

DiFuMo atlas (Dadi et al. 2020); we ship 64/256/512 resolutions. A subset
of components are non-neural (WM tracts, ventricles, CSF, venous sinuses)
and ``exclude_non_neural=True`` removes them.
"""
from __future__ import annotations

from typing import Dict, List

# DiFuMo-64 labels in standard atlas order, after dropping the two global signal columns.

DIFUMO_64_LABELS: List[str] = [
    "Superior frontal sulcus", "Fusiform gyrus",
    "Calcarine cortex posterior", "Cingulate cortex posterior",
    "Parieto-occipital sulcus superior", "Insula antero-superior",
    "Superior temporal sulcus with angular gyrus", "Planum temporale",
    "Cerebellum Crus II",
    "Superior parts of Postcentral and Precentral gyri",
    "Transverse sinus", "Paracentral gyrus RH",
    "Superior occipital gyrus", "Cingulate gyrus mid-posterior",
    "ventricles", "Fusiform gyrus posterior",
    "Superior frontal gyrus medial", "Precuneus superior",
    "Planum polare", "Parieto-occipital sulcus middle",
    "Cerebellum I-V", "Superior fornix and isthmus",
    "Anterior Cingulate Cortex", "Descending occipital gyrus",
    "Putamen", "Cingulate gyrus mid-anterior",
    "Superior parietal lobule posterior", "Paracentral lobule",
    "Inferior occipital gyrus", "Superior rostral gyrus",
    "Calcarine sulcus anterior", "Intraparietal sulcus",
    "Superior parietal lobule anterior", "Precentral gyrus medial",
    "Lingual gyrus anterior", "Angular gyrus superior",
    "Supramarginal gyrus", "Intraparietal sulcus LH",
    "Dorsomedial prefrontal cortex antero-superior",
    "Precentral gyrus superior", "Postcentral gyrus inferior",
    "Lateral occipital cortex", "Callosomarginal sulcus",
    "Paracentral lobule superior", "Heschl’s gyrus",
    "Occipital pole", "Thalamus", "Intraparietal sulcus RH",
    "Inferior frontal sulcus", "Postcentral gyrus LH",
    "Middle frontal gyrus", "Inferior frontal gyrus",
    "Parieto-occipital sulcus anterior", "Precuneus anterior",
    "Lingual gyrus", "Superior occipital sulcus",
    "Superior parietal lobule", "Middle frontal gyrus anterior",
    "Angular gyrus inferior", "Cuneus", "Middle temporal gyrus",
    "Superior frontal gyrus", "Central sulcus", "Caudate",
]

# Non-neural component indices per resolution.

_NON_NEURAL_64: Dict[int, str] = {
    10: "Transverse sinus",          # venous sinus
    14: "ventricles",                # CSF
    21: "Superior fornix and isthmus",
}

# 256/512 lists from the paper preprocessing pipeline (WM>0.7 / CSF>0.4 / GM<0.3
# plus label patterns). Names are documentation only; code uses the index sets.
_NON_NEURAL_256: Dict[int, str] = {
    2: "CSF (between superior parietal lobule and skull)",
    4: "Superior longitudinal fasciculus II middle",
    6: "Lateral ventricles anterior horns",
    10: "CSF (between intraparietal sulcus and skull)",
    17: "CSF (between precuneus and skull)",
    18: "Dorsal visual stream superior",
    33: "Corpus callosum forceps minor LH",
    51: "CSF (between parietal lobe and skull)",
    54: "Corona radiata anterior",
    55: "Suborbital cortex",
    59: "Genu of callosal body",
    70: "Isthmus of corpus callosum",
    81: "CSF (between superior frontal gyrus anterior and skull)",
    84: "Ventral visual stream",
    91: "Frontomarginal gyrus",
    92: "Optic radiation LH",
    97: "Forceps minor RH",
    100: "Internal capsule anterior horn RH",
    102: "Lateral ventricles middle",
    110: "Retrosplenial cortex",
    113: "Corona radiata",
    130: "Suborbital cortex medial",
    134: "Superior corona radiata RH",
    135: "CSF (between middle frontal gyrus posterior and skull)",
    139: "Lateral ventricles posterior horns",
    149: "CSF (between precentral gyrus and skull)",
    152: "CSF (between parieto-occipital sulcus and skull)",
    156: "Corona radiata anterior RH",
    157: "Thalamus lateral",
    168: "Fornix anterior",
    169: "Temporal pole",
    174: "CSF (between middle frontal gyrus anterior and skull)",
    179: "CSF (between precentral gyrus superior and skull)",
    181: "CSF (between superior frontal gyrus posterior and skull)",
    197: "CSF (between paracentral lobule and skull)",
    198: "Putamen and globus pallidus",
    201: "CSF (between middle frontal gyrus and skull RH)",
    219: "Thalamic radiation mid-posterior",
    230: "Corona radiata posterior LH",
    244: "CSF (between middle frontal gyrus RH and skull)",
    245: "CSF (between precentral and postcentral gyri LH and skull)",
    246: "Superior longitudinal fasciculus I posterior RH",
    248: "Forceps major RH",
    249: "Superior longitudinal fasciculus II LH",
    253: "Superior longitudinal fasciculus I LH",
}

_NON_NEURAL_512: Dict[int, str] = {
    21: "Superior longitudinal fasciculus II LH",
    22: "Corona radiata antero-superior",
    24: "Superior longitudinal fasciculus III anterior LH",
    27: "Corpus callosum isthmus anterior",
    28: "Internal capsule posterior limb RH",
    35: "Superior longitudinal fasciculus I posterior RH",
    36: "Superior longitudinal fasciculus II middle RH",
    38: "CSF (between middle frontal gyrus and skull)",
    41: "CSF (between interhemispheric fissure and superior frontal gyrus)",
    53: "Caudate superior",
    56: "Superior longitudinal fasciculus I anterior LH",
    57: "Superior longitudinal fasciculus I anterior RH",
    62: "CSF (between middle frontal gyrus superior and skull LH)",
    71: "Globus pallidus RH",
    79: "Corona radiata posterior LH",
    88: "Corticospinal tract superior",
    97: "Arcuate fasciculus postero-inferior LH",
    100: "Temporal pole LH",
    107: "Callosomarginal sulcus middle",
    109: "Putamen superior",
    112: "Forceps minor",
    125: "Cingulum anterior",
    128: "Optic radiation LH",
    129: "Corpus callosum genu inferior",
    135: "Temporal pole",
    138: "CSF (between superior precentral gyrus and skull)",
    148: "Middle frontal sulcus anterior",
    153: "CSF (between angular gyrus and skull LH)",
    155: "Arcuate fasciculus postero-inferior RH",
    162: "Superior longitudinal fasciculus I posterior LH",
    164: "Corona radiata superior",
    167: "Third ventricle",
    172: "Superior longitudinal fasciculus II posterior RH",
    180: "CSF (between superior parietal lobule and skull RH)",
    181: "CSF (between superior frontal gyrus and skull)",
    187: "CSF (between superior frontal gyrus lateral RH and skull)",
    190: "Cingulum middle",
    193: "CSF (between superior cerebellum and limbic lobe)",
    196: "Corpus callosum genu",
    207: "Corpus callosum genu superior",
    208: "Fourth ventricle",
    209: "Calcarine sulcus middle",
    221: "Inferior longitudinal fasciculus",
    228: "Paracentral sulcus",
    244: "CSF (between middle frontal gyrus posterior and skull RH)",
    253: "Subsplenial area",
    258: "Cerebellum III superior",
    274: "CSF (between superior frontal gyrus superior and skull)",
    278: "Internal capsule posterior RH",
    307: "Globus pallidus",
    310: "CSF (between supramarginal gyrus and skull LH)",
    312: "Corticospinal tract middle RH",
    322: "Superior longitudinal fasciculus II posterior LH",
    332: "CSF (between middle frontal gyrus and skull LH)",
    344: "CSF (between middle frontal gyrus anterior and skull RH)",
    360: "CSF (between intraparietal sulcus and skull LH)",
    376: "Internal capsule middle LH",
    379: "CSF (between central sulcus and skull LH)",
    381: "CSF (between central sulcus and skull RH)",
    382: "Corona radiata anterior LH",
    391: "Retrosplenial cortex inferior",
    392: "Suborbital sulcus",
    395: "Superior longitudinal fasciculus I posterior",
    397: "CSF (between callosomarginal sulcus and skull RH)",
    406: "Lateral ventricles posterior horns",
    407: "Callosal sulcus mid-posterior",
    422: "Caudate nucleus tail",
    425: "Temporal pole RH",
    427: "CSF (between postcentral gyrus and skull)",
    429: "Corpus callosum anterior body",
    433: "Hippocampus posterior",
    439: "Superior longitudinal fasciculus I",
    443: "CSF (between intraparietal sulcus and skull RH)",
    444: "Forceps minor RH",
    445: "Forceps major",
    453: "Superior longitudinal fasciculus III posterior RH",
    455: "Superior longitudinal fasciculus III middle RH",
    457: "CSF (between superior frontal gyrus middle superior and skull)",
    463: "Lateral ventricles anterior horns",
    464: "CSF (between postcentral sulcus and skull RH)",
    468: "CSF (between superior parietal lobule posterior and skull RH)",
    471: "Optic radiation RH",
    476: "CSF (between superior parietal lobule and skull LH)",
    482: "Superior longitudinal fasciculus III anterior RH",
    488: "Corpus callosum rostrum",
    489: "Orbitofrontal cortex",
    495: "Corpus callosum isthmus posterior",
    500: "Inferior fronto-occipital fasciculus posterior LH",
    502: "CSF (between inferior frontal sulcus and skull LH)",
    505: "Corpus callosum splenium",
}

DIFUMO_NON_NEURAL: Dict[int, Dict[int, str]] = {
    64: _NON_NEURAL_64,
    256: _NON_NEURAL_256,
    512: _NON_NEURAL_512,
}


def non_neural_indices(n_rois: int) -> set[int]:
    """Return the set of non-neural component indices for a DiFuMo resolution."""
    return set(DIFUMO_NON_NEURAL.get(n_rois, {}).keys())


def normalize_apostrophes(s: str) -> str:
    """Normalize Unicode apostrophes to ASCII for consistent label matching."""
    return s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")

"""BOLDFlow: EEG-to-fMRI synthesis via conditional flow matching.

>>> from boldflow import BoldFlow
>>> model = BoldFlow()                              # ~96 M parameters
>>> prediction = model(eeg)                         # (B, 64) DiFuMo-64 ROIs
>>> samples = model.sample_ensemble(eeg, n_samples=50)   # (50, B, 64) for UQ
"""
from boldflow.model import BoldFlow

__all__ = ["BoldFlow"]
__version__ = "1.0.0"

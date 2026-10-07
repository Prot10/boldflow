"""BOLDFlow: network-level recovery of fMRI connectivity from EEG via
conditional flow matching.

>>> from boldflow import BoldFlow
>>> model = BoldFlow()                              # ~96 M parameters
>>> prediction = model(eeg)                         # (B, 256) = 4 x DiFuMo-64, one source draw
>>> samples = model.sample_ensemble(eeg, n_samples=50)   # (50, B, 256) for UQ
"""
from boldflow.model import BoldFlow

__all__ = ["BoldFlow"]
__version__ = "1.0.0"

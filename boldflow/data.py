"""Paired EEG/fMRI dataset loaders for NeuroBOLT and OpenNeuroSleep.

On-disk layout::

    data_root/
      EEG/{scan_name}_eeg.set / .fdt
      fMRI_difumo_{n_rois}/{scan_name}_difumo{n_rois}_roi.pkl

The pickle is a ``pandas`` DataFrame (timepoints x ROIs); ``global signal``
columns are dropped. The loader epochs EEG around fMRI triggers (``R149`` for
NeuroBOLT, ``R128`` for sleep), filters EEG (0.5 Hz HP) and fMRI (0.15 Hz LP),
z-scores the EEG per channel and clips it to +/-15 SD, and normalises each ROI
by per-scan absolute 95th percentile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, TensorDataset

from boldflow.difumo import non_neural_indices, normalize_apostrophes
from boldflow.splits import CVFold

logger = logging.getLogger(__name__)


# Non-EEG sensors written into the same file; dropped before training.
NEUROBOLT_EXCLUDE = ["EOG1", "EOG2", "EMG1", "EMG2", "EMG3", "ECG",
                     "CWL1", "CWL2", "CWL3", "CWL4"]
SLEEP_EXCLUDE = ["EOG", "ECG"]


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset constants for the loader."""

    name: str
    event_name: str           # MNE event marker for the fMRI trigger
    exclude_channels: List[str]
    expected_sfreq: float = 200.0


NEUROBOLT_SPEC = DatasetSpec(
    name="neurobolt", event_name="R149", exclude_channels=NEUROBOLT_EXCLUDE,
)
SLEEP_SPEC = DatasetSpec(
    name="sleep", event_name="R128", exclude_channels=SLEEP_EXCLUDE,
)
DATASETS = {"neurobolt": NEUROBOLT_SPEC, "sleep": SLEEP_SPEC}


def load_scan(
    data_root: str | Path,
    scan_name: str,
    *,
    dataset: str = "neurobolt",
    n_rois: int = 64,
    target_roi: Optional[str] = None,
    multi_roi: bool = True,
    apply_eeg_filter: bool = True,
    apply_fmri_filter: bool = True,
    normalize_eeg: bool = True,
    clip_eeg: Optional[float] = 15.0,
    tr: float = 2.1,
    tmin: float = -32.0,
    tmax: float = 0.0,
    crop: int = 6400,
    eeg_lowpass: Optional[float] = None,
    exclude_non_neural: bool = False,
    n_out_timesteps: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """Load and preprocess one paired EEG/fMRI scan.

    Returns ``(eeg_epochs, fmri_epochs, metadata)``: one entry per fMRI
    anchor TR with at least ``-tmin`` seconds of EEG history. EEG shape is
    ``(C, crop)``.

    With ``n_out_timesteps=1`` (seq2one) each fMRI target is a single volume
    of shape ``(R,)``. With ``n_out_timesteps=T>1`` (seq2seq) the target is
    the block of ``T`` consecutive volumes ending at the anchor TR, shape
    ``(T, R)``; anchors lacking ``T-1`` prior TRs are dropped.
    """
    spec = DATASETS[dataset]
    data_root = Path(data_root)

    eeg_path = data_root / "EEG" / f"{scan_name}_eeg.set"
    fmri_path = data_root / f"fMRI_difumo_{n_rois}" / f"{scan_name}_difumo{n_rois}_roi.pkl"
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG not found: {eeg_path}")
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI not found: {fmri_path}")

    try:
        import mne
    except ImportError as exc:
        raise ImportError("MNE is required for EEG loading; install via `pip install mne`") from exc

    raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
    if raw.info["sfreq"] != spec.expected_sfreq:
        raw.resample(spec.expected_sfreq, verbose=False)

    drop = [ch for ch in spec.exclude_channels if ch in raw.ch_names]
    if drop:
        raw.drop_channels(drop)

    if apply_eeg_filter:
        raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    if eeg_lowpass is not None:
        raw.filter(l_freq=None, h_freq=eeg_lowpass, verbose=False)

    fmri_df = pd.read_pickle(fmri_path)
    roi_labels = [normalize_apostrophes(c) for c in fmri_df.columns.tolist()]
    fmri_np = fmri_df.to_numpy().T.copy()  # (n_rois_total, n_timepoints)

    selected_rois: List[str]
    if multi_roi:
        roi_mask = [i for i, label in enumerate(roi_labels)
                    if "global signal" not in label.lower()]
        if exclude_non_neural:
            non_neural = non_neural_indices(n_rois)
            roi_mask = [i for i in roi_mask if i not in non_neural]
            logger.info(
                "Excluded %d non-neural components from DiFuMo-%d (kept %d)",
                len(non_neural), n_rois, len(roi_mask),
            )
        fmri_np = fmri_np[roi_mask]
        selected_rois = [roi_labels[i] for i in roi_mask]
    elif target_roi is not None:
        target_norm = normalize_apostrophes(target_roi)
        try:
            roi_idx = roi_labels.index(target_norm)
        except ValueError as exc:
            matches = [i for i, l in enumerate(roi_labels) if target_norm.lower() in l.lower()]
            if not matches:
                raise ValueError(
                    f"ROI {target_roi!r} not found; available: {roi_labels}"
                ) from exc
            roi_idx = matches[0]
            logger.info("Matched %r to %r", target_roi, roi_labels[roi_idx])
        fmri_np = fmri_np[roi_idx : roi_idx + 1]
        selected_rois = [roi_labels[roi_idx]]
    else:
        selected_rois = roi_labels

    if apply_fmri_filter:
        nyquist = 0.5 / tr
        b, a = butter(N=5, Wn=0.15 / nyquist, btype="low")
        fmri_np = filtfilt(b, a, fmri_np, axis=1)

    # Per-ROI absmax normalisation (paper convention).
    fmri_np = fmri_np - fmri_np.mean(axis=-1, keepdims=True)
    scale = np.quantile(np.abs(fmri_np), q=0.95, axis=-1, keepdims=True) + 1e-8
    fmri_np = fmri_np / scale

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if spec.event_name not in event_id:
        raise ValueError(
            f"event {spec.event_name!r} not found in {scan_name}; "
            f"available: {list(event_id.keys())}"
        )
    keep_id = {spec.event_name: event_id[spec.event_name]}
    events = mne.pick_events(events, include=keep_id[spec.event_name])

    epochs = mne.Epochs(
        raw, events, event_id=keep_id, tmin=tmin, tmax=tmax,
        preload=True, baseline=(None, None), reject=None, verbose=False,
    )
    eeg_data = epochs.get_data(units="uV")

    if normalize_eeg:
        ch_mean = eeg_data.mean(axis=(0, 2), keepdims=True)
        ch_std = eeg_data.std(axis=(0, 2), keepdims=True) + 1e-8
        eeg_data = (eeg_data - ch_mean) / ch_std
        # Clip to +/- clip_eeg standard deviations, matching the distribution
        # seen during REVE pretraining (paper App. C). This only caps rare
        # residual gradient/BCG artefacts: on NeuroBOLT it touches <0.02% of
        # samples, and most scans never reach +/-15 at all.
        if clip_eeg is not None:
            eeg_data = np.clip(eeg_data, -clip_eeg, clip_eeg)

    # Pair each EEG window with its fMRI target. ``epochs.selection`` gives the
    # fMRI-trigger index of every retained EEG epoch, in scan order.
    n_out = max(1, min(int(n_out_timesteps), 4))
    n_tp = fmri_np.shape[1]
    cropped = [s[:, :crop] if crop > 0 else s for s in eeg_data]

    eeg_epochs: List[np.ndarray] = []
    fmri_epochs: List[np.ndarray] = []
    for i, idx in enumerate(epochs.selection):
        if idx >= n_tp:
            continue                       # trigger past the end of the fMRI run
        if idx < n_out - 1:
            continue                       # seq2seq: no T_out-1 prior TRs
        eeg_epochs.append(cropped[i])
        if n_out > 1:
            # Block of T_out consecutive volumes ending at the anchor TR.
            fmri_epochs.append(fmri_np[:, idx - (n_out - 1):idx + 1].T.copy())  # (T, R)
        else:
            fmri_epochs.append(fmri_np[:, idx].copy())                          # (R,)

    metadata = {
        "scan_name": scan_name,
        "n_epochs": len(eeg_epochs),
        "n_channels": eeg_epochs[0].shape[0] if eeg_epochs else 0,
        "n_samples": eeg_epochs[0].shape[1] if eeg_epochs else 0,
        "n_rois": len(selected_rois),
        "n_out_timesteps": n_out,
        "roi_names": selected_rois,
        "channel_names": epochs.ch_names,
        "sfreq": epochs.info["sfreq"],
    }
    logger.info(
        "Loaded %s: %d anchors, %d channels, %d ROIs, T_out=%d",
        scan_name, metadata["n_epochs"], metadata["n_channels"],
        metadata["n_rois"], n_out,
    )
    return eeg_epochs, fmri_epochs, metadata


def _stack(items: List[np.ndarray]) -> torch.Tensor:
    if not items:
        return torch.empty(0)
    return torch.stack([torch.tensor(x, dtype=torch.float32) for x in items])


def create_cv_dataloaders(
    data_root: str | Path,
    fold: CVFold,
    *,
    dataset: str = "neurobolt",
    n_rois: int = 64,
    target_roi: Optional[str] = None,
    multi_roi: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_eval_batch_size: Optional[int] = None,
    **load_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Train/val/test loaders for one CV fold.

    Extra ``load_kwargs`` are forwarded to :func:`load_scan` (e.g. ``tmin``,
    ``tmax``, ``exclude_non_neural``, ``n_out_timesteps``).

    For seq2seq (``n_out_timesteps>1``) the per-anchor ``(T, R)`` targets are
    flattened to ``(T*R,)`` so the model flows in ``T*R`` space. The returned
    metadata carries ``n_out_timesteps`` and, for the val/test splits,
    ``*_scan_sizes`` (an ordered ``[(scan_name, n_anchors), ...]`` list) so the
    evaluator can overlap-average the predicted blocks per scan.
    """
    splits = {"train": fold.train_scans, "val": fold.val_scans, "test": fold.test_scans}
    eeg = {k: [] for k in splits}
    fmri = {k: [] for k in splits}
    scan_sizes = {k: [] for k in splits}
    metadata: Optional[Dict[str, Any]] = None

    for split_name, scans in splits.items():
        for scan in scans:
            scan_eeg, scan_fmri, meta = load_scan(
                data_root,
                scan,
                dataset=dataset,
                n_rois=n_rois,
                target_roi=target_roi,
                multi_roi=multi_roi,
                **load_kwargs,
            )
            eeg[split_name].extend(scan_eeg)
            fmri[split_name].extend(scan_fmri)
            scan_sizes[split_name].append((scan, len(scan_eeg)))
            metadata = metadata or meta

    train_eeg, train_fmri = _stack(eeg["train"]), _stack(fmri["train"])
    val_eeg, val_fmri = _stack(eeg["val"]), _stack(fmri["val"])
    test_eeg, test_fmri = _stack(eeg["test"]), _stack(fmri["test"])

    n_out = max(1, min(int(load_kwargs.get("n_out_timesteps", 1) or 1), 4))
    if n_out > 1:
        # Flatten the (N, T, R) blocks to (N, T*R): the model flows in T*R space.
        train_fmri = train_fmri.reshape(train_fmri.shape[0], -1)
        val_fmri = val_fmri.reshape(val_fmri.shape[0], -1)
        test_fmri = test_fmri.reshape(test_fmri.shape[0], -1)
    elif not multi_roi:
        if train_fmri.ndim == 2 and train_fmri.shape[1] == 1:
            train_fmri = train_fmri.squeeze(1)
            val_fmri = val_fmri.squeeze(1)
            test_fmri = test_fmri.squeeze(1)

    cap = max_eval_batch_size or 256
    eval_bs = min(batch_size * 4, cap)

    def loader(ds: TensorDataset, shuffle: bool, drop_last: bool, bs: int) -> DataLoader:
        return DataLoader(
            ds, batch_size=max(1, min(bs, len(ds))), shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        )

    out_meta = dict(metadata or {})
    out_meta["n_out_timesteps"] = n_out
    out_meta["val_scan_sizes"] = scan_sizes["val"]
    out_meta["test_scan_sizes"] = scan_sizes["test"]

    return (
        loader(TensorDataset(train_eeg, train_fmri), shuffle=True, drop_last=True, bs=batch_size),
        loader(TensorDataset(val_eeg, val_fmri), shuffle=False, drop_last=False, bs=eval_bs),
        loader(TensorDataset(test_eeg, test_fmri), shuffle=False, drop_last=False, bs=eval_bs),
        out_meta,
    )

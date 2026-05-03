"""Subject-level K-fold splitter.

All scans from the same subject stay on the same side of every split.
Folds are produced by permuting subjects with a fixed seed and rotating
the test segment; the remainder is split train/val by ``val_ratio``.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """A single CV fold (train/val/test partition by subject and by scan)."""

    fold_idx: int
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]
    train_scans: List[str] = field(default_factory=list)
    val_scans: List[str] = field(default_factory=list)
    test_scans: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        train, val, test = map(set, (self.train_subjects, self.val_subjects, self.test_subjects))
        if not train.isdisjoint(val):
            raise ValueError("train and val subjects overlap")
        if not train.isdisjoint(test):
            raise ValueError("train and test subjects overlap")

    @property
    def n_train_scans(self) -> int: return len(self.train_scans)
    @property
    def n_val_scans(self) -> int: return len(self.val_scans)
    @property
    def n_test_scans(self) -> int: return len(self.test_scans)


class SubjectLevelCVSplitter:
    """K-fold CV that keeps a subject's scans on the same side of every split.

    Parameters
    ----------
    data_root
        Root with ``EEG/`` and ``fMRI_difumo_{n_rois}/`` subdirectories.
    k_folds, seed, val_ratio
        Self-explanatory; paper uses 5 folds, seed=12345, val_ratio=0.2.
    dataset
        ``"neurobolt"`` or ``"sleep"``; selects the EEG filename regex.
    task_filter
        For sleep, restrict to scans containing ``task-<filter>``.
    n_rois
        DiFuMo resolution, used only to verify EEG/fMRI pairing.
    """

    _PATTERNS: Dict[str, re.Pattern] = {
        "neurobolt": re.compile(r"(sub\d+)-(.+?)_eeg\.set"),
        "sleep":     re.compile(r"(sub-\d+)_(task-\w+_run-\d+)_eeg\.set"),
    }

    def __init__(
        self,
        data_root: str | Path,
        *,
        k_folds: int = 5,
        seed: int = 42,
        val_ratio: float = 0.2,
        dataset: str = "neurobolt",
        task_filter: Optional[str] = None,
        n_rois: int = 64,
    ):
        if dataset not in self._PATTERNS:
            raise ValueError(f"unknown dataset {dataset!r}; supported: {list(self._PATTERNS)}")
        self.data_root = Path(data_root)
        self.k_folds = k_folds
        self.seed = seed
        self.val_ratio = val_ratio
        self.dataset = dataset
        self.task_filter = task_filter
        self.n_rois = n_rois
        self.eeg_pattern = self._PATTERNS[dataset]

        self.scans: List[str] = []
        self.subjects: Dict[str, List[str]] = defaultdict(list)
        self._discover_data()

        self.folds: List[CVFold] = []
        self._create_folds()

    def _discover_data(self) -> None:
        eeg_dir = self.data_root / "EEG"
        if not eeg_dir.exists():
            raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")

        fmri_dir = self.data_root / f"fMRI_difumo_{self.n_rois}"

        for f in sorted(eeg_dir.iterdir()):
            if f.suffix != ".set":
                continue
            match = self.eeg_pattern.match(f.name)
            if match is None:
                continue
            subject_id, scan_id = match.group(1), match.group(2)

            if self.dataset == "sleep":
                scan_name = f"{subject_id}_{scan_id}"
                if self.task_filter and f"task-{self.task_filter}" not in scan_name:
                    continue
                fmri_path = fmri_dir / f"{scan_name}_difumo{self.n_rois}_roi.pkl"
                if not fmri_path.exists():
                    logger.warning("Skipping %s: missing fMRI at %s", scan_name, fmri_path)
                    continue
            else:  # neurobolt
                scan_name = f"{subject_id}-{scan_id}"

            self.scans.append(scan_name)
            self.subjects[subject_id].append(scan_name)

        suffix = f" (task={self.task_filter})" if self.task_filter else ""
        logger.info("Found %d scans from %d subjects%s",
                    len(self.scans), len(self.subjects), suffix)

    def _create_folds(self) -> None:
        rng = np.random.RandomState(self.seed)
        subjects = sorted(self.subjects.keys())
        rng.shuffle(subjects)

        n = len(subjects)
        if n < self.k_folds:
            raise ValueError(f"need at least {self.k_folds} subjects, have {n}")

        fold_size = n // self.k_folds
        fold_indices = [
            list(range(i * fold_size, n if i == self.k_folds - 1 else (i + 1) * fold_size))
            for i in range(self.k_folds)
        ]

        for fold_i, test_idx in enumerate(fold_indices):
            test_subj = [subjects[i] for i in test_idx]
            remaining = [s for j, s in enumerate(subjects) if j not in set(test_idx)]
            rng.shuffle(remaining)
            n_val = max(1, int(len(remaining) * self.val_ratio))
            val_subj = remaining[:n_val]
            train_subj = remaining[n_val:]

            self.folds.append(CVFold(
                fold_idx=fold_i + 1,
                train_subjects=sorted(train_subj),
                val_subjects=sorted(val_subj),
                test_subjects=sorted(test_subj),
                train_scans=sorted(s for subj in train_subj for s in self.subjects[subj]),
                val_scans=sorted(s for subj in val_subj for s in self.subjects[subj]),
                test_scans=sorted(s for subj in test_subj for s in self.subjects[subj]),
            ))

    def get_folds(self) -> List[CVFold]: return self.folds

    def get_fold(self, fold_idx: int) -> CVFold:
        if not 1 <= fold_idx <= self.k_folds:
            raise ValueError(f"fold_idx must be between 1 and {self.k_folds}")
        return self.folds[fold_idx - 1]

    def summary(self) -> str:
        lines = [
            f"data_root: {self.data_root}",
            f"dataset: {self.dataset}",
            f"k_folds: {self.k_folds}, seed: {self.seed}",
            f"subjects: {len(self.subjects)}, scans: {len(self.scans)}",
        ]
        for fold in self.folds:
            lines.append(
                f"  fold {fold.fold_idx}: train={fold.n_train_scans} "
                f"val={fold.n_val_scans} test={fold.n_test_scans}"
            )
        return "\n".join(lines)

# Checkpoints

This directory holds:

* `reve-base.safetensors` -- pretrained REVE encoder weights (~265 MB), used
  to initialise `model.encoder`. Get them with::

      python ../scripts/download_pretrained.py

  This pulls `model.safetensors` from the HuggingFace repo `brain-bzh/reve-base`
  and renames it to `reve-base.safetensors` here.

* `boldflow_neurobolt_fold{1..5}.pt` and `boldflow_sleep_fold{1..5}.pt` --
  optional trained BoldFlow checkpoints that reproduce the paper headline
  numbers in Table 1. These are not bundled with the source release; if you
  trained your own checkpoints with `scripts/train.py`, the per-fold `best.pt`
  files live under `outputs/<run_name>/fold_<i>/best.pt` and you can copy
  them here to run `scripts/evaluate.py` and `scripts/run_uncertainty.py`.

The `.gitignore` excludes the actual checkpoint blobs so this directory only
ever contains this README in the source tree.

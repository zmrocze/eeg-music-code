from typing import Dict, List, Optional, Tuple

import numpy as np
from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import (
  Preprocessor,
  create_windows_from_events,
  preprocess,
)
from sklearn.preprocessing import scale as standard_scale
from torch.utils.data import Dataset


def load_sleepedf_dataset(
  path: str,
  subject_ids: List[int],
  recording_ids: Optional[List[int]] = None,
  crop_wake_mins: int = 30,
  high_cut_hz: float = 30,
  window_size_s: int = 30,
  sfreq: int = 100,
  preload: bool = True,
  channel_wise_scale: bool = True,
) -> Tuple[Dataset, Dict[str, int]]:
  """
  Load SleepEDF dataset using braindecode and prepare it as PyTorch dataset.

  Args:
      path: Path to the dataset directory (not used by braindecode, downloads automatically)
      subject_ids: List of subject IDs to load (0-indexed)
      recording_ids: List of recording IDs to load (default: [2] for night 2)
      crop_wake_mins: Minutes of wake time to crop from start/end
      high_cut_hz: High-pass filter cutoff frequency
      window_size_s: Window size in seconds for epoching
      sfreq: Target sampling frequency
      preload: Whether to preload data into memory
      channel_wise_scale: Whether to apply channel-wise z-score normalization

  Returns:
      windows_dataset: Braindecode WindowsDataset (PyTorch compatible)
      mapping: Dictionary mapping sleep stage names to integer labels
  """
  if recording_ids is None:
    recording_ids = [2]

  dataset = SleepPhysionet(
    subject_ids=subject_ids,
    recording_ids=recording_ids,
    crop_wake_mins=crop_wake_mins,
  )

  factor = 1e6
  preprocessors = [
    Preprocessor(lambda data: np.multiply(data, factor), apply_on_array=True),
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz),
  ]

  preprocess(dataset, preprocessors)

  mapping = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
  }

  window_size_samples = window_size_s * sfreq

  windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    preload=preload,
    mapping=mapping,
  )

  if channel_wise_scale:
    preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])

  return windows_dataset, mapping

"""
Module for creating train/val/test dataloaders for sleep EEG data with subject-based splitting.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from braindecode.datasets import BaseConcatDataset, SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events
from typing import Tuple, Dict, Optional


class SleepStageDataset(Dataset):
  """
  Dataset wrapper that returns windows with labels and subject IDs.

  Args:
      windows_dataset: Braindecode windows dataset
      n_channels: Number of EEG channels to use
  """

  def __init__(self, windows_dataset, n_channels):
    self.windows_dataset = windows_dataset
    self.n_channels = n_channels

  def __len__(self):
    return len(self.windows_dataset)

  def __getitem__(self, idx):
    """
    Returns:
        window: EEG window of shape (n_channels, window_length)
        label: Sleep stage label (0-5)
        subject_id: Subject identifier
    """
    X, y, window_ind = self.windows_dataset[idx]

    # Find which dataset this window belongs to by using idx
    dataset_idx = 0
    cumsum = 0
    for i, ds in enumerate(self.windows_dataset.datasets):
      if idx < cumsum + len(ds):
        dataset_idx = i
        break
      cumsum += len(ds)

    subject_id = self.windows_dataset.datasets[dataset_idx].description["subject"]

    # Select only the specified number of channels
    X = X[: self.n_channels, :]

    return X, y, subject_id


def create_sleep_dataloaders(
  windows_dataset,
  window_length,
  n_channels,
  batch_size=32,
  test_size=0.4,
  val_split=0.5,
  random_state=42,
  num_workers=0,
):
  """
  Create train, validation, and test dataloaders with subject-based splitting.

  Args:
      windows_dataset: Braindecode BaseConcatDataset with windowed EEG data
      window_length: Length of each window in samples (used for validation)
      n_channels: Number of EEG channels to load
      batch_size: Batch size for dataloaders
      test_size: Proportion of subjects for test+validation (default: 0.4)
      val_split: Proportion of test+val subjects for validation (default: 0.5)
      random_state: Random seed for reproducibility
      num_workers: Number of workers for dataloaders

  Returns:
      train_loader: DataLoader for training set
      val_loader: DataLoader for validation set
      test_loader: DataLoader for test set

  Each batch contains:
      - windows: Tensor of shape (batch_size, n_channels, window_length)
      - labels: Tensor of shape (batch_size,) with sleep stage labels (0-5)
      - subject_ids: List of subject identifiers
  """

  # Get unique subjects
  subjects = np.unique(windows_dataset.description["subject"])

  # Split subjects into train, validation, and test
  subj_train, subj_test = train_test_split(
    subjects, test_size=test_size, random_state=random_state
  )
  subj_val, subj_test = train_test_split(
    subj_test, test_size=val_split, random_state=random_state
  )

  # Create datasets for each split
  split_ids = {"train": subj_train, "val": subj_val, "test": subj_test}
  datasets = {}

  for name, subject_list in split_ids.items():
    # Filter datasets by subject
    filtered_datasets = [
      ds for ds in windows_dataset.datasets if ds.description["subject"] in subject_list
    ]
    concat_dataset = BaseConcatDataset(filtered_datasets)
    datasets[name] = SleepStageDataset(concat_dataset, n_channels)

  # Create dataloaders
  train_loader = DataLoader(
    datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
  )

  val_loader = DataLoader(
    datasets["val"],
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
  )

  test_loader = DataLoader(
    datasets["test"],
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
  )

  print(f"Train subjects: {len(subj_train)}, samples: {len(datasets['train'])}")
  print(f"Val subjects: {len(subj_val)}, samples: {len(datasets['val'])}")
  print(f"Test subjects: {len(subj_test)}, samples: {len(datasets['test'])}")

  return train_loader, val_loader, test_loader


def load_and_create_sleep_dataloaders(
  subject_ids: list[int],
  window_size_s: int,
  sfreq: int,
  n_channels: int,
  batch_size: int = 32,
  test_size: float = 0.4,
  val_split: float = 0.5,
  random_state: int = 42,
  num_workers: int = 0,
  l_freq: float = 0.5,
  h_freq: float = 30.0,
  recording_ids: list[int] | None = None,
  crop_wake_mins: int = 30,
  sleep_stage_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
  """
  Load Sleep Physionet dataset, preprocess, create windows, and return dataloaders.

  This function combines all steps:
  1. Load SleepPhysionet dataset
  2. Preprocess (filter, resample)
  3. Create windows from sleep stage events
  4. Split by subjects into train/val/test
  5. Create dataloaders

  Args:
      subject_ids: List of subject IDs to load
      recording_ids: List of recording IDs (1 or 2 for first/second night)
      crop_wake_mins: Minutes of wake to crop before/after sleep
      window_size_s: Window size in seconds
      sfreq: Target sampling frequency in Hz
      n_channels: Number of EEG channels to use
      batch_size: Batch size for dataloaders
      test_size: Proportion of subjects for test+validation
      val_split: Proportion of test+val subjects for validation
      random_state: Random seed for reproducibility
      num_workers: Number of workers for dataloaders
      l_freq: Low frequency for bandpass filter (Hz)
      h_freq: High frequency for bandpass filter (Hz)
      sleep_stage_mapping: Custom sleep stage to label mapping. If None, uses default:
          {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
           "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}

  Returns:
      Tuple of (train_loader, val_loader, test_loader)

  Each batch contains:
      - windows: Tensor of shape (batch_size, n_channels, window_length)
      - labels: Tensor of shape (batch_size,) with sleep stage labels
      - subject_ids: List of subject identifiers
  """
  # 1. Load dataset
  print(f"Loading SleepPhysionet dataset for subjects {subject_ids}...")
  dataset = SleepPhysionet(
    subject_ids=subject_ids, recording_ids=recording_ids, crop_wake_mins=crop_wake_mins
  )
  print(f"Loaded {len(dataset)} recordings")

  # 2. Preprocess
  print(f"Preprocessing: filtering ({l_freq}-{h_freq} Hz), resampling to {sfreq} Hz...")
  preprocessors = [
    Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
    Preprocessor("resample", sfreq=sfreq),
  ]
  preprocess(dataset, preprocessors)
  print("Preprocessing complete")

  # 3. Create windows
  window_size_samples = window_size_s * sfreq

  if sleep_stage_mapping is None:
    sleep_stage_mapping = {
      "Sleep stage W": 0,
      "Sleep stage 1": 1,
      "Sleep stage 2": 2,
      "Sleep stage 3": 3,
      "Sleep stage 4": 3,  # Merge stage 3 and 4 (AASM standard)
      "Sleep stage R": 4,
    }

  print(f"Creating {window_size_s}s windows ({window_size_samples} samples)...")
  windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,  # Non-overlapping
    preload=True,
    mapping=sleep_stage_mapping,
  )
  print(f"Created {len(windows_dataset)} windows")

  # 4. Create dataloaders with subject-based splitting
  return create_sleep_dataloaders(
    windows_dataset=windows_dataset,
    window_length=window_size_samples,
    n_channels=n_channels,
    batch_size=batch_size,
    test_size=test_size,
    val_split=val_split,
    random_state=random_state,
    num_workers=num_workers,
  )

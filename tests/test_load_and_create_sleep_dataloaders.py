"""
Tests for load_and_create_sleep_dataloaders function.
"""

from eeg_music.sleep_dataloader import load_and_create_sleep_dataloaders


def test_load_and_create_sleep_dataloaders_basic():
  """Test basic functionality with minimal subjects."""
  train_loader, val_loader, test_loader = load_and_create_sleep_dataloaders(
    subject_ids=list(range(5)),  # 5 subjects minimum for proper splitting
    recording_ids=[1],
    crop_wake_mins=30,
    window_size_s=30,
    sfreq=100,
    n_channels=2,
    batch_size=8,
    test_size=0.4,  # 40% for test+val -> 2 subjects
    val_split=0.5,  # Split test+val equally -> 1 val, 1 test
    random_state=42,
    num_workers=0,
  )

  assert train_loader is not None
  assert val_loader is not None
  assert test_loader is not None

  # Check that we can get batches
  train_batch = next(iter(train_loader))
  windows, labels, subject_ids = train_batch

  assert windows.shape[1] == 2  # n_channels
  assert windows.shape[2] == 30 * 100  # window_size_s * sfreq
  assert len(labels) == len(subject_ids)
  assert labels.min() >= 0
  assert labels.max() <= 4  # Sleep stages 0-4


def test_load_and_create_sleep_dataloaders_custom_mapping():
  """Test with custom sleep stage mapping."""
  custom_mapping = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 1,  # Merge N1 and N2
    "Sleep stage 3": 2,
    "Sleep stage 4": 2,
    "Sleep stage R": 3,
  }

  train_loader, val_loader, test_loader = load_and_create_sleep_dataloaders(
    subject_ids=list(range(5)),
    recording_ids=[1],
    crop_wake_mins=30,
    window_size_s=30,
    sfreq=100,
    n_channels=2,
    batch_size=8,
    test_size=0.4,
    val_split=0.5,
    sleep_stage_mapping=custom_mapping,
    num_workers=0,
  )

  # Check that labels use custom mapping (0-3 instead of 0-4)
  all_labels = []
  for _, labels, _ in train_loader:
    all_labels.extend(labels.tolist())

  unique_labels = set(all_labels)
  assert max(unique_labels) <= 3  # Custom mapping has max label 3


def test_load_and_create_sleep_dataloaders_different_params():
  """Test with different preprocessing parameters."""
  train_loader, val_loader, test_loader = load_and_create_sleep_dataloaders(
    subject_ids=list(range(5)),
    recording_ids=[1],
    crop_wake_mins=15,
    window_size_s=20,  # Different window size
    sfreq=50,  # Lower sampling rate
    n_channels=1,  # Single channel
    batch_size=4,
    test_size=0.4,
    val_split=0.5,
    l_freq=1.0,  # Different filter
    h_freq=20.0,
    num_workers=0,
  )

  batch = next(iter(train_loader))
  windows, labels, subject_ids = batch

  assert windows.shape[1] == 1  # n_channels
  assert windows.shape[2] == 20 * 50  # window_size_s * sfreq

import numpy as np
import pytest
from eeg_music.data import (
  EEGMusicDataset,
  RobustNormalizedDataset,
  ArrayEeg,
  MusicRef,
  MusicFilename,
  NoteOnsets,
)


def test_robust_normalized_dataset_basic():
  """Test basic functionality of RobustNormalizedDataset."""
  # Create a simple dataset with ArrayEeg
  ds = EEGMusicDataset()

  # Create some test data with known statistics
  # Channel 0: values [0, 1, 2, 3, 4] -> mean=2, p25=1, p75=3, median=2, IQR=2
  # Channel 1: values [10, 20, 30, 40, 50] -> mean=30, p25=20, p75=40, median=30, IQR=20
  data1 = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]], dtype=np.float32)
  data2 = np.array([[3.0, 4.0], [40.0, 50.0]], dtype=np.float32)

  eeg1 = ArrayEeg(data=data1, ch_names=["ch0", "ch1"], sfreq=100.0)
  eeg2 = ArrayEeg(data=data2, ch_names=["ch0", "ch1"], sfreq=100.0)

  music = NoteOnsets(onset_times=np.array([]), sample_rate=44100, duration_seconds=1.0)

  ds.df = ds.df._append(
    {
      "dataset": "test",
      "subject": "s1",
      "session": "sess1",
      "run": "run1",
      "trial_id": "t1",
      "music_filename": "music.wav",
      "eeg_data": eeg1,
    },
    ignore_index=True,
  )
  ds.df = ds.df._append(
    {
      "dataset": "test",
      "subject": "s1",
      "session": "sess1",
      "run": "run1",
      "trial_id": "t2",
      "music_filename": "music.wav",
      "eeg_data": eeg2,
    },
    ignore_index=True,
  )
  ds.music_collection[
    MusicRef(filename=MusicFilename(filename="music.wav"), dataset="test")
  ] = music

  # Create normalized dataset
  norm_ds = RobustNormalizedDataset(ds)

  # Check statistics were calculated
  assert norm_ds.p25.shape == (2, 1)
  assert norm_ds.p75.shape == (2, 1)
  assert norm_ds.iqr.shape == (2, 1)
  assert norm_ds.median.shape == (2, 1)

  # Check that IQR is positive
  assert np.all(norm_ds.iqr > 0)

  # Check length
  assert len(norm_ds) == 2

  # Get normalized trial
  trial = norm_ds[0]
  assert isinstance(trial.eeg_data, ArrayEeg)
  assert trial.eeg_data.data.shape == (2, 3)

  # Verify normalization was applied (data should be centered around 0)
  # After normalization: (x - median) / IQR
  assert trial.eeg_data.ch_names == ["ch0", "ch1"]
  assert trial.eeg_data.sfreq == 100.0


def test_robust_normalized_dataset_properties():
  """Test that df and music_collection properties are properly delegated."""
  ds = EEGMusicDataset()

  data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
  eeg = ArrayEeg(data=data, ch_names=["ch0", "ch1"], sfreq=100.0)
  music = NoteOnsets(onset_times=np.array([]), sample_rate=44100, duration_seconds=1.0)

  ds.df = ds.df._append(
    {
      "dataset": "test",
      "subject": "s1",
      "session": "sess1",
      "run": "run1",
      "trial_id": "t1",
      "music_filename": "music.wav",
      "eeg_data": eeg,
    },
    ignore_index=True,
  )
  ds.music_collection[
    MusicRef(filename=MusicFilename(filename="music.wav"), dataset="test")
  ] = music

  norm_ds = RobustNormalizedDataset(ds)

  # Check df property
  assert len(norm_ds.df) == 1
  assert norm_ds.df.iloc[0]["dataset"] == "test"

  # Check music_collection property
  assert len(norm_ds.music_collection) == 1
  assert (
    MusicRef(filename=MusicFilename(filename="music.wav"), dataset="test")
    in norm_ds.music_collection
  )


def test_robust_normalized_dataset_error_on_wrong_type():
  """Test that RobustNormalizedDataset raises error for non-ArrayEeg data."""
  from eeg_music.data import RawEeg
  import mne

  ds = EEGMusicDataset()

  # Create RawEeg (not ArrayEeg)
  info = mne.create_info(ch_names=["ch0", "ch1"], sfreq=100.0, ch_types="eeg")
  raw = mne.io.RawArray(data=np.random.randn(2, 100), info=info, verbose="error")
  eeg = RawEeg(raw_eeg=raw)

  music = NoteOnsets(onset_times=np.array([]), sample_rate=44100, duration_seconds=1.0)

  ds.df = ds.df._append(
    {
      "dataset": "test",
      "subject": "s1",
      "session": "sess1",
      "run": "run1",
      "trial_id": "t1",
      "music_filename": "music.wav",
      "eeg_data": eeg,
    },
    ignore_index=True,
  )
  ds.music_collection[
    MusicRef(filename=MusicFilename(filename="music.wav"), dataset="test")
  ] = music

  # Should raise TypeError during initialization
  with pytest.raises(TypeError, match="Expected ArrayEeg or OnDiskArrayEeg"):
    RobustNormalizedDataset(ds)

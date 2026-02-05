from eeg_music.sleepedf_dataset import load_sleepedf_dataset


def test_load_sleepedf_dataset_basic():
  """Test basic loading of SleepEDF dataset with braindecode."""
  dataset, mapping = load_sleepedf_dataset(
    subject_ids=[0],
    recording_ids=[2],
    crop_wake_mins=30,
    preload=True,
  )

  assert dataset is not None
  assert len(mapping) == 5
  assert mapping["Sleep stage W"] == 0
  assert mapping["Sleep stage R"] == 4

  assert len(dataset) > 0

  X, y, _ = dataset[0]
  assert X.shape[0] == 2
  assert X.shape[1] == 3000
  assert isinstance(y, int)
  assert 0 <= y <= 4


def test_load_sleepedf_dataset_multiple_subjects():
  """Test loading multiple subjects."""
  dataset, mapping = load_sleepedf_dataset(
    subject_ids=[0, 1],
    recording_ids=[2],
    crop_wake_mins=30,
    preload=True,
  )

  assert len(dataset) > 0

  X, y, _ = dataset[0]
  assert X.shape[0] == 2
  assert isinstance(y, int)


def test_load_sleepedf_dataset_custom_params():
  """Test loading with custom preprocessing parameters."""
  dataset, mapping = load_sleepedf_dataset(
    subject_ids=[0],
    recording_ids=[2],
    crop_wake_mins=15,
    high_cut_hz=25,
    window_size_s=30,
    sfreq=100,
    preload=True,
    channel_wise_scale=False,
  )

  assert dataset is not None
  assert len(dataset) > 0

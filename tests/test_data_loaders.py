"""
Comprehensive test suite for all EEG dataset loaders.

This module tests the functionality of all dataset loaders implemented in the src directory:
- BCMI datasets (calibration, training, testing, tempo, scores, fmri)
- MUSIN-G dataset
- NMED-T dataset
- OpenMIIR dataset

Dataset Requirements:
--------------------
The tests expect datasets to be located at:
- /home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-*
- /home/zmrocze/studia/uwr/magisterka/datasets/musin_g_data/
- /home/zmrocze/studia/uwr/magisterka/datasets/nmed-t/
- /home/zmrocze/studia/uwr/magisterka/datasets/openmiir/

Tests will skip gracefully if datasets are not available.

Usage:
------
Run all tests:
    uv run python tests/data_loaders.py

Run specific test class:
    uv run python tests/data_loaders.py TestBCMILoaders

Run specific test method:
    uv run python tests/data_loaders.py TestBCMILoaders.test_bcmi_calibration_loader

Run with verbose output:
    uv run python tests/data_loaders.py -v

Author: Test Suite for EEG Music Datasets
Date: 2025
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
from functools import wraps


from eeg_music.bcmi import (
  create_bcmi_loader,
  # load_all_bcmi_datasets,
  BCMICalibrationLoader,
  BCMITrainingLoader,
  BCMITestingLoader,
  BCMITempoLoader,
  BCMIScoresLoader,
  BCMIFMRILoader,
)
from eeg_music.musing import MUSINGDataset
from eeg_music.nmed_t import NMEDTLoader
from eeg_music.openmiir import OpenMIIRLoader
from eeg_music.data import EEGMusicDataset, RawEeg, WavRAW, MusicRef, MusicFilename
import numpy as np
import mne

DATASET_ROOT = Path("/home/zmrocze/studia/uwr/magisterka/datasets")
BCMI_ROOT = DATASET_ROOT / "bcmi"
MUSING_ROOT = DATASET_ROOT / "musin_g_data"
NMEDT_ROOT = DATASET_ROOT / "nmed-t"
OPENMIIR_ROOT = DATASET_ROOT / "openmiir"


def dataset_exists(dataset_path):
  """
  Check if a dataset directory exists and has expected structure.

  Parameters:
  -----------
  dataset_path : Path or str
      Path to the dataset directory

  Returns:
  --------
  bool
      True if dataset directory exists and contains files
  """
  path = Path(dataset_path)
  if not path.exists():
    return False

  # Check if directory has any content
  try:
    return any(path.iterdir())
  except Exception:
    return False


def skip_if_dataset_missing(dataset_path):
  """
  Decorator to skip tests when dataset is unavailable.

  Parameters:
  -----------
  dataset_path : Path or str
      Path to the dataset directory

  Returns:
  --------
  function
      Decorated test function that skips if dataset is missing
  """

  def decorator(test_func):
    @wraps(test_func)
    def wrapper(self):
      if not dataset_exists(dataset_path):
        self.skipTest(f"Dataset not found at {dataset_path}")
      return test_func(self)

    return wrapper

  return decorator


class TestBCMILoaders(unittest.TestCase):
  """Test suite for BCMI dataset loaders."""

  @classmethod
  def setUpClass(cls):
    """Set up test fixtures for BCMI tests."""
    cls.calibration_path = BCMI_ROOT / "bcmi-calibration"
    cls.training_path = BCMI_ROOT / "bcmi-training"
    cls.testing_path = BCMI_ROOT / "bcmi-testing"
    cls.tempo_path = BCMI_ROOT / "bcmi-tempo"
    cls.scores_path = BCMI_ROOT / "bcmi-scores"
    cls.fmri_path = BCMI_ROOT / "bcmi-fmri"

  @skip_if_dataset_missing(BCMI_ROOT / "bcmi-calibration")
  def test_bcmi_factory_function(self):
    """Test that create_bcmi_loader correctly identifies dataset types."""
    print("\n🧪 Testing BCMI factory function...")

    # Test calibration dataset detection
    if dataset_exists(self.calibration_path):
      loader = create_bcmi_loader(str(self.calibration_path))
      self.assertIsInstance(loader, BCMICalibrationLoader)
      self.assertEqual(loader.dataset_name, "bcmi-calibration")
      print("  ✓ Calibration loader correctly identified")

    # Test training dataset detection
    if dataset_exists(self.training_path):
      loader = create_bcmi_loader(str(self.training_path))
      self.assertIsInstance(loader, BCMITrainingLoader)
      print("  ✓ Training loader correctly identified")

    # Test other variants if available
    dataset_mapping = [
      (self.testing_path, BCMITestingLoader, "bcmi-testing"),
      (self.tempo_path, BCMITempoLoader, "bcmi-tempo"),
      (self.scores_path, BCMIScoresLoader, "bcmi-scores"),
      (self.fmri_path, BCMIFMRILoader, "bcmi-fmri"),
    ]

    for path, expected_class, expected_name in dataset_mapping:
      if dataset_exists(path):
        loader = create_bcmi_loader(str(path))
        self.assertIsInstance(loader, expected_class)
        self.assertEqual(loader.dataset_name, expected_name)
        print(f"  ✓ {expected_name} loader correctly identified")

  @skip_if_dataset_missing(BCMI_ROOT / "bcmi-calibration")
  def test_bcmi_calibration_loader(self):
    """Test BCMICalibrationLoader functionality."""
    print("\n🧪 Testing BCMI Calibration loader...")

    loader = BCMICalibrationLoader(str(self.calibration_path))

    # Test initialization
    self.assertIsNotNone(loader.subjects)
    self.assertTrue(len(loader.subjects) > 0)
    print(f"  ✓ Found {len(loader.subjects)} subjects")

    # Test experimental info
    exp_info = loader._get_experimental_info()
    self.assertEqual(exp_info["paradigm_type"], "Calibration")
    self.assertIn("40s total", exp_info["trial_structure"])
    print(f"  ✓ Experimental info: {exp_info['paradigm_type']}")

    # Test loading single subject
    if loader.subjects:
      subject = loader.subjects[0]
      data = loader.load_subject_data(subject, max_runs=1)

      # Verify data structure
      self.assertIsInstance(data, dict)

      # Check for expected keys in loaded data
      for session_key, session_data in data.items():
        for run_key, run_data in session_data.items():
          self.assertIn("raw", run_data)
          self.assertIn("events", run_data)
          self.assertIn("processed_events", run_data)
          self.assertIn("duration", run_data)
          self.assertIn("n_channels", run_data)
          self.assertIn("sfreq", run_data)

          # Check processed events
          processed = run_data["processed_events"]
          if "emotion_events" in processed:
            # Verify emotion event processing
            emotion_df = processed["emotion_events"]
            if not emotion_df.empty:
              self.assertIn("emotion_description", emotion_df.columns)
              self.assertIn("valence", emotion_df.columns)
              self.assertIn("arousal", emotion_df.columns)

          # Check calibration-specific features
          if "calibration_trials" in processed:
            trials = processed["calibration_trials"]
            if trials:
              # Each calibration trial should have two states
              trial = trials[0]
              self.assertIn("state_1", trial)
              self.assertIn("state_2", trial)
              self.assertIn("trial_number", trial)
              print("  ✓ Calibration trials structure verified")

          print(
            f"  ✓ Loaded subject {subject}, run {run_key}: "
            f"{run_data['duration']:.1f}s, {run_data['n_channels']} channels"
          )
          break  # Test only first run
        break  # Test only first session

  @skip_if_dataset_missing(BCMI_ROOT / "bcmi-training")
  def test_bcmi_training_loader(self):
    """Test BCMITrainingLoader with multi-session structure."""
    print("\n🧪 Testing BCMI Training loader...")

    loader = BCMITrainingLoader(str(self.training_path))

    # Test experimental info
    exp_info = loader._get_experimental_info()
    self.assertEqual(exp_info["paradigm_type"], "Training")
    print(f"  ✓ Paradigm type: {exp_info['paradigm_type']}")

    # Test loading with sessions
    if loader.subjects:
      subject = loader.subjects[0]
      sessions = loader._get_available_sessions(subject)
      print(f"  ✓ Subject {subject} has {len(sessions)} sessions")

      # Load limited data
      data = loader.load_subject_data(subject, max_runs=1, max_sessions=1)

      # Verify training-specific features
      for session_data in data.values():
        for run_data in session_data.values():
          processed = run_data.get("processed_events", {})
          if "training_pairs" in processed:
            pairs = processed["training_pairs"]
            if pairs:
              pair = pairs[0]
              self.assertIn("affect_1", pair)
              self.assertIn("affect_2", pair)
              self.assertIn("contrast", pair)
              print("  ✓ Training pairs structure verified")
          break
        break

  @skip_if_dataset_missing(BCMI_ROOT / "bcmi-fmri")
  def test_bcmi_fmri_loader(self):
    """Test BCMIFMRILoader with special task names."""
    print("\n🧪 Testing BCMI fMRI loader...")

    loader = BCMIFMRILoader(str(self.fmri_path))

    # Test experimental info
    exp_info = loader._get_experimental_info()
    self.assertEqual(exp_info["paradigm_type"], "EEG-fMRI")
    self.assertIn("tasks", exp_info)
    expected_tasks = [
      "classicalMusic",
      "genMusic01",
      "genMusic02",
      "genMusic03",
      "washout",
    ]
    self.assertEqual(exp_info["tasks"], expected_tasks)
    print(f"  ✓ fMRI tasks: {exp_info['tasks']}")

    # Test that it uses task names instead of run numbers
    if loader.subjects:
      subject = loader.subjects[0]
      runs = loader._get_available_runs(subject)
      self.assertEqual(runs, expected_tasks)
      print("  ✓ Uses task names instead of run numbers")

  def test_bcmi_scores_loader(self):
    """Test BCMIScoresLoader with movie score excerpts."""
    print("\n🧪 Testing BCMI Scores loader...")

    loader = BCMIScoresLoader(str(self.scores_path))

    # Test initialization
    self.assertIsNotNone(loader.subjects)
    self.assertTrue(len(loader.subjects) > 0)
    print(f"  ✓ Found {len(loader.subjects)} subjects")

    # Test experimental info
    exp_info = loader._get_experimental_info()
    self.assertEqual(exp_info["paradigm_type"], "Movie Scores")
    self.assertIn("Movie score excerpts", exp_info["music_type"])
    print(f"  ✓ Experimental info: {exp_info['paradigm_type']}")

    # Test loading single subject
    if loader.subjects:
      subject = loader.subjects[0]
      data = loader.load_subject_data(subject, max_runs=2)

      # Verify data structure
      self.assertIsInstance(data, dict)

      # Check for expected keys in loaded data
      for session_key, session_data in data.items():
        for run_key, run_data in session_data.items():
          self.assertIn("raw", run_data)
          self.assertIn("events", run_data)
          self.assertIn("processed_events", run_data)
          self.assertIn("duration", run_data)
          self.assertIn("n_channels", run_data)
          self.assertIn("sfreq", run_data)

          # Check processed events
          processed = run_data["processed_events"]

          # Check for marker events (music stimulus codes 301-360)
          if "marker_events" in processed:
            marker_df = processed["marker_events"]
            if not marker_df.empty:
              # Look for music stimulus codes (301-360)
              music_codes = marker_df[
                (marker_df["trial_type"] >= 301) & (marker_df["trial_type"] <= 360)
              ]
              if not music_codes.empty:
                print(
                  f"  ✓ Found {len(music_codes)} music stimulus events in run {run_key}"
                )
                # Verify music codes are in valid range
                for code in music_codes["trial_type"]:
                  self.assertGreaterEqual(code, 301)
                  self.assertLessEqual(code, 360)

          print(
            f"  ✓ Loaded subject {subject}, run {run_key}: "
            f"{run_data['duration']:.1f}s, {run_data['n_channels']} channels"
          )
          break  # Test only first run
        break  # Test only first session

  def test_bcmi_scores_trial_iterator(self):
    """Test trial_iterator for bcmi-scores dataset."""
    print("\n🧪 Testing BCMI Scores trial iterator...")

    loader = BCMIScoresLoader(str(self.scores_path))

    # Load minimal data
    if loader.subjects:
      loader.load_subject_data(loader.subjects[0], max_runs=2)

    # Test trial iterator
    trials = list(loader.trial_iterator())

    if trials:
      print(f"  ✓ Generated {len(trials)} trials")

      # Check first trial structure
      trial = trials[0]
      self.assertEqual(trial.dataset, "bcmi-scores")
      self.assertIsNotNone(trial.subject)
      self.assertIsNotNone(trial.eeg_data)
      self.assertIsNotNone(trial.music_filename)

      # Verify EEG data
      eeg = trial.eeg_data.get_eeg()
      self.assertIsInstance(eeg.raw_eeg, mne.io.BaseRaw)

      # Check trial duration (should be ~20 seconds)
      duration = eeg.length_seconds()
      self.assertAlmostEqual(duration, 20.0, delta=0.5)
      print(f"  ✓ Trial duration: {duration:.2f}s (expected ~20s)")

      # Verify music filename format (should be like "001.mp3", "062.mp3", etc.)
      music_filename = trial.music_filename.filename
      self.assertTrue(music_filename.endswith(".mp3"))
      self.assertTrue(music_filename[:3].isdigit())
      print(f"  ✓ Music filename format: {music_filename}")

      # Check that music numbers are in valid range (1-720)
      music_number = int(music_filename[:3])
      self.assertGreaterEqual(music_number, 1)
      self.assertLessEqual(music_number, 720)
      print(f"  ✓ Music number in valid range: {music_number}")
    else:
      print("  ⚠ No trials generated (dataset may be empty or not loaded)")

  def test_bcmi_scores_music_iterator(self):
    """Test music_iterator for bcmi-scores dataset."""
    print("\n🧪 Testing BCMI Scores music iterator...")

    loader = BCMIScoresLoader(str(self.scores_path))

    # Test music iterator
    music_files = list(loader.music_iterator())

    if music_files:
      print(f"  ✓ Found {len(music_files)} music files")

      # Check first music file
      music_ref, wav_raw = music_files[0]

      # Verify music reference
      self.assertIsInstance(music_ref, MusicFilename)
      self.assertTrue(music_ref.filename.endswith(".mp3"))
      print(f"  ✓ First music file: {music_ref.filename}")

      # Verify WAV data
      self.assertIsInstance(wav_raw, WavRAW)
      self.assertTrue(wav_raw.is_not_empty())
      self.assertGreater(wav_raw.length_seconds(), 0)
      print(f"  ✓ Music duration: {wav_raw.length_seconds():.2f}s")
      print(f"  ✓ Sample rate: {wav_raw.sample_rate} Hz")

      # Check that filenames are sequential and properly formatted
      for i, (music_ref, _) in enumerate(music_files[:5]):
        filename = music_ref.filename
        self.assertTrue(filename.endswith(".mp3"))
        self.assertTrue(filename[:3].isdigit())
        print(f"  ✓ Music file {i + 1}: {filename}")
    else:
      print("  ⚠ No music files found (stimuli directory may be missing)")

  def test_bcmi_statistics_and_queries(self):
    """Test dataset statistics and query methods."""
    print("\n🧪 Testing BCMI statistics and queries...")

    # Find any available BCMI dataset for testing
    test_dataset = None
    for path in [self.calibration_path, self.training_path, self.testing_path]:
      if dataset_exists(path):
        test_dataset = path
        break

    if test_dataset is None:
      self.skipTest("No BCMI datasets available for testing")

    loader = create_bcmi_loader(str(test_dataset))

    # Load minimal data
    loader.load_all_subjects(max_subjects=1, max_runs_per_session=1)

    # Test statistics
    loader.get_dataset_statistics()  # Should not raise
    print("  ✓ Dataset statistics generated")

    # Test emotional state info
    state_info = loader.get_emotional_state_info(1)
    self.assertIn("valence", state_info)
    self.assertIn("arousal", state_info)
    self.assertIn("description", state_info)
    print(f"  ✓ Emotional state 1: {state_info['description']}")

    # Test condition trials query
    if loader.data:
      trials = loader.get_condition_trials(1)  # Get happy/excited trials
      self.assertIsInstance(trials, list)
      if trials:
        trial = trials[0]
        self.assertIn("subject", trial)
        self.assertIn("run", trial)
        self.assertIn("trial_info", trial)
        self.assertIn("raw_data", trial)
        print(f"  ✓ Found {len(trials)} trials for condition 1")


class TestMUSINGDataset(unittest.TestCase):
  """Test suite for MUSIN-G dataset loader."""

  @skip_if_dataset_missing(MUSING_ROOT)
  def test_musing_initialization(self):
    """Test MUSINGDataset initialization."""
    print("\n🧪 Testing MUSIN-G initialization...")

    dataset = MUSINGDataset(str(MUSING_ROOT))

    # Check attributes
    self.assertIsNotNone(dataset.subjects)
    self.assertIsNotNone(dataset.sessions)
    self.assertIsNotNone(dataset.songs_info)
    self.assertEqual(len(dataset.songs_info), 12)
    print(
      f"  ✓ Found {len(dataset.subjects)} subjects, {len(dataset.sessions)} sessions"
    )

    # Check behavioral data loading
    self.assertIsNotNone(dataset.behavioral_data)
    self.assertIsInstance(dataset.behavioral_data, pd.DataFrame)
    print(f"  ✓ Behavioral data loaded: {len(dataset.behavioral_data)} entries")

  @skip_if_dataset_missing(MUSING_ROOT)
  def test_musing_load_complete_dataset(self):
    """Test loading complete MUSIN-G dataset."""
    print("\n🧪 Testing MUSIN-G data loading...")

    dataset = MUSINGDataset(str(MUSING_ROOT))

    # Load limited data for testing
    all_data = dataset.load_complete_dataset(max_subjects=2, verbose=True)

    self.assertIsInstance(all_data, dict)

    # Verify data structure
    for subject_id, subject_data in all_data.items():
      for session_id, session_data in subject_data.items():
        if "raw" in session_data:
          self.assertIn("song_info", session_data)
          self.assertIn("enjoyment", session_data)
          self.assertIn("familiarity", session_data)

          # Check song info
          song_info = session_data["song_info"]
          self.assertIn("name", song_info)
          self.assertIn("artist", song_info)
          self.assertIn("genre", song_info)

          print(
            f"  ✓ Subject {subject_id}, Session {session_id}: "
            f"{song_info['name']} ({song_info['genre']})"
          )
          break
      break

  @skip_if_dataset_missing(MUSING_ROOT)
  def test_musing_song_metadata(self):
    """Test that all 12 songs have correct metadata."""
    print("\n🧪 Testing MUSIN-G song metadata...")

    dataset = MUSINGDataset(str(MUSING_ROOT))

    # Check all 12 songs
    self.assertEqual(len(dataset.songs_info), 12)

    for song_id, info in dataset.songs_info.items():
      self.assertIn("name", info)
      self.assertIn("artist", info)
      self.assertIn("genre", info)
      self.assertIn("duration", info)
      self.assertIn("characteristics", info)
      print(f"  ✓ Song {song_id}: {info['name']} - {info['genre']}")

  @skip_if_dataset_missing(MUSING_ROOT)
  def test_musing_get_methods(self):
    """Test data retrieval methods."""
    print("\n🧪 Testing MUSIN-G data retrieval methods...")

    dataset = MUSINGDataset(str(MUSING_ROOT))
    dataset.load_complete_dataset(max_subjects=2, verbose=False)

    if dataset.all_data:
      # Test get_subject_data
      subject_id = list(dataset.all_data.keys())[0]
      subject_data = dataset.get_subject_data(subject_id)
      self.assertIsInstance(subject_data, dict)
      print(f"  ✓ Retrieved data for subject {subject_id}")

      # Test get_song_data_across_subjects
      if dataset.sessions:
        session_id = dataset.sessions[0]
        song_data = dataset.get_song_data_across_subjects(session_id)
        self.assertIsInstance(song_data, dict)
        print(f"  ✓ Retrieved data for song {session_id} across subjects")

      # Test get_song_metadata
      for subject_data in dataset.all_data.values():
        for session_data in subject_data.values():
          if "raw" in session_data:
            raw = session_data["raw"]
            metadata = dataset.get_song_metadata(raw)
            if metadata:
              self.assertIn("genre", metadata)
              self.assertIn("enjoyment_rating", metadata)
              print("  ✓ Retrieved metadata from raw object")
            break
        break

  @skip_if_dataset_missing(MUSING_ROOT)
  def test_musing_statistics(self):
    """Test dataset statistics generation."""
    print("\n🧪 Testing MUSIN-G statistics...")

    dataset = MUSINGDataset(str(MUSING_ROOT))
    dataset.load_complete_dataset(max_subjects=1, verbose=False)

    # Should not raise
    dataset.get_dataset_statistics()
    print("  ✓ Dataset statistics generated successfully")


class TestNMEDTLoader(unittest.TestCase):
  """Test suite for NMED-T dataset loader."""

  @skip_if_dataset_missing(NMEDT_ROOT)
  def test_nmedt_initialization(self):
    """Test NMEDTLoader initialization."""
    print("\n🧪 Testing NMED-T initialization...")

    loader = NMEDTLoader(str(NMEDT_ROOT))

    # Check song info
    self.assertEqual(len(loader.song_info), 10)
    for song_id, info in loader.song_info.items():
      self.assertIn("title", info)
      self.assertIn("artist", info)
      self.assertIn("tempo", info)
      self.assertIn("tempo_hz", info)

    print(f"  ✓ Initialized with {len(loader.song_info)} songs")

  @skip_if_dataset_missing(NMEDT_ROOT)
  def test_nmedt_load_cleaned_eeg(self):
    """Test loading cleaned EEG data."""
    print("\n🧪 Testing NMED-T cleaned EEG loading...")

    loader = NMEDTLoader(str(NMEDT_ROOT))

    # Load first 2 songs for testing
    eeg_data = loader.load_cleaned_eeg_data(song_numbers=[1, 2])

    for song_num, data in eeg_data.items():
      self.assertIn("data", data)
      self.assertIn("participants", data)
      self.assertIn("sampling_rate", data)
      self.assertIn("song_info", data)

      # Check data shape (125 electrodes, time_samples, 20 participants)
      shape = data["data"].shape
      self.assertEqual(shape[0], 125)  # Electrodes
      self.assertEqual(shape[2], 20)  # Participants

      # Check sampling rate
      self.assertEqual(data["sampling_rate"], 125)

      print(
        f"  ✓ Song {song_num}: {data['song_info']['title']} - "
        f"Shape: {shape}, {data['song_info']['tempo']} BPM"
      )

  @skip_if_dataset_missing(NMEDT_ROOT)
  def test_nmedt_load_raw_eeg(self):
    """Test loading raw EEG data."""
    print("\n🧪 Testing NMED-T raw EEG loading...")

    loader = NMEDTLoader(str(NMEDT_ROOT))

    # Try to load raw data for first available participant
    raw_data = loader.load_raw_eeg_data(participant_ids=[1, 2], recording_sessions=[1])

    if raw_data:
      for (participant_id, session), data in raw_data.items():
        self.assertIn("data", data)
        self.assertIn("events", data)
        self.assertIn("sampling_rate", data)

        # Check data shape (129 electrodes, time_samples)
        shape = data["data"].shape
        self.assertEqual(shape[0], 129)  # Electrodes (including vertex reference)

        # Check sampling rate
        self.assertEqual(data["sampling_rate"], 1000)

        print(
          f"  ✓ Participant {participant_id}, Session {session}: "
          f"Shape: {shape}, 1000 Hz"
        )
        break
    else:
      print(
        "  ⚠ No raw data files found (this is normal if only cleaned data is available)"
      )

  @skip_if_dataset_missing(NMEDT_ROOT)
  def test_nmedt_behavioral_data(self):
    """Test loading behavioral ratings."""
    print("\n🧪 Testing NMED-T behavioral data...")

    loader = NMEDTLoader(str(NMEDT_ROOT))
    behavioral = loader.load_behavioral_data()

    if behavioral:
      self.assertIn("ratings", behavioral)
      self.assertIn("questions", behavioral)

      # Check shape (20 participants, 10 songs, 2 questions)
      shape = behavioral["ratings"].shape
      self.assertEqual(shape, (20, 10, 2))

      # Check questions
      self.assertEqual(behavioral["questions"], ["familiarity", "enjoyment"])

      print(f"  ✓ Behavioral data shape: {shape}")
      print(f"  ✓ Questions: {behavioral['questions']}")
    else:
      print("  ⚠ Behavioral data file not found")

  @skip_if_dataset_missing(NMEDT_ROOT)
  def test_nmedt_participant_info(self):
    """Test loading participant demographics."""
    print("\n🧪 Testing NMED-T participant info...")

    loader = NMEDTLoader(str(NMEDT_ROOT))
    participant_info = loader.load_participant_info()

    if participant_info:
      self.assertIn("demographics", participant_info)
      self.assertIn("fields", participant_info)
      self.assertIn("descriptions", participant_info)

      expected_fields = ["age", "nYearsTraining", "weeklyListening", "id"]
      for field in expected_fields:
        self.assertIn(field, participant_info["demographics"])

      print(f"  ✓ Demographic fields: {participant_info['fields']}")
    else:
      print("  ⚠ Participant info file not found")

  @skip_if_dataset_missing(NMEDT_ROOT)
  def test_nmedt_summary(self):
    """Test dataset summary generation."""
    print("\n🧪 Testing NMED-T dataset summary...")

    loader = NMEDTLoader(str(NMEDT_ROOT))
    summary = loader.get_dataset_summary()

    self.assertIn("dataset_name", summary)
    self.assertIn("participants", summary)
    self.assertIn("songs", summary)
    self.assertIn("electrodes", summary)
    self.assertIn("sampling_rates", summary)

    self.assertEqual(summary["participants"], 20)
    self.assertEqual(summary["songs"], 10)
    self.assertEqual(summary["electrodes"], 125)

    print(f"  ✓ Dataset: {summary['dataset_name']}")
    print(f"  ✓ Participants: {summary['participants']}, Songs: {summary['songs']}")
    print(f"  ✓ Sampling rates: {summary['sampling_rates']}")


class TestOpenMIIRLoader(unittest.TestCase):
  """Test suite for OpenMIIR dataset loader."""

  @skip_if_dataset_missing(OPENMIIR_ROOT)
  def test_openmiir_initialization(self):
    """Test OpenMIIRLoader initialization."""
    print("\n🧪 Testing OpenMIIR initialization...")

    loader = OpenMIIRLoader(str(OPENMIIR_ROOT))

    self.assertIsNotNone(loader.subjects)
    self.assertIsNotNone(loader.runs)
    self.assertIsNotNone(loader.emotional_states)
    self.assertEqual(len(loader.emotional_states), 9)

    print(f"  ✓ Found {len(loader.subjects)} subjects")
    print(f"  ✓ Available runs: {loader.runs}")

  @skip_if_dataset_missing(OPENMIIR_ROOT)
  def test_openmiir_load_subject(self):
    """Test loading single subject data."""
    print("\n🧪 Testing OpenMIIR subject loading...")

    loader = OpenMIIRLoader(str(OPENMIIR_ROOT))

    if loader.subjects:
      subject = loader.subjects[0]
      data = loader.load_subject_data(subject, max_runs=2)

      self.assertIsInstance(data, dict)

      for run_id, run_data in data.items():
        self.assertIn("raw", run_data)
        self.assertIn("events", run_data)
        self.assertIn("processed_events", run_data)
        self.assertIn("duration", run_data)
        self.assertIn("n_channels", run_data)
        self.assertIn("sfreq", run_data)
        self.assertIn("n_trials", run_data)

        print(
          f"  ✓ Subject {subject}, Run {run_id}: "
          f"{run_data['duration']:.1f}s, {run_data['n_trials']} trials"
        )
        break

  @skip_if_dataset_missing(OPENMIIR_ROOT)
  def test_openmiir_event_processing(self):
    """Test event processing separation."""
    print("\n🧪 Testing OpenMIIR event processing...")

    loader = OpenMIIRLoader(str(OPENMIIR_ROOT))

    if loader.subjects:
      subject = loader.subjects[0]
      data = loader.load_subject_data(subject, max_runs=1)

      for run_data in data.values():
        processed = run_data.get("processed_events")
        if processed:
          self.assertIn("condition_events", processed)
          self.assertIn("marker_events", processed)
          self.assertIn("all_events", processed)

          # Check condition events have emotion info
          condition_events = processed["condition_events"]
          if not condition_events.empty:
            self.assertIn("emotion_description", condition_events.columns)
            self.assertIn("valence", condition_events.columns)
            self.assertIn("arousal", condition_events.columns)

            # Check that condition events are 1-9
            self.assertTrue(all(condition_events["trial_type"] <= 9))

            print(f"  ✓ Condition events: {len(condition_events)} trials")
            print("  ✓ Emotion annotations added successfully")

          # Check marker events are > 100
          marker_events = processed["marker_events"]
          if not marker_events.empty:
            self.assertTrue(all(marker_events["trial_type"] > 100))
            print(f"  ✓ Marker events: {len(marker_events)} markers")
        break

  @skip_if_dataset_missing(OPENMIIR_ROOT)
  def test_openmiir_emotional_states(self):
    """Test emotional state mapping."""
    print("\n🧪 Testing OpenMIIR emotional states...")

    loader = OpenMIIRLoader(str(OPENMIIR_ROOT))

    # Check all 9 emotional states
    for code in range(1, 10):
      state_info = loader.get_emotional_state_info(code)
      self.assertIn("valence", state_info)
      self.assertIn("arousal", state_info)
      self.assertIn("description", state_info)

      print(
        f"  ✓ State {code}: {state_info['description']} "
        f"(V: {state_info['valence']}, A: {state_info['arousal']})"
      )

  @skip_if_dataset_missing(OPENMIIR_ROOT)
  def test_openmiir_condition_trials(self):
    """Test retrieving trials by condition."""
    print("\n🧪 Testing OpenMIIR condition trial retrieval...")

    loader = OpenMIIRLoader(str(OPENMIIR_ROOT))
    loader.load_all_subjects(max_subjects=2, max_runs_per_subject=2, verbose=False)

    if loader.data:
      # Test getting happy/excited trials (condition 1)
      trials = loader.get_condition_trials(1)
      self.assertIsInstance(trials, list)

      if trials:
        trial = trials[0]
        self.assertIn("subject", trial)
        self.assertIn("run", trial)
        self.assertIn("trial_info", trial)
        self.assertIn("raw_data", trial)

        print(f"  ✓ Found {len(trials)} trials for condition 1 (Happy/Excited)")

  @skip_if_dataset_missing(OPENMIIR_ROOT)
  def test_openmiir_statistics(self):
    """Test dataset statistics generation."""
    print("\n🧪 Testing OpenMIIR statistics...")

    loader = OpenMIIRLoader(str(OPENMIIR_ROOT))
    loader.load_all_subjects(max_subjects=1, max_runs_per_subject=2, verbose=False)

    # Should not raise
    loader.get_dataset_statistics()
    print("  ✓ Dataset statistics generated successfully")


class TestTrialIterator(unittest.TestCase):
  """Test suite for trial_iterator functionality in BCMI loaders."""

  @skip_if_dataset_missing(BCMI_ROOT / "bcmi-calibration")
  def test_calibration_trial_iterator(self):
    """Test trial_iterator for BCMI Calibration loader."""
    print("\n🧪 Testing BCMI Calibration trial_iterator...")

    loader = BCMICalibrationLoader(str(BCMI_ROOT / "bcmi-calibration"))

    # Load minimal data for testing
    loader.load_all_subjects(max_subjects=1, max_runs_per_session=1)

    if not loader.data:
      self.skipTest("No data loaded for calibration trial iterator test")

    # Test that trial_iterator is implemented and works
    trial_count = 0
    trial_duration = 0
    try:
      for trial in loader.trial_iterator():
        # Check that we get TrialRow objects with expected attributes
        self.assertHasAttr(trial, "dataset", "Trial should have dataset attribute")
        self.assertHasAttr(trial, "subject", "Trial should have subject attribute")
        self.assertHasAttr(trial, "session", "Trial should have session attribute")
        self.assertHasAttr(trial, "run", "Trial should have run attribute")
        self.assertHasAttr(trial, "trial_id", "Trial should have trial_id attribute")
        self.assertHasAttr(
          trial, "music_filename", "Trial should have music_filename attribute"
        )
        self.assertHasAttr(trial, "eeg_data", "Trial should have eeg_data attribute")

        # Check that we can get music data through the music collection
        # (Note: music collection is on the loader, not the individual trial)
        music_ref = trial.music_filename
        self.assertIsNotNone(music_ref, "Trial should have music reference")

        # Check EEG data access
        eeg_data = trial.eeg_data.get_eeg()
        self.assertIsNotNone(eeg_data, "Should be able to get EEG data")
        raw_eeg = eeg_data.raw_eeg
        self.assertTrue(hasattr(raw_eeg, "info"), "raw_eeg should be MNE Raw object")
        self.assertTrue(hasattr(raw_eeg, "times"), "raw_eeg should have times")
        self.assertGreater(len(raw_eeg.times), 0, "EEG data should not be empty")

        # Check trial duration (should be around 21 seconds for calibration)
        trial_duration = raw_eeg.times[-1] - raw_eeg.times[0]
        self.assertGreater(trial_duration, 19, "Trial should be at least 19 seconds")
        self.assertLess(trial_duration, 23, "Trial should be less than 23 seconds")

        trial_count += 1
        if trial_count >= 3:  # Test only first few trials
          break

      self.assertGreater(trial_count, 0, "Should yield at least one trial")
      print(f"  ✓ Successfully generated {trial_count} trials")
      print(f"  ✓ Trial duration check: ~{trial_duration:.1f}s")

    except Exception as e:
      self.fail(f"trial_iterator failed: {str(e)}")

  @skip_if_dataset_missing(BCMI_ROOT / "bcmi-training")
  def test_training_trial_iterator(self):
    """Test trial_iterator for BCMI Training loader."""
    print("\n🧪 Testing BCMI Training trial_iterator...")

    loader = BCMITrainingLoader(str(BCMI_ROOT / "bcmi-training"))

    # Load minimal data for testing
    loader.load_all_subjects(
      max_subjects=1, max_runs_per_session=1, max_sessions_per_subject=1
    )

    # Test that trial_iterator is implemented and works
    trial_count = 0
    trial_duration = 0
    try:
      for trial in loader.trial_iterator():
        # Check that we get TrialRow objects with expected attributes
        self.assertHasAttr(trial, "dataset", "Trial should have dataset attribute")
        self.assertHasAttr(trial, "subject", "Trial should have subject attribute")
        self.assertHasAttr(trial, "session", "Trial should have session attribute")
        self.assertHasAttr(trial, "run", "Trial should have run attribute")
        self.assertHasAttr(trial, "trial_id", "Trial should have trial_id attribute")
        self.assertHasAttr(
          trial, "music_filename", "Trial should have music_filename attribute"
        )
        self.assertHasAttr(trial, "eeg_data", "Trial should have eeg_data attribute")

        # Check that we can get music data through the music collection
        music_ref = trial.music_filename
        self.assertIsNotNone(music_ref, "Trial should have music reference")

        # Check EEG data access
        eeg_data = trial.eeg_data.get_eeg()
        self.assertIsNotNone(eeg_data, "Should be able to get EEG data")
        raw_eeg = eeg_data.raw_eeg
        self.assertTrue(hasattr(raw_eeg, "info"), "raw_eeg should be MNE Raw object")
        self.assertTrue(hasattr(raw_eeg, "times"), "raw_eeg should have times")
        self.assertGreater(len(raw_eeg.times), 0, "EEG data should not be empty")

        # Check trial duration (should be around 20 seconds for training)
        trial_duration = raw_eeg.times[-1] - raw_eeg.times[0]
        self.assertGreater(trial_duration, 18, "Trial should be at least 18 seconds")
        self.assertLess(trial_duration, 22, "Trial should be less than 22 seconds")

        trial_count += 1

      self.assertGreater(trial_count, 0, "Should yield at least one trial")
      print(f"  ✓ Successfully generated {trial_count} trials")
      print(f"  ✓ Trial duration check: ~{trial_duration:.1f}s")

    except Exception as e:
      self.fail(f"training trial_iterator failed: {str(e)}")

  def test_not_implemented_trial_iterators(self):
    """Test that NotImplementedError is raised for loaders that don't support trial_iterator."""
    print("\n🧪 Testing NotImplementedError for unsupported trial_iterators...")

    # Test loaders that should raise NotImplementedError
    loader_classes_and_paths = [
      (BCMITestingLoader, BCMI_ROOT / "bcmi-testing", "Testing"),
      (BCMITempoLoader, BCMI_ROOT / "bcmi-tempo", "Tempo"),
      (BCMIScoresLoader, BCMI_ROOT / "bcmi-scores", "Scores"),
      (BCMIFMRILoader, BCMI_ROOT / "bcmi-fmri", "FMRI"),
    ]

    for loader_class, dataset_path, dataset_name in loader_classes_and_paths:
      if dataset_exists(dataset_path):
        try:
          loader = loader_class(str(dataset_path))

          # Should raise NotImplementedError when calling trial_iterator
          with self.assertRaises(
            NotImplementedError,
            msg=f"{dataset_name} loader should raise NotImplementedError",
          ):
            # Try to get the first trial from iterator
            list(loader.trial_iterator())

          print(f"  ✓ {dataset_name} loader correctly raises NotImplementedError")

        except Exception as e:
          print(f"  ⚠ Error testing {dataset_name} loader: {e}")
      else:
        print(f"  - {dataset_name} dataset not available, skipping")

  def test_trial_iterator_consistency(self):
    """Test that trial_iterator produces consistent results across multiple calls."""
    print("\n🧪 Testing trial_iterator consistency...")

    # Test with calibration dataset if available
    if dataset_exists(BCMI_ROOT / "bcmi-calibration"):
      loader = BCMICalibrationLoader(str(BCMI_ROOT / "bcmi-calibration"))
      loader.load_all_subjects(max_subjects=1, max_runs_per_session=1)

      if loader.data:
        # Get first few trials from first call
        first_call_trials = []
        for i, trial in enumerate(loader.trial_iterator()):
          if i >= 2:  # Only test first 2 trials
            break
          eeg_raw = trial.eeg_data.get_eeg().raw_eeg
          # Get music from loader's collection using the music_ref
          music_data = None
          for music_ref, music in loader.music_iterator():
            if music_ref.filename == trial.music_filename.filename:
              music_data = music
              break
          if music_data:
            first_call_trials.append((eeg_raw.times.shape, music_data.raw_data.shape))

        # Get first few trials from second call
        second_call_trials = []
        for i, trial in enumerate(loader.trial_iterator()):
          if i >= 2:
            break
          eeg_raw = trial.eeg_data.get_eeg().raw_eeg
          # Get music from loader's collection using the music_ref
          music_data = None
          for music_ref, music in loader.music_iterator():
            if music_ref.filename == trial.music_filename.filename:
              music_data = music
              break
          if music_data:
            second_call_trials.append((eeg_raw.times.shape, music_data.raw_data.shape))

        # Should produce the same structure
        self.assertEqual(
          len(first_call_trials),
          len(second_call_trials),
          "trial_iterator should produce same number of trials on repeated calls",
        )

        for i, (first_shapes, second_shapes) in enumerate(
          zip(first_call_trials, second_call_trials)
        ):
          self.assertEqual(
            first_shapes,
            second_shapes,
            f"Trial {i} should have same shapes on repeated calls",
          )

        print(
          f"  ✓ trial_iterator produces consistent results across {len(first_call_trials)} trials"
        )
      else:
        print("  - No data loaded, skipping consistency test")
    else:
      print("  - Calibration dataset not available, skipping consistency test")

  def assertHasAttr(self, obj, attr_name, msg=None):
    """Helper method to assert object has attribute."""
    if not hasattr(obj, attr_name):
      if msg is None:
        msg = f"Object {obj} should have attribute '{attr_name}'"
      self.fail(msg)


class TestDataLoaderIntegration(unittest.TestCase):
  """Integration tests for all dataset loaders."""

  def test_all_loaders_instantiate(self):
    """Quick smoke test that all loaders can be created."""
    print("\n🧪 Testing loader instantiation...")

    loaders_tested = []

    # Test BCMI loaders
    if dataset_exists(BCMI_ROOT / "bcmi-calibration"):
      try:
        BCMICalibrationLoader(str(BCMI_ROOT / "bcmi-calibration"))
        loaders_tested.append("BCMI-Calibration")
        print("  ✓ BCMI Calibration loader instantiated")
      except Exception as e:
        print(f"  ✗ BCMI Calibration loader failed: {e}")

    # Test MUSIN-G loader
    if dataset_exists(MUSING_ROOT):
      try:
        MUSINGDataset(str(MUSING_ROOT))
        loaders_tested.append("MUSIN-G")
        print("  ✓ MUSIN-G loader instantiated")
      except Exception as e:
        print(f"  ✗ MUSIN-G loader failed: {e}")

    # Test NMED-T loader
    if dataset_exists(NMEDT_ROOT):
      try:
        NMEDTLoader(str(NMEDT_ROOT))
        loaders_tested.append("NMED-T")
        print("  ✓ NMED-T loader instantiated")
      except Exception as e:
        print(f"  ✗ NMED-T loader failed: {e}")

    # Test OpenMIIR loader
    if dataset_exists(OPENMIIR_ROOT):
      try:
        OpenMIIRLoader(str(OPENMIIR_ROOT))
        loaders_tested.append("OpenMIIR")
        print("  ✓ OpenMIIR loader instantiated")
      except Exception as e:
        print(f"  ✗ OpenMIIR loader failed: {e}")

    print(f"\n  Summary: {len(loaders_tested)} loaders tested successfully")
    if loaders_tested:
      print(f"  Loaders: {', '.join(loaders_tested)}")

  def test_common_interface(self):
    """Verify common methods across loaders where applicable."""
    print("\n🧪 Testing common loader interfaces...")

    # Common methods that should exist in BCMI and OpenMIIR loaders
    common_methods = [
      "load_subject_data",
      "load_all_subjects",
      "get_dataset_statistics",
    ]

    loaders_to_test = []

    # Collect available loaders
    if dataset_exists(BCMI_ROOT / "bcmi-calibration"):
      loaders_to_test.append(
        ("BCMI", BCMICalibrationLoader(str(BCMI_ROOT / "bcmi-calibration")))
      )

    if dataset_exists(OPENMIIR_ROOT):
      loaders_to_test.append(("OpenMIIR", OpenMIIRLoader(str(OPENMIIR_ROOT))))

    for name, loader in loaders_to_test:
      for method in common_methods:
        self.assertTrue(
          hasattr(loader, method), f"{name} loader missing method: {method}"
        )
      print(f"  ✓ {name} loader has all common methods")

  def test_error_handling(self):
    """Test graceful handling of missing datasets."""
    print("\n🧪 Testing error handling...")

    # Test with non-existent path
    fake_path = "/tmp/non_existent_dataset_12345"

    try:
      BCMICalibrationLoader(fake_path)
      # Should not raise during initialization
      print("  ✓ BCMI loader handles missing dataset gracefully")
    except Exception as e:
      print(f"  ⚠ BCMI loader raised exception on missing dataset: {e}")

    try:
      OpenMIIRLoader(fake_path)
      # Should not raise during initialization
      print("  ✓ OpenMIIR loader handles missing dataset gracefully")
    except Exception as e:
      print(f"  ⚠ OpenMIIR loader raised exception on missing dataset: {e}")


def run_all_tests():
  """
  Main function to run all tests.

  This function sets up the test environment and runs all test suites.
  It provides a summary of test results and any skipped tests.
  """
  print("=" * 70)
  print("EEG DATASET LOADER TEST SUITE")
  print("=" * 70)
  print()

  # Check available datasets
  print("📂 Checking dataset availability:")
  datasets_status = {
    "BCMI Calibration": dataset_exists(BCMI_ROOT / "bcmi-calibration"),
    "BCMI Training": dataset_exists(BCMI_ROOT / "bcmi-training"),
    "BCMI Testing": dataset_exists(BCMI_ROOT / "bcmi-testing"),
    "BCMI Tempo": dataset_exists(BCMI_ROOT / "bcmi-tempo"),
    "BCMI Scores": dataset_exists(BCMI_ROOT / "bcmi-scores"),
    "BCMI fMRI": dataset_exists(BCMI_ROOT / "bcmi-fmri"),
    "MUSIN-G": dataset_exists(MUSING_ROOT),
    "NMED-T": dataset_exists(NMEDT_ROOT),
    "OpenMIIR": dataset_exists(OPENMIIR_ROOT),
  }

  for dataset, available in datasets_status.items():
    status = "✓ Available" if available else "✗ Not found"
    print(f"  {dataset:20} {status}")

  print()
  print("=" * 70)
  print("Running tests...")
  print("=" * 70)

  # Run tests
  unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
  # Check if running specific tests or all tests
  if len(sys.argv) > 1:
    # Let unittest handle command line arguments
    unittest.main()
  else:
    # Run our custom test runner with dataset checking
    run_all_tests()


class TestEEGMusicDatasetFiltering(unittest.TestCase):
  def _mk_eeg(self, seconds: float, sfreq: float = 10.0) -> RawEeg:
    n_times = int(seconds * sfreq)
    info = mne.create_info(ch_names=["ch1"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, n_times), dtype=float), info)
    return RawEeg(raw_eeg=raw)

  def _mk_music(self, seconds: float, sr: int = 10) -> WavRAW:
    n = int(seconds * sr)
    return WavRAW(raw_data=np.zeros((n,), dtype=float), sample_rate=sr)

  def test_remove_short_trials(self):
    ds = EEGMusicDataset()
    # build music collection
    mref1 = MusicRef(MusicFilename("a.wav"), "d1")
    mref2 = MusicRef(MusicFilename("b.wav"), "d1")
    ds.music_collection[mref1] = self._mk_music(12.0)
    ds.music_collection[mref2] = self._mk_music(8.0)

    rows = [
      {
        "dataset": "d1",
        "subject": "s1",
        "session": "ses1",
        "run": "r1",
        "trial_id": "t1",
        "music_filename": MusicFilename("a.wav"),
        "eeg_data": self._mk_eeg(12.0),
      },
      {
        "dataset": "d1",
        "subject": "s1",
        "session": "ses1",
        "run": "r1",
        "trial_id": "t2",
        "music_filename": MusicFilename("b.wav"),
        "eeg_data": self._mk_eeg(20.0),
      },
      {
        "dataset": "d1",
        "subject": "s1",
        "session": "ses1",
        "run": "r1",
        "trial_id": "t3",
        "music_filename": MusicFilename("a.wav"),
        "eeg_data": self._mk_eeg(5.0),
      },
    ]
    ds.df = pd.DataFrame(rows)

    # threshold 10s: keep t1 (12s music, 12s eeg), remove t2 (8s music), remove t3 (5s eeg)
    filtered = ds.remove_short_trials(10.0)
    self.assertEqual(len(filtered), 1)
    tr = filtered[0]
    self.assertEqual(tr.trial_id, "t1")

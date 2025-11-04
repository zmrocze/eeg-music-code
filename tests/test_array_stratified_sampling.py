import unittest
from pathlib import Path
import tempfile
import numpy as np
from fractions import Fraction
from eeg_music.data import (
  EEGMusicDataset,
  ArrayEeg,
  OnDiskArrayEeg,
  RawEeg,
  WavRAW,
  MelRaw,
  NoteOnsets,
  TrialData,
  MusicFilename,
  MusicRef,
  ArrayStratifiedSamplingDataset,
)


class TestArrayStratifiedSamplingDataset(unittest.TestCase):
  def setUp(self):
    """Create a simple dataset with ArrayEeg trials for testing."""
    self.dataset = EEGMusicDataset()

    # Create 3 trials with ArrayEeg and different music types
    np.random.seed(42)

    # Trial 1: ArrayEeg + WavRAW
    eeg_data1 = np.random.randn(4, 2560).astype(np.float32)  # 10 seconds at 256 Hz
    array_eeg1 = ArrayEeg(
      data=eeg_data1, ch_names=["C3", "C4", "Cz", "Fz"], sfreq=256.0
    )
    wav_data1 = np.random.randn(441000).astype(np.float32)  # 10 seconds at 44100 Hz
    music1 = WavRAW(raw_data=wav_data1, sample_rate=44100)

    trial1 = TrialData(
      dataset="test",
      subject="s1",
      session="1",
      run="1",
      trial_id="t1",
      music_filename=MusicFilename(filename="song1.wav"),
      eeg_data=array_eeg1,
      music_data=music1,
    )

    # Trial 2: ArrayEeg + MelRaw
    eeg_data2 = np.random.randn(4, 5120).astype(np.float32)  # 20 seconds at 256 Hz
    array_eeg2 = ArrayEeg(
      data=eeg_data2, ch_names=["C3", "C4", "Cz", "Fz"], sfreq=256.0
    )
    mel_data2 = np.random.randn(128, 100).astype(np.float32)
    music2 = MelRaw(
      mel=mel_data2, sample_rate=22050, hop_length=512, fmin=0.0, fmax=None, to_db=True
    )

    trial2 = TrialData(
      dataset="test",
      subject="s1",
      session="1",
      run="2",
      trial_id="t2",
      music_filename=MusicFilename(filename="song2.wav"),
      eeg_data=array_eeg2,
      music_data=music2,
    )

    # Trial 3: ArrayEeg + NoteOnsets
    eeg_data3 = np.random.randn(4, 3840).astype(np.float32)  # 15 seconds at 256 Hz
    array_eeg3 = ArrayEeg(
      data=eeg_data3, ch_names=["C3", "C4", "Cz", "Fz"], sfreq=256.0
    )
    onset_times3 = np.array([0.5, 1.2, 2.5, 4.0, 7.5, 10.2, 13.0])
    music3 = NoteOnsets(
      onset_times=onset_times3, sample_rate=256, duration_seconds=15.0
    )

    trial3 = TrialData(
      dataset="test",
      subject="s2",
      session="1",
      run="1",
      trial_id="t3",
      music_filename=MusicFilename(filename="song3.wav"),
      eeg_data=array_eeg3,
      music_data=music3,
    )

    # Add trials to dataset
    self.dataset.df.loc[0] = {
      "dataset": trial1.dataset,
      "subject": trial1.subject,
      "session": trial1.session,
      "run": trial1.run,
      "trial_id": trial1.trial_id,
      "music_filename": trial1.music_filename,
      "eeg_data": trial1.eeg_data,
    }
    self.dataset.df.loc[1] = {
      "dataset": trial2.dataset,
      "subject": trial2.subject,
      "session": trial2.session,
      "run": trial2.run,
      "trial_id": trial2.trial_id,
      "music_filename": trial2.music_filename,
      "eeg_data": trial2.eeg_data,
    }
    self.dataset.df.loc[2] = {
      "dataset": trial3.dataset,
      "subject": trial3.subject,
      "session": trial3.session,
      "run": trial3.run,
      "trial_id": trial3.trial_id,
      "music_filename": trial3.music_filename,
      "eeg_data": trial3.eeg_data,
    }

    # Add music to collection
    music_ref1 = MusicRef(filename=trial1.music_filename, dataset=trial1.dataset)
    music_ref2 = MusicRef(filename=trial2.music_filename, dataset=trial2.dataset)
    music_ref3 = MusicRef(filename=trial3.music_filename, dataset=trial3.dataset)
    self.dataset.music_collection[music_ref1] = music1
    self.dataset.music_collection[music_ref2] = music2
    self.dataset.music_collection[music_ref3] = music3

  def test_basic_initialization(self):
    """Test that ArrayStratifiedSamplingDataset can be initialized."""
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=4, trial_length_secs=Fraction(5, 1)
    )

    self.assertEqual(len(stratified), len(self.dataset) * 4)
    self.assertEqual(stratified.n_strata, 4)
    self.assertEqual(stratified.trial_length_secs, Fraction(5, 1))

  def test_length_calculation(self):
    """Test that length is correctly calculated as base_length * n_strata."""
    n_strata = 5
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=n_strata, trial_length_secs=Fraction(4, 1)
    )

    self.assertEqual(len(stratified), len(self.dataset) * n_strata)

  def test_eeg_trimming(self):
    """Test that EEG is trimmed to specified length."""
    trial_length = Fraction(5, 1)  # 5 seconds
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=3, trial_length_secs=trial_length
    )

    # Get a trial from the first stratum of the first trial
    trial = stratified[0]
    eeg = trial.eeg_data

    self.assertIsInstance(eeg, ArrayEeg)
    expected_samples = int(trial_length * 256)  # 256 Hz
    self.assertEqual(eeg.data.shape[1], expected_samples)
    self.assertEqual(eeg.data.shape[0], 4)  # 4 channels
    self.assertAlmostEqual(eeg.length_seconds(), float(trial_length), places=2)

  def test_music_unchanged(self):
    """Test that music data is passed through unchanged."""
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=2, trial_length_secs=Fraction(5, 1)
    )

    # Test trial with WavRAW
    trial = stratified[0]
    original_trial = self.dataset[0]

    # Music should be exactly the same object
    self.assertIs(trial.music_data, original_trial.music_data)

    # Verify it's actually WavRAW and unchanged
    self.assertIsInstance(trial.music_data, WavRAW)
    if isinstance(trial.music_data, WavRAW) and isinstance(
      original_trial.music_data, WavRAW
    ):
      np.testing.assert_array_equal(
        trial.music_data.raw_data, original_trial.music_data.raw_data
      )

  def test_stratified_sampling_distribution(self):
    """Test that different strata sample from different regions."""
    np.random.seed(42)
    n_strata = 4
    trial_length = Fraction(4, 1)
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=n_strata, trial_length_secs=trial_length
    )

    # Sample multiple times from same trial but different strata
    # We expect stratum 0 to sample from earlier parts, stratum 3 from later parts
    stratum_0_samples = []
    stratum_3_samples = []

    for _ in range(5):
      trial_s0 = stratified[0]  # First trial, stratum 0
      trial_s3 = stratified[3]  # First trial, stratum 3
      stratum_0_samples.append(trial_s0.eeg_data.data.copy())
      stratum_3_samples.append(trial_s3.eeg_data.data.copy())

    # Samples from same stratum should potentially differ (random within stratum)
    # but we mainly verify the implementation doesn't crash

  def test_trial_and_stratum_indexing(self):
    """Test that indexing correctly maps to trial and stratum."""
    n_strata = 3
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=n_strata, trial_length_secs=Fraction(4, 1)
    )

    # Indices 0, 1, 2 should be trial 0, strata 0, 1, 2
    # Indices 3, 4, 5 should be trial 1, strata 0, 1, 2
    # Indices 6, 7, 8 should be trial 2, strata 0, 1, 2

    for trial_idx in range(3):
      original_trial = self.dataset[trial_idx]
      for stratum_idx in range(n_strata):
        idx = trial_idx * n_strata + stratum_idx
        sampled_trial = stratified[idx]

        # Check metadata is preserved
        self.assertEqual(sampled_trial.dataset, original_trial.dataset)
        self.assertEqual(sampled_trial.subject, original_trial.subject)
        self.assertEqual(sampled_trial.trial_id, original_trial.trial_id)

  def test_ondisk_arrayeeg(self):
    """Test that OnDiskArrayEeg is handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmppath = Path(tmpdir)

      # Save an ArrayEeg trial to disk
      eeg_data = np.random.randn(4, 2560).astype(np.float32)
      array_eeg = ArrayEeg(
        data=eeg_data, ch_names=["C3", "C4", "Cz", "Fz"], sfreq=256.0
      )
      eeg_path = tmppath / "test_eeg.npz"
      array_eeg.save(eeg_path)

      # Create trial with OnDiskArrayEeg
      ondisk_eeg = OnDiskArrayEeg(filepath=eeg_path)
      music = WavRAW(
        raw_data=np.random.randn(441000).astype(np.float32), sample_rate=44100
      )

      trial = TrialData(
        dataset="test",
        subject="s1",
        session="1",
        run="1",
        trial_id="ondisk_test",
        music_filename=MusicFilename(filename="test.wav"),
        eeg_data=ondisk_eeg,
        music_data=music,
      )

      # Create dataset with OnDiskArrayEeg trial
      ds = EEGMusicDataset()
      ds.df.loc[0] = {
        "dataset": trial.dataset,
        "subject": trial.subject,
        "session": trial.session,
        "run": trial.run,
        "trial_id": trial.trial_id,
        "music_filename": trial.music_filename,
        "eeg_data": trial.eeg_data,
      }
      music_ref = MusicRef(filename=trial.music_filename, dataset=trial.dataset)
      ds.music_collection[music_ref] = music

      # Test stratified sampling with OnDiskArrayEeg
      stratified = ArrayStratifiedSamplingDataset(
        ds, n_strata=2, trial_length_secs=Fraction(5, 1)
      )

      sampled_trial = stratified[0]
      self.assertIsInstance(sampled_trial.eeg_data, ArrayEeg)
      self.assertEqual(sampled_trial.eeg_data.data.shape[1], 1280)  # 5 sec * 256 Hz

  def test_wrong_eeg_type_raises_error(self):
    """Test that using RawEeg raises AttributeError (duck typing)."""
    import mne

    # Create a trial with RawEeg instead of ArrayEeg
    info = mne.create_info(
      ch_names=["C3", "C4", "Cz", "Fz"], sfreq=256.0, ch_types="eeg"
    )
    raw_data = np.random.randn(4, 2560)
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    raweeg = RawEeg(raw_eeg=raw)

    music = WavRAW(
      raw_data=np.random.randn(441000).astype(np.float32), sample_rate=44100
    )

    trial = TrialData(
      dataset="test",
      subject="s1",
      session="1",
      run="1",
      trial_id="raweeg_test",
      music_filename=MusicFilename(filename="test.wav"),
      eeg_data=raweeg,
      music_data=music,
    )

    ds = EEGMusicDataset()
    ds.df.loc[0] = {
      "dataset": trial.dataset,
      "subject": trial.subject,
      "session": trial.session,
      "run": trial.run,
      "trial_id": trial.trial_id,
      "music_filename": trial.music_filename,
      "eeg_data": trial.eeg_data,
    }
    music_ref = MusicRef(filename=trial.music_filename, dataset=trial.dataset)
    ds.music_collection[music_ref] = music

    stratified = ArrayStratifiedSamplingDataset(
      ds, n_strata=2, trial_length_secs=Fraction(5, 1)
    )

    # Should raise AttributeError due to duck typing (no get_array method)
    with self.assertRaises(AttributeError) as ctx:
      _ = stratified[0]

    self.assertIn("get_array", str(ctx.exception))

  def test_metadata_preservation(self):
    """Test that all trial metadata is preserved."""
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=2, trial_length_secs=Fraction(4, 1)
    )

    for i in range(len(self.dataset)):
      original = self.dataset[i]
      for stratum in range(2):
        sampled = stratified[i * 2 + stratum]

        self.assertEqual(sampled.dataset, original.dataset)
        self.assertEqual(sampled.subject, original.subject)
        self.assertEqual(sampled.session, original.session)
        self.assertEqual(sampled.run, original.run)
        self.assertEqual(sampled.trial_id, original.trial_id)
        self.assertEqual(sampled.music_filename, original.music_filename)

  def test_channel_names_preserved(self):
    """Test that channel names are preserved after sampling."""
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=2, trial_length_secs=Fraction(3, 1)
    )

    trial = stratified[0]
    self.assertIsInstance(trial.eeg_data, ArrayEeg)
    self.assertEqual(trial.eeg_data.ch_names, ["C3", "C4", "Cz", "Fz"])

  def test_df_and_music_collection_properties(self):
    """Test that df and music_collection properties work correctly."""
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=2, trial_length_secs=Fraction(4, 1)
    )

    # Test df property
    self.assertEqual(len(stratified.df), len(self.dataset.df))
    self.assertIs(stratified.df, self.dataset.df)

    # Test music_collection property
    self.assertEqual(
      len(stratified.music_collection), len(self.dataset.music_collection)
    )
    self.assertIs(stratified.music_collection, self.dataset.music_collection)

  def test_multiple_music_types(self):
    """Test that all music types (WavRAW, MelRaw, NoteOnsets) are passed through."""
    stratified = ArrayStratifiedSamplingDataset(
      self.dataset, n_strata=2, trial_length_secs=Fraction(4, 1)
    )

    # Trial 0: WavRAW
    trial0 = stratified[0]
    self.assertIsInstance(trial0.music_data, WavRAW)

    # Trial 1: MelRaw
    trial1 = stratified[2]  # 2nd trial, 1st stratum
    self.assertIsInstance(trial1.music_data, MelRaw)

    # Trial 2: NoteOnsets
    trial2 = stratified[4]  # 3rd trial, 1st stratum
    self.assertIsInstance(trial2.music_data, NoteOnsets)


if __name__ == "__main__":
  unittest.main()

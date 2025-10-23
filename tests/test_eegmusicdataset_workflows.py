import unittest
from pathlib import Path
import tempfile
import itertools
import numpy as np
from typing import cast

from eeg_music.bcmi import BCMICalibrationLoader, BCMITrainingLoader
from eeg_music.data import (
  EEGMusicDataset,
  copy_from_dataloader_into_dir,
  OnDiskMusic,
  OnDiskMel,
  OnDiskOnsets,
  wavraw_to_melspectrogram,
  MelRaw,
  WavRAW,
  NoteOnsets,
  MappedDataset,
  StratifiedSamplingDataset,
  prepare_trial,
  rereference_trial,
  MelParams,
  RepeatedDataset,
)
from eeg_music.onset_conversion import (
  trial_wavraw_to_noteonsets,
  trial_melraw_to_noteonsets,
)
from eeg_music.dataloader import mel_create_collate_fn
from fractions import Fraction


DATASET_ROOT = Path("/home/zmrocze/studia/uwr/magisterka/datasets")
BCMI_CAL_PATH = DATASET_ROOT / "bcmi" / "bcmi-calibration"
BCMI_TRN_PATH = DATASET_ROOT / "bcmi" / "bcmi-training"

# Exemplar WAV files for testing mel spectrogram roundtrip
EXEMPLAR_WAVS = {
  "bcmi/bcmi-calibration": [
    "stimuli/hvha5.wav",
    "stimuli/lvla1.wav",
    "stimuli/nvna3.wav",
  ],
  "bcmi/bcmi-training": ["stimuli/2-9_1.wav", "stimuli/4-1_3.wav"],
  "bcmi/bcmi-fmri": ["stimuli/generated/2-9_1.wav", "stimuli/generated/6-5_2.wav"],
  "musin_g_data/Code": ["ESongs/1.esh.wav", "ESongs/5.esh.wav"],
  "openmiir": ["stimuli/hvha1.wav", "stimuli/lvla5.wav"],
  "bcmi_preprocessed/bcmi_caltrain_256": [
    "stimuli/bcmi-calibration/hvha1.wav",
    "stimuli/bcmi-training/2-9_1_first.wav",
  ],
}


def dataset_exists(p: Path) -> bool:
  try:
    return p.exists() and any(p.iterdir())
  except Exception:
    return False


def _collect_trials_with_min_subjects(
  trial_iterator, min_subjects, trials_per_subject=2
):
  """Collect trials ensuring minimum number of unique subjects.

  Args:
    trial_iterator: Iterator over trials
    min_subjects: Minimum number of unique subjects required
    trials_per_subject: Target number of trials per subject (default: 2)

  Returns:
    List of trials with at least min_subjects unique subjects
  """
  collected_trials = []
  subject_counts = {}  # subject -> count of trials

  for trial in trial_iterator:
    subject = trial.subject

    # Track how many trials we have for this subject
    if subject not in subject_counts:
      subject_counts[subject] = 0

    # Only add trial if we need more trials for this subject
    if subject_counts[subject] < trials_per_subject:
      collected_trials.append(trial)
      subject_counts[subject] += 1

    # Check if we have enough subjects
    if len(subject_counts) >= min_subjects:
      # Check if all subjects have at least 1 trial (they should by construction)
      # and we've collected enough trials
      if all(count >= 1 for count in subject_counts.values()):
        break

  return collected_trials


def limit_loader_iterators(loader, max_trials: int = 6, min_subjects=None):
  """Limit trials and ensure music iterator yields exactly what trials need.

  We first materialize a limited list of trials, then filter music to only
  the referenced filenames. This guarantees no missing stimuli for trials.

  Args:
    loader: Dataset loader with trial_iterator and music_iterator methods
    max_trials: Maximum number of trials to include (used when min_subjects is None)
    min_subjects: If provided, collect trials to ensure at least this many unique subjects
  """

  # Load subjects - adjust based on min_subjects requirement
  if min_subjects is not None:
    loader.load_all_subjects(max_subjects=max(min_subjects, 3))
  else:
    loader.load_all_subjects(max_subjects=3)

  orig_trials = loader.trial_iterator
  orig_music = loader.music_iterator

  # Precompute limited trials
  if min_subjects is not None:
    # Use subject-aware collection
    limited_trials = _collect_trials_with_min_subjects(
      orig_trials(), min_subjects=min_subjects, trials_per_subject=2
    )
  else:
    # Original behavior: simple trial limit
    limited_trials = list(itertools.islice(orig_trials(), max_trials))

  needed = {t.music_filename for t in limited_trials}

  def trial_iter():
    for t in limited_trials:
      yield t

  def music_iter():
    yielded = set()
    for ref, music in orig_music():
      if ref in needed and ref not in yielded:
        yielded.add(ref)
        yield ref, music

  loader.trial_iterator = trial_iter
  loader.music_iterator = music_iter
  return loader


class TestEEGMusicDatasetWorkflows(unittest.TestCase):
  """End-to-end persistence scenarios for EEGMusicDataset.

  Scenarios implemented (run for both calibration and training when available):
  1) copy_from_dataloader_into_dir -> load_ondisk -> open files and check lengths
  2) like (1) + load_to_mem
  3) copy two datasets -> load -> merge -> verify
  4) copy same dataset twice into same dir -> load -> expect duplicates (current impl appends)
  5) remove_short_trials on merged dataset
  6) subject_wise_split on merged dataset
  7) load_ondisk -> save() -> load_ondisk -> compare identifiers
  8) Real-life flow: copy two datasets -> load -> load_to_mem -> save -> reload -> spot-check

  If any test fails:
  - (4) If duplicates are not doubled, implementation may be de-duplicating trials; adjust expectation.
  - (8) Failures often indicate RawEeg.save using mne.export.export_raw cannot write EDF in env
        (missing writer backend or MNE version). In that case, save() on RawEeg needs alternate export.
  """

  def setUp(self):
    self.available = {
      "calibration": dataset_exists(BCMI_CAL_PATH),
      "training": dataset_exists(BCMI_TRN_PATH),
    }

  def _mk_loader(self, kind: str):
    if kind == "calibration":
      return BCMICalibrationLoader(str(BCMI_CAL_PATH), dataset_name="bcmi-calibration")
    if kind == "training":
      return BCMITrainingLoader(str(BCMI_TRN_PATH), dataset_name="bcmi-training")
    raise ValueError(kind)

  def _ensure_available(self, kind: str):
    if not self.available.get(kind, False):
      self.skipTest(f"Dataset not found for {kind}")

  def _copy_and_merge_combined_dataset(
    self, f, min_subjects=None
  ):  # todo for copilot: add type annotation here
    """Helper function to copy and load combined calibration + training dataset.

    Returns:
        Combined EEGMusicDataset with both calibration and training data loaded and merged.
        Data is loaded to memory to avoid temporary directory cleanup issues.
    """
    if not (self.available["calibration"] and self.available["training"]):
      self.skipTest("Both calibration and training datasets required")

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
      cal_loader = self._mk_loader("calibration")
      trn_loader = self._mk_loader("training")
      limit_loader_iterators(cal_loader, min_subjects=min_subjects)
      limit_loader_iterators(trn_loader, min_subjects=min_subjects)

      copy_from_dataloader_into_dir(cal_loader, Path(d1))
      copy_from_dataloader_into_dir(trn_loader, Path(d2))

      cal_ds = EEGMusicDataset.load_ondisk(Path(d1))
      trn_ds = EEGMusicDataset.load_ondisk(Path(d2))
      merged = cal_ds.merge(trn_ds)

      # Load to memory to avoid temporary directory cleanup issues
      merged.load_to_mem()
      return f(merged)

  def _copy_and_load_combined_dataset(
    self, f, min_subjects=None
  ):  # todo for copilot: add type annotation here
    """Helper function to copy and load combined calibration + training dataset.

    Returns:
        Combined EEGMusicDataset with both calibration and training data loaded and merged.
        Data is loaded to memory to avoid temporary directory cleanup issues.
    """
    if not (self.available["calibration"] and self.available["training"]):
      self.skipTest("Both calibration and training datasets required")

    with tempfile.TemporaryDirectory() as d1:
      cal_loader = self._mk_loader("calibration")
      trn_loader = self._mk_loader("training")
      limit_loader_iterators(cal_loader, min_subjects=min_subjects)
      limit_loader_iterators(trn_loader, min_subjects=min_subjects)
      copy_from_dataloader_into_dir(cal_loader, Path(d1))
      copy_from_dataloader_into_dir(trn_loader, Path(d1))
      ds = EEGMusicDataset.load_ondisk(Path(d1))
      return f(ds)

  # 1. copy -> load_ondisk; inspect counts and sample lengths
  def test_copy_and_load_ondisk_lengths(self):
    for kind in ("calibration", "training"):
      with self.subTest(kind=kind):
        self._ensure_available(kind)
        loader = self._mk_loader(kind)
        limit_loader_iterators(loader)
        with tempfile.TemporaryDirectory() as d:
          base = Path(d)
          copy_from_dataloader_into_dir(loader, base)
          ds = EEGMusicDataset.load_ondisk(base)
          self.assertGreater(len(ds), 0)
          # Verify music and eeg can be opened and have positive durations
          t = ds[0]
          mus = t.music_data.get_music()
          self.assertGreater(mus.length_seconds(), 0.0)
          eeg = t.eeg_data.get_eeg().raw_eeg
          sf = float(eeg.info.get("sfreq", 0.0))
          self.assertGreater(sf, 0.0)
          self.assertGreater(eeg.n_times, 0)

  # 2. copy -> load_ondisk -> load_to_mem
  def test_copy_and_load_to_mem(self):
    for kind in ("calibration", "training"):
      with self.subTest(kind=kind):
        self._ensure_available(kind)
        loader = self._mk_loader(kind)
        limit_loader_iterators(loader)
        with tempfile.TemporaryDirectory() as d:
          base = Path(d)
          copy_from_dataloader_into_dir(loader, base)
          ds = EEGMusicDataset.load_ondisk(base)
          n = len(ds)
          ds.load_to_mem()
          self.assertEqual(len(ds), n)
          # Spot check types via behavior
          t = ds[0]
          self.assertGreater(t.music_data.get_music().length_seconds(), 0.0)
          eeg = t.eeg_data.get_eeg().raw_eeg
          self.assertGreater(eeg.n_times, 0)

  # 3. copy two datasets -> load both -> merge -> verify
  def test_merge_datasets(self):
    def action(ds):
      # We can't easily verify the exact split since helper merges internally,
      # but we can verify basic properties
      self.assertGreater(len(ds), 0)
      # Music collection should contain items from both datasets
      self.assertGreater(len(ds.music_collection), 0)

    self._copy_and_merge_combined_dataset(action)

  # 5. remove_short_trials on combined dataset
  def test_remove_short_trials(self):
    def action(ds):
      filtered = ds.remove_short_trials(5.0)
      self.assertLessEqual(len(filtered), len(ds))
      # Sanity: all remaining trials have >= threshold durations
      for i in range(len(filtered)):
        t = filtered[i]
        eeg = t.eeg_data.get_eeg().raw_eeg
        sf = float(eeg.info.get("sfreq", 0.0))
        self.assertGreaterEqual(eeg.n_times / sf, 5.0)

    return self._copy_and_load_combined_dataset(action)

  # 6. subject_wise_split on merged dataset
  def test_subject_wise_split(self):
    def action(ds):
      tr, va, te = ds.subject_wise_split(0.5, 0.0, seed=123)
      self.assertEqual(len(tr) + len(va) + len(te), len(ds))

      def pairs(dataset: EEGMusicDataset) -> set[tuple[str, str]]:
        cols = dataset.df.reset_index(drop=True)[["dataset", "subject"]]
        return {
          (cast(str, row.dataset), cast(str, row.subject))  # type: ignore[reportAttributeAccessIssue]
          for row in cols.drop_duplicates().itertuples(index=False)
        }

      tr_pairs = pairs(tr)
      va_pairs = pairs(va)
      te_pairs = pairs(te)
      all_pairs = pairs(ds)

      self.assertTrue(tr_pairs.isdisjoint(va_pairs))
      self.assertTrue(tr_pairs.isdisjoint(te_pairs))
      self.assertTrue(va_pairs.isdisjoint(te_pairs))
      self.assertSetEqual(tr_pairs | va_pairs | te_pairs, all_pairs)

    self._copy_and_load_combined_dataset(action, min_subjects=8)

  # 7. load -> save -> reload -> compare basic invariants
  def test_save_roundtrip(self):
    for kind in ("calibration", "training"):
      with self.subTest(kind=kind):
        self._ensure_available(kind)
        loader = self._mk_loader(kind)
        limit_loader_iterators(loader)
        with (
          tempfile.TemporaryDirectory() as dsrc,
          tempfile.TemporaryDirectory() as ddst,
        ):
          base = Path(dsrc)
          copy_from_dataloader_into_dir(loader, base)
          ds = EEGMusicDataset.load_ondisk(base)
          out = Path(ddst) / "saved"
          ds.save(out)
          ds2 = EEGMusicDataset.load_ondisk(out)
          self.assertEqual(len(ds2), len(ds))
          # Compare per-row identifiers
          a = (
            ds.df[
              ["dataset", "subject", "session", "run", "trial_id", "music_filename"]
            ]
            .copy()
            .reset_index(drop=True)
          )
          b = (
            ds2.df[
              ["dataset", "subject", "session", "run", "trial_id", "music_filename"]
            ]
            .copy()
            .reset_index(drop=True)
          )
          self.assertEqual(len(a), len(b))
          # Use set of tuples to avoid ordering sensitivity
          self.assertEqual(
            set(map(tuple, a.values.tolist())), set(map(tuple, b.values.tolist()))
          )

  # 8. Mel spectrogram save/load roundtrip test
  def test_mel_spectrogram_roundtrip(self):
    """Test mel spectrogram computation, saving, and loading roundtrip.

    This test verifies that:
    1. WAV files can be loaded correctly
    2. Mel spectrograms can be computed from WAV data
    3. Mel spectrograms can be saved to disk as compressed .npz files
    4. Saved mel spectrograms can be reloaded correctly
    5. Reloaded mel spectrograms match the original computation within tolerance
    """
    for ds_name, wav_paths in EXEMPLAR_WAVS.items():
      base = DATASET_ROOT / ds_name
      if not dataset_exists(base):
        self.skipTest(f"Dataset not found: {ds_name}")

      # Test up to 2 files per dataset for speed
      for wav_file in wav_paths[:2]:
        with self.subTest(dataset=ds_name, file=wav_file):
          wav_path = base / wav_file
          if not wav_path.exists():
            self.skipTest(f"WAV file not found: {wav_path}")

          # 1) Load WAV -> WavRAW
          wav_raw = OnDiskMusic(wav_path).get_music()
          self.assertTrue(wav_raw.is_not_empty())
          self.assertGreater(wav_raw.length_seconds(), 0.0)
          self.assertGreater(wav_raw.sample_rate, 0)

          # 2) Compute mel spectrogram with different parameter sets
          test_params = [
            # Standard parameters
            {
              "n_mels": 128,
              "hop_length": 512,
              "fmin": 0.0,
              "fmax": None,
              "to_db": True,
            },
            # Alternative parameters
            {
              "n_mels": 64,
              "hop_length": 256,
              "fmin": 80.0,
              "fmax": 8000.0,
              "to_db": False,
            },
          ]

          for params in test_params:
            with self.subTest(params=params):
              mel_raw = wavraw_to_melspectrogram(wav_raw, **params)

              # Verify mel spectrogram properties
              self.assertEqual(mel_raw.mel.shape[0], params["n_mels"])
              self.assertGreater(mel_raw.mel.shape[1], 0)  # n_frames
              self.assertEqual(mel_raw.sample_rate, wav_raw.sample_rate)
              self.assertEqual(mel_raw.hop_length, params["hop_length"])
              self.assertEqual(mel_raw.fmin, params["fmin"])
              self.assertEqual(mel_raw.fmax, params["fmax"])
              self.assertEqual(mel_raw.to_db, params["to_db"])

              with tempfile.TemporaryDirectory() as temp_dir:
                # 3) Save mel spectrogram
                mel_path = Path(temp_dir) / "test_mel.npz"
                mel_raw.save(mel_path)
                self.assertTrue(mel_path.exists())

                # 4) Reload mel spectrogram
                reloaded_mel = OnDiskMel(mel_path).get_music()

                # 5) Verify roundtrip consistency
                self.assertTrue(
                  np.allclose(reloaded_mel.mel, mel_raw.mel, atol=1e-5, rtol=1e-3),
                  "Mel spectrogram data should match after save/load roundtrip",
                )
                self.assertEqual(reloaded_mel.sample_rate, mel_raw.sample_rate)
                self.assertEqual(reloaded_mel.hop_length, mel_raw.hop_length)
                self.assertEqual(reloaded_mel.fmin, mel_raw.fmin)
                self.assertEqual(reloaded_mel.fmax, mel_raw.fmax)
                self.assertEqual(reloaded_mel.to_db, mel_raw.to_db)

                # 6) Verify mel spectrogram shape and properties
                self.assertEqual(reloaded_mel.mel.shape, mel_raw.mel.shape)
                self.assertGreater(reloaded_mel.length_seconds(), 0.0)

  # 8. Real-life flow: copy two datasets -> load -> load_to_mem -> save -> reload -> check
  def test_real_life_flow(self):
    def action(ds):
      ds.load_to_mem()
      with tempfile.TemporaryDirectory() as ddst:
        out = Path(ddst) / "out"
        # NOTE: If this save fails, most likely mne.export.export_raw (used by RawEeg.save)
        # cannot write EDF in the current environment. This indicates a missing/export backend
        # or incompatible MNE version, not a logic error in dataset wiring.
        ds.save(out)
        ds2 = EEGMusicDataset.load_ondisk(out)
        self.assertEqual(len(ds2), len(ds))
        # Spot-check random item
        t = ds2[0]
        self.assertGreater(t.music_data.get_music().length_seconds(), 0.0)
        eeg = t.eeg_data.get_eeg().raw_eeg
        self.assertGreater(eeg.n_times, 0)

    self._copy_and_load_combined_dataset(action)

  # 9. Test prepare_trial workflows with/without mel and with/without stratification
  def test_prepare_trial_workflows(self):
    """Test prepare_trial with/without mel + with/without stratification.

    This test covers 4 scenarios in a 2x2 matrix:
    - Dimension 1: Mel Transform (None vs MelParams)
    - Dimension 2: Stratified Sampling (Direct vs StratifiedSamplingDataset)

    For each scenario, we test:
    1. Data processing with prepare_trial
    2. Optional stratified sampling
    3. Save/load roundtrip of sample trials
    4. Property verification (sample rates, shapes, lengths, etc.)
    """

    # Test scenarios in 2x2 matrix
    scenarios = [
      {"apply_mel": None, "use_stratified": False, "name": "wav_direct"},
      {"apply_mel": None, "use_stratified": True, "name": "wav_stratified"},
      {
        "apply_mel": MelParams(n_mels=128, hop_length=512, fmax=10240.0, to_db=False),
        "use_stratified": False,
        "name": "mel_direct",
      },
      {
        "apply_mel": MelParams(n_mels=32, hop_length=256, fmax=5000.0),
        "use_stratified": True,
        "name": "mel_stratified",
      },
    ]

    for scenario in scenarios:
      with self.subTest(scenario=scenario["name"]):
        # 1. Get combined dataset
        def action(base_dataset):
          # 2. Apply prepare_trial processing
          mapped_dataset = MappedDataset(
            base_dataset,
            lambda t: prepare_trial(
              t,
              eeg_resample=256,
              eeg_l_freq=0.0,
              eeg_h_freq=50.0,
              apply_mel=scenario["apply_mel"],
            ),
          )

          # 3. Optional stratification
          if scenario["use_stratified"]:
            processed_dataset = StratifiedSamplingDataset(
              mapped_dataset, n_strata=3, trial_length_secs=Fraction(4, 1)
            )
          else:
            processed_dataset = mapped_dataset

          # 4. Verify dataset properties
          self.assertGreater(len(processed_dataset), 0)

          if scenario["use_stratified"]:
            # For stratified: length should be base_length * n_strata
            self.assertEqual(len(processed_dataset), len(mapped_dataset) * 3)

          # 5. Test a few sample trials
          test_indices = [
            0,
            min(len(processed_dataset) - 1, 5),
            min(len(processed_dataset) - 1, 10),
          ]

          for idx in test_indices:
            if idx >= len(processed_dataset):
              continue

            trial = processed_dataset[idx]

            # 6. Verify EEG properties after prepare_trial
            eeg = trial.eeg_data.get_eeg().raw_eeg
            self.assertEqual(float(eeg.info["sfreq"]), 256.0)  # Resampled to 256 Hz
            self.assertGreater(eeg.n_times, 0)

            # 7. Verify music properties based on mel transform
            music = trial.music_data.get_music()
            self.assertGreater(music.length_seconds(), 0.0)

            if scenario["apply_mel"] is None:
              # Should be WavRAW
              self.assertIsInstance(music, WavRAW)
              self.assertGreater(music.sample_rate, 0)
            else:
              # Should be MelRaw
              self.assertIsInstance(music, MelRaw)
              mel_music = cast(MelRaw, music)
              mel_params = scenario["apply_mel"]
              self.assertEqual(mel_music.mel.shape[0], mel_params.n_mels)
              self.assertGreater(mel_music.mel.shape[1], 0)  # n_frames > 0
              self.assertEqual(mel_music.hop_length, mel_params.hop_length)
              self.assertEqual(mel_music.fmax, mel_params.fmax)

            # 8. For stratified datasets, verify time slice properties
            if scenario["use_stratified"]:
              # EEG duration should be approximately 4 seconds (trial_length_secs)
              eeg_duration = eeg.n_times / float(eeg.info["sfreq"])
              self.assertAlmostEqual(eeg_duration, 4.0, delta=0.1)

          # 9. Test save/load roundtrip for a sample
          if len(processed_dataset) > 0:
            sample_trial = processed_dataset[0]

            with tempfile.TemporaryDirectory() as temp_dir:
              # Save EEG data
              eeg_path = Path(temp_dir) / "test_eeg.edf"
              sample_trial.eeg_data.save(eeg_path)
              self.assertTrue(eeg_path.exists())

              # Save music data
              if scenario["apply_mel"] is None:
                music_path = Path(temp_dir) / "test_music.wav"
              else:
                music_path = Path(temp_dir) / "test_music.npz"

              sample_trial.music_data.save(music_path)
              self.assertTrue(music_path.exists())

              # Reload and verify basic properties match
              if scenario["apply_mel"] is None:
                reloaded_music = OnDiskMusic(music_path).get_music()
                original_music = sample_trial.music_data.get_music()
                self.assertAlmostEqual(
                  reloaded_music.length_seconds(),
                  original_music.length_seconds(),
                  delta=0.01,
                )
              else:
                reloaded_music = OnDiskMel(music_path).get_music()
                original_music = cast(MelRaw, sample_trial.music_data.get_music())
                reloaded_mel = cast(MelRaw, reloaded_music)
                self.assertEqual(reloaded_mel.mel.shape, original_music.mel.shape)
                self.assertTrue(
                  np.allclose(
                    reloaded_mel.mel, original_music.mel, atol=1e-5, rtol=1e-3
                  )
                )

        return self._copy_and_load_combined_dataset(action)

  # 10a. Test NoteOnsets with WAV
  def test_noteonsets_from_wav(self):
    """Test NoteOnsets: prepare_trial (wav) -> convert to onsets -> save/load -> stratified sampling."""

    def action(base_dataset):
      self._test_noteonsets_pipeline(
        base_dataset,
        prepare_fn=lambda t: prepare_trial(t, eeg_resample=256),
        convert_fn=trial_wavraw_to_noteonsets,
      )

    return self._copy_and_load_combined_dataset(action)

  # 10b. Test NoteOnsets with Mel
  @unittest.skip("Ignoring this test")
  def test_noteonsets_from_mel(self):
    """Test NoteOnsets: prepare_trial (mel) -> convert to onsets -> save/load -> stratified sampling."""

    def action(base_dataset):
      self._test_noteonsets_pipeline(
        base_dataset,
        prepare_fn=lambda t: prepare_trial(
          t, eeg_resample=256, apply_mel=MelParams(n_mels=128)
        ),
        convert_fn=trial_melraw_to_noteonsets,
      )

    return self._copy_and_load_combined_dataset(action)

  def _test_noteonsets_pipeline(self, base_dataset, prepare_fn, convert_fn):
    """Helper to test NoteOnsets pipeline with different music types."""
    # Step 1: Prepare trial and convert to NoteOnsets
    prepared_trial = prepare_fn(base_dataset[0])
    onset_trial = convert_fn(prepared_trial)
    music = onset_trial.music_data.get_music()
    self.assertIsInstance(music, NoteOnsets)

    if isinstance(music, NoteOnsets):  # Type narrowing for linter
      self.assertGreater(len(music.onset_times), 0)
      self.assertTrue(np.all(music.onset_times >= 0.0))
      self.assertTrue(np.all(music.onset_times < music.duration_seconds))

      # Step 2: Save/load roundtrip for NoteOnsets
      with tempfile.TemporaryDirectory() as temp_dir:
        onset_path = Path(temp_dir) / "test.npz"
        music.save(onset_path)
        reloaded = OnDiskOnsets(onset_path).get_music()
        self.assertTrue(np.allclose(reloaded.onset_times, music.onset_times))
        self.assertAlmostEqual(
          reloaded.duration_seconds, music.duration_seconds, places=5
        )

    # Step 3: NoteOnsets with StratifiedSamplingDataset
    onset_dataset = MappedDataset(
      base_dataset,
      lambda t: convert_fn(prepare_fn(t)),  # type: ignore[arg-type]
    )
    stratified = StratifiedSamplingDataset(
      onset_dataset, n_strata=2, trial_length_secs=Fraction(4, 1)
    )
    trial_strat = stratified[0]
    music_strat = trial_strat.music_data.get_music()
    self.assertIsInstance(music_strat, NoteOnsets)

    if isinstance(music_strat, NoteOnsets):  # Type narrowing for linter
      self.assertAlmostEqual(music_strat.duration_seconds, 4.0, delta=0.1)
      if len(music_strat.onset_times) > 0:
        self.assertTrue(np.all(music_strat.onset_times >= 0.0))
        self.assertTrue(np.all(music_strat.onset_times < music_strat.duration_seconds))

  # 11. Test complex pipeline: mapped(prepare_trial) -> stratified -> mapped(rereference_trial)
  def test_complex_pipeline_with_save_load(self):
    """Test a complex dataset processing pipeline with save/load roundtrip.

    This test verifies:
    1. A complex pipeline: MappedDataset(prepare_trial) -> StratifiedSamplingDataset -> MappedDataset(rereference_trial)
    2. Save/load roundtrip preserves data integrity
    3. Original dataset is not accidentally modified during processing
    """

    def action(original_ds):
      # Store original dataset state for comparison
      original_len = len(original_ds)
      original_first_trial_id = original_ds.df.iloc[0]["trial_id"]
      original_music_collection_size = len(original_ds.music_collection)

      # Create a copy and load to memory to avoid modification during processing
      original_ds.load_to_mem()

      # Store a reference to original data for comparison
      original_first_trial = original_ds[0]
      original_eeg_data_id = id(original_first_trial.eeg_data.get_eeg().raw_eeg)
      original_music_data_id = id(original_first_trial.music_data.get_music())

      # Test Step 1: Apply prepare_trial mapping ONLY
      # First let's just test prepare_trial without mel to see if the mapping works
      mapped_ds1 = MappedDataset(
        original_ds,
        lambda trial: prepare_trial(
          trial,
          eeg_resample=256,
          eeg_l_freq=1.0,
          eeg_h_freq=50.0,
          wav_resample=22050,
          apply_mel=None,  # Start without mel to isolate the issue
        ),
      )

      # Test that mapping works
      first_mapped = mapped_ds1[0]
      self.assertEqual(first_mapped.eeg_data.get_eeg().raw_eeg.info["sfreq"], 256.0)
      self.assertIsInstance(first_mapped.music_data.get_music(), WavRAW)

      # Now try with mel
      mapped_ds_mel = MappedDataset(
        original_ds,
        lambda trial: prepare_trial(
          trial,
          eeg_resample=256,
          eeg_l_freq=1.0,
          eeg_h_freq=50.0,
          wav_resample=22050,
          apply_mel=MelParams(n_mels=64, hop_length=256, to_db=True),
        ),
      )

      # Test that mel transformation works
      first_mel = mapped_ds_mel[0]
      if not isinstance(first_mel.music_data.get_music(), MelRaw):
        # Skip the rest of the test if mel transform fails

        # self.skipTest(
        #   f"Mel transformation failed, got {type(first_mel.music_data.get_music())} instead of MelRaw"
        # )

        # Step 2: Apply stratified sampling
        stratified_ds = StratifiedSamplingDataset(
          mapped_ds_mel,
          n_strata=3,
          trial_length_secs=Fraction(2, 1),  # 2 seconds
        )

        # Step 3: Apply rereference_trial mapping
        final_ds = MappedDataset(stratified_ds, rereference_trial)

        # Verify we can access items from the final dataset
        self.assertGreater(len(final_ds), 0)
        sample_trial = final_ds[0]

        # Verify the pipeline worked correctly
        music_data = sample_trial.music_data.get_music()
        self.assertIsInstance(music_data, MelRaw)
        self.assertEqual(sample_trial.eeg_data.get_eeg().raw_eeg.info["sfreq"], 256.0)

        # Verify sample shapes and basic properties
        mel_data = cast(MelRaw, music_data)
        self.assertEqual(mel_data.mel.shape[0], 64)  # n_mels
        self.assertTrue(mel_data.to_db)

        eeg_data = sample_trial.eeg_data.get_eeg().raw_eeg
        expected_samples = 2 * 256  # 2 seconds at 256 Hz
        self.assertEqual(eeg_data.n_times, expected_samples)

        # Test save/load roundtrip
        with tempfile.TemporaryDirectory() as temp_dir:
          save_path = Path(temp_dir) / "complex_pipeline_dataset"

          # Save the final processed dataset
          final_ds.save(save_path)

          # Load it back
          reloaded_ds = EEGMusicDataset.load_ondisk(save_path)

          # Compare basic properties
          self.assertEqual(len(reloaded_ds), len(final_ds))

          # Compare a sample trial
          original_sample = final_ds[0]
          reloaded_sample = reloaded_ds[0]

          # Check trial metadata
          self.assertEqual(original_sample.dataset, reloaded_sample.dataset)
          self.assertEqual(original_sample.subject, reloaded_sample.subject)
          self.assertEqual(original_sample.trial_id, reloaded_sample.trial_id)
          self.assertEqual(
            original_sample.music_filename.filename,
            reloaded_sample.music_filename.filename,
          )

          # Check EEG data properties
          orig_eeg = original_sample.eeg_data.get_eeg().raw_eeg
          reload_eeg = reloaded_sample.eeg_data.get_eeg().raw_eeg
          self.assertEqual(orig_eeg.info["sfreq"], reload_eeg.info["sfreq"])
          self.assertEqual(orig_eeg.n_times, reload_eeg.n_times)
          self.assertEqual(len(orig_eeg.ch_names), len(reload_eeg.ch_names))

          # Check music data properties
          orig_music = original_sample.music_data.get_music()
          reload_music = reloaded_sample.music_data.get_music()
          self.assertIsInstance(reload_music, MelRaw)

          # Cast to MelRaw for property comparisons
          orig_mel = cast(MelRaw, orig_music)
          reload_mel = cast(MelRaw, reload_music)

          self.assertEqual(orig_mel.mel.shape, reload_mel.mel.shape)
          self.assertEqual(orig_mel.sample_rate, reload_mel.sample_rate)
          self.assertEqual(orig_mel.hop_length, reload_mel.hop_length)
          self.assertEqual(orig_mel.to_db, reload_mel.to_db)

          # Check mel data is approximately equal (allowing for float precision)
          np.testing.assert_allclose(orig_mel.mel, reload_mel.mel, rtol=1e-5, atol=1e-6)

        # Verify original dataset was not modified
        self.assertEqual(len(original_ds), original_len)
        self.assertEqual(original_ds.df.iloc[0]["trial_id"], original_first_trial_id)
        self.assertEqual(
          len(original_ds.music_collection), original_music_collection_size
        )

        # Verify original data objects are still the same (not deep copied accidentally)
        current_first_trial = original_ds[0]
        current_eeg_data_id = id(current_first_trial.eeg_data.get_eeg().raw_eeg)
        current_music_data_id = id(current_first_trial.music_data.get_music())

        # Note: These checks verify that the original dataset's data objects
        # are not accidentally replaced during pipeline processing
        self.assertEqual(original_eeg_data_id, current_eeg_data_id)
        self.assertEqual(original_music_data_id, current_music_data_id)

    return self._copy_and_load_combined_dataset(action, min_subjects=4)

  def test_full_dataloader_pipeline_with_repeated_dataset(self):
    """Test the complete dataloader pipeline including RepeatedDataset.

    This test verifies the full workflow similar to create_dataloaders:
    1. Load combined dataset (limited for testing)
    2. Apply prepare_trial with mel transformation
    3. Save and reload dataset
    4. Apply subject_wise_split
    5. Apply StratifiedSamplingDataset and rereference_trial mapping
    6. Apply RepeatedDataset to test dataset (repeat 20 times)
    7. Create dataloaders
    8. Verify batch contents and lengths
    """
    from torch.utils.data import DataLoader
    from fractions import Fraction

    def action(base_dataset):
      # Step 1: Apply prepare_trial with mel transformation
      mel_params = MelParams(
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fmin=0,
        fmax=8192,
        to_db=True,
      )

      prepared_ds = MappedDataset(
        base_dataset,
        lambda t: prepare_trial(
          t,
          eeg_resample=256,
          eeg_l_freq=0.1,
          eeg_h_freq=100.0,
          wav_resample=32768,  # 64 * 512
          apply_mel=mel_params,
        ),
      )

      # Verify mel transformation worked
      first_sample = prepared_ds[0]
      self.assertIsInstance(first_sample.music_data.get_music(), MelRaw)

      # Step 2: Save and reload
      with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "prepared_dataset"
        prepared_ds.save(save_path)
        reloaded_ds = EEGMusicDataset.load_ondisk(save_path)

        # Step 3: Apply subject_wise_split
        train_ds, val_ds, test_ds = reloaded_ds.subject_wise_split(
          p_train=0.5, p_val=0.25, seed=42
        )

        # Verify splits exist
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertGreater(len(test_ds), 0)

        # Step 4: Apply StratifiedSamplingDataset and rereference_trial
        def apply_processing(ds):
          stratified = StratifiedSamplingDataset(
            ds,
            n_strata=10,
            trial_length_secs=Fraction(4, 1),
          )
          dereferenced = MappedDataset(stratified, rereference_trial)
          return dereferenced

        train_processed = apply_processing(train_ds)
        val_processed = apply_processing(val_ds)
        test_processed = apply_processing(test_ds)

        # Step 5: Apply RepeatedDataset to test dataset (20 times)
        test_repeated = RepeatedDataset(test_processed, num_repeats=20)

        # Verify RepeatedDataset length
        expected_test_length = len(test_processed) * 20
        self.assertEqual(len(test_repeated), expected_test_length)

        # Verify RepeatedDataset returns correct items
        # First item should be the same as the original first item
        original_first = test_processed[0]
        repeated_first = test_repeated[0]
        self.assertEqual(original_first.trial_id, repeated_first.trial_id)
        self.assertEqual(original_first.subject, repeated_first.subject)

        # Item at len(test_processed) should wrap around to index 0
        repeated_wrapped = test_repeated[len(test_processed)]
        self.assertEqual(original_first.trial_id, repeated_wrapped.trial_id)

        # Step 6: Create dataloaders
        batch_size = 1

        # Use collate function from dataloader module (with metadata)
        collate_fn = mel_create_collate_fn(include_info=True)

        train_loader = DataLoader(
          train_processed,
          batch_size=batch_size,
          shuffle=True,
          collate_fn=collate_fn,
          drop_last=True,
        )

        val_loader = DataLoader(
          val_processed,
          batch_size=batch_size,
          shuffle=False,
          collate_fn=collate_fn,
          drop_last=False,
        )

        test_loader = DataLoader(
          test_repeated,
          batch_size=batch_size,
          shuffle=False,
          collate_fn=collate_fn,
          drop_last=False,
        )

        # Step 7: Test dataloaders - verify batch contents and lengths

        # Test train loader
        train_batches = list(train_loader)
        self.assertGreater(len(train_batches), 0)

        for batch in train_batches:
          # Verify batch structure
          self.assertIn("eeg", batch)
          self.assertIn("music", batch)
          self.assertIn(
            "info", batch
          )  # mel_create_collate_fn uses 'info' not 'metadata'

          # Verify batch dimensions
          self.assertEqual(batch["eeg"].shape[0], batch_size)  # batch dimension
          self.assertEqual(batch["music"].shape[0], batch_size)

          # Verify metadata
          self.assertEqual(len(batch["info"]["trial_id"]), batch_size)
          self.assertEqual(len(batch["info"]["subject"]), batch_size)

          # Verify emotion codes are present
          self.assertIn("emotion", batch["info"])
          self.assertEqual(len(batch["info"]["emotion"]), batch_size)

          # Verify emotion codes are parsed (should be integers 1-9 or None)
          for emotion in batch["info"]["emotion"]:
            if emotion is not None:
              self.assertIsInstance(emotion, int)
              self.assertGreaterEqual(emotion, 1)
              self.assertLessEqual(emotion, 9)

          # Verify EEG has correct number of channels (28 channels after picking)
          self.assertLessEqual(batch["eeg"].shape[1], 64)  # channels
          self.assertGreater(batch["eeg"].shape[2], 0)  # time samples

          # Verify music has correct shape
          self.assertEqual(batch["music"].shape[1], 128)  # n_mels
          self.assertGreater(batch["music"].shape[2], 0)  # time frames

        # Test validation loader
        val_batches = list(val_loader)
        self.assertGreater(len(val_batches), 0)

        # Test repeated test loader
        test_batches = list(test_loader)
        self.assertGreater(len(test_batches), 0)

        # Verify test loader has approximately 20x more batches than without repetition
        # Calculate expected number of batches
        non_repeated_batches = (len(test_processed) + batch_size - 1) // batch_size
        expected_repeated_batches = (len(test_repeated) + batch_size - 1) // batch_size

        # Due to drop_last=False, the last batch might not be full
        self.assertGreaterEqual(len(test_batches), non_repeated_batches * 19)
        self.assertLessEqual(len(test_batches), expected_repeated_batches)

        # Verify that we see repeated data in test loader
        # Collect all trial_ids from test loader
        test_trial_ids = []
        for batch in test_batches:
          test_trial_ids.extend(
            batch["info"]["trial_id"]
          )  # mel_create_collate_fn uses 'info'

        # Count occurrences of first trial_id
        first_trial_id = test_processed[0].trial_id
        first_trial_count = test_trial_ids.count(first_trial_id)

        # Should appear approximately 20 times (might vary due to batching)
        self.assertGreaterEqual(
          first_trial_count,
          15,
          f"Expected trial to repeat ~20 times, got {first_trial_count}",
        )

        print(f"✓ Train batches: {len(train_batches)}")
        print(f"✓ Val batches: {len(val_batches)}")
        print(f"✓ Test batches: {len(test_batches)} (20x repeated)")
        print(
          f"✓ Test samples total: {len(test_trial_ids)} (expected ~{len(test_repeated)})"
        )
        print(f"✓ First trial appeared {first_trial_count} times in test set")

    return self._copy_and_load_combined_dataset(action, min_subjects=8)


if __name__ == "__main__":
  unittest.main()

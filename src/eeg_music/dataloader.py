from fractions import Fraction
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Callable, Any, Optional, Sequence
from dataclasses import dataclass
from .subject_specific import SubjectDatasetMapper
from .data import (
  ArrayStratifiedSamplingDataset,
  EEGMusicDataset,
  MappedDataset,
  MelRaw,
  MusicData,
  NoteOnsets,
  RepeatedDataset,
  RobustNormalizationStats,
  RobustNormalizedDataset,
  StratifiedSamplingDataset,
  TrialData,
  EegData,
  WavRAW,
  rereference_trial,
  trial_to_arrayeeg,
  robust_normalize_trial,
)
from .emotion_utils import parse_music_emotion
from pathlib import Path


@dataclass
class SubjectWiseSplit:
  pass


@dataclass
class TrialWiseSplit:
  pass


def after_loaded_ds(ds, trial_length_secs=Fraction(4, 1)):
  stratified = StratifiedSamplingDataset(
    ds,
    n_strata=10,
    trial_length_secs=trial_length_secs,
  )

  dereferenced = MappedDataset(stratified, rereference_trial)

  return dereferenced


def load_and_create_dataloaders(
  ds_path: Path,
  config,
  collate_fn=None,
  include_mapper: bool = False,
  split_type: SubjectWiseSplit | TrialWiseSplit = SubjectWiseSplit(),
  after_loaded_ds=after_loaded_ds,
) -> Dict[str, Any]:
  # Path("./datasets/bcmi_combined_prepared_mel_28ch")
  ds = EEGMusicDataset.load_ondisk(ds_path)

  # Choose split method based on split_type
  match split_type:
    case SubjectWiseSplit():
      split_result = ds.subject_wise_split(
        p_train=config.ds_p_train,
        p_val=config.ds_p_val,
        seed=config.ds_split_seed,
      )
    case TrialWiseSplit():
      split_result = ds.trial_wise_split(
        p_train=config.ds_p_train,
        p_val=config.ds_p_val,
        seed=config.ds_split_seed,
      )

  train_ds = split_result["train"]
  val_ds = split_result.get("val", split_result["test"])  # fallback to test if no val
  test_ds = split_result["test"]

  mapper = None
  if include_mapper:
    mapper = SubjectDatasetMapper()
    for _, row in ds.df[["dataset", "subject"]].drop_duplicates().iterrows():
      mapper.add_subject(str(row["dataset"]), str(row["subject"]))

  # Get chunk width from config (default to 4 seconds if not specified)
  trial_length_secs = getattr(config, "ds_chunk_width", Fraction(4, 1))

  dereferenced = after_loaded_ds(train_ds, trial_length_secs=trial_length_secs)
  dereferenced_val = after_loaded_ds(val_ds, trial_length_secs=trial_length_secs)
  dereferenced_tst = after_loaded_ds(test_ds, trial_length_secs=trial_length_secs)
  if config.ds_use_test_for_val:  # for when p_val=0
    dereferenced_val = dereferenced_tst

  ds_train_repeated_mul = getattr(config, "ds_train_repeated_mul", 1)
  ds_val_repeated_mul = getattr(config, "ds_val_repeated_mul", 1)
  ds_test_repeated_mul = getattr(config, "ds_test_repeated_mul", 1)

  if ds_train_repeated_mul > 1:
    dereferenced = RepeatedDataset(dereferenced, ds_train_repeated_mul)
  if ds_val_repeated_mul > 1:
    dereferenced_val = RepeatedDataset(dereferenced_val, ds_val_repeated_mul)
  if ds_test_repeated_mul > 1:
    dereferenced_tst = RepeatedDataset(dereferenced_tst, ds_test_repeated_mul)

  include_info = getattr(config, "include_info", False)
  if collate_fn is None:
    collate_fn = mel_create_collate_fn(include_info=include_info)

  pin_memory = getattr(config, "pin_memory", True)

  train_dl = create_dataloader(
    dereferenced,
    is_training=True,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )
  val_dl = create_dataloader(
    dereferenced_val,
    is_training=False,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )
  test_dl = create_dataloader(
    dereferenced_tst,
    is_training=False,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )
  result: Dict[str, Any] = {"train": train_dl, "val": val_dl, "test": test_dl}
  if include_mapper and mapper is not None:
    result["mapper"] = mapper
  if "num_skipped_trials" in split_result:
    result["num_skipped_trials"] = split_result["num_skipped_trials"]

  return result


def create_dataloaders_but_with_normalization(
  ds_path: Path,
  config,
  collate_fn=None,
  include_mapper: bool = False,
  split_type: SubjectWiseSplit | TrialWiseSplit = SubjectWiseSplit(),
  use_global_normalization=False,
  use_local_normalization=True,
) -> Dict[str, Any]:
  def after_loaded_ds(ds, trial_length_secs=Fraction(4, 1), pre_calculated_stats=None):
    # Apply rereferencing
    dereferenced = MappedDataset(ds, lambda x: trial_to_arrayeeg(rereference_trial(x)))
    # Apply robust normalization
    if use_global_normalization:
      normalized = RobustNormalizedDataset(
        dereferenced, pre_calculated_stats=pre_calculated_stats
      )
      robust_stats = RobustNormalizationStats(
        p25=normalized.p25,
        p75=normalized.p75,
        iqr=normalized.iqr,
        median=normalized.median,
      )
    else:
      robust_stats = None
      normalized = dereferenced
    # Apply stratified sampling (last, after preprocessing)
    stratified = ArrayStratifiedSamplingDataset(
      normalized,
      n_strata=10,
      trial_length_secs=trial_length_secs,
    )

    if use_local_normalization:
      mapped = MappedDataset(stratified, robust_normalize_trial)
    else:
      mapped = stratified

    return mapped, robust_stats

  ds = EEGMusicDataset.load_ondisk(ds_path)

  # Choose split method based on split_type
  match split_type:
    case SubjectWiseSplit():
      split_result = ds.subject_wise_split(
        p_train=config.ds_p_train,
        p_val=config.ds_p_val,
        seed=config.ds_split_seed,
      )
    case TrialWiseSplit():
      split_result = ds.trial_wise_split(
        p_train=config.ds_p_train,
        p_val=config.ds_p_val,
        seed=config.ds_split_seed,
      )

  train_ds = split_result["train"]
  val_ds = split_result.get("val", split_result["test"])  # fallback to test if no val
  test_ds = split_result["test"]

  mapper = None
  if include_mapper:
    mapper = SubjectDatasetMapper()
    for _, row in ds.df[["dataset", "subject"]].drop_duplicates().iterrows():
      mapper.add_subject(str(row["dataset"]), str(row["subject"]))

  # Get chunk width from config (default to 4 seconds if not specified)
  trial_length_secs = getattr(config, "ds_chunk_width", Fraction(4, 1))

  dereferenced, normalization_stats = after_loaded_ds(
    train_ds, trial_length_secs=trial_length_secs, pre_calculated_stats=None
  )
  dereferenced_val, _ = after_loaded_ds(
    val_ds,
    trial_length_secs=trial_length_secs,
    pre_calculated_stats=normalization_stats,
  )
  dereferenced_tst, _ = after_loaded_ds(
    test_ds,
    trial_length_secs=trial_length_secs,
    pre_calculated_stats=normalization_stats,
  )
  if config.ds_use_test_for_val:  # for when p_val=0
    dereferenced_val = dereferenced_tst

  ds_train_repeated_mul = getattr(config, "ds_train_repeated_mul", 1)
  ds_val_repeated_mul = getattr(config, "ds_val_repeated_mul", 1)
  ds_test_repeated_mul = getattr(config, "ds_test_repeated_mul", 1)

  if ds_train_repeated_mul > 1:
    dereferenced = RepeatedDataset(dereferenced, ds_train_repeated_mul)
  if ds_val_repeated_mul > 1:
    dereferenced_val = RepeatedDataset(dereferenced_val, ds_val_repeated_mul)
  if ds_test_repeated_mul > 1:
    dereferenced_tst = RepeatedDataset(dereferenced_tst, ds_test_repeated_mul)

  include_info = getattr(config, "include_info", False)
  if collate_fn is None:
    collate_fn = mel_create_collate_fn(include_info=include_info)

  pin_memory = getattr(config, "pin_memory", True)

  train_dl = create_dataloader(
    dereferenced,
    is_training=True,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )
  val_dl = create_dataloader(
    dereferenced_val,
    is_training=False,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )
  test_dl = create_dataloader(
    dereferenced_tst,
    is_training=False,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )
  result: Dict[str, Any] = {"train": train_dl, "val": val_dl, "test": test_dl}
  if include_mapper and mapper is not None:
    result["mapper"] = mapper
  if "num_skipped_trials" in split_result:
    result["num_skipped_trials"] = split_result["num_skipped_trials"]

  return result


def create_collate_fn(
  music_batch_fn: Callable[[Sequence[MelRaw | WavRAW | NoteOnsets]], Any],
  include_info: bool = False,
  eeg_batch_fn: Optional[Callable[[Sequence[EegData]], Any]] = None,
) -> Callable[
  [List[TrialData[EegData, MusicData]]], Dict[str, torch.Tensor | Dict[str, Any]]
]:
  """
  Create a collate function that gathers trial data into batches.

  Args:
      include_info: If True, also return a dictionary with metadata and trial info
  Returns:
      Collate function that converts list of TrialData[EegData, MelRaw] into batched tensors
  """

  if eeg_batch_fn is None:

    def default_eeg_batch_fn(eegs):
      return torch.stack(
        [
          torch.tensor(trial.get_eeg().raw_eeg.get_data(), dtype=torch.float32)
          for trial in eegs
        ]
      )

    eeg_batch_fn = default_eeg_batch_fn

  def collate_fn(
    trials: List[TrialData[EegData, MusicData]],
  ) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    # Extract EEG and music data as torch tensors
    eegs = [trial.eeg_data for trial in trials]
    music = [trial.music_data.get_music() for trial in trials]

    # Stack tensors along batch dimension
    eeg_batch = eeg_batch_fn(eegs)
    music_batch = music_batch_fn(music)

    if include_info:
      # Gather metadata and trial info for tracing/debugging

      info_dict = {
        "dataset": [trial.dataset for trial in trials],
        "subject": [trial.subject for trial in trials],
        "session": [trial.session for trial in trials],
        "run": [trial.run for trial in trials],
        "trial_id": [trial.trial_id for trial in trials],
        "music_filename": [trial.music_filename.filename for trial in trials],
        "batch_size": len(trials),
        "emotion": [
          parse_music_emotion(trial.music_filename.filename, trial.dataset)
          for trial in trials
        ],
      }
      # Return dict with eeg, mel, and info
      return {"eeg": eeg_batch, "music": music_batch, "info": info_dict}
    else:
      # Return dict with just eeg and mel
      return {"eeg": eeg_batch, "music": music_batch}

  return collate_fn


def mel_create_collate_fn(
  include_info: bool = False,
) -> Callable[
  [List[TrialData[EegData, MelRaw]]], Dict[str, torch.Tensor | Dict[str, Any]]
]:
  """
  Create a collate function that gathers trial data into batches.

  Args:
      include_info: If True, also return a dictionary with metadata and trial info
  Returns:
      Collate function that converts list of TrialData[EegData, MelRaw] into batched tensors
  """

  def mel_batch_fn(music_list):
    music = [torch.tensor(getattr(x, "mel"), dtype=torch.float32) for x in music_list]
    return torch.stack(music)

  return create_collate_fn(mel_batch_fn, include_info)  # type: ignore[return-value]


def create_dataloader(
  dataset,
  batch_size=8,
  num_workers=4,
  pin_memory=True,
  is_training=True,
  prefetch_factor=2,
  collate_fn=None,
):
  """
  Create an optimized DataLoader using parameters from training notes.

  Args:
      dataset: PyTorch dataset
      batch_size: Batch size (default from your config: 8)
      num_workers: Number of worker processes (default from your config: 4)
      pin_memory: Whether to use pinned memory (recommended: True)
      is_training: If True, shuffle=True and drop_last=True; if False, shuffle=False and drop_last=False
      prefetch_factor: Prefetch factor (default: 2, from training notes)
      collate_fn: default mel_create_collate_fn with include_info=False
  Returns:
      DataLoader configured for training or validation with custom collate function
  """

  if collate_fn is None:
    collate_fn = mel_create_collate_fn(include_info=False)

  # Configure DataLoader with optimal parameters from training.md notes
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=is_training,  # Shuffle for training, no shuffle for validation
    drop_last=is_training,  # Drop last batch if incomplete during training
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,  # Only with multiprocessing
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    collate_fn=collate_fn,
  )

  return dataloader

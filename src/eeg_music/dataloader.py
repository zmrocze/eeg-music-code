from fractions import Fraction
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Callable, Any
from .data import (
  EEGMusicDataset,
  MappedDataset,
  MelRaw,
  RepeatedDataset,
  StratifiedSamplingDataset,
  TrialData,
  EegData,
  rereference_trial,
)
from .emotion_utils import parse_music_emotion
from pathlib import Path


def after_loaded_ds(ds):
  stratified = StratifiedSamplingDataset(
    ds,
    n_strata=10,
    trial_length_secs=Fraction(4, 1),
  )

  dereferenced = MappedDataset(stratified, rereference_trial)

  return dereferenced


def load_and_create_dataloaders(ds_path: Path, config) -> Dict[str, DataLoader]:
  # Path("./datasets/bcmi_combined_prepared_mel_28ch")
  ds = EEGMusicDataset.load_ondisk(ds_path)
  train_ds, val_ds, test_ds = ds.subject_wise_split(
    p_train=config.ds_p_train, p_val=config.ds_p_val, seed=config.ds_split_seed
  )
  dereferenced = after_loaded_ds(train_ds)
  dereferenced_val = after_loaded_ds(val_ds)
  dereferenced_tst = after_loaded_ds(test_ds)
  if config.ds_use_test_for_val:  # for when p_val=0
    dereferenced_val = dereferenced_tst
  if config.ds_test_repeated_mul > 1:
    dereferenced_tst = RepeatedDataset(dereferenced_tst, config.ds_test_repeated_mul)

  include_info = getattr(config, "include_info", False)

  train_dl = create_dataloader(
    dereferenced,
    is_training=True,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    include_info=include_info,
  )
  val_dl = create_dataloader(
    dereferenced_val,
    is_training=False,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    include_info=include_info,
  )
  test_dl = create_dataloader(
    dereferenced_tst,
    is_training=False,
    batch_size=config.batch_size,
    num_workers=config.data_loader_num_workers,
    prefetch_factor=config.prefetch_factor,
    include_info=include_info,
  )

  return {"train": train_dl, "val": val_dl, "test": test_dl}


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

  def collate_fn(
    trials: List[TrialData[EegData, MelRaw]],
  ) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    # Extract EEG and music data as torch tensors
    eegs = [
      torch.tensor(trial.eeg_data.get_eeg().raw_eeg.get_data(), dtype=torch.float32)
      for trial in trials
    ]
    music = [
      torch.tensor(getattr(trial.music_data.get_music(), "mel"), dtype=torch.float32)
      for trial in trials
    ]

    # Stack tensors along batch dimension
    eeg_batch = torch.stack(eegs)
    music_batch = torch.stack(music)

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
      return {"eeg": eeg_batch, "mel": music_batch, "info": info_dict}
    else:
      # Return dict with just eeg and mel
      return {"eeg": eeg_batch, "mel": music_batch}

  return collate_fn


def create_dataloader(
  dataset,
  batch_size=8,
  num_workers=4,
  pin_memory=True,
  is_training=True,
  prefetch_factor=2,
  include_info=False,
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
      include_info: If True, collate function will also return metadata dict

  Returns:
      DataLoader configured for training or validation with custom collate function
  """

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
    collate_fn=mel_create_collate_fn(include_info=include_info),
  )

  return dataloader

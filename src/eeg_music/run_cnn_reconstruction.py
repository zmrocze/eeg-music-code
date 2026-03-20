"""Train CNNReconstruction to predict mel spectrograms from EEG, analogous to run_bcmi_emotion.py."""

from pathlib import Path
from fractions import Fraction
import pandas as pd

from eeg_music.data import (
  EEGMusicDataset,
  ArrayStratifiedSamplingDataset,
  temporal_train_test_split,
)
from eeg_music.eegpt import LRCosine, UseAdamW
from eeg_music.mel_training import (
  MelModelConfig,
  MelTrainingConfig,
  MelTraining,
  CNNReconstructionConfig,
)


if __name__ == "__main__":
  trial_length_secs = Fraction(1, 1)

  ds = EEGMusicDataset.load_ondisk(
    Path("./datasets/musing_preprocessed/musing_mel64_60ch/")
  )

  filtered_ds = EEGMusicDataset()
  filtered_ds.df = pd.DataFrame(
    ds.df[(ds.df["subject"] == "001")].reset_index(drop=True)
  )
  filtered_ds.music_collection = ds.music_collection
  ds = filtered_ds

  # splitted = ds.subject_wise_split(p_train=0.6, p_val=0.2)
  train_ds, test_ds = temporal_train_test_split(ds, length_sec=Fraction(20, 1))
  val_ds, test_ds = temporal_train_test_split(ds, length_sec=Fraction(20, 1))

  # train_ds = ArrayStratifiedSamplingDataset(
  #   splitted["train"], 10, trial_length_secs=trial_length_secs
  # )
  # val_ds = ArrayStratifiedSamplingDataset(
  #   splitted["val"], 10, trial_length_secs=trial_length_secs
  # )
  # test_ds = ArrayStratifiedSamplingDataset(
  #   splitted["test"].merge(splitted["val"]), 10, trial_length_secs=trial_length_secs
  # )

  train_ds = ArrayStratifiedSamplingDataset(
    train_ds, 10, trial_length_secs=trial_length_secs
  )
  val_ds = ArrayStratifiedSamplingDataset(
    val_ds, 10, trial_length_secs=trial_length_secs
  )
  test_ds = ArrayStratifiedSamplingDataset(
    test_ds, 10, trial_length_secs=trial_length_secs
  )

  print(len(train_ds), len(val_ds), len(test_ds))

  config = MelTrainingConfig(
    model_config=MelModelConfig(
      model_config=CNNReconstructionConfig(in_channels=1, out_channels=1, dropout=0.25),
      lr_config=LRCosine(max_lr=1e-4, T_0=10, T_mult=2),
      optimizer=UseAdamW(),
    ),
    auroc_every_n_epochs=10,
    batch_size=120,
    num_epochs=12000,
  )

  training = MelTraining(config, train_ds, val_ds, test_ds)
  model, trainer, dataloaders = training.run()

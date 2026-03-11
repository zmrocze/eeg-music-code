"""Train CNNReconstruction to predict mel spectrograms from EEG, analogous to run_bcmi_emotion.py."""

from pathlib import Path
from fractions import Fraction

from eeg_music.data import (
  EEGMusicDataset,
  ArrayStratifiedSamplingDataset,
)
from eeg_music.eegpt import LRCosine, UseAdamW
from eeg_music.mel_training import (
  MelModelConfig,
  MelTrainingConfig,
  MelTraining,
  CNNReconstructionConfig,
)


if __name__ == "__main__":
  trial_length_secs = Fraction(2, 1)

  ds = EEGMusicDataset.load_ondisk(
    Path("./datasets/bcmi_preprocessed/bcmi_mel64_60ch/")
  )

  splitted = ds.subject_wise_split(p_train=0.6, p_val=0.2)

  train_ds = ArrayStratifiedSamplingDataset(
    splitted["train"], 10, trial_length_secs=trial_length_secs
  )
  val_ds = ArrayStratifiedSamplingDataset(
    splitted["val"], 10, trial_length_secs=trial_length_secs
  )
  test_ds = ArrayStratifiedSamplingDataset(
    splitted["test"], 10, trial_length_secs=trial_length_secs
  )

  config = MelTrainingConfig(
    model_config=MelModelConfig(
      model_config=CNNReconstructionConfig(in_channels=1, out_channels=1, dropout=0.25),
      lr_config=LRCosine(max_lr=3e-4, T_0=10, T_mult=2),
      optimizer=UseAdamW(),
    ),
    batch_size=64,
    num_epochs=200,
  )

  training = MelTraining(config, train_ds, val_ds, test_ds)
  model, trainer, dataloaders = training.run()

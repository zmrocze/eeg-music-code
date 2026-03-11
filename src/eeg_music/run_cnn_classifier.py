"""Train CNNClassifier to predict labels from EEG (MusingMusicIdData datasets)."""

from pathlib import Path
from fractions import Fraction

from eeg_music.data import (
  EEGMusicDataset,
  ArrayStratifiedSamplingDataset,
)
from eeg_music.eegpt import LRCosine, UseAdamW
from eeg_music.mel_training import (
  CNNClassifierConfig,
  ClassifierModelConfig,
  ClassifierTrainingConfig,
  ClassifierTraining,
)


if __name__ == "__main__":
  trial_length_secs = Fraction(1, 1)

  ds = EEGMusicDataset.load_ondisk(
    Path("./datasets/musing_preprocessed/musing_pre_60ch/")
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

  # --- Cross-entropy variant (multi-class) ---
  config = ClassifierTrainingConfig(
    model_config=ClassifierModelConfig(
      model_config=CNNClassifierConfig(in_channels=1, dropout=0.25),
      num_classes=12,
      loss="ce",
      lr_config=LRCosine(max_lr=3e-4, T_0=10, T_mult=2),
      optimizer=UseAdamW(),
    ),
    batch_size=64,
    num_epochs=200,
    project_name="cnn-classifier-ce",
    run_name="cnn-classifier-ce",
    save_path="cnn-classifier-ce-ckpt",
  )

  training = ClassifierTraining(config, train_ds, val_ds, test_ds)
  model, trainer, dataloaders = training.run()

  # --- Binary cross-entropy variant (uncomment to use) ---
  # config_bce = ClassifierTrainingConfig(
  #   model_config=ClassifierModelConfig(
  #     model_config=CNNClassifierConfig(in_channels=1, dropout=0.25),
  #     num_classes=2,
  #     loss="bce",
  #     lr_config=LRCosine(max_lr=3e-4, T_0=10, T_mult=2),
  #     optimizer=UseAdamW(),
  #   ),
  #   batch_size=64,
  #   num_epochs=200,
  #   project_name="cnn-classifier-bce",
  #   run_name="cnn-classifier-bce",
  #   save_path="cnn-classifier-bce-ckpt",
  # )
  #
  # training_bce = ClassifierTraining(config_bce, train_ds, val_ds, test_ds)
  # model_bce, trainer_bce, dataloaders_bce = training_bce.run()

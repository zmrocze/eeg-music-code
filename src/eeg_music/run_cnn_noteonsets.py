"""Train CNNClassifier to predict note onset density from EEG (binary classification)."""

from pathlib import Path
from fractions import Fraction

import torch

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
from eeg_music.dataloader import create_collate_fn, create_dataloader


class NoteOnsetsClassifierTraining(ClassifierTraining):
  """ClassifierTraining variant that labels based on note onset count."""

  def __init__(
    self,
    config: ClassifierTrainingConfig,
    train_ds,
    val_ds,
    test_ds,
    median_noteonsets: int,
  ):
    self.median_noteonsets = median_noteonsets
    super().__init__(config, train_ds, val_ds, test_ds)

  def create_dataloaders(self):
    collate_fn = create_collate_fn(
      include_info=self.config.include_info,
      music_batch_fn=lambda xs: torch.tensor(
        [1 if len(x.onset_times) >= self.median_noteonsets else 0 for x in xs],
        dtype=torch.long,
      ),
      eeg_batch_fn=lambda x: torch.stack(
        [torch.from_numpy(a.get_array().data) for a in x]  # pyright: ignore[reportAttributeAccessIssue]
      ),
    )
    self.dataloaders = {
      split: create_dataloader(
        ds,
        batch_size=self.config.batch_size,
        num_workers=self.config.data_loader_num_workers,
        pin_memory=self.config.pin_memory,
        is_training=(split == "train"),
        prefetch_factor=self.config.prefetch_factor,
        collate_fn=collate_fn,
      )
      for split, ds in [
        ("train", self._train_ds),
        ("val", self._val_ds),
        ("test", self._test_ds),
      ]
    }


if __name__ == "__main__":
  trial_length_secs = Fraction(1, 1)

  ds = EEGMusicDataset.load_ondisk(Path("./datasets/bcmi_preprocessed/bcmi_notes_60ch"))

  splitted = ds.subject_wise_split(p_train=0.6, p_val=0.2)

  train_ds = ArrayStratifiedSamplingDataset(
    splitted["train"], 10, trial_length_secs=trial_length_secs
  )
  val_ds = ArrayStratifiedSamplingDataset(
    splitted["val"], 10, trial_length_secs=trial_length_secs
  )
  test_ds = ArrayStratifiedSamplingDataset(
    splitted["test"].merge(splitted["val"]), 10, trial_length_secs=trial_length_secs
  )

  median_noteonsets = 2

  config = ClassifierTrainingConfig(
    model_config=ClassifierModelConfig(
      model_config=CNNClassifierConfig(in_channels=1, dropout=0.4),
      num_classes=2,
      loss="bce",
      lr_config=LRCosine(max_lr=1e-5, T_0=10, T_mult=2),
      optimizer=UseAdamW(),
    ),
    batch_size=256,
    num_epochs=747,
    project_name="cnn-classifier-noteonsets-bce",
    run_name="cnn-classifier-noteonsets-bce",
    save_path="cnn-classifier-noteonsets-bce-ckpt",
  )

  training = NoteOnsetsClassifierTraining(
    config, train_ds, val_ds, test_ds, median_noteonsets=median_noteonsets
  )
  model, trainer, dataloaders = training.run()

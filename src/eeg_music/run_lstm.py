"""Train BiLSTM on ODF (onset detection function) dataset."""

from pathlib import Path
from fractions import Fraction
import pandas as pd
import torch

from eeg_music.data import (
  EEGMusicDataset,
  ArrayStratifiedSamplingDataset,
  temporal_train_test_split,
)
from eeg_music.eegpt import LRCosine, UseAdamW
from eeg_music.mel_training import (
  MelModelConfig,
  MelTrainingConfig,
  OdfTraining,
  BiLSTMConfig,
)


if __name__ == "__main__":
  trial_length_secs = Fraction(1, 1)

  ds = EEGMusicDataset.load_ondisk(
    # Path("./datasets/musing_preprocessed/musing_rawnoica_odf_129/")
    Path("./datasets/musing_preprocessed/musing_rawica_odf/")
  )

  filtered_ds = EEGMusicDataset()
  # filtered_ds.df = pd.DataFrame(
  #   ds.df[(ds.df["subject"] == "001")].reset_index(drop=True)
  # )
  filtered_ds.df = pd.DataFrame(
    ds.df.iloc[:1].reset_index(drop=True)
    # ds.df.iloc[12:13].reset_index(drop=True)
    # ds.df.iloc[24:25].reset_index(drop=True)
    # ds.df.iloc[26:37].reset_index(drop=True)
    # ds.df.iloc[38:39].reset_index(drop=True)
  )
  filtered_ds.music_collection = ds.music_collection
  ds = filtered_ds

  train_ds, test_ds = temporal_train_test_split(ds, length_sec=Fraction(80, 1))
  val_ds, test_ds = temporal_train_test_split(test_ds, length_sec=Fraction(20, 1))

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

  # ODF data has shape (1, n_frames). Upsample by 4x to match EEG sample rate (256 Hz vs 64 Hz)
  def odf_batch_fn(xs):
    return torch.stack(
      [torch.from_numpy(x.mel).float().repeat_interleave(4, dim=1) for x in xs]
    ).unsqueeze(1)  # (B, 1, 1, n_frames*4) to match mel format

  config = MelTrainingConfig(
    model_config=MelModelConfig(
      model_config=BiLSTMConfig(
        input_size=12,
        # input_size=129,
        # hidden_size=256,
        hidden_size=64,
        num_layers=2,
        output_size=1,
      ),
      lr_config=LRCosine(max_lr=1e-5, T_0=10, T_mult=2),
      optimizer=UseAdamW(),
    ),
    music_batch_fn=odf_batch_fn,
    auroc_every_n_epochs=10,
    # batch_size=120
    batch_size=10,
    num_epochs=1750,
    project_name="odf-reconstruction-lstm",
    run_name="bilstm-odf",
    save_path="bilstm-odf-ckpt",
  )

  training = OdfTraining(config, train_ds, val_ds, test_ds)
  model, trainer, dataloaders = training.run()

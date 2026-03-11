from pathlib import Path
from fractions import Fraction

from eeg_music.data import (
  EEGMusicDataset,
  ArrayStratifiedSamplingDataset,
)
from eeg_music.eegpt import LRCosine, LRStepLR, UseAdamW
from eeg_music.emotion_eegnet import (
  EmotionEEGNetTrainingConfig,
  EmotionEEGNetModelConfig,
  BCMIEmotionEEGNetTraining,
)
from eeg_music.eegnet import EEGNetConfig


def create_config(
  model_config=None,
  optimizer=None,
  lr_config: float | LRStepLR | LRCosine = 3e-4,
  num_epochs=200,
  batch_size=512,
  trial_length_secs=4,
  num_channels=28,
  eeg_sample_rate=256,
):
  if model_config is None:
    model_config = EEGNetConfig()
  if optimizer is None:
    optimizer = UseAdamW()

  return EmotionEEGNetTrainingConfig(
    model_config=EmotionEEGNetModelConfig(
      model_config=model_config,
      chunk_width=eeg_sample_rate * trial_length_secs,
      num_channels=num_channels,
      eeg_sample_rate=eeg_sample_rate,
      num_classes=9,
      lr_config=lr_config,
      optimizer=optimizer,
    ),
    batch_size=batch_size,
    data_loader_num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    num_epochs=num_epochs,
    project_name="emotion-classification-eegnet",
    run_name="eegnet-bcmi-emotion-9class",
    save_path="eegnet-bcmi-emotion-9class-ckpt",
  )


if __name__ == "__main__":
  trial_length_secs = Fraction(4, 1)

  ds = EEGMusicDataset.load_ondisk(
    # Path("./datasets/bcmi_preprocessed/bcmi_full_ica_40ch")
    Path("./datasets/bcmi_preprocessed/bcmi_pre_60ch/")
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

  config = create_config(
    model_config=EEGNetConfig(),
    lr_config=LRCosine(max_lr=3e-5, T_0=10, T_mult=2),
    num_epochs=200,
    eeg_sample_rate=10,
    batch_size=64,
    trial_length_secs=int(trial_length_secs),
  )

  training = BCMIEmotionEEGNetTraining(config, train_ds, val_ds, test_ds)
  model, trainer, dataloaders = training.run()

from pathlib import Path
from fractions import Fraction

from eeg_music.data import (
  EEGMusicDataset,
  MappedDataset,
  ArrayStratifiedSamplingDataset,
)
from eeg_music.onset_conversion import trial_wavraw_to_noteonsets
from eeg_music.eegpt import LRCosine, LRStepLR, UseAdamW
from eeg_music.emotion_eegnet import (
  EmotionEEGNetTrainingConfig,
  EmotionEEGNetModelConfig,
  MusingEEGNetTraining,
)
from eeg_music.eegnet import EEGNetConfig


def prepare_ds(ds, trial_length_secs: Fraction):
  nds = MappedDataset(ds, trial_wavraw_to_noteonsets)
  return ArrayStratifiedSamplingDataset(nds, 10, trial_length_secs=trial_length_secs)


def create_config(
  model_config=None,
  optimizer=None,
  lr_config: float | LRStepLR | LRCosine = 3e-4,
  num_epochs=500,
  batch_size=512,
  trial_length_secs=3,
  num_channels=40,
  eeg_sample_rate=250,
  median_num_noteonsets=35,
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
      num_classes=12,
      lr_config=lr_config,
      optimizer=optimizer,
      median_num_noteonsets=median_num_noteonsets,
    ),
    batch_size=batch_size,
    data_loader_num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    num_epochs=num_epochs,
    run_name="eegnet-noteonsets-binary",
    save_path="eegnet-noteonsets-binary-ckpt",
  )


if __name__ == "__main__":
  # ds = EEGMusicDataset.load_ondisk(Path("./datasets/bcmi_preprocessed/bcmi_onesubj_ica_40ch/"))
  # splitted = ds.trial_wise_split(p_train=0.75, p_val=0.0)
  trial_length_secs = Fraction(5, 1)
  # train_ds = prepare_ds(splitted["train"], trial_length_secs)
  # test_ds = prepare_ds(splitted["test"], trial_length_secs)

  # train_ds = prepare_ds(EEGMusicDataset.load_ondisk(Path("./datasets/bcmi_preprocessed/bcmi_train_onesubj_ica_40ch/")), trial_length_secs)
  # test_ds = prepare_ds(EEGMusicDataset.load_ondisk(Path("./datasets/bcmi_preprocessed/bcmi_test_onesubj_ica_40ch/")), trial_length_secs)
  ds = EEGMusicDataset.load_ondisk(
    Path("./datasets/musing_preprocessed/musing_ica_8ch")
  )
  splitted = ds.subject_wise_split(p_train=0.75, p_val=0.0)
  train_ds = ArrayStratifiedSamplingDataset(
    splitted["train"], 10, trial_length_secs=trial_length_secs
  )
  test_ds = ArrayStratifiedSamplingDataset(
    splitted["test"], 10, trial_length_secs=trial_length_secs
  )

  config = create_config(
    model_config=EEGNetConfig(),
    lr_config=LRCosine(max_lr=3e-5, T_0=10, T_mult=2),
    num_epochs=200,
    eeg_sample_rate=10,
    batch_size=512,
    trial_length_secs=int(trial_length_secs),
    median_num_noteonsets=5,
  )

  training = MusingEEGNetTraining(config, train_ds, test_ds)
  model, trainer, dataloaders = training.run()

from pathlib import Path
from eeg_music.eegpt import LRCosine, LRStepLR, UseAdamW
from eeg_music.emotion_eegnet import (
  EmotionEEGNetTrainingConfig,
  EmotionEEGNetModelConfig,
  MusingEEGNetTraining,
)
from eeg_music.eegnet import EEGNetConfig
from eeg_music.dataloader import TrialWiseSplit, SubjectWiseSplit
from fractions import Fraction
import os
import shutil


def create_musing_config(
  model_config=None,
  ds_split_type=None,
  use_subject_specific=False,
  optimizer=None,
  lr_config: float | LRStepLR | LRCosine = 1e-4,
  num_epochs=200,
  batch_size=64,
  trial_length_secs=4,
  data_path=Path("./datasets/bcmi_preprocessed/musin_g_data"),
):
  """Create configuration for MUSING dataset song classification.

  Args:
      model_config: Model architecture config (EEGNet, TSCeption, ATCNet)
      ds_split_type: Data split strategy (TrialWiseSplit or SubjectWiseSplit)
      use_subject_specific: Enable subject-specific linear preprocessing
      optimizer: Optimizer configuration
      lr_config: Learning rate or scheduler config
      num_epochs: Number of training epochs
      batch_size: Batch size for training
      trial_length_secs: Length of each trial in seconds
      data_path: Path to MUSING dataset

  Returns:
      EmotionEEGNetTrainingConfig configured for MUSING (12 song classes)
  """
  if model_config is None:
    model_config = EEGNetConfig()
  if optimizer is None:
    optimizer = UseAdamW()
  if ds_split_type is None:
    ds_split_type = TrialWiseSplit()

  config = EmotionEEGNetTrainingConfig(
    model_config=EmotionEEGNetModelConfig(
      model_config=model_config,
      chunk_width=10 * trial_length_secs,  # 250Hz sampling rate for MUSING
      num_channels=40,
      eeg_sample_rate=10,
      num_classes=12,  # 12 songs in MUSING dataset
      lr_config=lr_config,
      use_subject_specific=use_subject_specific,
      optimizer=optimizer,
    ),
    # data_path=data_path,
    # data_path=Path("./datasets/musing_preprocessed/musing_8ch"),
    data_path=Path("./datasets/musing_preprocessed/musing_ica_8ch"),
    batch_size=batch_size,
    data_loader_num_workers=4,
    prefetch_factor=2,
    ds_p_train=0.75,
    pin_memory=True,
    ds_split_seed=42,
    num_epochs=num_epochs,
    use_global_normalization=False,
    use_local_normalization=False,
    ds_test_repeated_mul=4,
    ds_train_repeated_mul=1,
    ds_chunk_width=Fraction(trial_length_secs, 1),
    ds_split_type=ds_split_type,
    run_name="eegnet-musing-classification",
    save_path="eegnet-musing-ckpt",
  )
  return config


# Example configurations for MUSING dataset
all_configs = [
  # EEGNet with cosine annealing
  create_musing_config(
    # model_config=TSCeptionConfig(),
    model_config=EEGNetConfig(),
    lr_config=LRCosine(max_lr=3e-4, T_0=10, T_mult=2),
    num_epochs=500,
    batch_size=512,
    trial_length_secs=30,
    ds_split_type=SubjectWiseSplit(),
  ),
]

if __name__ == "__main__":
  i = 0
  for config in all_configs:
    directory_to_remove = "eegnet-musing-ckpt"
    if os.path.exists(directory_to_remove):
      shutil.rmtree(directory_to_remove)
      print(f"Directory '{directory_to_remove}' removed successfully.")
    else:
      print(f"Directory '{directory_to_remove}' does not exist.")

    training = MusingEEGNetTraining(config)
    model, trainer, dataloaders = training.run()
    i += 1

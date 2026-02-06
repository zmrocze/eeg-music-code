"""Main script for running sleep stage classification training."""

from eeg_music.eegpt import LRCosine, LRStepLR, UseAdamW
from eeg_music.sleepedf_eegnet import (
  SleepEDFTraining,
  SleepEDFTrainingConfig,
  SleepEDFDataConfig,
  EmotionEEGNetModelConfig,
)
from eeg_music.eegnet import EEGNetConfig


def create_sleep_config(
  model_config=None,
  optimizer=None,
  lr_config: float | LRStepLR | LRCosine = 1e-4,
  num_epochs=100,
  batch_size=512,
  use_subject_specific=False,
  subject_ids=None,
  recording_ids=None,
  train_ratio=0.5,
  val_ratio=0.25,
):
  """Create configuration for sleep stage classification training.

  Args:
    model_config: Model architecture config (EEGNetConfig, TSCeptionConfig, etc.)
    optimizer: Optimizer config (UseAdamW, UseSGD)
    lr_config: Learning rate or learning rate schedule
    num_epochs: Number of training epochs
    batch_size: Batch size
    use_subject_specific: Whether to use subject-specific linear preprocessing
    subject_ids: List of subject IDs to use (default: first 20 subjects)
    recording_ids: Recording IDs to use (default: [2] for night 2)
    train_ratio: Proportion of subjects for training (default: 0.5)
    val_ratio: Proportion of subjects for validation (default: 0.25)
      Note: test_ratio = 1 - train_ratio - val_ratio
  """
  if model_config is None:
    model_config = EEGNetConfig()
  if optimizer is None:
    optimizer = UseAdamW()
  if subject_ids is None:
    subject_ids = list(range(20))
  if recording_ids is None:
    recording_ids = [2]  # Night 2

  # Create train/val/test indices based on ratios
  total_subjects = len(subject_ids)
  num_train = int(total_subjects * train_ratio)
  num_val = int(total_subjects * val_ratio)
  num_test = total_subjects - num_train - num_val  # Remaining = test

  train_indices = list(range(0, num_train))
  val_indices = list(range(num_train, num_train + num_val))
  test_indices = list(range(num_train + num_val, num_train + num_val + num_test))

  config = SleepEDFTrainingConfig(
    model_config=EmotionEEGNetModelConfig(
      model_config=model_config,
      chunk_width=3000,  # 30s * 100Hz
      num_channels=2,  # Sleep EEG typically uses 2 channels
      eeg_sample_rate=100,
      num_classes=5,  # 5 sleep stages
      lr_config=lr_config,
      use_subject_specific=use_subject_specific,
      optimizer=optimizer,
    ),
    data_config=SleepEDFDataConfig(
      subject_ids=subject_ids,
      recording_ids=recording_ids,
      crop_wake_mins=30,
      high_cut_hz=30.0,
      window_size_s=30,
      sfreq=100,
      train_subject_indices=train_indices,
      val_subject_indices=val_indices,
      test_subject_indices=test_indices,
    ),
    batch_size=batch_size,
    data_loader_num_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    num_epochs=num_epochs,
    val_every_n_epoch=1,
    save_model_per_epochs=5,
    project_name="sleep-stage-classification-eegnet",
    run_name="sleep-eegnet-5class",
  )
  return config


# Example configurations
# Default split: 50% train, 25% val, 25% test (train_ratio=0.5, val_ratio=0.25)
all_configs = [
  # EEGNet baseline
  # create_sleep_config(
  #   model_config=EEGNetConfig(),
  #   lr_config=LRStepLR(initial_lr=1e-3, step_size=20, gamma=0.9),
  #   num_epochs=200,
  #   batch_size=1024,
  # ),
  # TSCeption model
  create_sleep_config(
    model_config=EEGNetConfig(),
    lr_config=LRStepLR(initial_lr=1e-3, step_size=30, gamma=0.9),
    num_epochs=500,
    batch_size=1024,
  ),
  # ATCNet model
  # create_sleep_config(
  #   model_config=ATCNetConfig(),
  #   lr_config=LRStepLR(initial_lr=1e-3, step_size=20, gamma=0.9),
  #   num_epochs=100,
  #   batch_size=64,
  # ),
]

if __name__ == "__main__":
  # Run all configurations sequentially
  for i, config in enumerate(all_configs):
    print(f"\n{'=' * 80}")
    print(f"Running configuration {i + 1}/{len(all_configs)}")
    print(f"Model: {type(config.model_config.model_config).__name__}")
    print(f"Learning rate: {config.model_config.lr_config}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"{'=' * 80}\n")

    training = SleepEDFTraining(config)
    model, trainer, dataloaders = training.run()

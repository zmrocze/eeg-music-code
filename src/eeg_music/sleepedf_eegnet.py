"""Sleep stage classification using EEGNet models.

This module provides training for 5-class sleep stage classification using EEGNet-based models.
Based on the emotion classification framework.
"""

import torch
import wandb
from dataclasses import dataclass, field

from .eegnet import (
  EEGNetConfig,
)
from .subject_specific import SubjectDatasetMapper
from .emotion_eegnet import (
  EmotionEEGNetModelConfig,
  EmotionEEGNetLightning,
  EmotionEEGNetTrainingConfig,
  EmotionEEGNetTraining,
)
from .sleep_dataloader import load_and_create_sleep_dataloaders


class SleepEDFLightning(EmotionEEGNetLightning):
  """PyTorch Lightning module for sleep stage classification using EEGNet.

  Inherits from EmotionEEGNetLightning and only overrides _compute_loss
  to handle the different batch format from braindecode WindowsDataset.
  """

  def _compute_loss(self, batch, batch_idx, stage: str):
    """Compute loss for a batch.

    Args:
        batch: Tuple of (eeg_data, targets, subject_ids) from sleep dataloader
        batch_idx: Batch index
        stage: 'train', 'val', or 'test'

    Returns:
        loss: Computed loss value
    """
    eeg_data, targets, subject_ids = batch

    # Get subject IDs if using subject-specific preprocessing
    subject_id_tensor = None
    if self.config.use_subject_specific and self.subject_mapper is not None:
      # subject_ids is a list from the dataloader, convert to tensor
      subject_id_tensor = torch.tensor(
        [self.subject_mapper.get_id("sleepedf", str(sid)) for sid in subject_ids],
        device=self.device,
      )

    # Forward pass
    logits = self(eeg_data, subject_ids=subject_id_tensor)

    # Compute loss
    loss = self.loss_fn(logits, targets)

    # Update metrics
    metrics_calc = {
      "train": self.train_metrics,
      "val": self.val_metrics,
      "test": self.test_metrics,
    }[stage]
    metrics_calc.update(logits.detach(), targets)

    # Log loss
    self.log(
      f"{stage}_loss",
      loss,
      on_step=True,
      on_epoch=True,
      prog_bar=True,
      batch_size=eeg_data.shape[0],
    )

    return loss


@dataclass
class SleepEDFDataConfig:
  """Configuration for SleepEDF dataset loading and preprocessing."""

  subject_ids: list[int] = field(
    default_factory=lambda: list(range(20))
  )  # First 20 subjects
  recording_ids: list[int] = field(default_factory=lambda: [2])  # Night 2
  crop_wake_mins: int = 30
  high_cut_hz: float = 30.0
  window_size_s: int = 30
  sfreq: int = 100
  # Train/val/test split using subject indices (e.g., train=[0,2,4,...], val=[1,5,9,...])
  train_subject_indices: list[int] = field(
    default_factory=lambda: list(range(0, 20, 2))
  )  # Even indices
  val_subject_indices: list[int] = field(
    default_factory=lambda: list(range(1, 20, 4))
  )  # 1,5,9,13,17
  test_subject_indices: list[int] = field(
    default_factory=lambda: list(range(3, 20, 4))
  )  # 3,7,11,15,19


@dataclass
class SleepEDFTrainingConfig(EmotionEEGNetTrainingConfig):
  """Configuration for sleep stage classification training using EEGNet.

  Inherits from EmotionEEGNetTrainingConfig and only overrides SleepEDF-specific settings.
  """

  model_config: EmotionEEGNetModelConfig = field(
    default_factory=lambda: EmotionEEGNetModelConfig(
      model_config=EEGNetConfig(),
      chunk_width=3000,  # 30s * 100Hz
      num_channels=2,
      eeg_sample_rate=100,
      num_classes=5,
      lr_config=1e-4,
    )
  )
  data_config: SleepEDFDataConfig = field(default_factory=SleepEDFDataConfig)
  project_name: str = "sleep-stage-classification-eegnet"
  run_name: str = "eegnet-sleep-5class"


class SleepEDFTraining(EmotionEEGNetTraining):
  """Training class for sleep stage classification using EEGNet models.

  Inherits from EmotionEEGNetTraining and overrides:
  - create_dataloaders: Load SleepEDF dataset using unified sleep dataloader
  - create_model: Use SleepEDFLightning instead of EmotionEEGNetLightning
  - log_hyperparameters: Add sleep-specific dataset parameters
  """

  def __init__(self, config: SleepEDFTrainingConfig):
    self.config = config
    # Don't call super().__init__ yet - we need to create dataloaders first

  def create_dataloaders(self):
    """Create dataloaders for SleepEDF dataset using unified function."""
    dc = self.config.data_config

    # Calculate train/val/test proportions from indices
    n_subjects = len(dc.subject_ids)
    _n_train = len(dc.train_subject_indices)
    n_val = len(dc.val_subject_indices)
    n_test = len(dc.test_subject_indices)

    # Convert to proportions for load_and_create_sleep_dataloaders
    # test_size includes both val and test
    test_val_size = (n_val + n_test) / n_subjects
    # val_split is proportion of test_val that goes to validation
    val_split = n_val / (n_val + n_test) if (n_val + n_test) > 0 else 0.5

    # Use unified dataloader function
    train_loader, val_loader, test_loader = load_and_create_sleep_dataloaders(
      subject_ids=dc.subject_ids,
      recording_ids=dc.recording_ids,
      crop_wake_mins=dc.crop_wake_mins,
      window_size_s=dc.window_size_s,
      sfreq=dc.sfreq,
      n_channels=self.config.model_config.num_channels,
      batch_size=self.config.batch_size,
      test_size=test_val_size,
      val_split=val_split,
      random_state=42,
      num_workers=self.config.data_loader_num_workers,
      l_freq=0.5,  # Standard low-pass for sleep EEG
      h_freq=dc.high_cut_hz,
    )

    self.dataloaders = {
      "train": train_loader,
      "val": val_loader,
      "test": test_loader,
    }

    # Create subject mapper if needed
    if self.config.model_config.use_subject_specific:
      mapper = SubjectDatasetMapper()
      for subj_id in dc.subject_ids:
        mapper.add_subject("sleepedf", str(subj_id))
      self.dataloaders["mapper"] = mapper

  def create_model(self):
    """Create SleepEDFLightning model."""
    mapper = (
      self.dataloaders.get("mapper")
      if self.config.model_config.use_subject_specific
      else None
    )

    if self.config.wandb_checkpoint is not None:
      print(f"Loading model from wandb checkpoint: {self.config.wandb_checkpoint}")
      run = wandb.init()
      artifact = run.use_artifact(self.config.wandb_checkpoint, type="model")
      artifact_dir = artifact.download()
      self.model = SleepEDFLightning.load_from_checkpoint(
        artifact_dir + "/model.ckpt",
        config=self.config.model_config,
        subject_mapper=mapper,
      )
    elif (
      self.config.checkpoint_path is not None and self.config.checkpoint_path.exists()
    ):
      print(f"Loading model from checkpoint: {self.config.checkpoint_path}")
      self.model = SleepEDFLightning.load_from_checkpoint(
        self.config.checkpoint_path,
        config=self.config.model_config,
        subject_mapper=mapper,
      )
    else:
      if self.config.checkpoint_path is not None:
        print(f"Checkpoint path specified but not found: {self.config.checkpoint_path}")
      print("Creating fresh SleepEDFLightning model")
      self.model = SleepEDFLightning(
        self.config.model_config,
        subject_mapper=mapper,
      )

  def log_hyperparameters(self):
    """Log sleep stage classification hyperparameters to wandb."""
    # Call parent to log common params (model, training, dataloader)
    super().log_hyperparameters()

    # Add SleepEDF-specific dataset params
    dataset_params = {
      "n_subjects": len(self.config.data_config.subject_ids),
      "n_train_subjects": len(self.config.data_config.train_subject_indices),
      "n_val_subjects": len(self.config.data_config.val_subject_indices),
      "n_test_subjects": len(self.config.data_config.test_subject_indices),
      "recording_ids": self.config.data_config.recording_ids,
      "window_size_s": self.config.data_config.window_size_s,
      "sfreq": self.config.data_config.sfreq,
      "high_cut_hz": self.config.data_config.high_cut_hz,
    }
    self.wandb_logger.log_hyperparams(dataset_params)

"""Emotion classification using EEGNet models.

This module provides training for 9-class emotion classification using EEGNet-based models.
It extends the note onset detection framework to handle multi-class classification.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union
from fractions import Fraction
from lightning.pytorch import LightningModule

from .eegnet import (
  EEGNetConfig,
  FBCNetConfig,
  TSCeptionConfig,
  ATCNetConfig,
  EEGNetWrapper,
)
from .eegpt import UseAdamW, UseSGD, mk_optimizer_and_lr_scheduler, LRCosine
from .training import NoteOnsetsTraining
from .subject_specific import SubjectDatasetMapper
from .data import (
  MappedDataset,
  StratifiedSamplingDataset,
  RobustNormalizedDataset,
  rereference_trial,
)
from .dataloader import TrialWiseSplit, create_collate_fn


class MultiClassAccuracyCalc:
  """Multi-class classification metrics calculator.

  Tracks predictions and targets to compute accuracy.
  Uses online calculation for efficiency.
  """

  def __init__(self, num_classes: int):
    self.num_classes = num_classes
    self.correct = 0
    self.total = 0

  def update(self, logits: torch.Tensor, targets: torch.Tensor):
    """Update metrics with new logits and targets.

    Args:
        logits: Model output logits (batch_size, num_classes)
        targets: Ground truth class labels (batch_size,) - integers 0 to num_classes-1
    """
    predictions = torch.argmax(logits, dim=1)
    self.correct += (predictions == targets).sum().item()
    self.total += targets.size(0)

  def compute(self) -> dict[str, float]:
    """Compute accuracy metric.

    Returns:
        Dictionary with key: accuracy
    """
    accuracy = self.correct / self.total if self.total > 0 else 0.0
    return {"accuracy": accuracy}

  def reset(self):
    """Reset all counters."""
    self.correct = 0
    self.total = 0


@dataclass
class EmotionEEGNetModelConfig:
  """Configuration for EEG emotion classification model using EEGNet.

  Args:
      model_config: Model-specific configuration (EEGNet, FBCNet, TSCeption, or ATCNet)
      chunk_width: Total width of input chunk in samples
      num_channels: Number of EEG channels
      eeg_sample_rate: EEG sampling rate in Hz
      num_classes: Number of emotion classes (default: 9)
      lr_config: Learning rate config - either a float or LRCosine scheduler config
      optimizer: Optimizer to use
      use_subject_specific: Enable subject-specific linear preprocessing
  """

  model_config: EEGNetConfig | FBCNetConfig | TSCeptionConfig | ATCNetConfig = field(
    default_factory=EEGNetConfig
  )
  chunk_width: int = 1024
  num_channels: int = 28
  eeg_sample_rate: int = 256
  num_classes: int = 9
  lr_config: float | LRCosine = 1e-4
  optimizer: UseAdamW | UseSGD = field(default_factory=UseAdamW)
  use_subject_specific: bool = False


class EmotionEEGNetLightning(LightningModule):
  """PyTorch Lightning module for emotion classification using EEGNet.

  This module wraps the EEGNetWrapper and handles training, validation,
  and testing with cross-entropy loss for multi-class classification.
  """

  def __init__(
    self,
    config: EmotionEEGNetModelConfig,
    subject_mapper: Optional[SubjectDatasetMapper] = None,
    **model_kwargs,
  ):
    super().__init__()
    self.config = config
    self.subject_mapper = subject_mapper
    if isinstance(self.config.lr_config, float):
      self.learning_rate = self.config.lr_config
    self.save_hyperparameters(ignore=["subject_mapper"])

    # Create the model
    self.model = EEGNetWrapper(
      chunk_width=config.chunk_width,
      num_channels=config.num_channels,
      eeg_sample_rate=config.eeg_sample_rate,
      model_config=config.model_config,
      num_classes=config.num_classes,
      subject_specific_mapper=subject_mapper,
      **model_kwargs,
    )

    # Loss function for multi-class classification
    self.loss_fn = nn.CrossEntropyLoss()

    # Multi-class metrics
    self.train_metrics = MultiClassAccuracyCalc(config.num_classes)
    self.val_metrics = MultiClassAccuracyCalc(config.num_classes)
    self.test_metrics = MultiClassAccuracyCalc(config.num_classes)

  def forward(self, x: torch.Tensor, subject_ids: Optional[torch.Tensor] = None):
    """Forward pass through the model."""
    return self.model(x, subject_ids)

  def _compute_loss(self, batch, batch_idx, stage: str):
    """Compute loss for a batch.

    Args:
        batch: Dictionary with keys:
            - 'eeg': (batch, channels, timepoints)
            - 'music': List of music filenames or objects
            - 'info': Dict with dataset and subject info (and music_filename)
        batch_idx: Batch index
        stage: 'train', 'val', or 'test'

    Returns:
        loss: Computed loss value
    """
    eeg = batch["eeg"]
    emotion_codes = batch["info"]["emotion"]  # List of emotion codes (1-9 or None)

    # Convert to class indices (0-8) for CrossEntropyLoss. Subtract 1 to map 1-9 to 0-8
    targets = torch.tensor(
      [code - 1 if code is not None else 0 for code in emotion_codes],
      dtype=torch.long,
      device=eeg.device,
    )

    # Get subject IDs if using subject-specific preprocessing
    subject_ids = None
    if self.subject_mapper is not None:
      info = batch["info"]
      subject_ids = torch.tensor(
        [
          self.subject_mapper.get_id(info["dataset"][i], info["subject"][i])
          for i in range(len(info["dataset"]))
        ],
        device=eeg.device,
      )

    # Forward pass
    logits = self(eeg, subject_ids)

    # Compute loss
    loss = self.loss_fn(logits, targets)

    # Update metrics
    if stage == "train":
      self.train_metrics.update(logits.detach(), targets)
    elif stage == "val":
      self.val_metrics.update(logits.detach(), targets)
    elif stage == "test":
      self.test_metrics.update(logits.detach(), targets)

    # Log loss
    self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def training_step(self, batch, batch_idx):
    """Training step."""
    return self._compute_loss(batch, batch_idx, "train")

  def validation_step(self, batch, batch_idx):
    """Validation step."""
    return self._compute_loss(batch, batch_idx, "val")

  def test_step(self, batch, batch_idx):
    """Test step."""
    return self._compute_loss(batch, batch_idx, "test")

  def _log_and_reset_metrics(self, stage: str):
    """Log metrics and reset counters."""
    if stage == "train":
      metrics = self.train_metrics.compute()
      self.train_metrics.reset()
    elif stage == "val":
      metrics = self.val_metrics.compute()
      self.val_metrics.reset()
    elif stage == "test":
      metrics = self.test_metrics.compute()
      self.test_metrics.reset()
    else:
      return

    for metric_name, metric_value in metrics.items():
      self.log(f"{stage}_{metric_name}", metric_value, prog_bar=True)

  def on_train_epoch_end(self):
    self._log_and_reset_metrics("train")

  def on_validation_epoch_end(self):
    self._log_and_reset_metrics("val")

  def on_test_epoch_end(self):
    self._log_and_reset_metrics("test")

  def configure_optimizers(self):
    optimizer, lr_scheduler = mk_optimizer_and_lr_scheduler(
      self.model.parameters(),
      self.config.lr_config,
      self.config.optimizer,
    )
    if lr_scheduler is None:
      return optimizer
    return [optimizer], [lr_scheduler]


@dataclass
class EmotionEEGNetTrainingConfig:
  """Configuration for emotion classification training using EEGNet.

  Inherits structure from NoteOnsetsTrainingConfig but adapted for emotion classification.
  """

  model_config: EmotionEEGNetModelConfig = field(
    default_factory=lambda: EmotionEEGNetModelConfig(
      model_config=EEGNetConfig(),
      chunk_width=4000,  # 256Hz * 4s
      num_channels=28,
      eeg_sample_rate=1000,
      num_classes=9,
      lr_config=1e-4,
    )
  )
  checkpoint_path: Optional[Path] = None
  data_path: Path = Path("./datasets/bcmi_preprocessed/bcmi_combined_prepared_mel_28ch")
  data_loader_num_workers: int = 2
  prefetch_factor: int = 2
  batch_size: int = 32
  num_epochs: int = 100
  save_model_per_epochs: int = 5
  val_every_n_epoch: int = 1
  ds_p_train: float = 0.85
  ds_p_val: float = 0.0
  ds_split_seed: int = 42
  ds_use_test_for_val: bool = True
  ds_train_repeated_mul: int = 1
  ds_val_repeated_mul: int = 1
  ds_test_repeated_mul: int = 10
  ds_chunk_width: Fraction = Fraction(4, 1)
  ds_split_type: TrialWiseSplit = field(default_factory=TrialWiseSplit)
  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "emotion-classification-eegnet"
  run_name: str = "eegnet-emotion-9class"
  run_extra_name: str = "0"
  randint: int = 0
  save_path: str = f"{run_name}-ckpt"
  use_learning_rate_finder: bool = False
  include_info: bool = True  # Required for emotion labels


class EmotionEEGNetTraining(NoteOnsetsTraining):
  """Training class for emotion classification using EEGNet models.

  Inherits from NoteOnsetsTraining and overrides:
  - create_dataloaders: Use rereference_trial, RobustNormalizedDataset, stratified_sampling
  - create_model: Use EmotionEEGNetLightning instead of EEGNetLightning
  """

  def __init__(self, config: EmotionEEGNetTrainingConfig):
    super().__init__(config)
    # Ensure include_info is True for emotion classification
    self.config.include_info = True

  def create_dataloaders(self):
    """Create dataloaders with rereference_trial, robust normalization, and stratified sampling."""
    from .dataloader import load_and_create_dataloaders

    # Define custom dataset preprocessing pipeline
    def after_loaded_ds(ds, trial_length_secs=Fraction(4, 1)):
      # Apply rereferencing
      dereferenced = MappedDataset(ds, rereference_trial)
      # Apply robust normalization
      normalized = RobustNormalizedDataset(dereferenced)
      # Apply stratified sampling (last, after preprocessing)
      stratified = StratifiedSamplingDataset(
        normalized,
        n_strata=10,
        trial_length_secs=trial_length_secs,
      )
      return stratified

    # Enable include_info when using subject-specific preprocessing (need dataset+subject)
    include_info = (
      self.config.include_info or self.config.model_config.use_subject_specific
    )

    self.dataloaders = load_and_create_dataloaders(
      self.config.data_path,
      self.config,
      collate_fn=create_collate_fn(
        include_info=include_info,
        music_batch_fn=lambda x: x,  # Keep music data as-is
      ),
      include_mapper=self.config.model_config.use_subject_specific,
      split_type=self.config.ds_split_type,
      after_loaded_ds=after_loaded_ds,
    )

  def create_model(self):
    """Create EmotionEEGNetLightning model."""
    # Get mapper from dataloaders if subject-specific preprocessing is enabled
    mapper = (
      self.dataloaders.get("mapper")
      if self.config.model_config.use_subject_specific
      else None
    )

    if self.config.checkpoint_path is not None and self.config.checkpoint_path.exists():
      # Load from checkpoint
      print(f"Loading model from checkpoint: {self.config.checkpoint_path}")
      self.model = EmotionEEGNetLightning.load_from_checkpoint(
        self.config.checkpoint_path,
        config=self.config.model_config,
        subject_mapper=mapper,
      )
    else:
      # Create fresh model
      if self.config.checkpoint_path is not None:
        print(f"Checkpoint path specified but not found: {self.config.checkpoint_path}")
      print("Creating fresh EmotionEEGNetLightning model")
      self.model = EmotionEEGNetLightning(
        self.config.model_config,
        subject_mapper=mapper,
      )

  def log_hyperparameters(self):
    """Log emotion classification hyperparameters to wandb."""
    from dataclasses import asdict
    from .training import count_n_params

    # Convert model-specific config to dict and prefix keys
    model_config_dict = {
      f"model_{k}": v for k, v in asdict(self.config.model_config.model_config).items()
    }

    params_to_log = {
      # Model structure
      "trainable_params_total": count_n_params(self.model),
      "model_config_type": type(self.config.model_config.model_config).__name__,
      "chunk_width": self.config.model_config.chunk_width,
      "num_channels": self.config.model_config.num_channels,
      "eeg_sample_rate": self.config.model_config.eeg_sample_rate,
      "num_classes": self.config.model_config.num_classes,
      # Training
      "lr_config": str(self.config.model_config.lr_config),
      "use_subject_specific": self.config.model_config.use_subject_specific,
      # Dataloader params
      "dataloader_train_size": len(self.dataloaders["train"]),
      "dataloader_val_size": len(self.dataloaders["val"]),
      "dataloader_test_size": len(self.dataloaders["test"]),
      "batch_size": self.config.batch_size,
      "num_workers": self.config.data_loader_num_workers,
      **model_config_dict,  # Add model-specific config values
    }
    self.wandb_logger.log_hyperparams(params_to_log)

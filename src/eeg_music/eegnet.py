import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field
from lightning.pytorch import LightningModule
from torcheeg.models import EEGNet as TorchEEGNet, FBCNet, TSCeption, ATCNet

from eeg_music.eegpt import mk_optimizer_and_lr_scheduler, LRCosine
from .data import NoteOnsets


class BinaryAccuracyCalc:
  """Binary classification metrics calculator with comprehensive evaluation metrics.

  Tracks true positives, true negatives, false positives, and false negatives
  to compute accuracy, recall (sensitivity), specificity, precision, and F1 score.
  """

  def __init__(self):
    self.tp = 0  # True Positives
    self.tn = 0  # True Negatives
    self.fp = 0  # False Positives
    self.fn = 0  # False Negatives

  def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Update metrics with new logits and targets.

    Args:
      logits: Model output logits (batch_size,) - single logit per sample
      targets: Ground truth binary labels (batch_size,) - 0 or 1 or bool
      threshold: Threshold for converting probabilities to predictions (default: 0.5)
    """
    # Apply sigmoid to get probabilities, then threshold to get binary predictions
    predictions = torch.sigmoid(logits) > threshold

    # Convert targets to bool for comparison
    targets_bool = targets.bool()

    # Update confusion matrix components
    self.tp += ((predictions) & (targets_bool)).sum().item()
    self.tn += ((~predictions) & (~targets_bool)).sum().item()
    self.fp += ((predictions) & (~targets_bool)).sum().item()
    self.fn += ((~predictions) & (targets_bool)).sum().item()

  def compute(self) -> dict[str, float]:
    """Compute all binary classification metrics.

    Returns:
      Dictionary with keys: accuracy, recall, specificity, precision, f1_score

    Formulas:
      - Accuracy = (TP + TN) / (TP + TN + FP + FN)
      - Recall (Sensitivity) = TP / (TP + FN)
      - Specificity = TN / (TN + FP)
      - Precision = TP / (TP + FP)
      - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    total = self.tp + self.tn + self.fp + self.fn

    # Accuracy: ratio of correct predictions
    accuracy = (self.tp + self.tn) / total if total > 0 else 0.0

    # Recall (Sensitivity, True Positive Rate): ratio of actual positives correctly identified
    recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    # Specificity (True Negative Rate): ratio of actual negatives correctly identified
    specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0

    # Precision (Positive Predictive Value): ratio of correct positive predictions
    precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    # F1 Score: harmonic mean of precision and recall
    f1_score = (
      (2 * precision * recall) / (precision + recall)
      if (precision + recall) > 0
      else 0.0
    )

    return {
      "accuracy": accuracy,
      "recall": recall,
      "specificity": specificity,
      "precision": precision,
      "f1_score": f1_score,
    }

  def reset(self):
    """Reset all counters."""
    self.tp = 0
    self.tn = 0
    self.fp = 0
    self.fn = 0


class EEGNetWrapper(nn.Module):
  """Wrapper around torcheeg EEG models for binary onset detection.

  This class wraps EEGNet, FBCNet, TSCeption, or ATCNet from torcheeg
  and adapts them for binary onset detection within a time window.

  Note: FBCCNN is not supported as it requires grid-like input preprocessing.

  Args:
      chunk_width: Total width of the input chunk in samples
      num_channels: Number of EEG channels (electrodes)
      model_config: Model-specific configuration (determines which model to use)
      **model_kwargs: Additional keyword arguments to override config values
  """

  def __init__(
    self,
    chunk_width: int,
    num_channels: int = 28,
    eeg_sample_rate: int = 256,
    model_config: Optional[
      "EEGNetConfig | FBCNetConfig | TSCeptionConfig | ATCNetConfig"
    ] = None,
    **model_kwargs,
  ):
    super().__init__()

    self.chunk_width = chunk_width
    self.num_channels = num_channels
    self.eeg_sample_rate = eeg_sample_rate
    self.model_config = model_config or EEGNetConfig()  # Default to EEGNet
    self.num_classes = 1  # Binary classification: single output

    # Initialize the chosen model
    self.model = self._create_model(**model_kwargs)

  def _create_model(self, **model_kwargs) -> nn.Module:
    """Create the underlying torcheeg model based on model_config type.

    Note: Input shapes
    - EEGNet/TSCeption/ATCNet: [batch, 1, num_electrodes, chunk_size]
    - FBCNet: [batch, in_channels, num_electrodes, chunk_size]
    """
    base = {"num_electrodes": self.num_channels, "num_classes": self.num_classes}

    match self.model_config:
      case EEGNetConfig():
        return TorchEEGNet(chunk_size=self.chunk_width, **base, **model_kwargs)

      case FBCNetConfig(num_bands=nb):
        return FBCNet(
          chunk_size=self.chunk_width, in_channels=nb, **base, **model_kwargs
        )

      case TSCeptionConfig(num_T=nt, num_S=ns, hid_channels=hc, dropout=d):
        return TSCeption(
          num_T=nt,
          num_S=ns,
          hid_channels=hc,
          dropout=d,
          sampling_rate=self.eeg_sample_rate,
          in_channels=1,
          **base,
          **model_kwargs,
        )

      case ATCNetConfig(
        num_windows=nw,
        F1=f1,
        D=d,
        tcn_kernel_size=tks,
        tcn_depth=td,
        conv_pool_size=cps,
      ):
        # Auto-adjust conv_pool_size for short sequences to preserve temporal resolution
        # Keep num_windows consistent for uniform attention granularity
        pool_size = (
          cps
          if cps is not None
          else (4 if self.chunk_width <= 256 else (6 if self.chunk_width <= 512 else 7))
        )
        return ATCNet(
          num_windows=nw or 5,  # Default to 5 windows for consistent attention
          F1=f1,
          D=d,
          tcn_kernel_size=tks,
          tcn_depth=td,
          conv_pool_size=pool_size,
          chunk_size=self.chunk_width,
          in_channels=1,
          **base,
          **model_kwargs,
        )

      case _:
        raise ValueError(f"Unknown model config type: {type(self.model_config)}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model.

    Args:
        x: Input EEG tensor of shape (batch, channels, timepoints)

    Returns:
        Output tensor of shape (batch,) with 1 logit per sample
        for binary classification (apply sigmoid for probability)
    """
    # Reshape input - add channel dimension for all models (they all expect 4D input)
    # Input: [batch, num_electrodes, chunk_size]
    # Output: [batch, 1, num_electrodes, chunk_size]
    if x.dim() == 3:
      x = x.unsqueeze(1)

    # Get raw model output: (batch, num_classes) where num_classes=1
    output = self.model(x)

    # Squeeze to (batch,) - single logit per sample
    return output.squeeze(-1)


@dataclass
class EEGNetConfig:
  """EEGNet model-specific configuration.

  EEGNet uses depthwise separable convolutions for efficiency.
  Minimal parameters, works well with short segments (32+ samples).
  """

  # No model-specific parameters needed - uses torcheeg defaults
  pass


@dataclass
class FBCNetConfig:
  """FBCNet model-specific configuration.

  FBCNet uses filter bank common spatial patterns.
  """

  num_bands: int = 1  # Number of frequency bands


@dataclass
class TSCeptionConfig:
  """TSCeption model-specific configuration.

  TSCeption uses multi-scale temporal convolutions + asymmetric spatial processing.
  Note: sampling_rate is taken from NoteOnsetModelConfig.eeg_sample_rate
  """

  num_T: int = 20  # Number of temporal filters (multi-scale)
  num_S: int = 15  # Number of spatial filters
  hid_channels: int = 48  # Hidden channels capacity
  dropout: float = 0.25  # Dropout rate


@dataclass
class ATCNetConfig:
  """ATCNet model-specific configuration.

  ATCNet uses attention + TCN for temporal modeling.
  Note: conv_pool_size is auto-adjusted based on chunk_width if not specified,
  to preserve temporal resolution for short sequences.
  """

  num_windows: int = 5  # Number of attention windows (kept consistent)
  F1: int = 24  # Number of temporal filters
  D: int = 2  # Spatial filter multiplier
  tcn_kernel_size: int = 5  # TCN kernel size
  tcn_depth: int = 3  # Number of TCN layers
  conv_pool_size: Optional[int] = None  # Auto-adjusted: 3/5/7 based on chunk_width


@dataclass
class NoteOnsetModelConfig:
  """Configuration for EEG note onset detection model.

  Args:
      model_config: Model-specific configuration (EEGNet, FBCNet, TSCeption, or ATCNet)
      chunk_width: Total width of input chunk in samples
      num_channels: Number of EEG channels
      eeg_sample_rate: EEG sampling rate in Hz
      window_start: Start sample index of target window (constant for all samples)
      window_end: End sample index of target window (constant for all samples)
      lr_config: Learning rate config - either a float or LRCosine scheduler config
      pos_weight: Positive class weight for BCEWithLogitsLoss (to handle class imbalance)
  """

  model_config: EEGNetConfig | FBCNetConfig | TSCeptionConfig | ATCNetConfig = field(
    default_factory=EEGNetConfig
  )
  chunk_width: int = 1024
  num_channels: int = 28
  eeg_sample_rate: int = 256
  window_start: int = 0
  window_end: int = 256
  lr_config: float | LRCosine = 1e-4
  pos_weight: Optional[float] = None


def has_onset_in_window(
  note_onsets: NoteOnsets,
  window_start: int,
  window_end: int,
  eeg_sample_rate: int,
) -> bool:
  """Check if any onset occurs within a specified time window (single sample).

  Args:
      note_onsets: NoteOnsets object containing onset times in seconds
      window_start: Start sample index
      window_end: End sample index
      eeg_sample_rate: EEG sampling rate in Hz

  Returns:
      True if any onset falls in [window_start, window_end), False otherwise
  """
  if len(note_onsets.onset_times) == 0:
    return False

  # Convert onset times from seconds to samples
  onset_samples = (
    torch.tensor(note_onsets.onset_times, dtype=torch.float32) * eeg_sample_rate
  )

  # Check if any onset falls within [window_start, window_end)
  return bool(
    ((onset_samples >= window_start) & (onset_samples < window_end)).any().item()
  )


class EEGNetLightning(LightningModule):
  """PyTorch Lightning module for training EEGNet onset detection model.

  This module wraps the EEGNetWrapper and handles training, validation,
  and testing with binary cross-entropy loss.
  """

  def __init__(self, config: NoteOnsetModelConfig, **model_kwargs):
    super().__init__()
    self.config = config
    if isinstance(self.config.lr_config, float):
      # to access by LearningRateFinder
      self.learning_rate = self.config.lr_config
    self.save_hyperparameters()

    # Create the model - EEGNetWrapper infers model type from config
    self.model = EEGNetWrapper(
      chunk_width=config.chunk_width,
      num_channels=config.num_channels,
      eeg_sample_rate=config.eeg_sample_rate,
      model_config=config.model_config,
      **model_kwargs,
    )

    # Loss function for binary classification
    # pos_weight can help with class imbalance (onsets are typically rare)
    if config.pos_weight is not None:
      pos_weight = torch.tensor([config.pos_weight])
      self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
      self.loss_fn = nn.BCEWithLogitsLoss()

    # Store window range constants
    self.window_start = config.window_start
    self.window_end = config.window_end

    # Binary classification metric calculators
    self.train_metrics = BinaryAccuracyCalc()
    self.val_metrics = BinaryAccuracyCalc()
    self.test_metrics = BinaryAccuracyCalc()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model."""
    return self.model(x)

  def _compute_loss(self, batch, batch_idx, stage: str):
    """Compute loss for a batch.

    Args:
        batch: Dictionary with keys:
            - 'eeg': (batch, channels, timepoints)
            - 'music': List of NoteOnsets objects
        batch_idx: Batch index
        stage: 'train', 'val', or 'test'

    Returns:
        loss: Computed loss value
    """
    x = batch["eeg"]  # (batch, channels, timepoints)
    note_onsets_list = batch["music"]  # List of NoteOnsets objects
    batch_size = x.shape[0]

    # Forward pass
    y_hat = self(x)  # (batch,) - single logit per sample

    # Check if onsets fall within the specified window for each sample
    y = torch.tensor(
      [
        has_onset_in_window(
          note_onsets_list[i],
          window_start=self.window_start,
          window_end=self.window_end,
          eeg_sample_rate=self.config.eeg_sample_rate,
        )
        for i in range(batch_size)
      ],
      dtype=torch.float32,
      device=x.device,
    )

    # Compute loss
    loss = self.loss_fn(y_hat, y)

    # Update metrics using reflection
    metrics = getattr(self, f"{stage}_metrics")
    metrics.update(y_hat, y)

    # Log loss
    self.log(
      f"{stage}_loss",
      loss,
      on_step=(stage == "train"),
      on_epoch=True,
      prog_bar=True,
      logger=True,
      batch_size=batch_size,
    )

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
    metrics = getattr(self, f"{stage}_metrics").compute()
    for metric_name, metric_value in metrics.items():
      self.log(
        f"{stage}_{metric_name}",
        metric_value,
        on_epoch=True,
        prog_bar=True,
        logger=True,
      )
    getattr(self, f"{stage}_metrics").reset()

  def on_train_epoch_end(self):
    self._log_and_reset_metrics("train")

  def on_validation_epoch_end(self):
    self._log_and_reset_metrics("val")

  def on_test_epoch_end(self):
    self._log_and_reset_metrics("test")

  def configure_optimizers(self):
    optimizer, lr_scheduler = mk_optimizer_and_lr_scheduler(
      self.model.parameters(), self.config.lr_config
    )
    if lr_scheduler is None:
      return optimizer
    return [optimizer], [lr_scheduler]

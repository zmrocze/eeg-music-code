"""Mel-spectrogram reconstruction and CNN classifier training."""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Literal, Union

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import (
  ModelCheckpoint,
  LearningRateMonitor,
  RichProgressBar,
)
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder

from .cnn import (
  CNNReconstruction,
  CNNReconstructionMel,
  CNNClassifier,
  CNNClassifierRaw,
)
from .lstm import BiLSTM
from .eegpt import LRCosine, LRStepLR, UseAdamW, UseSGD, mk_optimizer_and_lr_scheduler
from .training import (
  MainTraining,
  OnExceptionCheckpoint,
  SpectrogramLoggingCallback,
  AUROCCallback,
  count_n_params,
)
from .dataloader import create_collate_fn, create_dataloader


@dataclass
class CNNClassifierConfig:
  """Config wrapper for CNNClassifier."""

  in_channels: int = 1
  dropout: float = 0.25
  channels: int = 32


@dataclass
class CNNClassifierRawConfig:
  """Config wrapper for CNNClassifierRaw (129×256 input, bigger kernels)."""

  in_channels: int = 1
  dropout: float = 0.25


@dataclass
class CNNReconstructionConfig:
  """Config wrapper for CNNReconstruction."""

  in_channels: int = 1
  out_channels: int = 1
  dropout: float = 0.25
  variant: Literal["full", "timeconstant"] = "full"


@dataclass
class BiLSTMConfig:
  """Config wrapper for BiLSTM (for ODF 1D sequences)."""

  input_size: int = 60
  hidden_size: int = 64
  num_layers: int = 2
  output_size: int = 1


@dataclass
class MelModelConfig:
  """Model config for mel reconstruction — only the fields MelLightning actually uses."""

  model_config: CNNReconstructionConfig | BiLSTMConfig = field(
    default_factory=CNNReconstructionConfig
  )
  lr_config: float | LRCosine | LRStepLR = 1e-4
  optimizer: UseAdamW | UseSGD = field(default_factory=UseAdamW)


@dataclass
class MelTrainingConfig:
  """Single config for mel-spectrogram reconstruction training.

  Contains all and only the fields needed by MelTraining.
  """

  # Model
  model_config: MelModelConfig = field(default_factory=MelModelConfig)

  # Dataloader
  batch_size: int = 64
  data_loader_num_workers: int = 4
  prefetch_factor: int = 2
  pin_memory: bool = True
  include_info: bool = True
  music_batch_fn: object = None

  # Training
  num_epochs: int = 200
  val_every_n_epoch: int = 1
  save_model_per_epochs: int = 5
  use_learning_rate_finder: bool = False

  # AUROC callbacks
  auroc_every_n_epochs: int = 2
  auroc_similarity_metric: list[Literal["cosine", "structural_similarity"]] = field(
    default_factory=lambda: ["cosine", "structural_similarity"]
  )
  auroc_prediction_batch_size: int = 128

  # Wandb / saving
  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "mel-reconstruction-cnn"
  run_name: str = "cnn-reconstruction-mel"
  run_extra_name: str = "0"
  randint: int = 0
  save_path: str = "cnn-reconstruction-mel-ckpt"


class MelLightning(LightningModule):
  """Lightning module for mel-spectrogram reconstruction with MSE loss."""

  def __init__(self, config: MelModelConfig):
    super().__init__()
    self.config = config
    if isinstance(self.config.lr_config, float):
      self.learning_rate = self.config.lr_config
    self.save_hyperparameters()

    mc = config.model_config
    match mc:
      case CNNReconstructionConfig(variant="full"):
        self.model = CNNReconstruction(
          in_channels=mc.in_channels, out_channels=mc.out_channels, dropout=mc.dropout
        )
      case CNNReconstructionConfig(variant="timeconstant"):
        self.model = CNNReconstructionMel(
          in_channels=mc.in_channels, dropout=mc.dropout
        )
      case BiLSTMConfig():
        self.model = BiLSTM(
          input_size=mc.input_size,
          hidden_size=mc.hidden_size,
          num_layers=mc.num_layers,
          output_size=mc.output_size,
        )
      case _:
        raise ValueError(f"Unsupported model_config for mel reconstruction: {type(mc)}")

    self.is_lstm = isinstance(mc, BiLSTMConfig)

    self.loss_fn = nn.MSELoss()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.is_lstm:
      # LSTM expects (B, seq_len, input_size): transpose (B, channels, time) -> (B, time, channels)
      out = self.model(x.transpose(1, 2))  # (B, time, output_size)
      # Reshape to match mel format: (B, time, 1) -> (B, 1, 1, time)
      return out.transpose(1, 2).unsqueeze(1)
    # CNN: EEG arrives as (B, channels, time) — add image channel dim -> (B, 1, channels, time)
    return self.model(x.unsqueeze(1))

  def _step(self, batch, stage: str):
    x = batch["eeg"]
    y = batch["music"]  # (B, 1, n_mels, n_frames) or (B, 1, 1, 1) for LSTM
    y_hat = self(x)  # (B, 1, 64, 64) or (B, 1, 1, 1) for LSTM
    loss = self.loss_fn(y_hat, y)
    self.log(
      f"{stage}_loss",
      loss,
      on_step=(stage == "train"),
      on_epoch=True,
      prog_bar=True,
      logger=True,
      batch_size=x.shape[0],
    )
    return loss

  def training_step(self, batch, batch_idx):
    return self._step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self._step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self._step(batch, "test")

  def configure_optimizers(self):
    optimizer, lr_scheduler = mk_optimizer_and_lr_scheduler(
      self.model.parameters(),
      self.config.lr_config,
      self.config.optimizer,
    )
    if lr_scheduler is None:
      return optimizer
    return [optimizer], [lr_scheduler]


class MelTraining(MainTraining):
  """Training class for mel reconstruction using CNNReconstruction.

  Takes MelTrainingConfig and pre-built datasets. Inherits logger, trainer,
  fit/test from MainTraining. Overrides create_dataloaders, create_model,
  create_callbacks and log_hyperparameters.
  """

  def __init__(self, config: MelTrainingConfig, train_ds, val_ds, test_ds):
    super().__init__(config)
    self._train_ds = train_ds
    self._val_ds = val_ds
    self._test_ds = test_ds

  def create_dataloaders(self):
    def default_music_batch_fn(xs):
      return torch.stack(
        [torch.from_numpy((x.mel + 40.0) / 40.0).float() for x in xs]
      ).unsqueeze(1)

    collate_fn = create_collate_fn(
      include_info=self.config.include_info,
      music_batch_fn=self.config.music_batch_fn
      if self.config.music_batch_fn is not None
      else default_music_batch_fn,
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

  def create_model(self):
    """Create MelLightning from config.model_config."""
    self.model = MelLightning(self.config.model_config)

  def create_callbacks(self):
    """Callbacks like MainTraining: val_loss checkpoint, AUROC, spectrogram logging."""
    save_on_exc = OnExceptionCheckpoint(f"{self.config.save_path}/exc_save")

    ckpt_callback = ModelCheckpoint(
      every_n_epochs=self.config.save_model_per_epochs,
      dirpath=self.config.save_path,
      save_top_k=2,
      monitor="val_loss",
      mode="min",
      save_last=True,
    )

    optional_lr_finder = (
      [LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100)]
      if self.config.use_learning_rate_finder
      else []
    )

    auroc_callbacks = [
      AUROCCallback(
        auroc_every_n_epochs=self.config.auroc_every_n_epochs,
        similarity_metric=metric,
        prediction_batch_size=self.config.auroc_prediction_batch_size,
      )
      for metric in self.config.auroc_similarity_metric
    ]

    self.callbacks = (
      [
        ckpt_callback,
        SpectrogramLoggingCallback(),
        RichProgressBar(),
        save_on_exc,
        LearningRateMonitor(logging_interval="step"),
      ]
      + auroc_callbacks
      + optional_lr_finder
    )

  def log_hyperparameters(self):
    from dataclasses import asdict

    model_config_dict = {
      f"model_{k}": v for k, v in asdict(self.config.model_config.model_config).items()
    }
    self.wandb_logger.log_hyperparams(
      {
        "trainable_params_total": count_n_params(self.model),
        "model_config_type": type(self.config.model_config.model_config).__name__,
        "batch_size": self.config.batch_size,
        "num_workers": self.config.data_loader_num_workers,
        "lr_config": str(self.config.model_config.lr_config),
        "dataloader_train_size": len(self.dataloaders["train"]),
        "dataloader_val_size": len(self.dataloaders["val"]),
        "dataloader_test_size": len(self.dataloaders["test"]),
        **model_config_dict,
      }
    )


class WeightedMSELoss(nn.Module):
  """Custom weighted MSE loss for ODF reconstruction.

  Uses a weighted MSE loss where:
  - w(x) = 20 for 0 <= x < 0.05
  - w(x) = linear from 4 to 1 for 0.05 <= x <= 1

  Loss formula: Sum_i w(yhat_i) * (y_i - yhat_i)^2
  """

  def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute weighted MSE loss.

    Args:
      y_hat: Predicted ODF values
      y: Target ODF values

    Returns:
      Weighted MSE loss
    """
    # Compute weights based on predicted values
    # w(x) = 20 for 0 <= x < 0.05
    # w(x) = 4 - 3*(x - 0.05)/(1 - 0.05) for 0.05 <= x <= 1
    # Simplified: w(x) = 4 - 3*(x - 0.05)/0.95 = 4 - 3.157894737*x + 0.157894737
    #           = 4.157894737 - 3.157894737*x

    weights = torch.where(
      y_hat < 0.05,
      torch.tensor(1.0, device=y_hat.device, dtype=y_hat.dtype),
      torch.tensor(100.0, device=y_hat.device, dtype=y_hat.dtype),
    )

    # Clamp weights to [1, 20] to handle values outside [0, 1]
    weights = torch.clamp(weights, min=1.0, max=20.0)

    # Compute weighted MSE
    squared_error = (y - y_hat) ** 2
    weighted_loss = weights * squared_error

    return weighted_loss.mean()


class OdfLightning(MelLightning):
  """Lightning module for ODF reconstruction with custom weighted MSE loss."""

  def __init__(self, config: MelModelConfig):
    super().__init__(config)
    self.loss_fn = WeightedMSELoss()


class OdfTraining(MelTraining):
  """Training for ODF (onset detection function) data.

  Inherits from MelTraining but uses OdfLightning with custom weighted loss
  and skips mel-spectrogram normalization since ODF values are already in [0, 1] range.
  """

  def create_model(self):
    """Create OdfLightning from config.model_config."""
    self.model = OdfLightning(self.config.model_config)

  def create_dataloaders(self):
    def default_music_batch_fn(xs):
      return torch.stack([torch.from_numpy(x.mel).float() for x in xs]).unsqueeze(1)

    collate_fn = create_collate_fn(
      include_info=self.config.include_info,
      music_batch_fn=self.config.music_batch_fn
      if self.config.music_batch_fn is not None
      else default_music_batch_fn,
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


# ---------------------------------------------------------------------------
# CNN Classifier (label prediction from EEG)
# ---------------------------------------------------------------------------


@dataclass
class ClassifierModelConfig:
  """Model config for CNN classifier — fields used by ClassifierLightning."""

  model_config: CNNClassifierConfig | CNNClassifierRawConfig = field(
    default_factory=CNNClassifierConfig
  )
  num_classes: int = 4
  loss: Literal["ce", "bce"] = "ce"
  lr_config: float | LRCosine | LRStepLR = 1e-4
  optimizer: UseAdamW | UseSGD = field(default_factory=UseAdamW)


@dataclass
class ClassifierTrainingConfig:
  """Single config for CNN classifier training.

  Contains all and only the fields needed by ClassifierTraining.
  """

  model_config: ClassifierModelConfig = field(default_factory=ClassifierModelConfig)

  # Dataloader
  batch_size: int = 64
  data_loader_num_workers: int = 4
  prefetch_factor: int = 2
  pin_memory: bool = True
  include_info: bool = True

  # Training
  num_epochs: int = 200
  val_every_n_epoch: int = 1
  save_model_per_epochs: int = 5
  use_learning_rate_finder: bool = False

  # Wandb / saving
  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "cnn-classifier"
  run_name: str = "cnn-classifier"
  run_extra_name: str = "0"
  randint: int = 0
  save_path: str = "cnn-classifier-ckpt"


class ClassifierLightning(LightningModule):
  """Lightning module for CNN label classification (CE or BCE loss)."""

  def __init__(self, config: ClassifierModelConfig):
    super().__init__()
    self.config = config
    if isinstance(self.config.lr_config, float):
      self.learning_rate = self.config.lr_config
    self.save_hyperparameters()

    mc = config.model_config
    num_out = 1 if config.loss == "bce" else config.num_classes
    match mc:
      case CNNClassifierRawConfig():
        self.model = CNNClassifierRaw(
          num_classes=num_out, in_channels=mc.in_channels, dropout=mc.dropout
        )
      case CNNClassifierConfig():
        self.model = CNNClassifier(
          num_classes=num_out,
          in_channels=mc.in_channels,
          dropout=mc.dropout,
          channels=mc.channels,
        )

    self.loss_fn: nn.Module = (
      nn.BCEWithLogitsLoss() if config.loss == "bce" else nn.CrossEntropyLoss()
    )
    self._reset_metrics()

  # ---- metrics bookkeeping ----
  def _reset_metrics(self):
    self._metrics: dict[str, dict[str, int | float]] = {}

  def _update_metrics(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
    if stage not in self._metrics:
      self._metrics[stage] = {"correct": 0, "total": 0}
    preds = (
      (logits.squeeze(-1) > 0).long()
      if self.config.loss == "bce"
      else logits.argmax(dim=1)
    )
    self._metrics[stage]["correct"] += (preds == targets).sum().item()
    self._metrics[stage]["total"] += targets.size(0)

  def _log_and_reset_metrics(self, stage: str):
    m = self._metrics.get(stage, {"correct": 0, "total": 0})
    acc = m["correct"] / m["total"] if m["total"] > 0 else 0.0
    self.log(f"{stage}_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
    self._metrics.pop(stage, None)

  # ---- forward / steps ----
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x.unsqueeze(1))  # (B, ch, time) -> (B, 1, ch, time)

  def _step(self, batch, stage: str):
    x = batch["eeg"]
    y = batch["music"]  # (B,) integer labels
    logits = self(x)
    if self.config.loss == "bce":
      loss = self.loss_fn(logits.squeeze(-1), y.float())
    else:
      loss = self.loss_fn(logits, y)
    self._update_metrics(logits, y, stage)
    self.log(
      f"{stage}_loss",
      loss,
      on_step=(stage == "train"),
      on_epoch=True,
      prog_bar=True,
      logger=True,
      batch_size=x.shape[0],
    )
    return loss

  def training_step(self, batch, batch_idx):
    return self._step(batch, "train")

  def on_train_epoch_end(self):
    self._log_and_reset_metrics("train")

  def validation_step(self, batch, batch_idx):
    return self._step(batch, "val")

  def on_validation_epoch_end(self):
    self._log_and_reset_metrics("val")

  def test_step(self, batch, batch_idx):
    return self._step(batch, "test")

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


class ClassifierTraining(MainTraining):
  """Training class for CNN label classification.

  Expects datasets whose music_data is MusingMusicIdData (or any object
  with a .music_id.song_id integer attribute).
  """

  def __init__(self, config: ClassifierTrainingConfig, train_ds, val_ds, test_ds):
    super().__init__(config)
    self._train_ds = train_ds
    self._val_ds = val_ds
    self._test_ds = test_ds

  def create_dataloaders(self):
    collate_fn = create_collate_fn(
      include_info=self.config.include_info,
      music_batch_fn=lambda xs: torch.tensor(
        [x.music_id.song_id - 1 for x in xs],
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

  def create_model(self):
    self.model = ClassifierLightning(self.config.model_config)

  def create_callbacks(self):
    save_on_exc = OnExceptionCheckpoint(f"{self.config.save_path}/exc_save")

    ckpt_loss = ModelCheckpoint(
      every_n_epochs=self.config.save_model_per_epochs,
      dirpath=self.config.save_path,
      save_top_k=1,
      monitor="val_loss",
      mode="min",
      filename="best-loss-{epoch:02d}-{val_loss:.3f}",
      save_last=True,
    )

    ckpt_acc = ModelCheckpoint(
      every_n_epochs=self.config.save_model_per_epochs,
      dirpath=self.config.save_path,
      save_top_k=1,
      monitor="val_accuracy",
      mode="max",
      filename="best-acc-{epoch:02d}-{val_accuracy:.3f}",
    )

    optional_lr_finder = (
      [LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100)]
      if self.config.use_learning_rate_finder
      else []
    )

    self.callbacks = [
      ckpt_loss,
      ckpt_acc,
      RichProgressBar(),
      save_on_exc,
      LearningRateMonitor(logging_interval="step"),
    ] + optional_lr_finder

  def log_hyperparameters(self):
    from dataclasses import asdict

    model_config_dict = {
      f"model_{k}": v for k, v in asdict(self.config.model_config.model_config).items()
    }
    self.wandb_logger.log_hyperparams(
      {
        "trainable_params_total": count_n_params(self.model),
        "model_config_type": type(self.config.model_config.model_config).__name__,
        "num_classes": self.config.model_config.num_classes,
        "loss": self.config.model_config.loss,
        "batch_size": self.config.batch_size,
        "num_workers": self.config.data_loader_num_workers,
        "lr_config": str(self.config.model_config.lr_config),
        "dataloader_train_size": len(self.dataloaders["train"]),
        "dataloader_val_size": len(self.dataloaders["val"]),
        "dataloader_test_size": len(self.dataloaders["test"]),
        **model_config_dict,
      }
    )

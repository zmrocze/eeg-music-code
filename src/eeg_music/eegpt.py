from pathlib import Path
from downstream.Modules.models.EEGPT_mcae_finetune import EEGPTClassifier
from lightning.pytorch import LightningModule
import wandb
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Union, Optional, List

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class AccuracyCalc:
  """Simple accuracy calculator that tracks correct predictions and total count."""

  def __init__(self):
    self.correct = 0
    self.total = 0

  def update(self, logits: torch.Tensor, targets: torch.Tensor):
    """Update accuracy with new logits and targets.

    Args:
      logits: Model output logits (batch_size, num_classes)
      targets: Ground truth class indices (batch_size,)
    """
    predictions = torch.argmax(logits, dim=1)
    self.correct += (predictions == targets).sum().item()
    self.total += targets.size(0)

  def compute(self) -> float:
    """Compute current accuracy."""
    return self.correct / self.total if self.total > 0 else 0.0

  def reset(self):
    """Reset counters."""
    self.correct = 0
    self.total = 0


finetuning_all_ch = [
  "FP1",
  "FPZ",
  "FP2",
  "AF7",
  "AF3",
  "AF4",
  "AF8",
  "F7",
  "F5",
  "F3",
  "F1",
  "FZ",
  "F2",
  "F4",
  "F6",
  "F8",
  "FT7",
  "FC5",
  "FC3",
  "FC1",
  "FCZ",
  "FC2",
  "FC4",
  "FC6",
  "FT8",
  "T7",
  "C5",
  "C3",
  "C1",
  "CZ",
  "C2",
  "C4",
  "C6",
  "T8",
  "TP7",
  "CP5",
  "CP3",
  "CP1",
  "CPZ",
  "CP2",
  "CP4",
  "CP6",
  "TP8",
  "P7",
  "P5",
  "P3",
  "P1",
  "PZ",
  "P2",
  "P4",
  "P6",
  "P8",
  "PO7",
  "PO5",
  "PO3",
  "POZ",
  "PO4",
  "PO6",
  "PO8",
  "O1",
  "OZ",
  "O2",
]

all_in_pretraining = [
  "FP2",
  "FPz",
  "FP1",
  "AF4",
  "AF3",
  "F7",
  "F5",
  "F3",
  "F6",
  "F1",
  "Fz",
  "F2",
  "F4",
  "F8",
  "FT7",
  "FC5",
  "FC3",
  "FC6",
  "FC1",
  "FCz",
  "FC2",
  "FC4",
  "FT8",
  "T7",
  "C5",
  "C3",
  "C6",
  "C1",
  "Cz",
  "C2",
  "C4",
  "T8",
  "TP7",
  "CP5",
  "CP3",
  "CP6",
  "CP1",
  "CPz",
  "CP2",
  "CP4",
  "TP8",
  "P7",
  "P5",
  "P3",
  "P6",
  "P1",
  "Pz",
  "P2",
  "P4",
  "P8",
  "O1",
  "PO7",
  "PO3",
  "O2",
  "Oz",
  "PO4",
  "PO8",
  "POz",
]


channels_calibration = [
  "FP1",
  "FPz",
  "FP2",
  "F7",
  "F3",
  "Fz",
  "F4",
  "F8",
  "FT9",
  "FC5",
  "FC1",
  "FC2",
  "FC6",
  "FT10",
  "T7",
  "C3",
  "Cz",
  "C4",
  "T8",
  "TP9",
  "CP5",
  "CP1",
  "CP2",
  "CP6",
  "TP10",
  "P7",
  "P3",
  "Pz",
  "P4",
  "P8",
  "O1",
  "O2",
]

# these are extra:   #  "GSR", "ECG", "VA1", "VA2", "VAtarg"]
# and these eeg:  FT9 FT10 TP9 TP10 (not in pretraining)
# and these not in pretraining but in finetuning, so let's include: 'TP9', 'FT10', 'FT9', 'FPz', 'Cz', 'Fz', 'TP10', 'Pz'
# and use chan_conv
# print(len(channels_calibration)) # 37
# picking from finetuning_all_ch
USING_CHANNELS = [ch for ch in channels_calibration if ch.upper() in finetuning_all_ch]

EEG_WIDTH = 256 * 4

# mising: target_encoder, predictor, fc, head

# Load the pretrained weights

# "./model_checkpoints/25866970/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"


def load_model(chpt_path, use_chan_conv, num_classes=0) -> EEGPTClassifier:
  model = EEGPTClassifier(
    num_classes,
    in_channels=len(USING_CHANNELS),
    img_size=[len(USING_CHANNELS), EEG_WIDTH],
    use_channels_names=USING_CHANNELS,
    patch_stride=64,
    use_chan_conv=use_chan_conv,
    use_predictor=True,
    desired_time_len=EEG_WIDTH,
    interpolate_factor=1.0,
  )
  checkpoint = torch.load(chpt_path, map_location="cpu", weights_only=False)
  model.load_state_dict(
    checkpoint["state_dict"], strict=False
  )  # strict=False to allow new classification head

  # # Ensure all parameters are float32 to avoid mixed precision issues
  model = model.float()

  return model


class EegptWithLinear(torch.nn.Module):
  def __init__(self, model, hidden_dim=256):
    super().__init__()
    self.model = model
    self.linear = ResidualLinear(model.embed_dim, 4 * 128, hidden_dim)

  def forward(self, x):
    x = self.model(x, return_patch_tokens=True)
    x = self.linear(x)  # B x patches x 512
    x = x.reshape(x.shape[0], x.shape[1] * 4, 128)  # B x 4 * patches x 128
    x = x.permute(0, 2, 1)  # B x 128 x 4 * patches
    return x

  @classmethod
  def load_from_checkpoint(cls, chpt_path, use_chan_conv=True):
    model = load_model(chpt_path, use_chan_conv=use_chan_conv)
    return cls(model)


class EegptWithLinearClassifier(torch.nn.Module):
  def __init__(self, model, hidden_dim=256, num_classes=9):
    super().__init__()
    self.model = model
    self.num_classes = num_classes
    self.hidden_dim = hidden_dim
    self.linear = ResidualLinear(model.embed_dim, num_classes, hidden_dim=hidden_dim)

  def forward(self, x):
    x = self.model(x, return_patch_tokens=False)
    x = self.linear(x)  # B x num_classes
    return x

  @classmethod
  def load_from_checkpoint(cls, chpt_path, use_chan_conv=True):
    model = load_model(chpt_path, use_chan_conv=use_chan_conv, num_classes=0)
    return cls(model)


class ResidualLinear(torch.nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=256):
    super(ResidualLinear, self).__init__()
    # like in eegptclassifier
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(input_dim, hidden_dim)
    # linear with weights l2 norm equal 1
    # self.linear3 = LinearWithConstraint(hidden_dim, output_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    residual = x  # input_dim
    x = self.linear2(residual) + torch.relu(self.linear1(x))  # hidden_dim
    x = self.linear3(x)  # output_dim
    return x


@dataclass
class LRCosine:
  max_lr: float
  T_0: int
  T_mult: int
  eta_min: float = 1e-8
  last_epoch: int = -1


@dataclass
class EegptConfig:
  chpt_path: Path
  trainable: Optional[
    List[str]
  ]  # None = all trainable; list can contain: "chan_conv", "linear", "head", "reconstructor", "predictor", "target_encoder"
  requiring_grad: Optional[
    List[str]
  ]  # None = all require grad; list can contain: "chan_conv", "linear", "head", "reconstructor", "predictor", "target_encoder"
  lr_config: Union[float, LRCosine] = 1e-4
  use_chan_conv: bool = True
  # batch_size: int = 8
  # num_workers: int = 4
  # prefetch_factor: int = 2


def mk_optimizer_and_lr_scheduler(trainable_params, config: EegptConfig):
  """
  Create optimizer and learning rate scheduler.

  Args:
    trainable_params: Iterator of parameters to optimize
    config: EegptConfig with lr_config settings
  """
  if isinstance(config.lr_config, float):
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr_config)
    return optimizer, None
  elif isinstance(config.lr_config, LRCosine):
    lr_config = config.lr_config
    optimizer = torch.optim.AdamW(trainable_params, lr=lr_config.max_lr)
    lr_scheduler = CosineAnnealingWarmRestarts(
      optimizer,
      T_0=lr_config.T_0,
      T_mult=lr_config.T_mult,
      eta_min=lr_config.eta_min,
      last_epoch=lr_config.last_epoch,
    )
    return optimizer, lr_scheduler
  else:
    raise ValueError(f"Unknown lr_config type: {type(config.lr_config)}")


class EegptLightning(LightningModule):
  def __init__(self, config: EegptConfig):
    super().__init__()
    self.config = config
    if isinstance(self.config.lr_config, float):
      # to access by LearningRateFinder
      self.learning_rate = self.config.lr_config
    self.save_hyperparameters()
    self.model = EegptWithLinear.load_from_checkpoint(
      self.config.chpt_path,
      use_chan_conv=self.config.use_chan_conv,
    )
    self.loss_fn = MSELoss()

    # Set trainable parameters based on config
    self._setup_trainable_parameters()

  @classmethod
  def load_from_wandb(cls, artifact_name):
    ### artifact_name i.e. 'zmrocze-uniwroc/neural-music-decoding/model-wttyehe9:v1'
    run = wandb.init()
    artifact = run.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()
    model = cls.load_from_checkpoint(artifact_dir + "/model.ckpt")
    return model

  def _setup_trainable_parameters(self):
    """
    Set requires_grad for parameters based on config.requiring_grad.
    If requiring_grad is None, all parameters have requires_grad=True.
    Otherwise, only the specified components have requires_grad=True.
    Handles both EegptWithLinear wrapper and direct EEGPTClassifier models.
    """
    # Check if model is wrapped (EegptWithLinear) or direct (EEGPTClassifier)
    if hasattr(self.model, "model") and hasattr(self.model, "linear"):
      # EegptWithLinear wrapper
      eegpt_with_linear = self.model
      eegpt_classifier = self.model.model
    else:
      # Direct EEGPTClassifier
      eegpt_with_linear = None
      eegpt_classifier = self.model

    if self.config.requiring_grad is None:
      # All parameters have requires_grad=True (default)
      for param in self.parameters():
        param.requires_grad = True
      return

    # Set requires_grad=False for everything first
    for param in self.parameters():
      param.requires_grad = False

    # Set requires_grad=True for specified components
    for component in self.config.requiring_grad:
      if component == "chan_conv":
        for param in eegpt_classifier.chan_conv.parameters():  # pyright: ignore[reportAttributeAccessIssue]
          param.requires_grad = True
      elif component == "linear":
        if eegpt_with_linear is not None:
          for param in eegpt_with_linear.linear.parameters():  # pyright: ignore[reportAttributeAccessIssue]
            param.requires_grad = True
      elif component == "head":
        for param in eegpt_classifier.head.parameters():  # pyright: ignore[reportAttributeAccessIssue]
          param.requires_grad = True
      elif component == "reconstructor":
        if hasattr(eegpt_classifier, "reconstructor"):
          for param in eegpt_classifier.reconstructor.parameters():  # pyright: ignore[reportAttributeAccessIssue]
            param.requires_grad = True
      elif component == "predictor":
        if hasattr(eegpt_classifier, "predictor"):
          for param in eegpt_classifier.predictor.parameters():  # pyright: ignore[reportAttributeAccessIssue]
            param.requires_grad = True
      elif component == "target_encoder":
        for param in eegpt_classifier.target_encoder.parameters():  # pyright: ignore[reportAttributeAccessIssue]
          param.requires_grad = True
      else:
        raise ValueError(f"Unknown requiring_grad component: {component}")

  def _get_trainable_parameters(self):
    """
    Get parameters to pass to optimizer based on config.trainable.
    If trainable is None, returns all parameters with requires_grad=True.
    Otherwise, returns only parameters from specified components.
    Handles both EegptWithLinear wrapper and direct EEGPTClassifier models.
    """
    # Check if model is wrapped (EegptWithLinear) or direct (EEGPTClassifier)
    if hasattr(self.model, "model") and hasattr(self.model, "linear"):
      # EegptWithLinear wrapper
      eegpt_with_linear = self.model
      eegpt_classifier = self.model.model
    else:
      # Direct EEGPTClassifier
      eegpt_with_linear = None
      eegpt_classifier = self.model

    if self.config.trainable is None:
      # Return all parameters with requires_grad=True
      return filter(lambda p: p.requires_grad, self.parameters())

    # Collect parameters from specified components
    trainable_params = []
    for component in self.config.trainable:
      if component == "chan_conv":
        trainable_params.extend(eegpt_classifier.chan_conv.parameters())  # pyright: ignore[reportAttributeAccessIssue]
      elif component == "linear":
        if eegpt_with_linear is not None:
          trainable_params.extend(eegpt_with_linear.linear.parameters())  # pyright: ignore[reportAttributeAccessIssue]
      elif component == "head":
        trainable_params.extend(eegpt_classifier.head.parameters())  # pyright: ignore[reportAttributeAccessIssue]
      elif component == "reconstructor":
        if hasattr(eegpt_classifier, "reconstructor"):
          trainable_params.extend(eegpt_classifier.reconstructor.parameters())  # pyright: ignore[reportAttributeAccessIssue]
      elif component == "predictor":
        if hasattr(eegpt_classifier, "predictor"):
          trainable_params.extend(eegpt_classifier.predictor.parameters())  # pyright: ignore[reportAttributeAccessIssue]
      elif component == "target_encoder":
        trainable_params.extend(eegpt_classifier.target_encoder.parameters())  # pyright: ignore[reportAttributeAccessIssue]
      else:
        raise ValueError(f"Unknown trainable component: {component}")
    return iter(trainable_params)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x = batch["eeg"]
    y = batch["mel"]
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    self.log(
      "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
    )
    return loss

  def validation_step(self, batch, batch_idx):
    x = batch["eeg"]
    y = batch["mel"]
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

  def test_step(self, batch, batch_idx):
    x = batch["eeg"]
    y = batch["mel"]
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    self.log(
      "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
    )

  def configure_optimizers(self):
    trainable_params = self._get_trainable_parameters()
    optimizer, lr_scheduler = mk_optimizer_and_lr_scheduler(
      trainable_params, self.config
    )
    if lr_scheduler is None:
      return optimizer
    return [optimizer], [lr_scheduler]


class EegptEmotionClassifier(EegptLightning):
  def __init__(self, config: EegptConfig, num_classes=9):
    super(EegptLightning, self).__init__()
    self.config = config
    if isinstance(self.config.lr_config, float):
      # to access by LearningRateFinder
      self.learning_rate = self.config.lr_config
    self.num_classes = num_classes
    self.save_hyperparameters()
    self.model = load_model(
      chpt_path=self.config.chpt_path,
      use_chan_conv=self.config.use_chan_conv,
      num_classes=self.num_classes,
    )
    self.loss_fn = CrossEntropyLoss()

    # Accuracy calculators
    self.train_accuracy = AccuracyCalc()
    self.val_accuracy = AccuracyCalc()
    self.test_accuracy = AccuracyCalc()

    # Set trainable parameters based on config
    self._setup_trainable_parameters()

  # _setup_trainable_parameters and _get_trainable_parameters are inherited from EegptLightning

  def training_step(self, batch, batch_idx):
    x = batch["eeg"]
    emotion_codes = batch["info"]["emotion"]  # List of emotion codes (1-9 or None)
    # Convert to class indices (0-8) for CrossEntropyLoss. Subtract 1 to map 1-9 to 0-8
    y = torch.tensor(
      [code - 1 if code is not None else 0 for code in emotion_codes],
      dtype=torch.long,
      device=x.device,
    )
    y_hat = self(x)
    assert y_hat.shape == (x.shape[0], self.num_classes)
    loss = self.loss_fn(y_hat, y)

    # Update accuracy
    self.train_accuracy.update(y_hat, y)

    self.log(
      "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
    )
    return loss

  def validation_step(self, batch, batch_idx):
    x = batch["eeg"]
    emotion_codes = batch["info"]["emotion"]  # List of emotion codes (1-9 or None)
    # Convert to class indices (0-8) for CrossEntropyLoss. Subtract 1 to map 1-9 to 0-8
    y = torch.tensor(
      [code - 1 if code is not None else 0 for code in emotion_codes],
      dtype=torch.long,
      device=x.device,
    )
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)

    # Update accuracy
    self.val_accuracy.update(y_hat, y)

    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

  def test_step(self, batch, batch_idx):
    x = batch["eeg"]
    emotion_codes = batch["info"]["emotion"]  # List of emotion codes (1-9 or None)
    # Convert to class indices (0-8) for CrossEntropyLoss. Subtract 1 to map 1-9 to 0-8
    y = torch.tensor(
      [code - 1 if code is not None else 0 for code in emotion_codes],
      dtype=torch.long,
      device=x.device,
    )
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)

    # Update accuracy
    self.test_accuracy.update(y_hat, y)

    self.log(
      "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
    )

  def on_train_epoch_end(self):
    """Log and reset training accuracy at epoch end."""
    accuracy = self.train_accuracy.compute()
    self.log("train_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
    self.train_accuracy.reset()

  def on_validation_epoch_end(self):
    """Log and reset validation accuracy at epoch end."""
    accuracy = self.val_accuracy.compute()
    self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
    self.val_accuracy.reset()

  def on_test_epoch_end(self):
    """Log and reset test accuracy at epoch end."""
    accuracy = self.test_accuracy.compute()
    self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
    self.test_accuracy.reset()


class EegptWithLinearEmotionClassifier(EegptEmotionClassifier):
  """
  Emotion classifier variant that uses EegptWithLinear (with ResidualLinear layer).
  Inherits from EegptEmotionClassifier and overrides only the __init__ method.
  """

  def __init__(self, config: EegptConfig, num_classes=9, hidden_dim=256):
    LightningModule.__init__(self)
    self.config = config
    if isinstance(self.config.lr_config, float):
      # to access by LearningRateFinder
      self.learning_rate = self.config.lr_config
    self.save_hyperparameters()
    self.model = EegptWithLinearClassifier.load_from_checkpoint(
      self.config.chpt_path,
      use_chan_conv=self.config.use_chan_conv,
    )
    self.loss_fn = CrossEntropyLoss()

    # Set trainable parameters based on config
    self._setup_trainable_parameters()

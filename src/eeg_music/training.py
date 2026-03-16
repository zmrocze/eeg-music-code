from fractions import Fraction
from pathlib import Path

from lightning import Callback, LightningModule, Trainer
from eeg_music.data import EEGMusicDataset, RobustNormalizedDataset
from eeg_music.dataloader import (
  create_collate_fn,
  load_and_create_dataloaders,
  SubjectWiseSplit,
  TrialWiseSplit,
)
import torch
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity

# import lightning as pl
# import lightning
from lightning.pytorch.loggers import WandbLogger

# .pytorch.loggers.wandb
import wandb
from dataclasses import dataclass, asdict, field
from typing import List, Literal, Optional, Tuple, Union
import random
from lightning.pytorch.callbacks import (
  LearningRateFinder,
  LearningRateMonitor,
  ModelCheckpoint,
  OnExceptionCheckpoint,
  RichProgressBar,
)
from eeg_music.eegpt import (
  EegptLightning,
  EegptEmotionClassifier,
  EegptWithLinearEmotionClassifier,
  EegptConfig,
  LRCosine,
  EEG_WIDTH,
  USING_CHANNELS,
)
from eeg_music.eegnet import NoteOnsetModelConfig, EEGNetConfig, EEGNetLightning
from eeg_music.freeze_utils import freeze_all_except_head_and_adapters


@dataclass
class TrainingConfig:
  eegpt_chpt_path: Path = Path(
    "./model_checkpoints/25866970/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
  )
  data_path: Path = Path("./datasets/bcmi_preprocessed/bcmi_combined_prepared_mel_28ch")
  data_loader_num_workers: int = 4
  prefetch_factor: int = 2
  batch_size: int = 8
  num_epochs: int = 100
  save_model_per_epochs: int = 5

  val_every_n_epoch: int = 1
  ds_p_train = 0.85
  ds_p_val = 0.0
  ds_split_seed = 42
  ds_use_test_for_val = True
  ds_train_repeated_mul = 1
  ds_val_repeated_mul = 1
  ds_test_repeated_mul = 10
  ds_split_type: SubjectWiseSplit | TrialWiseSplit = field(
    default_factory=SubjectWiseSplit
  )

  # ckpt_load_path: Optional[str] = None  # 'best', 'last', <path]>

  # Wandb checkpoint loading (e.g., 'user/project/model-id:version')
  wandb_checkpoint: Optional[str] = None

  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "neural-music-decoding"
  run_name: str = "eegpt-2layer-mel"
  run_extra_name: str = "lr_find"
  randint: int = random.randint(0, 1000)
  save_path: str = f"{run_name}-ckpt"

  lr_config: Union[float, LRCosine] = 1e-4

  use_learning_rate_finder: bool = False

  trainable: Optional[List[str]] = field(default_factory=lambda: ["linear", "head"])
  requiring_grad: Optional[List[str]] = None

  # use_chan_conv: bool = True
  use_chan_conv: bool = False

  # AUROC callback settings
  auroc_every_n_epochs: int = 2
  auroc_similarity_metric: list[Literal["cosine", "structural_similarity"]] = field(
    default_factory=lambda: ["cosine", "structural_similarity"]
  )
  auroc_prediction_batch_size: int = 128

  # Dataloader settings
  include_info: bool = (
    False  # If True, dataloaders include metadata (e.g., emotion labels)
  )

  # Emotion classifier model selection
  emotion_classifier_model: Literal["base", "linear"] = (
    "base"  # "base" for EegptEmotionClassifier, "linear" for EegptWithLinearEmotionClassifier
  )


config = TrainingConfig()


@dataclass
class NoteOnsetsTrainingConfig:
  """Configuration for note onsets detection training."""

  # Model config
  model_config: NoteOnsetModelConfig = field(
    default_factory=lambda: NoteOnsetModelConfig(
      model_config=EEGNetConfig(),
      chunk_width=128,  # 256Hz * 1/2s
      num_channels=len(USING_CHANNELS),  # 28 channels
      eeg_sample_rate=256,
      window_start=32,
      window_end=32 + 64,
      lr_config=1e-4,
      pos_weight=None,  # Can be tuned for class imbalance
    )
  )

  # Checkpoint path (if available, otherwise train from scratch)
  checkpoint_path: Optional[Path] = None

  # Wandb checkpoint loading (e.g., 'user/project/model-id:version')
  wandb_checkpoint: Optional[str] = None

  # Data settings
  data_path: Path = Path("./datasets/bcmi_preprocessed/bcmi_combined_noteonsets_28ch")
  data_loader_num_workers: int = 4
  prefetch_factor: int = 2
  pin_memory: bool = True
  batch_size: int = 32

  # Training settings
  num_epochs: int = 100
  save_model_per_epochs: int = 5
  val_every_n_epoch: int = 1

  # Dataset split settings
  ds_p_train: float = 0.85
  ds_p_val: float = 0.0
  ds_split_seed: int = 42
  ds_use_test_for_val: bool = True
  ds_train_repeated_mul: int = 1
  ds_val_repeated_mul: int = 1
  ds_test_repeated_mul: int = 10
  ds_chunk_width: Fraction = Fraction(1, 2)
  ds_split_type: SubjectWiseSplit | TrialWiseSplit = field(
    default_factory=SubjectWiseSplit
  )

  # Wandb logging
  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "neural-noteonsets-decoding"
  run_name: str = "eegnet-onset-detection"
  run_extra_name: str = "0"
  randint: int = random.randint(0, 1000)
  save_path: str = f"{run_name}-ckpt"

  # Learning rate finder
  use_learning_rate_finder: bool = False

  # Dataloader settings
  include_info: bool = False


def count_n_params(model):
  """Counts the number of trainable parameters in a model."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_spectrograms(pl_module, y_hat, y, batch_idx, stage: str, n_samples=4):
  """Log a batch of predicted and ground truth spectrograms to wandb."""
  import matplotlib.pyplot as plt

  y_hat = y_hat.detach().cpu()[:n_samples].squeeze(1).numpy()
  y = y.detach().cpu()[:n_samples].squeeze(1).numpy()

  images = []
  for i, (pred_spec, true_spec) in enumerate(zip(y_hat, y)):
    vmin, vmax = true_spec.min(), true_spec.max()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(
      pred_spec, aspect="auto", origin="lower", cmap="inferno", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("Predicted")
    axes[1].imshow(
      true_spec, aspect="auto", origin="lower", cmap="inferno", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("True")
    fig.tight_layout()
    images.append(wandb.Image(fig, caption=f"Sample {i}"))
    plt.close(fig)

  pl_module.logger.experiment.log({f"{stage}/spectrograms": images})


class SpectrogramLoggingCallback(Callback):
  def __init__(self):
    super().__init__()
    self.val_log_batch_idx = 0
    self.test_log_batch_idx = 0

  def on_validation_epoch_start(self, trainer, pl_module):
    """Choose a random batch to log for this validation epoch."""
    if trainer.val_dataloaders:
      num_batches = len(trainer.val_dataloaders)
      if num_batches > 0:
        self.val_log_batch_idx = random.randint(0, num_batches - 1)

  def on_test_epoch_start(self, trainer, pl_module):
    """Choose a random batch to log for this test epoch."""
    if trainer.test_dataloaders:
      # Assuming single test dataloader
      num_batches = len(trainer.test_dataloaders)
      if num_batches > 0:
        self.test_log_batch_idx = random.randint(0, num_batches - 1)

  def on_validation_batch_end(
    self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
  ):
    """Log spectrograms at the end of each validation batch."""
    if batch_idx == self.val_log_batch_idx:
      x = batch["eeg"]
      y = batch["music"]
      y_hat = pl_module(x)
      log_spectrograms(pl_module, y_hat, y, batch_idx, stage="val")

  def on_test_batch_end(
    self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
  ):
    """Log spectrograms at the end of each test batch."""
    if batch_idx == self.test_log_batch_idx:
      x = batch["eeg"]
      y = batch["music"]
      y_hat = pl_module(x)
      log_spectrograms(pl_module, y_hat, y, batch_idx, stage="test")


class AUROCCallback(Callback):
  """Calculate AUROC-like retrieval metric on validation set.

  For each EEG sample, compares predicted mel spectrogram to all validation
  mel spectrograms and computes the rank of the correct match.
  """

  def __init__(
    self,
    auroc_every_n_epochs: int = 5,
    similarity_metric: str = "cosine",
    prediction_batch_size: int = 128,
  ):
    """
    Args:
      auroc_every_n_epochs: Calculate AUROC score every N epochs
      similarity_metric: Either 'cosine' or 'structural_similarity'
      prediction_batch_size: Batch size for generating predictions to control GPU memory
    """
    super().__init__()
    self.auroc_every_n_epochs = auroc_every_n_epochs
    self.similarity_metric = similarity_metric
    self.prediction_batch_size = prediction_batch_size
    self.auroc_history = []  # Store last 10 scores for moving average
    # Create suffix for metric names to distinguish different similarity metrics
    self.metric_suffix = (
      "" if similarity_metric == "cosine" else f"_{similarity_metric}"
    )

  def on_validation_epoch_end(self, trainer, pl_module):
    """Calculate AUROC score at the end of validation epoch."""
    # Only calculate every N epochs
    if (trainer.current_epoch + 1) % self.auroc_every_n_epochs != 0:
      return

    # Check if validation dataloader exists
    if not trainer.val_dataloaders:
      return

    # Collect all validation data
    all_x = []
    all_y = []

    pl_module.eval()
    with torch.no_grad():
      for batch in trainer.val_dataloaders:
        x = batch["eeg"].to(pl_module.device)
        y = batch["music"].to(pl_module.device)
        all_x.append(x)
        all_y.append(y)

    # Concatenate all batches
    all_x = torch.cat(all_x, dim=0)  # Shape: (N, channels, time)
    all_y = torch.cat(all_y, dim=0)  # Shape: (N, freq, time)

    # Generate predictions in batches to control GPU memory
    all_y_hat = []
    with torch.no_grad():
      for i in range(0, all_x.shape[0], self.prediction_batch_size):
        batch_x = all_x[i : i + self.prediction_batch_size]
        batch_y_hat = pl_module(batch_x)
        all_y_hat.append(batch_y_hat)
    all_y_hat = torch.cat(all_y_hat, dim=0)  # Shape: (N, freq, time)

    n_samples = all_y.shape[0]
    ranks = []

    # For each sample, compute similarity to all targets
    for i in range(n_samples):
      y_hat_i = all_y_hat[i]  # Shape: (freq, time)
      similarities = []

      # Compare to all ground truth spectrograms
      for j in range(n_samples):
        y_j = all_y[j]  # Shape: (freq, time)
        sim = self._compute_similarity(y_hat_i, y_j)
        similarities.append(sim)

      # Sort similarities in descending order (higher similarity = better match)
      similarities = torch.tensor(similarities)
      sorted_indices = torch.argsort(similarities, descending=True)

      # Find rank of correct match (where sorted_indices == i)
      rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
      ranks.append(rank)

    # Calculate AUROC-like score
    # If correct match is rank 0 (best), score should be 1.0
    # If correct match is rank (n_samples-1) (worst), score should be 0.0
    auroc_scores = [1.0 - (rank / (n_samples - 1)) for rank in ranks]
    mean_auroc = sum(auroc_scores) / len(auroc_scores)

    # Update history for moving average
    self.auroc_history.append(mean_auroc)
    if len(self.auroc_history) > 10:
      self.auroc_history.pop(0)

    moving_avg = sum(self.auroc_history) / len(self.auroc_history)

    # Log metrics
    pl_module.log(
      f"auroc_score{self.metric_suffix}",
      mean_auroc,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_ma10{self.metric_suffix}",
      moving_avg,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    # Log distribution of ranks for debugging
    median_rank = sorted(ranks)[len(ranks) // 2]
    pl_module.log(
      f"auroc_median_rank{self.metric_suffix}", median_rank, on_epoch=True, logger=True
    )

    # Log top-k accuracy metrics
    pl_module.log(
      f"auroc_top1_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r == 0) / len(ranks),
      on_epoch=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_top10_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r < 10) / len(ranks),
      on_epoch=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_top25_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r < 25) / len(ranks),
      on_epoch=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_top100_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r < 100) / len(ranks),
      on_epoch=True,
      logger=True,
    )

  def _compute_similarity(self, pred, target):
    """Compute similarity between two spectrograms.

    Args:
      pred: Predicted spectrogram (freq, time)
      target: Target spectrogram (freq, time)

    Returns:
      Similarity score (higher = more similar)
    """
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    if self.similarity_metric == "cosine":
      return cosine_similarity(pred_np.reshape(1, -1), target_np.reshape(1, -1))[0, 0]

    elif self.similarity_metric == "structural_similarity":
      win_size = min(7, *pred_np.shape) | 1
      return structural_similarity(
        pred_np,
        target_np,
        win_size=win_size,
        data_range=max(
          pred_np.max() - pred_np.min(), target_np.max() - target_np.min()
        ),
      )

    else:
      raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")


def log_hyperparameters(model, dataloaders, config, wandb_logger):
  params_to_log = {}

  # Parameter counts
  params_to_log["trainable_params_total"] = count_n_params(model)

  # Determine model structure: EegptLightning uses EegptWithLinear wrapper, EegptEmotionClassifier doesn't
  if hasattr(model.model, "linear"):
    # EegptLightning: model.model = EegptWithLinear, model.model.model = EEGPTClassifier
    eegpt_classifier = model.model.model
    params_to_log["trainable_params_residual_linear"] = count_n_params(
      model.model.linear
    )
    params_to_log["residual_linear_in_dim"] = model.model.linear.linear1.in_features
    params_to_log["residual_linear_out_dim"] = model.model.linear.linear2.out_features
  else:
    # EegptEmotionClassifier: model.model = EEGPTClassifier directly
    eegpt_classifier = model.model

  # Common EEGPTClassifier params
  if hasattr(eegpt_classifier, "chan_conv"):
    params_to_log["trainable_params_chan_conv"] = count_n_params(
      eegpt_classifier.chan_conv
    )
  params_to_log["trainable_params_head"] = count_n_params(eegpt_classifier.head)
  params_to_log["eegpt_classifier_use_chan_conv"] = eegpt_classifier.use_chan_conv
  if hasattr(eegpt_classifier, "num_classes"):
    params_to_log["num_classes"] = eegpt_classifier.num_classes

  # Target Encoder params
  target_encoder = eegpt_classifier.target_encoder
  params_to_log["target_encoder_img_size"] = str(target_encoder.patch_embed.img_size)
  params_to_log["target_encoder_patch_size"] = target_encoder.patch_embed.patch_size
  params_to_log["target_encoder_embed_dim"] = target_encoder.embed_dim
  params_to_log["target_encoder_depth"] = len(target_encoder.blocks)
  params_to_log["target_encoder_num_heads"] = target_encoder.num_heads
  params_to_log["target_encoder_patch_stride"] = target_encoder.patch_embed.patch_stride

  # Predictor params
  if eegpt_classifier.use_predictor:
    predictor = eegpt_classifier.predictor
    params_to_log["predictor_embed_dim"] = predictor.predictor_embed.in_features
    params_to_log["predictor_depth"] = len(predictor.predictor_blocks)
    params_to_log["predictor_num_heads"] = predictor.predictor_blocks[0].attn.num_heads

  # EEG data params
  params_to_log["eeg_width"] = EEG_WIDTH
  params_to_log["using_channels"] = USING_CHANNELS

  # Dataloader params
  params_to_log["dataloader_train_size"] = len(dataloaders["train"])
  params_to_log["dataloader_val_size"] = len(dataloaders["val"])
  params_to_log["dataloader_test_size"] = len(dataloaders["test"])
  params_to_log["batch_size"] = config.batch_size
  params_to_log["num_workers"] = config.data_loader_num_workers

  wandb_logger.log_hyperparams(params_to_log)


class MainTraining:
  def __init__(self, config):
    self.config = config
    self.dataloaders: dict
    self.model: LightningModule
    self.wandb_logger: WandbLogger
    self.callbacks: list[Callback]
    self.trainer: Trainer
    assert (
      isinstance(self.config.lr_config, float)
      if self.config.use_learning_rate_finder
      else True
    )

  def create_dataloaders(self):
    self.dataloaders = load_and_create_dataloaders(
      self.config.data_path, self.config, split_type=self.config.ds_split_type
    )

  def create_model(self):
    eegpt_config = EegptConfig(
      chpt_path=self.config.eegpt_chpt_path,
      lr_config=self.config.lr_config,
      use_chan_conv=self.config.use_chan_conv,
      trainable=self.config.trainable,
      requiring_grad=self.config.requiring_grad,
    )
    self.model = EegptLightning(eegpt_config)
    freeze_all_except_head_and_adapters(self.model, verbose=True)

  def initialize_logger(self):
    wandb.finish()
    self.wandb_logger = WandbLogger(
      project=self.config.project_name,
      name=f"{self.config.run_name}-{self.config.run_extra_name}-{self.config.randint}",
      log_model=self.config.wandb_log_model,
      config=asdict(self.config),
    )
    self.wandb_logger.watch(self.model, log="all")

  def create_callbacks(self):
    save_on_exc = OnExceptionCheckpoint(
      f"{self.config.save_path}/exc_save",
    )

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

    # Create AUROC callbacks for each similarity metric
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

  def create_trainer(self):
    self.trainer = Trainer(
      callbacks=self.callbacks,
      logger=self.wandb_logger,
      check_val_every_n_epoch=self.config.val_every_n_epoch,
      max_epochs=self.config.num_epochs,
      accelerator="auto",
      log_every_n_steps=1,
      # precision="16-mixed"
      # precision="32-true",
    )

  def log_hyperparameters(self):
    """Log hyperparameters to wandb. Can be overridden by subclasses."""
    log_hyperparameters(self.model, self.dataloaders, self.config, self.wandb_logger)

  def trainer_fit(self):
    print(f"Model trainable params: {count_n_params(self.model)}")
    print(
      "Note that val and test dataloaders augmentation/randomness in the form of choosing the i.e. 4s fragment."
    )

    self.trainer.fit(
      self.model,
      train_dataloaders=self.dataloaders["train"],
      val_dataloaders=self.dataloaders["val"],
      # ckpt_path=config.ckpt_load_path,
      ckpt_path=None,
    )

  def trainer_test(self):
    self.trainer.test(
      self.model,
      dataloaders=self.dataloaders["test"],
    )
    for ckpt_cb in (cb for cb in self.callbacks if isinstance(cb, ModelCheckpoint)):
      best_path = ckpt_cb.best_model_path
      if best_path and Path(best_path).exists():
        print(f"Testing best checkpoint ({ckpt_cb.monitor}): {best_path}")
        self.trainer.test(
          self.model,
          dataloaders=self.dataloaders["test"],
          ckpt_path=best_path,
        )
      elif best_path:
        print(f"Best checkpoint ({ckpt_cb.monitor}) not found, skipping: {best_path}")

  def run(self):
    self.create_dataloaders()
    self.create_model()
    self.initialize_logger()
    self.create_callbacks()
    self.create_trainer()
    self.log_hyperparameters()
    self.trainer_fit()
    self.trainer_test()
    return self.model, self.trainer, self.dataloaders


class EmotionClassifierTraining(MainTraining):
  """Training class for emotion classification using EegptEmotionClassifier."""

  def __init__(self, config):
    super().__init__(config)
    # Ensure include_info is True for emotion classification
    self.config.include_info = True

  def create_model(self):
    eegpt_config = EegptConfig(
      chpt_path=self.config.eegpt_chpt_path,
      lr_config=self.config.lr_config,
      use_chan_conv=self.config.use_chan_conv,
      trainable=self.config.trainable,
      requiring_grad=self.config.requiring_grad,
    )

    # Select model based on config
    if self.config.emotion_classifier_model == "linear":
      self.model = EegptWithLinearEmotionClassifier(eegpt_config, num_classes=9)
    else:
      self.model = EegptEmotionClassifier(eegpt_config, num_classes=9)
    # freeze_all_except_head_and_adapters(self.model, verbose=True)

  def create_callbacks(self):
    """Create callbacks without AUROC and spectrogram logging (not applicable for classification)."""
    save_on_exc = OnExceptionCheckpoint(
      f"{self.config.save_path}/exc_save",
    )

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

    self.callbacks = [
      ckpt_callback,
      RichProgressBar(),
      save_on_exc,
      LearningRateMonitor(logging_interval="step"),
    ] + optional_lr_finder


class NoteOnsetsTraining(MainTraining):
  """Training class for note onsets detection using EEGNet."""

  def __init__(self, config):
    super().__init__(config)

  def create_dataloaders(self):
    # Enable include_info when using subject-specific preprocessing (need dataset+subject)
    include_info = (
      self.config.include_info or self.config.model_config.use_subject_specific
    )
    self.dataloaders = load_and_create_dataloaders(
      self.config.data_path,
      self.config,
      collate_fn=create_collate_fn(
        include_info=include_info, music_batch_fn=lambda x: x
      ),
      include_mapper=self.config.model_config.use_subject_specific,
      split_type=self.config.ds_split_type,
    )
    if self.config.ds_split_type == TrialWiseSplit:
      assert self.dataloaders["num_skipped_trials"] == 0

  def create_model_aux(self, lightning_class):
    """Create model using the specified Lightning class, loading from checkpoint if available.

    Args:
        lightning_class: The Lightning module class to instantiate (e.g., EEGNetLightning)
    """
    # Get mapper from dataloaders if subject-specific preprocessing is enabled
    mapper = (
      self.dataloaders.get("mapper")
      if self.config.model_config.use_subject_specific
      else None
    )

    if self.config.wandb_checkpoint is not None:
      # Load from wandb checkpoint (takes priority)
      print(f"Loading model from wandb checkpoint: {self.config.wandb_checkpoint}")
      run = wandb.init()
      artifact = run.use_artifact(self.config.wandb_checkpoint, type="model")
      artifact_dir = artifact.download()
      self.model = lightning_class.load_from_checkpoint(
        artifact_dir + "/model.ckpt",
        config=self.config.model_config,
        subject_mapper=mapper,
      )
    elif (
      self.config.checkpoint_path is not None and self.config.checkpoint_path.exists()
    ):
      # Load from local checkpoint
      print(f"Loading model from checkpoint: {self.config.checkpoint_path}")
      self.model = lightning_class.load_from_checkpoint(
        self.config.checkpoint_path,
        config=self.config.model_config,
        subject_mapper=mapper,
      )
    else:
      # Create fresh model
      if self.config.checkpoint_path is not None:
        print(f"Checkpoint path specified but not found: {self.config.checkpoint_path}")
      print(f"Creating fresh {lightning_class.__name__} model")
      self.model = lightning_class(self.config.model_config, subject_mapper=mapper)

  def create_model(self):
    """Create EEGNet model, loading from checkpoint if available."""
    self.create_model_aux(EEGNetLightning)

  def log_hyperparameters(self):
    """Log EEGNet-specific hyperparameters to wandb."""
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
      # Window parameters
      "window_start": self.config.model_config.window_start,
      "window_end": self.config.model_config.window_end,
      # Training
      "lr_config": str(self.config.model_config.lr_config),
      "pos_weight": self.config.model_config.pos_weight,
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

  def create_callbacks(self):
    """Create callbacks for binary onset detection training."""
    save_on_exc = OnExceptionCheckpoint(
      f"{self.config.save_path}/exc_save",
    )

    ckpt_callback_f1 = ModelCheckpoint(
      every_n_epochs=self.config.save_model_per_epochs,
      dirpath=self.config.save_path,
      save_top_k=1,
      monitor="val_f1_score",
      mode="max",
      filename="best-f1-{epoch:02d}-{val_f1_score:.3f}",
    )

    ckpt_callback_loss = ModelCheckpoint(
      every_n_epochs=self.config.save_model_per_epochs,
      dirpath=self.config.save_path,
      save_top_k=1,
      monitor="val_loss",
      mode="min",
      filename="best-loss-{epoch:02d}-{val_loss:.3f}",
      save_last=True,
    )

    optional_lr_finder = (
      [LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100)]
      if self.config.use_learning_rate_finder
      else []
    )

    self.callbacks = [
      ckpt_callback_f1,
      ckpt_callback_loss,
      RichProgressBar(),
      save_on_exc,
      LearningRateMonitor(logging_interval="step"),
    ] + optional_lr_finder


class OverfitNoteOnsetsTraining(NoteOnsetsTraining):
  """Custom training class that overrides dataloader creation."""

  def create_dataloaders(self):
    # Custom dataloader implementation
    # Example: you can modify include_info, collate_fn, or other parameters
    def after_loaded_ds(data: EEGMusicDataset, trial_length_secs) -> EEGMusicDataset:
      # mapped = MappedDataset(data, rereference_trial)
      mapped = RobustNormalizedDataset(data)
      return mapped

    include_info = (
      self.config.include_info or self.config.model_config.use_subject_specific
    )
    self.dataloaders = load_and_create_dataloaders(
      self.config.data_path,
      self.config,
      collate_fn=create_collate_fn(
        include_info=include_info,
        music_batch_fn=lambda x: x,
        eeg_batch_fn=lambda x: torch.stack(
          [torch.from_numpy(a.get_array().data) for a in x]  # pyright: ignore[reportAttributeAccessIssue]
        ),
      ),
      include_mapper=self.config.model_config.use_subject_specific,
      split_type=self.config.ds_split_type,
      after_loaded_ds=after_loaded_ds,
    )
    if self.config.ds_split_type == TrialWiseSplit:
      assert self.dataloaders["num_skipped_trials"] == 0


def main(config=config) -> Tuple[LightningModule, Trainer, dict]:
  training = MainTraining(config)
  return training.run()


if __name__ == "__main__":
  main()

"""Tests for emotion classification using EEGNet models."""

import torch
import pytest
from pathlib import Path
from fractions import Fraction

from eeg_music.emotion_eegnet import (
  MultiClassAccuracyCalc,
  EmotionEEGNetModelConfig,
  EmotionEEGNetLightning,
  EmotionEEGNetTrainingConfig,
  EmotionEEGNetTraining,
)
from eeg_music.eegnet import EEGNetConfig, FBCNetConfig, TSCeptionConfig, ATCNetConfig


def test_multiclass_accuracy_calc_basic():
  """Test basic functionality of MultiClassAccuracyCalc."""
  calc = MultiClassAccuracyCalc(num_classes=9)

  # Perfect predictions
  logits = torch.tensor(
    [
      [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
  )
  targets = torch.tensor([0, 1, 2])

  calc.update(logits, targets)
  metrics = calc.compute()

  assert metrics["accuracy"] == 1.0
  assert calc.correct == 3
  assert calc.total == 3


def test_multiclass_accuracy_calc_imperfect():
  """Test MultiClassAccuracyCalc with some errors."""
  calc = MultiClassAccuracyCalc(num_classes=9)

  # 2 correct out of 4
  logits = torch.tensor(
    [
      [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # correct
      [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # wrong (should be 0)
      [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # correct
      [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # wrong (should be 2)
    ]
  )
  targets = torch.tensor([0, 0, 2, 2])

  calc.update(logits, targets)
  metrics = calc.compute()

  assert metrics["accuracy"] == 0.5
  assert calc.correct == 2
  assert calc.total == 4


def test_multiclass_accuracy_calc_reset():
  """Test reset functionality."""
  calc = MultiClassAccuracyCalc(num_classes=9)

  logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
  targets = torch.tensor([0])

  calc.update(logits, targets)
  assert calc.correct == 1
  assert calc.total == 1

  calc.reset()
  assert calc.correct == 0
  assert calc.total == 0


def test_emotion_eegnet_model_config_defaults():
  """Test EmotionEEGNetModelConfig default values."""
  config = EmotionEEGNetModelConfig()

  assert isinstance(config.model_config, EEGNetConfig)
  assert config.chunk_width == 1024
  assert config.num_channels == 28
  assert config.eeg_sample_rate == 256
  assert config.num_classes == 9
  assert config.lr_config == 1e-4
  assert config.use_subject_specific is False


def test_emotion_eegnet_model_config_custom():
  """Test EmotionEEGNetModelConfig with custom values."""
  config = EmotionEEGNetModelConfig(
    model_config=FBCNetConfig(num_bands=2),
    chunk_width=512,
    num_channels=32,
    num_classes=5,
  )

  assert isinstance(config.model_config, FBCNetConfig)
  assert config.chunk_width == 512
  assert config.num_channels == 32
  assert config.num_classes == 5


def test_emotion_eegnet_lightning_initialization():
  """Test EmotionEEGNetLightning initialization."""
  config = EmotionEEGNetModelConfig(
    chunk_width=256,
    num_channels=28,
    num_classes=9,
  )
  model = EmotionEEGNetLightning(config)

  assert model.config.num_classes == 9
  assert isinstance(model.loss_fn, torch.nn.CrossEntropyLoss)
  assert isinstance(model.train_metrics, MultiClassAccuracyCalc)
  assert isinstance(model.val_metrics, MultiClassAccuracyCalc)
  assert isinstance(model.test_metrics, MultiClassAccuracyCalc)


def test_emotion_eegnet_lightning_forward():
  """Test forward pass through EmotionEEGNetLightning."""
  config = EmotionEEGNetModelConfig(
    chunk_width=256,
    num_channels=28,
    num_classes=9,
  )
  model = EmotionEEGNetLightning(config)

  batch_size = 4
  eeg = torch.randn(batch_size, 28, 256)

  output = model(eeg)

  assert output.shape == (batch_size, 9)


def test_emotion_eegnet_lightning_forward_different_models():
  """Test forward pass with different model architectures."""
  configs = [
    EmotionEEGNetModelConfig(
      model_config=EEGNetConfig(), chunk_width=256, num_classes=9
    ),
    EmotionEEGNetModelConfig(
      model_config=FBCNetConfig(num_bands=1), chunk_width=256, num_classes=9
    ),
    EmotionEEGNetModelConfig(
      model_config=TSCeptionConfig(), chunk_width=256, num_classes=9
    ),
    EmotionEEGNetModelConfig(
      model_config=ATCNetConfig(), chunk_width=256, num_classes=9
    ),
  ]

  batch_size = 2
  eeg = torch.randn(batch_size, 28, 256)

  for config in configs:
    model = EmotionEEGNetLightning(config)
    output = model(eeg)
    assert output.shape == (batch_size, 9), (
      f"Failed for {type(config.model_config).__name__}"
    )


def test_emotion_eegnet_lightning_training_step():
  """Test training step with mock batch."""
  config = EmotionEEGNetModelConfig(
    chunk_width=256,
    num_channels=28,
    num_classes=9,
  )
  model = EmotionEEGNetLightning(config)

  # Mock batch
  batch = {
    "eeg": torch.randn(4, 28, 256),
    "music": [
      "hvla3.wav",
      "lvha5.wav",
      "lvla1.wav",
      "hvha9.wav",
    ],  # Not used in emotion classification
    "info": {
      "dataset": [
        "bcmi-calibration",
        "bcmi-calibration",
        "bcmi-calibration",
        "bcmi-calibration",
      ],
      "subject": ["S01", "S01", "S01", "S01"],
      "session": ["session_1", "session_1", "session_1", "session_1"],
      "run": ["run_1", "run_1", "run_1", "run_1"],
      "trial_id": ["trial_1", "trial_2", "trial_3", "trial_4"],
      "music_filename": ["hvla3.wav", "lvha5.wav", "lvla1.wav", "hvha9.wav"],
      "batch_size": 4,
      "emotion": [3, 7, 1, 9],
    },
  }

  # Training step should not raise
  loss = model.training_step(batch, 0)

  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0  # Scalar
  assert not torch.isnan(loss)


def test_emotion_eegnet_training_config_defaults():
  """Test EmotionEEGNetTrainingConfig default values."""
  config = EmotionEEGNetTrainingConfig()

  assert isinstance(config.model_config, EmotionEEGNetModelConfig)
  assert config.data_path == Path(
    "./datasets/bcmi_preprocessed/bcmi_combined_prepared_mel_28ch"
  )
  assert config.batch_size == 32
  assert config.num_epochs == 100
  assert config.ds_chunk_width == Fraction(4, 1)
  assert config.include_info is True
  assert config.project_name == "emotion-classification-eegnet"


def test_multiclass_accuracy_calc_multiple_batches():
  """Test MultiClassAccuracyCalc with multiple update calls."""
  calc = MultiClassAccuracyCalc(num_classes=9)

  # First batch: 2/3 correct
  logits1 = torch.tensor(
    [
      [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
  )
  targets1 = torch.tensor([0, 0, 2])  # Second one is wrong
  calc.update(logits1, targets1)

  # Second batch: 3/3 correct
  logits2 = torch.tensor(
    [
      [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    ]
  )
  targets2 = torch.tensor([3, 4, 5])
  calc.update(logits2, targets2)

  metrics = calc.compute()
  # Total: 5/6 correct
  assert metrics["accuracy"] == pytest.approx(5 / 6)
  assert calc.correct == 5
  assert calc.total == 6


def test_emotion_eegnet_model_num_classes_propagation():
  """Test that num_classes propagates correctly through the model stack."""
  for num_classes in [3, 5, 9, 12]:
    config = EmotionEEGNetModelConfig(
      chunk_width=256,
      num_channels=28,
      num_classes=num_classes,
    )
    model = EmotionEEGNetLightning(config)

    batch_size = 2
    eeg = torch.randn(batch_size, 28, 256)
    output = model(eeg)

    assert output.shape == (batch_size, num_classes)
    assert model.model.num_classes == num_classes


def test_emotion_eegnet_training_log_hyperparameters():
  """Test that log_hyperparameters works correctly."""
  from unittest.mock import MagicMock

  config = EmotionEEGNetTrainingConfig(
    num_epochs=1,
    batch_size=2,
  )
  training = EmotionEEGNetTraining(config)

  # Mock the necessary components
  training.model = EmotionEEGNetLightning(config.model_config)
  training.dataloaders = {
    "train": MagicMock(__len__=lambda self: 10),
    "val": MagicMock(__len__=lambda self: 5),
    "test": MagicMock(__len__=lambda self: 5),
  }
  training.wandb_logger = MagicMock()

  # This should not raise an error
  training.log_hyperparameters()

  # Verify that log_hyperparams was called
  assert training.wandb_logger.log_hyperparams.called

  # Check that correct parameters were logged
  logged_params = training.wandb_logger.log_hyperparams.call_args[0][0]
  assert "trainable_params_total" in logged_params
  assert "num_classes" in logged_params
  assert logged_params["num_classes"] == 9
  assert "chunk_width" in logged_params
  assert "num_channels" in logged_params
  # Should not have onset-specific params
  assert "window_start" not in logged_params
  assert "window_end" not in logged_params
  assert "pos_weight" not in logged_params


if __name__ == "__main__":
  pytest.main([__file__, "-v"])

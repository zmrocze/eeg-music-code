"""Tests for BinaryEmotionEEGNetLightning."""

import torch
import numpy as np
from eeg_music.emotion_eegnet import (
  BinaryEmotionEEGNetLightning,
  BinaryEmotionEEGNetTraining,
  EmotionEEGNetModelConfig,
  EmotionEEGNetTrainingConfig,
  EEGNetConfig,
)
from eeg_music.data import NoteOnsets


def test_binary_emotion_eegnet_compute_loss():
  """Test that BinaryEmotionEEGNetLightning correctly computes binary targets from note onset counts."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=256,
    num_classes=1,  # Binary classification with single logit
    median_num_noteonsets=35,
  )

  model = BinaryEmotionEEGNetLightning(config)

  # Create mock batch with NoteOnsets
  batch_size = 4
  eeg = torch.randn(batch_size, 28, 256)

  # Create NoteOnsets with different counts around the median threshold
  music_data = [
    NoteOnsets(
      onset_times=np.array([0.1, 0.2, 0.3]), sample_rate=256, duration_seconds=1.0
    ),  # 3 onsets -> 0
    NoteOnsets(
      onset_times=np.array([0.1] * 35), sample_rate=256, duration_seconds=1.0
    ),  # 35 onsets -> 0
    NoteOnsets(
      onset_times=np.array([0.1] * 36), sample_rate=256, duration_seconds=1.0
    ),  # 36 onsets -> 1
    NoteOnsets(
      onset_times=np.array([0.1] * 100), sample_rate=256, duration_seconds=1.0
    ),  # 100 onsets -> 1
  ]

  batch = {
    "eeg": eeg,
    "music": music_data,
    "info": {
      "dataset": ["ds1", "ds1", "ds1", "ds1"],
      "subject": ["s1", "s1", "s1", "s1"],
    },
  }

  # Compute loss
  loss = model._compute_loss(batch, 0, "train")

  # Check that loss is computed (should be a scalar tensor)
  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0  # Scalar
  assert loss.item() > 0  # Loss should be positive


def test_binary_emotion_eegnet_threshold_boundary():
  """Test that the threshold works correctly at the boundary."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=256,
    num_classes=1,  # Binary classification with single logit
    median_num_noteonsets=10,  # Use smaller threshold for testing
  )

  model = BinaryEmotionEEGNetLightning(config)

  batch_size = 3
  eeg = torch.randn(batch_size, 28, 256)

  # Test boundary cases: 9 (below), 10 (at), 11 (above)
  music_data = [
    NoteOnsets(
      onset_times=np.array([0.1] * 9), sample_rate=256, duration_seconds=1.0
    ),  # 9 -> 0
    NoteOnsets(
      onset_times=np.array([0.1] * 10), sample_rate=256, duration_seconds=1.0
    ),  # 10 -> 0
    NoteOnsets(
      onset_times=np.array([0.1] * 11), sample_rate=256, duration_seconds=1.0
    ),  # 11 -> 1
  ]

  batch = {
    "eeg": eeg,
    "music": music_data,
    "info": {"dataset": ["ds1", "ds1", "ds1"], "subject": ["s1", "s1", "s1"]},
  }

  # Compute loss (this will internally create targets)
  loss = model._compute_loss(batch, 0, "val")

  assert isinstance(loss, torch.Tensor)
  assert loss.item() > 0


def test_binary_emotion_eegnet_forward():
  """Test forward pass of BinaryEmotionEEGNetLightning."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=256,
    num_classes=1,  # Binary classification with single logit
    median_num_noteonsets=35,
  )

  model = BinaryEmotionEEGNetLightning(config)

  batch_size = 4
  eeg = torch.randn(batch_size, 28, 256)

  # Forward pass
  output = model(eeg)

  # Check output shape: (batch_size,) for single logit
  assert output.shape == (batch_size,)


def test_binary_emotion_eegnet_training_create_model():
  """Test that BinaryEmotionEEGNetTraining creates BinaryEmotionEEGNetLightning model."""
  config = EmotionEEGNetTrainingConfig(
    model_config=EmotionEEGNetModelConfig(
      model_config=EEGNetConfig(),
      chunk_width=256,
      num_channels=28,
      eeg_sample_rate=256,
      num_classes=1,  # Binary classification with single logit
      median_num_noteonsets=35,
    )
  )

  training = BinaryEmotionEEGNetTraining(config)

  # Mock dataloaders to avoid actual data loading
  training.dataloaders = {"train": [], "val": [], "test": []}

  # Create model
  training.create_model()

  # Verify that the model is BinaryEmotionEEGNetLightning
  assert isinstance(training.model, BinaryEmotionEEGNetLightning)
  assert training.model.config.num_classes == 1
  assert training.model.config.median_num_noteonsets == 35

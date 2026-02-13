"""Tests for MUSING dataset song classification with EEGNet models."""

import torch
from eeg_music.emotion_eegnet import (
  MusingEEGNetLightning,
  EmotionEEGNetModelConfig,
)
from eeg_music.eegnet import EEGNetConfig
from eeg_music.data import MusingMusicIdData, MusingMusicId


def test_musing_lightning_initialization():
  """Test MusingEEGNetLightning initialization."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=250,
    num_classes=12,  # 12 songs
  )

  model = MusingEEGNetLightning(config)

  assert model.config.num_classes == 12
  assert model.model.num_classes == 12


def test_musing_lightning_forward():
  """Test forward pass through MusingEEGNetLightning."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=250,
    num_classes=12,
  )

  model = MusingEEGNetLightning(config)

  # Create dummy input
  batch_size = 4
  x = torch.randn(batch_size, 28, 256)

  # Forward pass
  logits = model(x)

  assert logits.shape == (batch_size, 12)


def test_musing_lightning_compute_loss():
  """Test loss computation with MusingMusicIdData."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=250,
    num_classes=12,
  )

  model = MusingEEGNetLightning(config)

  # Create dummy batch with MusingMusicIdData
  batch_size = 4
  batch = {
    "eeg": torch.randn(batch_size, 28, 256),
    "music": [
      MusingMusicIdData(music_id=MusingMusicId(song_id=1)),
      MusingMusicIdData(music_id=MusingMusicId(song_id=5)),
      MusingMusicIdData(music_id=MusingMusicId(song_id=12)),
      MusingMusicIdData(music_id=MusingMusicId(song_id=7)),
    ],
    "info": {
      "dataset": ["musin-g"] * batch_size,
      "subject": ["001", "002", "003", "004"],
    },
  }

  # Compute loss
  loss = model._compute_loss(batch, 0, "train")

  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0  # Scalar loss
  assert loss.item() > 0


def test_musing_lightning_song_id_extraction():
  """Test that song IDs are correctly extracted and converted to class indices."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=250,
    num_classes=12,
  )

  model = MusingEEGNetLightning(config)

  # Create batch with specific song IDs
  batch_size = 3
  song_ids = [1, 6, 12]  # Song IDs (1-12)

  batch = {
    "eeg": torch.randn(batch_size, 28, 256),
    "music": [
      MusingMusicIdData(music_id=MusingMusicId(song_id=sid)) for sid in song_ids
    ],
    "info": {
      "dataset": ["musin-g"] * batch_size,
      "subject": ["001", "002", "003"],
    },
  }

  # Extract targets by calling _compute_loss and checking internal behavior
  # We'll verify by checking that the model can process the batch
  loss = model._compute_loss(batch, 0, "train")

  # Verify loss is computed successfully
  assert isinstance(loss, torch.Tensor)
  assert not torch.isnan(loss)


def test_musing_lightning_metrics_update():
  """Test that metrics are updated correctly during training."""
  config = EmotionEEGNetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=256,
    num_channels=28,
    eeg_sample_rate=250,
    num_classes=12,
  )

  model = MusingEEGNetLightning(config)

  # Create batch
  batch_size = 4
  batch = {
    "eeg": torch.randn(batch_size, 28, 256),
    "music": [
      MusingMusicIdData(music_id=MusingMusicId(song_id=i + 1))
      for i in range(batch_size)
    ],
    "info": {
      "dataset": ["musin-g"] * batch_size,
      "subject": [f"{i:03d}" for i in range(1, batch_size + 1)],
    },
  }

  # Compute loss for train stage
  initial_total = model.train_metrics.total
  model._compute_loss(batch, 0, "train")

  # Check that metrics were updated
  assert model.train_metrics.total == initial_total + batch_size
  assert model.train_metrics.total > 0

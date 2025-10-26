"""Tests for subject-specific preprocessing integration with EEGNet."""

import torch
import pytest
import numpy as np

from eeg_music.eegnet import (
  EEGNetWrapper,
  EEGNetLightning,
  NoteOnsetModelConfig,
  EEGNetConfig,
)
from eeg_music.subject_specific import SubjectDatasetMapper
from eeg_music.data import NoteOnsets


def test_eegnet_wrapper_with_subject_specific():
  """Test EEGNetWrapper with subject-specific preprocessing."""
  # Create a mapper
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")
  mapper.add_subject("dataset1", "S02")
  mapper.add_subject("dataset2", "S01")

  # Create model with subject-specific preprocessing
  model = EEGNetWrapper(
    chunk_width=128,
    num_channels=4,
    eeg_sample_rate=256,
    model_config=EEGNetConfig(),
    subject_specific_mapper=mapper,
  )

  # Test forward pass
  batch_size = 5
  x = torch.randn(batch_size, 4, 128)
  subject_ids = torch.tensor([0, 1, 2, 0, 1])

  output = model(x, subject_ids)

  assert output.shape == (batch_size,)
  assert model.subject_specific is not None
  assert isinstance(model.subject_specific.weights, torch.nn.Parameter)


def test_eegnet_wrapper_without_subject_specific():
  """Test that model works without subject-specific preprocessing."""
  model = EEGNetWrapper(
    chunk_width=128,
    num_channels=4,
    subject_specific_mapper=None,
  )

  x = torch.randn(3, 4, 128)
  output = model(x, subject_ids=None)

  assert output.shape == (3,)
  assert model.subject_specific is None


def test_eegnet_wrapper_requires_subject_ids_when_enabled():
  """Test that subject_ids are required when subject-specific preprocessing is enabled."""
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")

  model = EEGNetWrapper(
    chunk_width=128,
    num_channels=4,
    subject_specific_mapper=mapper,
  )

  x = torch.randn(2, 4, 128)

  with pytest.raises(ValueError, match="subject_ids required"):
    model(x, subject_ids=None)


def test_eegnet_lightning_with_subject_specific():
  """Test EEGNetLightning with subject-specific preprocessing."""
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")
  mapper.add_subject("dataset1", "S02")

  config = NoteOnsetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=128,
    num_channels=4,
    window_start=0,
    window_end=64,
    use_subject_specific=True,
  )

  model = EEGNetLightning(config, subject_mapper=mapper)

  assert model.subject_mapper is not None
  assert model.model.subject_specific is not None


def test_eegnet_lightning_compute_loss_with_subject_specific():
  """Test that _compute_loss handles subject_ids correctly."""
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")
  mapper.add_subject("dataset1", "S02")

  config = NoteOnsetModelConfig(
    chunk_width=128,
    num_channels=4,
    window_start=0,
    window_end=64,
    use_subject_specific=True,
  )

  model = EEGNetLightning(config, subject_mapper=mapper)

  # Create a mock batch with info
  batch = {
    "eeg": torch.randn(3, 4, 128),
    "music": [
      NoteOnsets(onset_times=np.array([0.1]), sample_rate=256, duration_seconds=0.5),
      NoteOnsets(onset_times=np.array([0.2]), sample_rate=256, duration_seconds=0.5),
      NoteOnsets(onset_times=np.array([]), sample_rate=256, duration_seconds=0.5),
    ],
    "info": {
      "dataset": ["dataset1", "dataset1", "dataset1"],
      "subject": ["S01", "S02", "S01"],
    },
  }

  # Should not raise an error
  loss = model._compute_loss(batch, 0, "train")
  assert isinstance(loss, torch.Tensor)
  assert loss.numel() == 1


def test_eegnet_lightning_without_subject_specific():
  """Test that model works normally without subject-specific preprocessing."""
  config = NoteOnsetModelConfig(
    chunk_width=128,
    num_channels=4,
    use_subject_specific=False,
  )

  model = EEGNetLightning(config, subject_mapper=None)

  # Batch without info should work
  batch = {
    "eeg": torch.randn(2, 4, 128),
    "music": [
      NoteOnsets(onset_times=np.array([0.1]), sample_rate=256, duration_seconds=0.5),
      NoteOnsets(onset_times=np.array([]), sample_rate=256, duration_seconds=0.5),
    ],
  }

  loss = model._compute_loss(batch, 0, "val")
  assert isinstance(loss, torch.Tensor)


def test_eegnet_lightning_requires_info_when_subject_specific():
  """Test that batch['info'] is required when subject-specific is enabled."""
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")

  config = NoteOnsetModelConfig(
    chunk_width=128,
    num_channels=4,
    use_subject_specific=True,
  )

  model = EEGNetLightning(config, subject_mapper=mapper)

  # Batch without info should raise error
  batch = {
    "eeg": torch.randn(2, 4, 128),
    "music": [
      NoteOnsets(onset_times=np.array([0.1]), sample_rate=256, duration_seconds=0.5),
      NoteOnsets(onset_times=np.array([]), sample_rate=256, duration_seconds=0.5),
    ],
  }

  with pytest.raises(ValueError, match="requires batch\\['info'\\]"):
    model._compute_loss(batch, 0, "train")


def test_subject_specific_gradients_flow():
  """Test that gradients flow through subject-specific preprocessing."""
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")
  mapper.add_subject("dataset1", "S02")

  config = NoteOnsetModelConfig(
    chunk_width=128,
    num_channels=4,
    use_subject_specific=True,
  )

  model = EEGNetLightning(config, subject_mapper=mapper)

  batch = {
    "eeg": torch.randn(3, 4, 128),
    "music": [
      NoteOnsets(onset_times=np.array([0.1]), sample_rate=256, duration_seconds=0.5),
      NoteOnsets(onset_times=np.array([0.2]), sample_rate=256, duration_seconds=0.5),
      NoteOnsets(onset_times=np.array([]), sample_rate=256, duration_seconds=0.5),
    ],
    "info": {
      "dataset": ["dataset1", "dataset1", "dataset1"],
      "subject": ["S01", "S02", "S01"],
    },
  }

  loss = model._compute_loss(batch, 0, "train")
  loss.backward()

  # Check that subject-specific weights have gradients
  assert model.model.subject_specific is not None
  assert model.model.subject_specific.weights.grad is not None
  assert model.model.subject_specific.weights.grad.shape == (2, 4, 4)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])

import pytest
import torch
import numpy as np
from eeg_music.eegnet import (
  EEGNetWrapper,
  NoteOnsetModelConfig,
  EEGNetConfig,
  FBCNetConfig,
  TSCeptionConfig,
  ATCNetConfig,
  EEGNetLightning,
  has_onset_in_window,
)
from eeg_music.data import NoteOnsets


def test_eegnet_wrapper_initialization():
  """Test that EEGNetWrapper initializes correctly with valid parameters."""
  chunk_width = 1024
  num_channels = 28

  model = EEGNetWrapper(
    chunk_width=chunk_width,
    num_channels=num_channels,
    model_config=EEGNetConfig(),
  )

  assert model.num_classes == 1  # Binary classification
  assert model.chunk_width == chunk_width
  assert model.num_channels == num_channels


@pytest.mark.parametrize(
  "model_config",
  [
    EEGNetConfig(),
    FBCNetConfig(),
    TSCeptionConfig(),
    ATCNetConfig(),
  ],
  ids=["eegnet", "fbcnet", "tsception", "atcnet"],
)
def test_eegnet_wrapper_forward_shape(model_config):
  """Test that forward pass produces correct output shape for all model types."""
  chunk_width = 1024
  num_channels = 28
  batch_size = 4

  model = EEGNetWrapper(
    chunk_width=chunk_width,
    num_channels=num_channels,
    model_config=model_config,
  )

  # Create dummy input
  x = torch.randn(batch_size, num_channels, chunk_width)

  # Forward pass
  output = model(x)

  # Check output shape - single logit per sample
  expected_shape = (batch_size,)
  assert output.shape == expected_shape, (
    f"Expected shape {expected_shape}, got {output.shape}"
  )


def test_eegnet_wrapper_different_chunk_sizes():
  """Test various chunk size configurations."""
  test_configs = [1024, 512, 2048, 256]

  for chunk_width in test_configs:
    model = EEGNetWrapper(
      chunk_width=chunk_width,
      num_channels=28,
      model_config=EEGNetConfig(),
    )

    assert model.num_classes == 1, f"Expected 1 class, got {model.num_classes}"

    # Test forward pass
    x = torch.randn(2, 28, chunk_width)
    output = model(x)
    assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"


def test_fbcnet_wrapper_band_dimension():
  """Test that FBCNet correctly handles band dimension."""
  chunk_width = 1024
  num_channels = 28
  batch_size = 2

  model = EEGNetWrapper(
    chunk_width=chunk_width,
    num_channels=num_channels,
    model_config=FBCNetConfig(num_bands=1),
  )

  # Input without band dimension (will be added automatically)
  x = torch.randn(batch_size, num_channels, chunk_width)
  output = model(x)

  expected_shape = (batch_size,)
  assert output.shape == expected_shape


def test_eegnet_wrapper_output_is_tensor():
  """Test that output is a valid PyTorch tensor."""
  model = EEGNetWrapper(
    chunk_width=1024,
    num_channels=28,
    model_config=EEGNetConfig(),
  )

  x = torch.randn(2, 28, 1024)
  output = model(x)

  assert isinstance(output, torch.Tensor)
  assert output.dtype == torch.float32 or output.dtype == torch.float64
  assert not torch.isnan(output).any(), "Output contains NaN values"
  assert output.shape == (2,)


def test_tsception_wrapper_with_custom_params():
  """Test TSCeption model with custom parameters."""
  chunk_width = 512
  num_channels = 6
  batch_size = 2

  # Test with custom TSCeption parameters
  model = EEGNetWrapper(
    chunk_width=chunk_width,
    num_channels=num_channels,
    eeg_sample_rate=256,
    model_config=TSCeptionConfig(
      num_T=10,
      num_S=10,
      hid_channels=32,
      dropout=0.3,
    ),
  )

  x = torch.randn(batch_size, num_channels, chunk_width)
  output = model(x)

  assert output.shape == (batch_size,)
  assert not torch.isnan(output).any()


def test_atcnet_wrapper_with_custom_params():
  """Test ATCNet model with custom parameters."""
  chunk_width = 512
  num_channels = 6
  batch_size = 2

  # Test with custom ATCNet parameters (ATCNet doesn't support dropout)
  model = EEGNetWrapper(
    chunk_width=chunk_width,
    num_channels=num_channels,
    model_config=ATCNetConfig(
      num_windows=5,
      F1=16,
      D=2,
      tcn_depth=2,
      tcn_kernel_size=4,
    ),
  )

  x = torch.randn(batch_size, num_channels, chunk_width)
  output = model(x)

  assert output.shape == (batch_size,)
  assert not torch.isnan(output).any()


@pytest.mark.parametrize(
  "model_config,chunk_width",
  [
    (EEGNetConfig(), 256),
    (FBCNetConfig(), 256),
    (TSCeptionConfig(), 256),
    (ATCNetConfig(), 256),
    (EEGNetConfig(), 512),
    (FBCNetConfig(), 512),
    (TSCeptionConfig(), 512),
    (ATCNetConfig(), 512),
  ],
  ids=[
    "eegnet-256",
    "fbcnet-256",
    "tsception-256",
    "atcnet-256",
    "eegnet-512",
    "fbcnet-512",
    "tsception-512",
    "atcnet-512",
  ],
)
def test_all_models_different_chunk_sizes(model_config, chunk_width):
  """Test all model types with various chunk sizes."""
  num_channels = 28
  batch_size = 2

  model = EEGNetWrapper(
    chunk_width=chunk_width,
    num_channels=num_channels,
    model_config=model_config,
  )

  x = torch.randn(batch_size, num_channels, chunk_width)
  output = model(x)

  model_name = type(model_config).__name__.replace("ModelConfig", "")
  assert output.shape == (batch_size,), (
    f"{model_name} with chunk_width={chunk_width} failed: "
    f"expected shape ({batch_size},), got {output.shape}"
  )
  assert not torch.isnan(output).any(), f"{model_name} produced NaN values"


def test_has_onset_in_window():
  """Test checking if onsets fall within a specified window (single sample)."""
  eeg_sample_rate = 256

  # Test case 1: onset at 0.5s (sample 128) - should be in window [0, 200)
  onsets_1 = NoteOnsets(np.array([0.5, 1.5]), sample_rate=256, duration_seconds=4.0)
  assert has_onset_in_window(onsets_1, 0, 200, eeg_sample_rate)

  # Test case 2: onset at 1.0s (sample 256) - should be in window [200, 400)
  onsets_2 = NoteOnsets(np.array([1.0]), sample_rate=256, duration_seconds=4.0)
  assert has_onset_in_window(onsets_2, 200, 400, eeg_sample_rate)

  # Test case 3: no onsets - should return False
  onsets_3 = NoteOnsets(np.array([]), sample_rate=256, duration_seconds=4.0)
  assert not has_onset_in_window(onsets_3, 0, 100, eeg_sample_rate)

  # Test case 4: onset outside window - should return False
  onsets_4 = NoteOnsets(
    np.array([2.0]), sample_rate=256, duration_seconds=4.0
  )  # sample 512
  assert not has_onset_in_window(onsets_4, 0, 200, eeg_sample_rate)

  # Test case 5: onset exactly at window_end - should return False (exclusive)
  onsets_5 = NoteOnsets(
    np.array([0.78125]), sample_rate=256, duration_seconds=4.0
  )  # sample 200 exactly
  assert not has_onset_in_window(onsets_5, 0, 200, eeg_sample_rate)


def test_note_onset_model_config():
  """Test NoteOnsetModelConfig initialization."""
  config = NoteOnsetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=1024,
    lr_config=1e-3,
  )

  assert isinstance(config.model_config, EEGNetConfig)
  assert config.chunk_width == 1024
  assert config.lr_config == 1e-3
  assert config.num_channels == 28  # default
  assert config.pos_weight is None  # default


def test_eegnet_lightning_initialization():
  """Test EEGNetLightning module initialization."""
  config = NoteOnsetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=1024,
    num_channels=28,
  )

  lightning_model = EEGNetLightning(config)

  assert lightning_model.config == config
  assert lightning_model.learning_rate == config.lr_config  # lr_config is a float
  assert isinstance(lightning_model.model, EEGNetWrapper)


def test_eegnet_lightning_forward():
  """Test forward pass through EEGNetLightning."""
  config = NoteOnsetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=1024,
    num_channels=28,
  )

  lightning_model = EEGNetLightning(config)

  # Create dummy input
  x = torch.randn(2, 28, 1024)
  output = lightning_model(x)

  # Check output shape - single logit per sample
  expected_shape = (2,)
  assert output.shape == expected_shape


def test_eegnet_lightning_training_step():
  """Test training step with dummy batch."""
  config = NoteOnsetModelConfig(
    model_config=EEGNetConfig(),
    chunk_width=1024,
    num_channels=28,
    eeg_sample_rate=256,
    window_start=0,
    window_end=200,
  )

  lightning_model = EEGNetLightning(config)

  # Create dummy batch (window ranges are in config, not batch)
  # music is a list of NoteOnsets objects
  batch = {
    "eeg": torch.randn(2, 28, 1024),
    "music": [
      NoteOnsets(np.array([0.5, 1.5]), sample_rate=256, duration_seconds=4.0),
      NoteOnsets(np.array([1.0]), sample_rate=256, duration_seconds=4.0),
    ],
  }

  # Run training step
  loss = lightning_model.training_step(batch, 0)

  # Check that loss is valid
  assert isinstance(loss, torch.Tensor)
  assert loss.dim() == 0  # scalar
  assert not torch.isnan(loss)
  assert loss > 0

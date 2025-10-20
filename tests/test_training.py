"""
Test suite for training.py main function.

Tests that training can run end-to-end with minimal configuration.
"""

import pytest

from eeg_music.training import main, TrainingConfig


@pytest.mark.skip(reason="Too long to run on cpu")
def test_training_main_one_epoch(tmp_path):
  """Test that training.main runs successfully with num_epochs=1."""

  # Create a minimal config for testing
  test_config = TrainingConfig(
    num_epochs=1,
    batch_size=2,
    save_model_per_epochs=1,
    val_every_n_epoch=1,
    data_loader_num_workers=2,
    prefetch_factor=2,
    save_path=str(tmp_path / "test_checkpoints"),
    wandb_log_model=False,
    project_name="test-project",
    run_name="test-run",
    run_extra_name="one-epoch-test",
    auroc_every_n_epochs=1,
    use_learning_rate_finder=False,
  )

  # Run training for 1 epoch
  main(config=test_config)

  # If we get here without exceptions, the test passes
  assert True


def test_model_output_shape():
  """Test that model output shape matches batch['mel'] shape."""
  from eeg_music.dataloader import load_and_create_dataloaders
  from eeg_music.eegpt import EegptWithLinear
  import torch

  config = TrainingConfig(
    batch_size=4,
    data_loader_num_workers=2,
    prefetch_factor=2,
  )

  dataloaders = load_and_create_dataloaders(config.data_path, config)
  train_dl = dataloaders["train"]

  batch = next(iter(train_dl))

  model = EegptWithLinear.load_from_checkpoint(
    config.eegpt_chpt_path,
    use_chan_conv=config.use_chan_conv,
  )
  model.eval()

  with torch.no_grad():
    output = model(batch["eeg"])

  assert output.shape == batch["music"].shape, (
    f"Model output shape {output.shape} does not match "
    f"music spectrogram shape {batch['music'].shape}"
  )

import torch
from unittest.mock import Mock

from eeg_music.training import AUROCCallback


def test_auroc_callback_batched_prediction():
  """Test that batched prediction in AUROCCallback works correctly."""
  # Mock pl_module (the model)
  pl_module = Mock()
  pl_module.device = torch.device("cpu")

  # Create some dummy data
  n_samples = 50
  all_x = torch.randn(n_samples, 28, 1024)  # (N, channels, time)
  all_y = torch.randn(n_samples, 128, 259)  # (N, freq, time)
  all_y_hat_expected = torch.randn(n_samples, 128, 259)

  # Configure the model mock to return the expected predictions
  pl_module.side_effect = lambda x: all_y_hat_expected[0 : x.shape[0]]

  # Mock the trainer and dataloaders
  trainer = Mock()
  trainer.current_epoch = 0
  trainer.val_dataloaders = [{"eeg": all_x, "music": all_y}]

  # Instantiate the callback with a small batch size for testing
  callback = AUROCCallback(auroc_every_n_epochs=1, prediction_batch_size=10)

  # Mock the _compute_similarity method to simplify the test
  callback._compute_similarity = Mock(return_value=0.5)

  # Run the callback method
  callback.on_validation_epoch_end(trainer, pl_module)

  # We can't easily check the output without a lot more mocking,
  # but if it runs without error, it's a good sign.
  # A more advanced test could check the logged values.
  assert pl_module.log.called, "pl_module.log() was not called."

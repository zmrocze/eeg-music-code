import torch
from eeg_music.eegnet import BinaryAccuracyCalc


def test_cumulative_mean_std_single_batch():
  """Test mean and std with a single batch."""
  calc = BinaryAccuracyCalc()

  logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
  targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

  calc.update(logits, targets)
  metrics = calc.compute()

  # Expected mean: (1 + 2 + 3 + 4) / 4 = 2.5
  assert abs(metrics["logits_mean"] - 2.5) < 1e-6

  # Expected std: sqrt(((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4)
  #             = sqrt((2.25 + 0.25 + 0.25 + 2.25) / 4) = sqrt(1.25) ≈ 1.118
  expected_std = ((1.5**2 + 0.5**2 + 0.5**2 + 1.5**2) / 4) ** 0.5
  assert abs(metrics["logits_std"] - expected_std) < 1e-6


def test_cumulative_mean_std_multiple_batches():
  """Test cumulative computation across multiple batches."""
  calc = BinaryAccuracyCalc()

  # First batch
  logits1 = torch.tensor([1.0, 2.0])
  targets1 = torch.tensor([1.0, 0.0])
  calc.update(logits1, targets1)

  # Second batch
  logits2 = torch.tensor([3.0, 4.0])
  targets2 = torch.tensor([1.0, 0.0])
  calc.update(logits2, targets2)

  metrics = calc.compute()

  # Combined: [1, 2, 3, 4], mean = 2.5
  assert abs(metrics["logits_mean"] - 2.5) < 1e-6

  # Combined std should match single-batch result
  expected_std = ((1.5**2 + 0.5**2 + 0.5**2 + 1.5**2) / 4) ** 0.5
  assert abs(metrics["logits_std"] - expected_std) < 1e-6


def test_cumulative_mean_std_reset():
  """Test that reset clears cumulative statistics."""
  calc = BinaryAccuracyCalc()

  logits = torch.tensor([1.0, 2.0, 3.0])
  targets = torch.tensor([1.0, 1.0, 0.0])
  calc.update(logits, targets)

  # Verify stats are non-zero
  metrics = calc.compute()
  assert metrics["logits_mean"] != 0.0
  assert calc.count > 0

  calc.reset()

  # Verify reset
  assert calc.count == 0
  assert calc.mean == 0.0
  assert calc.m2 == 0.0

  metrics = calc.compute()
  assert metrics["logits_mean"] == 0.0
  assert metrics["logits_std"] == 0.0


def test_cumulative_stats_with_negative_logits():
  """Test with negative logits."""
  calc = BinaryAccuracyCalc()

  logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
  targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])

  calc.update(logits, targets)
  metrics = calc.compute()

  # Mean: (-2 + -1 + 0 + 1 + 2) / 5 = 0
  assert abs(metrics["logits_mean"] - 0.0) < 1e-6

  # Std: sqrt((4 + 1 + 0 + 1 + 4) / 5) = sqrt(2) ≈ 1.414
  expected_std = (10 / 5) ** 0.5
  assert abs(metrics["logits_std"] - expected_std) < 1e-6


def test_cumulative_stats_empty():
  """Test with no updates."""
  calc = BinaryAccuracyCalc()
  metrics = calc.compute()

  assert metrics["logits_mean"] == 0.0
  assert metrics["logits_std"] == 0.0

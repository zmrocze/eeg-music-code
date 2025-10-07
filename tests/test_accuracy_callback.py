import torch
from eeg_music.eegpt import AccuracyCalc


def test_accuracy_calc_initialization():
  """Test that AccuracyCalc initializes correctly."""
  acc = AccuracyCalc()
  assert acc.correct == 0
  assert acc.total == 0


def test_accuracy_calc_update():
  """Test that AccuracyCalc updates correctly."""
  acc = AccuracyCalc()

  # Create logits and targets
  # Logits where class with highest value matches target for first 3, not for last
  logits = torch.tensor(
    [
      [10.0, 0.0, 0.0, 0.0],  # Predicts class 0, target is 0 ✓
      [0.0, 10.0, 0.0, 0.0],  # Predicts class 1, target is 1 ✓
      [0.0, 0.0, 10.0, 0.0],  # Predicts class 2, target is 2 ✓
      [0.0, 0.0, 0.0, 10.0],  # Predicts class 3, target is 0 ✗
    ]
  )
  targets = torch.tensor([0, 1, 2, 0])  # Last one is wrong

  acc.update(logits, targets)

  # Check counters
  assert acc.correct == 3  # 3 out of 4 correct
  assert acc.total == 4


def test_accuracy_calc_compute():
  """Test that AccuracyCalc computes accuracy correctly."""
  acc = AccuracyCalc()

  # Set known values
  acc.correct = 3
  acc.total = 4

  accuracy = acc.compute()
  assert accuracy == 0.75  # 3/4 = 0.75


def test_accuracy_calc_compute_empty():
  """Test that AccuracyCalc handles empty case."""
  acc = AccuracyCalc()

  accuracy = acc.compute()
  assert accuracy == 0.0  # Empty case returns 0


def test_accuracy_calc_reset():
  """Test that AccuracyCalc resets correctly."""
  acc = AccuracyCalc()

  # Set some values
  acc.correct = 10
  acc.total = 20

  # Reset
  acc.reset()

  # Check that counters are reset
  assert acc.correct == 0
  assert acc.total == 0


def test_accuracy_calc_multiple_updates():
  """Test accuracy calculation with multiple update calls."""
  acc = AccuracyCalc()

  # First batch: 2/3 correct
  logits1 = torch.tensor(
    [
      [10.0, 0.0, 0.0],  # Predicts 0, target 0 ✓
      [0.0, 10.0, 0.0],  # Predicts 1, target 1 ✓
      [0.0, 0.0, 10.0],  # Predicts 2, target 0 ✗
    ]
  )
  targets1 = torch.tensor([0, 1, 0])  # Last one wrong
  acc.update(logits1, targets1)

  # Second batch: 3/3 correct
  logits2 = torch.zeros(3, 6)
  logits2[0, 3] = 10.0  # Predicts 3
  logits2[1, 4] = 10.0  # Predicts 4
  logits2[2, 5] = 10.0  # Predicts 5
  targets2 = torch.tensor([3, 4, 5])  # All correct
  acc.update(logits2, targets2)

  # Total: 5/6 correct
  assert acc.correct == 5
  assert acc.total == 6
  assert abs(acc.compute() - 5 / 6) < 1e-6


def test_accuracy_calc_perfect_accuracy():
  """Test perfect accuracy case."""
  acc = AccuracyCalc()

  # Create logits where argmax matches targets
  logits = torch.zeros(4, 4)
  for i in range(4):
    logits[i, i] = 10.0
  targets = torch.tensor([0, 1, 2, 3])  # All correct
  acc.update(logits, targets)

  assert acc.compute() == 1.0


def test_accuracy_calc_zero_accuracy():
  """Test zero accuracy case."""
  acc = AccuracyCalc()

  # Create logits where predictions are all wrong
  logits = torch.zeros(4, 4)
  logits[0, 0] = 10.0  # Predicts 0, target 1
  logits[1, 1] = 10.0  # Predicts 1, target 2
  logits[2, 2] = 10.0  # Predicts 2, target 3
  logits[3, 3] = 10.0  # Predicts 3, target 0
  targets = torch.tensor([1, 2, 3, 0])  # All wrong
  acc.update(logits, targets)

  assert acc.compute() == 0.0


def test_accuracy_calc_reset_and_reuse():
  """Test that AccuracyCalc can be reset and reused."""
  acc = AccuracyCalc()

  # First epoch
  logits1 = torch.zeros(3, 3)
  for i in range(3):
    logits1[i, i] = 10.0
  targets1 = torch.tensor([0, 1, 2])
  acc.update(logits1, targets1)
  assert acc.compute() == 1.0

  # Reset for new epoch
  acc.reset()
  assert acc.compute() == 0.0

  # Second epoch
  logits2 = torch.tensor(
    [
      [10.0, 0.0],  # Predicts 0, target 0 ✓
      [10.0, 0.0],  # Predicts 0, target 1 ✗
    ]
  )
  targets2 = torch.tensor([0, 1])  # 1/2 correct
  acc.update(logits2, targets2)
  assert acc.compute() == 0.5


def test_accuracy_calc_with_large_batch():
  """Test AccuracyCalc with larger batch."""
  acc = AccuracyCalc()

  # Large batch - create logits where argmax gives specific predictions
  logits = torch.zeros(100, 9)
  predictions = torch.randint(0, 9, (100,))
  for i in range(100):
    logits[i, predictions[i]] = 10.0

  targets = predictions.clone()
  targets[::2] = (targets[::2] + 1) % 9  # Make half wrong

  acc.update(logits, targets)

  assert acc.total == 100
  assert acc.correct == 50  # Half correct
  assert acc.compute() == 0.5

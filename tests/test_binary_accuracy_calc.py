import torch
from eeg_music.eegnet import BinaryAccuracyCalc


def test_binary_accuracy_calc_perfect_predictions():
  """Test with perfect predictions."""
  calc = BinaryAccuracyCalc()

  # All predictions correct
  logits = torch.tensor([2.0, -2.0, 2.0, -2.0])  # High positive, high negative
  targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

  calc.update(logits, targets)
  metrics = calc.compute()

  assert metrics["accuracy"] == 1.0
  assert metrics["recall"] == 1.0
  assert metrics["specificity"] == 1.0
  assert metrics["precision"] == 1.0
  assert metrics["f1_score"] == 1.0


def test_binary_accuracy_calc_all_wrong():
  """Test with all predictions wrong."""
  calc = BinaryAccuracyCalc()

  # All predictions wrong
  logits = torch.tensor([2.0, -2.0, 2.0, -2.0])
  targets = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Opposite of predictions

  calc.update(logits, targets)
  metrics = calc.compute()

  assert metrics["accuracy"] == 0.0
  assert metrics["recall"] == 0.0
  assert metrics["specificity"] == 0.0
  # Precision and F1 are undefined when TP = 0 and FP = 0 or TP + FP = 0
  # In this case: TP=0, TN=0, FP=2, FN=2
  # Precision = 0 / (0 + 2) = 0
  assert metrics["precision"] == 0.0
  assert metrics["f1_score"] == 0.0


def test_binary_accuracy_calc_mixed():
  """Test with mixed predictions."""
  calc = BinaryAccuracyCalc()

  # TP=2, TN=1, FP=1, FN=0
  # Predictions: [1, 1, 1, 0] (using threshold 0.5)
  # Targets:     [1, 1, 0, 0]
  logits = torch.tensor([2.0, 2.0, 2.0, -2.0])
  targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

  calc.update(logits, targets)
  metrics = calc.compute()

  # Manual calculation:
  # TP=2 (first two), TN=1 (last one), FP=1 (third), FN=0
  # Accuracy = (2 + 1) / 4 = 0.75
  assert metrics["accuracy"] == 0.75

  # Recall = TP / (TP + FN) = 2 / (2 + 0) = 1.0
  assert metrics["recall"] == 1.0

  # Specificity = TN / (TN + FP) = 1 / (1 + 1) = 0.5
  assert metrics["specificity"] == 0.5

  # Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.6667
  assert abs(metrics["precision"] - 2 / 3) < 1e-6

  # F1 = 2 * (precision * recall) / (precision + recall)
  #    = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 2 * 2/3 / 5/3 = 4/3 * 3/5 = 4/5 = 0.8
  assert abs(metrics["f1_score"] - 0.8) < 1e-6


def test_binary_accuracy_calc_accumulation():
  """Test accumulation across multiple batches."""
  calc = BinaryAccuracyCalc()

  # First batch: TP=1, TN=1
  logits1 = torch.tensor([2.0, -2.0])
  targets1 = torch.tensor([1.0, 0.0])
  calc.update(logits1, targets1)

  # Second batch: FP=1, FN=1
  logits2 = torch.tensor([2.0, -2.0])
  targets2 = torch.tensor([0.0, 1.0])
  calc.update(logits2, targets2)

  metrics = calc.compute()

  # Total: TP=1, TN=1, FP=1, FN=1
  # Accuracy = 2 / 4 = 0.5
  assert metrics["accuracy"] == 0.5

  # Recall = 1 / (1 + 1) = 0.5
  assert metrics["recall"] == 0.5

  # Specificity = 1 / (1 + 1) = 0.5
  assert metrics["specificity"] == 0.5

  # Precision = 1 / (1 + 1) = 0.5
  assert metrics["precision"] == 0.5

  # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
  assert metrics["f1_score"] == 0.5


def test_binary_accuracy_calc_reset():
  """Test reset functionality."""
  calc = BinaryAccuracyCalc()

  logits = torch.tensor([2.0, -2.0])
  targets = torch.tensor([1.0, 0.0])
  calc.update(logits, targets)

  # Check counters are non-zero
  assert calc.tp > 0 or calc.tn > 0

  calc.reset()

  # Check all counters are zero
  assert calc.tp == 0
  assert calc.tn == 0
  assert calc.fp == 0
  assert calc.fn == 0

  metrics = calc.compute()
  # All metrics should be 0 after reset
  assert metrics["accuracy"] == 0.0


def test_binary_accuracy_calc_threshold():
  """Test with different threshold values."""
  calc = BinaryAccuracyCalc()

  # Logits around 0 (sigmoid ≈ 0.5)
  logits = torch.tensor([0.1, -0.1, 0.2, -0.2])
  targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

  # With default threshold 0.5
  calc.update(logits, targets, threshold=0.5)
  metrics = calc.compute()

  # sigmoid(0.1) ≈ 0.525 > 0.5 → pred=1
  # sigmoid(-0.1) ≈ 0.475 < 0.5 → pred=0
  # sigmoid(0.2) ≈ 0.550 > 0.5 → pred=1
  # sigmoid(-0.2) ≈ 0.450 < 0.5 → pred=0
  # All predictions match targets
  assert metrics["accuracy"] == 1.0


def test_binary_accuracy_calc_formulas():
  """Test specific confusion matrix values against formulas."""
  calc = BinaryAccuracyCalc()

  # Manually set confusion matrix values
  # We'll create a scenario: TP=10, TN=20, FP=5, FN=3
  # Positive predictions: TP + FP = 15
  # Negative predictions: TN + FN = 23
  # Total: 38

  # Create logits and targets to achieve this
  # TP=10: 10 positive samples predicted as positive
  tp_logits = torch.full((10,), 2.0)
  tp_targets = torch.ones(10)

  # TN=20: 20 negative samples predicted as negative
  tn_logits = torch.full((20,), -2.0)
  tn_targets = torch.zeros(20)

  # FP=5: 5 negative samples predicted as positive
  fp_logits = torch.full((5,), 2.0)
  fp_targets = torch.zeros(5)

  # FN=3: 3 positive samples predicted as negative
  fn_logits = torch.full((3,), -2.0)
  fn_targets = torch.ones(3)

  all_logits = torch.cat([tp_logits, tn_logits, fp_logits, fn_logits])
  all_targets = torch.cat([tp_targets, tn_targets, fp_targets, fn_targets])

  calc.update(all_logits, all_targets)
  metrics = calc.compute()

  # Verify confusion matrix
  assert calc.tp == 10
  assert calc.tn == 20
  assert calc.fp == 5
  assert calc.fn == 3

  # Test formulas
  # Accuracy = (TP + TN) / (TP + TN + FP + FN) = 30 / 38
  assert abs(metrics["accuracy"] - 30 / 38) < 1e-6

  # Recall = TP / (TP + FN) = 10 / 13
  assert abs(metrics["recall"] - 10 / 13) < 1e-6

  # Specificity = TN / (TN + FP) = 20 / 25
  assert abs(metrics["specificity"] - 20 / 25) < 1e-6

  # Precision = TP / (TP + FP) = 10 / 15
  assert abs(metrics["precision"] - 10 / 15) < 1e-6

  # F1 = 2 * precision * recall / (precision + recall)
  precision = 10 / 15
  recall = 10 / 13
  expected_f1 = 2 * precision * recall / (precision + recall)
  assert abs(metrics["f1_score"] - expected_f1) < 1e-6

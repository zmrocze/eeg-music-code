"""
Test script to demonstrate the freeze_all_except_head_and_adapters function.
"""

from pathlib import Path

from eeg_music.eegpt import EegptLightning, EegptConfig
from eeg_music.freeze_utils import freeze_all_except_head_and_adapters, count_parameters


def test_freeze_function():
  """Test the freeze function with a sample model."""

  # Load a model (adjust the path to your checkpoint)
  # This is just an example - replace with your actual checkpoint path
  chpt_path = Path(
    "/home/zmrocze/studia/uwr/magisterka/model_checkpoints/25866970/model_state_dict.pt"
  )

  if not chpt_path.exists():
    print(f"Checkpoint not found at {chpt_path}")
    print("Please update the checkpoint path in test_freeze_utils.py")
    return

  # Create model
  config = EegptConfig(
    chpt_path=chpt_path,
    lr_config=1e-4,
    trainable=["linear", "head"],
    requiring_grad=["linear", "head"],
    use_chan_conv=False,
  )
  model = EegptLightning(config)

  print("BEFORE FREEZING:")
  print("-" * 80)
  params_before = count_parameters(model)
  print(f"Trainable: {params_before['trainable']:,}")
  print(f"Frozen: {params_before['frozen']:,}")
  print(f"Total: {params_before['total']:,}")
  print(f"Trainable ratio: {params_before['trainable_ratio']:.2%}")

  # Apply freezing
  freeze_all_except_head_and_adapters(model, verbose=True)

  print("\nAFTER FREEZING:")
  print("-" * 80)
  params_after = count_parameters(model)
  print(f"Trainable: {params_after['trainable']:,}")
  print(f"Frozen: {params_after['frozen']:,}")
  print(f"Total: {params_after['total']:,}")
  print(f"Trainable ratio: {params_after['trainable_ratio']:.2%}")

  # Verify that the correct layers are trainable
  print("\n" + "=" * 80)
  print("VERIFICATION - Checking specific layers:")
  print("=" * 80)

  eegpt_classifier = model.model.model
  eegpt_with_linear = model.model

  # Check chan_conv
  chan_conv_trainable = any(
    p.requires_grad for p in eegpt_classifier.chan_conv.parameters()
  )
  print(
    f"chan_conv trainable: {chan_conv_trainable} {'✓' if chan_conv_trainable else '✗'}"
  )

  # Check head
  head_trainable = any(p.requires_grad for p in eegpt_classifier.head.parameters())
  print(f"head trainable: {head_trainable} {'✓' if head_trainable else '✗'}")

  # Check ResidualLinear
  linear_trainable = any(p.requires_grad for p in eegpt_with_linear.linear.parameters())
  print(
    f"linear (ResidualLinear) trainable: {linear_trainable} {'✓' if linear_trainable else '✗'}"
  )

  # Check encoder (should be frozen)
  encoder_trainable = any(
    p.requires_grad for p in eegpt_classifier.target_encoder.parameters()
  )
  print(
    f"target_encoder frozen: {not encoder_trainable} {'✓' if not encoder_trainable else '✗'}"
  )

  # Check reconstructor/predictor (should be frozen)
  if hasattr(eegpt_classifier, "reconstructor"):
    rec_trainable = any(
      p.requires_grad for p in eegpt_classifier.reconstructor.parameters()
    )
    print(
      f"reconstructor frozen: {not rec_trainable} {'✓' if not rec_trainable else '✗'}"
    )
  elif hasattr(eegpt_classifier, "predictor"):
    pred_trainable = any(
      p.requires_grad for p in eegpt_classifier.predictor.parameters()
    )
    print(
      f"predictor frozen: {not pred_trainable} {'✓' if not pred_trainable else '✗'}"
    )

  print("=" * 80)


if __name__ == "__main__":
  test_freeze_function()

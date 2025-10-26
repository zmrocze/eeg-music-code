import torch
import pytest
from dataclasses import dataclass

from eeg_music.subject_specific import (
  SubjectSpecificLinear,
  SubjectDatasetMapper,
  create_subject_ids_from_trials,
  create_onehot_from_ids,
)


@dataclass
class MockTrialData:
  """Minimal mock of TrialData for testing."""

  dataset: str
  subject: str


def test_subject_dataset_mapper():
  """Test SubjectDatasetMapper basic functionality."""
  mapper = SubjectDatasetMapper()

  # Add first subject-dataset pair
  id1 = mapper.add_subject("dataset1", "S01")
  assert id1 == 0
  assert mapper.num_subjects == 1

  # Add second pair - different subject, same dataset
  id2 = mapper.add_subject("dataset1", "S02")
  assert id2 == 1
  assert mapper.num_subjects == 2

  # Add third pair - same subject, different dataset (should get new ID)
  id3 = mapper.add_subject("dataset2", "S01")
  assert id3 == 2
  assert mapper.num_subjects == 3

  # Adding existing pair should return same ID
  id1_again = mapper.add_subject("dataset1", "S01")
  assert id1_again == id1
  assert mapper.num_subjects == 3

  # Test get_id
  assert mapper.get_id("dataset1", "S01") == 0
  assert mapper.get_id("dataset1", "S02") == 1
  assert mapper.get_id("dataset2", "S01") == 2

  # Test get_mapping
  mapping = mapper.get_mapping()
  assert mapping == {
    ("dataset1", "S01"): 0,
    ("dataset1", "S02"): 1,
    ("dataset2", "S01"): 2,
  }


def test_subject_dataset_mapper_key_error():
  """Test that get_id raises KeyError for unknown pairs."""
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")

  with pytest.raises(KeyError):
    mapper.get_id("dataset1", "S02")


def test_create_subject_ids_from_trials():
  """Test creating subject IDs from trial data."""
  trials = [
    MockTrialData(dataset="bcmi_eeg1", subject="S01"),
    MockTrialData(dataset="bcmi_eeg1", subject="S02"),
    MockTrialData(dataset="bcmi_eeg1", subject="S01"),  # Duplicate
    MockTrialData(
      dataset="bcmi_eeg2", subject="S01"
    ),  # Same subject, different dataset
  ]

  subject_ids, mapper = create_subject_ids_from_trials(trials)

  # Check shape and values
  assert subject_ids.shape == (4,)
  assert subject_ids.tolist() == [
    0,
    1,
    0,
    2,
  ]  # S01 in dataset1, S02 in dataset1, S01 again, S01 in dataset2

  # Check mapper
  assert mapper.num_subjects == 3
  assert mapper.get_id("bcmi_eeg1", "S01") == 0
  assert mapper.get_id("bcmi_eeg1", "S02") == 1
  assert mapper.get_id("bcmi_eeg2", "S01") == 2


def test_create_subject_ids_with_existing_mapper():
  """Test that we can use an existing mapper."""
  # Create mapper with some existing subjects
  mapper = SubjectDatasetMapper()
  mapper.add_subject("dataset1", "S01")
  mapper.add_subject("dataset1", "S02")

  # Create trials with new and existing subjects
  trials = [
    MockTrialData(dataset="dataset1", subject="S01"),  # Existing
    MockTrialData(dataset="dataset2", subject="S01"),  # New
  ]

  subject_ids, updated_mapper = create_subject_ids_from_trials(trials, mapper=mapper)

  assert subject_ids.tolist() == [
    0,
    2,
  ]  # S01 in dataset1 (existing), S01 in dataset2 (new)
  assert updated_mapper.num_subjects == 3
  assert updated_mapper is mapper  # Same object


def test_create_onehot_from_ids():
  """Test one-hot encoding creation."""
  ids = torch.tensor([0, 2, 1, 0])
  onehot = create_onehot_from_ids(ids, num_subjects=3)

  expected = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0],
      [1.0, 0.0, 0.0],
    ]
  )

  assert torch.allclose(onehot, expected)
  assert onehot.shape == (4, 3)


def test_subject_specific_linear_basic():
  """Test basic functionality of SubjectSpecificLinear."""
  num_subjects = 3
  num_channels = 4
  timepoints = 100
  batch_size = 5

  model = SubjectSpecificLinear(
    num_subjects=num_subjects,
    num_channels=num_channels,
  )

  # Create dummy EEG data
  eeg = torch.randn(batch_size, num_channels, timepoints)
  subject_ids = torch.tensor([0, 1, 2, 0, 1])

  # Forward pass
  output = model(eeg, subject_ids)

  # Check output shape
  assert output.shape == (batch_size, num_channels, timepoints)

  # With identity initialization, output should equal input
  assert torch.allclose(output, eeg, rtol=1e-5)


def test_subject_specific_linear_with_onehot():
  """Test SubjectSpecificLinear with one-hot encoded IDs."""
  num_subjects = 3
  num_channels = 4
  timepoints = 50
  batch_size = 4

  model = SubjectSpecificLinear(
    num_subjects=num_subjects,
    num_channels=num_channels,
  )

  eeg = torch.randn(batch_size, num_channels, timepoints)

  # Test with one-hot encoded IDs
  subject_ids = torch.tensor([0, 1, 2, 0])
  subject_ids_onehot = create_onehot_from_ids(subject_ids, num_subjects)

  output_int = model(eeg, subject_ids)
  output_onehot = model(eeg, subject_ids_onehot)

  # Both should give same result
  assert torch.allclose(output_int, output_onehot)


def test_subject_specific_linear_trainable_weights():
  """Test that trainable weights can be updated."""
  model = SubjectSpecificLinear(
    num_subjects=2,
    num_channels=3,
  )

  # Check that weights are parameters
  assert isinstance(model.weights, torch.nn.Parameter)

  # Modify weights
  with torch.no_grad():
    model.weights[0, 0, 1] = 2.0  # Subject 0: channel 0 gets 2x channel 1
    model.weights[0, 1, 1] = 0.0  # Subject 0: channel 1 contribution to channel 1 is 0

  eeg = torch.zeros(2, 3, 10)
  eeg[0, 1, :] = 1.0  # Subject 0: only channel 1 is active
  eeg[1, 0, :] = 1.0  # Subject 1: only channel 0 is active

  subject_ids = torch.tensor([0, 1])
  output = model(eeg, subject_ids)

  # For subject 0: channel 0 should have value 2.0 (from 2x channel 1)
  assert torch.allclose(output[0, 0, :], torch.full((10,), 2.0))
  # For subject 0: channel 1 should be 0 (weight set to 0)
  assert torch.allclose(output[0, 1, :], torch.zeros(10))

  # For subject 1: should still be identity (channel 0 unchanged)
  assert torch.allclose(output[1, 0, :], torch.ones(10))


def test_subject_specific_linear_different_transformations():
  """Test that different subjects get different transformations."""
  num_subjects = 2
  num_channels = 2
  timepoints = 5

  model = SubjectSpecificLinear(
    num_subjects=num_subjects,
    num_channels=num_channels,
  )

  # Set up different transformations for each subject
  with torch.no_grad():
    # Subject 0: swap channels
    model.weights[0] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    # Subject 1: average channels
    model.weights[1] = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

  # Create test data where channels have different values
  eeg = torch.zeros(2, 2, timepoints)
  eeg[:, 0, :] = 1.0  # Channel 0 = 1
  eeg[:, 1, :] = 2.0  # Channel 1 = 2

  subject_ids = torch.tensor([0, 1])
  output = model(eeg, subject_ids)

  # Subject 0: channels should be swapped
  assert torch.allclose(
    output[0, 0, :], torch.full((timepoints,), 2.0)
  )  # Got channel 1
  assert torch.allclose(
    output[0, 1, :], torch.full((timepoints,), 1.0)
  )  # Got channel 0

  # Subject 1: channels should be averaged
  assert torch.allclose(output[1, 0, :], torch.full((timepoints,), 1.5))  # (1+2)/2
  assert torch.allclose(output[1, 1, :], torch.full((timepoints,), 1.5))  # (1+2)/2


def test_full_pipeline_example():
  """Full example demonstrating the complete pipeline."""
  # Step 1: Create mock trial data
  trials = [
    MockTrialData(dataset="bcmi_eeg1", subject="S01"),
    MockTrialData(dataset="bcmi_eeg1", subject="S02"),
    MockTrialData(dataset="bcmi_eeg2", subject="S01"),
    MockTrialData(dataset="bcmi_eeg1", subject="S01"),
  ]

  # Step 2: Create subject IDs and mapper
  subject_ids, mapper = create_subject_ids_from_trials(trials)

  # Step 3: Create the model
  num_channels = 28  # Typical EEG setup
  model = SubjectSpecificLinear(
    num_subjects=mapper.num_subjects,
    num_channels=num_channels,
  )

  # Step 4: Create dummy EEG batch
  batch_size = len(trials)
  timepoints = 256
  eeg_batch = torch.randn(batch_size, num_channels, timepoints)

  # Step 5: Apply transformation
  output = model(eeg_batch, subject_ids)

  # Verify output shape
  assert output.shape == (batch_size, num_channels, timepoints)

  # With identity initialization, should preserve input
  assert torch.allclose(output, eeg_batch, rtol=1e-5)

  # Step 6: Can also use one-hot encoding
  subject_ids_onehot = create_onehot_from_ids(subject_ids, mapper.num_subjects)
  output_onehot = model(eeg_batch, subject_ids_onehot)
  assert torch.allclose(output, output_onehot)


def test_gradient_flow_with_trainable_weights():
  """Test that gradients flow properly through trainable weights."""
  model = SubjectSpecificLinear(
    num_subjects=2,
    num_channels=3,
  )

  eeg = torch.randn(4, 3, 10, requires_grad=True)
  subject_ids = torch.tensor([0, 1, 0, 1])

  output = model(eeg, subject_ids)
  loss = output.sum()
  loss.backward()

  # Check that gradients exist
  assert model.weights.grad is not None
  assert eeg.grad is not None

  # Check gradient shape
  assert model.weights.grad.shape == (2, 3, 3)


if __name__ == "__main__":
  # Run a simple demonstration
  print("=" * 60)
  print("SubjectSpecificLinear Model Demonstration")
  print("=" * 60)

  # Create sample trials
  trials = [
    MockTrialData(dataset="bcmi_eeg1", subject="S01"),
    MockTrialData(dataset="bcmi_eeg1", subject="S02"),
    MockTrialData(dataset="bcmi_eeg2", subject="S01"),
  ]

  # Create subject IDs
  subject_ids, mapper = create_subject_ids_from_trials(trials)

  print("\nSubject-Dataset Mapping:")
  for (dataset, subject), idx in mapper.get_mapping().items():
    print(f"  ({dataset}, {subject}) -> ID {idx}")

  print(f"\nSubject IDs for trials: {subject_ids.tolist()}")

  # Create model
  model = SubjectSpecificLinear(
    num_subjects=mapper.num_subjects,
    num_channels=4,
  )

  print(f"\nModel created with {mapper.num_subjects} subjects, 4 channels")
  print(f"Weight matrix shape: {model.weights.shape}")

  # Create dummy data
  eeg = torch.randn(3, 4, 100)
  output = model(eeg, subject_ids)

  print(f"\nInput shape: {eeg.shape}")
  print(f"Output shape: {output.shape}")

  # Test with one-hot
  onehot = create_onehot_from_ids(subject_ids, mapper.num_subjects)
  print(f"\nOne-hot encoding shape: {onehot.shape}")
  print(f"One-hot encoding:\n{onehot}")

  print("\n" + "=" * 60)
  print("All tests passed! ✓")
  print("=" * 60)

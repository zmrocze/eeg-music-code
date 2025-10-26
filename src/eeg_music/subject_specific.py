"""Subject-specific linear transformations for EEG data.

This module provides tools for applying different linear transformations
to EEG data based on subject-dataset identifiers. This is useful for:
- Subject-specific preprocessing/normalization
- Learning subject-adaptive spatial filters
- Accounting for inter-subject variability in EEG recordings
"""

import torch
import torch.nn as nn
from typing import Optional, Sequence, Protocol


class SubjectSpecificLinear(nn.Module):
  """Subject-specific linear transformation for EEG data.

  This model applies different linear channel-mixing transformations
  based on subject-dataset identifiers. Each subject-dataset combination
  has its own learnable (or fixed) weight matrix that linearly combines
  the input EEG channels.

  Args:
      num_subjects: Number of unique subject-dataset combinations
      num_channels: Number of EEG channels
      trainable_weights: If True, the transformation matrices are learnable parameters.
                        If False, they are fixed (identity initialization). Default: False.

  Input shapes:
      eeg: (batch, num_channels, timepoints) - EEG data
      subject_ids: (batch,) - Integer indices identifying subject-dataset combination,
                   or (batch, num_subjects) - One-hot encoded identifiers

  Output shape:
      (batch, num_channels, timepoints) - Transformed EEG data

  The transformation for sample i is:
      output[i] = W[subject_id[i]] @ input[i]
  where W[j] is a (num_channels, num_channels) matrix and @ is matrix multiplication
  on the channel dimension.
  """

  def __init__(self, num_subjects: int, num_channels: int):
    super().__init__()
    self.num_subjects = num_subjects
    self.num_channels = num_channels

    # Initialize weight matrices as identity (preserves input by default)
    # Shape: (num_subjects, num_channels, num_channels)
    weight_init = torch.eye(num_channels).unsqueeze(0).repeat(num_subjects, 1, 1)

    self.weights = nn.Parameter(weight_init)

  def forward(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
    """Apply subject-specific linear transformation.

    Args:
        eeg: (batch, num_channels, timepoints)
        subject_ids: (batch,) integer indices or (batch, num_subjects) one-hot

    Returns:
        Transformed EEG of shape (batch, num_channels, timepoints)
    """
    # Handle one-hot encoded input
    if subject_ids.dim() == 2:
      # Convert one-hot to indices: (batch, num_subjects) -> (batch,)
      subject_ids = subject_ids.argmax(dim=1)

    # Select weight matrices for this batch: (batch, num_channels, num_channels)
    batch_weights = self.weights[subject_ids]

    # Apply transformation: (batch, num_channels, num_channels) @ (batch, num_channels, timepoints)
    # = (batch, num_channels, timepoints)
    return torch.bmm(batch_weights, eeg)


class SubjectDatasetMapper:
  """Maps (dataset, subject) pairs to unique integer identifiers.

  Since subject IDs may repeat across datasets, we need to combine
  both dataset and subject to create unique identifiers.

  Example:
      >>> mapper = SubjectDatasetMapper()
      >>> mapper.add_subject('bcmi_eeg1', 'S01')
      0
      >>> mapper.add_subject('bcmi_eeg2', 'S01')  # Different dataset, different ID
      1
      >>> mapper.add_subject('bcmi_eeg1', 'S01')  # Same pair, same ID
      0
      >>> mapper.get_id('bcmi_eeg1', 'S01')
      0
      >>> mapper.num_subjects
      2
  """

  def __init__(self):
    self._mapping: dict[tuple[str, str], int] = {}
    self._next_id = 0

  def add_subject(self, dataset: str, subject: str) -> int:
    """Add a (dataset, subject) pair and return its unique ID.

    If the pair already exists, returns the existing ID.

    Args:
        dataset: Dataset identifier
        subject: Subject identifier

    Returns:
        Integer ID for this subject-dataset combination
    """
    key = (dataset, subject)
    if key not in self._mapping:
      self._mapping[key] = self._next_id
      self._next_id += 1
    return self._mapping[key]

  def get_id(self, dataset: str, subject: str) -> int:
    return self._mapping[(dataset, subject)]

  @property
  def num_subjects(self) -> int:
    """Return the total number of unique subject-dataset combinations."""
    return len(self._mapping)

  def get_mapping(self) -> dict[tuple[str, str], int]:
    """Return a copy of the mapping dictionary."""
    return self._mapping.copy()


class HasDatasetSubject(Protocol):
  """Protocol for objects that have dataset and subject attributes."""

  dataset: str
  subject: str


def create_subject_ids_from_trials(
  trials: Sequence[HasDatasetSubject], mapper: Optional[SubjectDatasetMapper] = None
) -> tuple[torch.Tensor, SubjectDatasetMapper]:
  """Create integer subject IDs from a sequence of objects with dataset and subject fields.

  Args:
      trials: Sequence of objects with dataset and subject attributes (e.g., TrialData)
      mapper: Optional existing mapper. If None, creates a new one.

  Returns:
      Tuple of:
          - subject_ids: (len(trials),) tensor of integer IDs
          - mapper: SubjectDatasetMapper containing the mapping

  Example:
      >>> trials = [trial1, trial2, trial3]  # TrialData objects
      >>> subject_ids, mapper = create_subject_ids_from_trials(trials)
      >>> subject_ids.shape
      torch.Size([3])
  """
  if mapper is None:
    mapper = SubjectDatasetMapper()

  ids = [mapper.add_subject(trial.dataset, trial.subject) for trial in trials]
  return torch.tensor(ids, dtype=torch.long), mapper


def create_onehot_from_ids(
  subject_ids: torch.Tensor, num_subjects: int
) -> torch.Tensor:
  """Convert integer subject IDs to one-hot encoding.

  Args:
      subject_ids: (batch,) tensor of integer IDs in range [0, num_subjects)
      num_subjects: Total number of unique subjects

  Returns:
      One-hot encoded tensor of shape (batch, num_subjects)

  Example:
      >>> ids = torch.tensor([0, 2, 1])
      >>> create_onehot_from_ids(ids, num_subjects=3)
      tensor([[1., 0., 0.],
              [0., 0., 1.],
              [0., 1., 0.]])
  """
  return torch.nn.functional.one_hot(subject_ids, num_classes=num_subjects).float()

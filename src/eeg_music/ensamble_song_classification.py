from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from eeg_music.data import (
  ArrayStratifiedSamplingDataset,
  EEGMusicDataset,
  MappedDataset,
  RepeatedDataset,
  StratifiedSamplingDataset,
  TrialData,
  trial_to_arrayeeg,
)
from eeg_music.dataloader import SubjectWiseSplit, TrialWiseSplit


LabelFn = Callable[[TrialData], int]
SplitType = SubjectWiseSplit | TrialWiseSplit
SamplingKind = Literal["array", "raw"]


@dataclass(frozen=True)
class VotingEvaluationResult:
  accuracy: float
  correct: int
  total: int
  predictions: dict[int, int]
  targets: dict[int, int]
  votes: dict[int, list[int]]


@dataclass(frozen=True)
class XGBoostVotingResult:
  snippet_accuracy: float
  song_accuracy: float
  model: XGBClassifier
  voting: VotingEvaluationResult


def song_id_label(trial: TrialData) -> int:
  return trial.music_data.get_music().music_id.song_id - 1


def create_test_set(
  ds: EEGMusicDataset,
  split_type: SplitType = TrialWiseSplit(),
  p_train: float = 0.6,
  p_val: float = 0.2,
  seed: int = 42,
) -> EEGMusicDataset:
  match split_type:
    case SubjectWiseSplit():
      return ds.subject_wise_split(p_train, p_val, seed)["test"]
    case TrialWiseSplit():
      return cast(EEGMusicDataset, ds.trial_wise_split(p_train, p_val, seed)["test"])


def _single_recording_dataset(ds: EEGMusicDataset, index: int) -> EEGMusicDataset:
  single = EEGMusicDataset()
  single.df = ds.df.iloc[[index]].reset_index(drop=True)
  single.music_collection = ds.music_collection
  return single


def _sampling_dataset(
  ds: EEGMusicDataset,
  n_snippets: int,
  trial_length_secs: Fraction,
  kind: SamplingKind,
) -> EEGMusicDataset:
  match kind:
    case "array":
      return ArrayStratifiedSamplingDataset(
        MappedDataset(ds, trial_to_arrayeeg), n_snippets, trial_length_secs
      )
    case "raw":
      return StratifiedSamplingDataset(ds, n_snippets, trial_length_secs)


def _array_sampling_dataset(
  ds: EEGMusicDataset,
  n_snippets: int,
  trial_length_secs: Fraction,
) -> ArrayStratifiedSamplingDataset:
  return ArrayStratifiedSamplingDataset(ds, n_snippets, trial_length_secs)


def create_X_y(dataset: Any, label_fn: LabelFn = song_id_label):
  xs, ys = zip(
    *[
      (cast(Any, dataset[i].eeg_data).get_array().data, label_fn(dataset[i]))
      for i in range(len(dataset))
    ]
  )
  return np.array(xs).reshape(len(xs), -1), np.array(ys)


def train_xgboost(
  train_ds: EEGMusicDataset,
  test_ds: EEGMusicDataset,
  label_fn: LabelFn = song_id_label,
  n_snippets: int = 100,
  trial_length_secs: Fraction = Fraction(3, 1),
  repeated_mul: int = 1,
  random_state: int = 42,
  n_estimators: int = 100,
  max_depth: int = 6,
  learning_rate: float = 0.1,
  n_jobs: int = -1,
) -> tuple[XGBClassifier, float]:
  train_snippets = RepeatedDataset(
    _array_sampling_dataset(train_ds, n_snippets, trial_length_secs), repeated_mul
  )
  test_snippets = RepeatedDataset(
    _array_sampling_dataset(test_ds, n_snippets, trial_length_secs), repeated_mul
  )
  x_train, y_train = create_X_y(train_snippets, label_fn)
  x_test, y_test = create_X_y(test_snippets, label_fn)
  model = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=random_state,
    n_jobs=n_jobs,
  )
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print(f"XGBoost snippet test accuracy: {accuracy_score(y_test, y_pred):.4f}")
  print(classification_report(y_test, y_pred))
  return model, float(accuracy_score(y_test, y_pred))


def voting_accuracy(
  model: XGBClassifier,
  test_ds: EEGMusicDataset,
  label_fn: LabelFn = song_id_label,
  n_snippets: int = 10,
  trial_length_secs: Fraction = Fraction(1, 1),
  seed: int = 42,
  sampling_kind: SamplingKind = "array",
) -> VotingEvaluationResult:
  predictions: dict[int, int] = {}
  targets: dict[int, int] = {}
  votes: dict[int, list[int]] = {}

  for recording_index in range(len(test_ds)):
    np.random.seed(seed + recording_index)
    snippets = _sampling_dataset(
      _single_recording_dataset(test_ds, recording_index),
      n_snippets,
      trial_length_secs,
      sampling_kind,
    )
    x, _ = create_X_y(snippets, label_fn)
    snippet_votes = model.predict(x).tolist()
    votes[recording_index] = snippet_votes
    predictions[recording_index] = Counter(snippet_votes).most_common(1)[0][0]
    targets[recording_index] = label_fn(test_ds[recording_index])

  correct = sum(predictions[i] == targets[i] for i in predictions)
  total = len(predictions)
  result = VotingEvaluationResult(
    accuracy=correct / total if total else 0.0,
    correct=correct,
    total=total,
    predictions=predictions,
    targets=targets,
    votes=votes,
  )
  print(f"XGBoost song voting test accuracy: {result.accuracy:.4f}")
  return result


def train_and_evaluate_xgboost_voting(
  ds: EEGMusicDataset,
  label_fn: LabelFn = song_id_label,
  split_type: SplitType = SubjectWiseSplit(),
  p_train: float = 0.6,
  p_val: float = 0.0,
  seed: int = 42,
  train_snippets: int = 100,
  vote_snippets: int = 100,
  trial_length_secs: Fraction = Fraction(3, 1),
  repeated_mul: int = 1,
  sampling_kind: SamplingKind = "array",
) -> XGBoostVotingResult:
  match split_type:
    case SubjectWiseSplit():
      split = ds.subject_wise_split(p_train, p_val, seed)
    case TrialWiseSplit():
      split = ds.trial_wise_split(p_train, p_val, seed)
  train_ds = cast(EEGMusicDataset, split["train"])
  test_ds = cast(EEGMusicDataset, split["test"])
  model, snippet_accuracy = train_xgboost(
    train_ds=train_ds,
    test_ds=test_ds,
    label_fn=label_fn,
    n_snippets=train_snippets,
    trial_length_secs=trial_length_secs,
    repeated_mul=repeated_mul,
    random_state=seed,
  )
  voting = voting_accuracy(
    model=model,
    test_ds=test_ds,
    label_fn=label_fn,
    n_snippets=vote_snippets,
    trial_length_secs=trial_length_secs,
    seed=seed,
    sampling_kind=sampling_kind,
  )
  return XGBoostVotingResult(
    snippet_accuracy=snippet_accuracy,
    song_accuracy=voting.accuracy,
    model=model,
    voting=voting,
  )


if __name__ == "__main__":
  result = train_and_evaluate_xgboost_voting(
    EEGMusicDataset.load_ondisk(Path("./datasets/musing_preprocessed/musing_pre_60ch")),
    split_type=SubjectWiseSplit(),
    p_train=0.6,
    p_val=0.0,
    seed=42,
    train_snippets=100,
    vote_snippets=100,
    trial_length_secs=Fraction(1, 1),
    # trial_length_secs=Fraction(3, 1),
  )
  print(
    {
      "snippet_accuracy": result.snippet_accuracy,
      "song_accuracy": result.song_accuracy,
    }
  )

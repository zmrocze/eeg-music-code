"""Train XGBoost to predict emotion codes from EEG snippets.

Reuses train_xgboost and voting_accuracy from ensamble_song_classification.
"""

from fractions import Fraction
from pathlib import Path

from eeg_music.data import EEGMusicDataset
from eeg_music.emotion_utils import parse_music_emotion
from eeg_music.ensamble_song_classification import train_xgboost, voting_accuracy


def emotion_label(trial) -> int:
  emotion = parse_music_emotion(trial.music_filename.filename, trial.dataset)
  assert emotion is not None
  return emotion - 1


if __name__ == "__main__":
  ds = EEGMusicDataset.load_ondisk(Path("./datasets/bcmi_preprocessed/bcmi_pre_60ch/"))
  splitted = ds.subject_wise_split(p_train=0.6, p_val=0.0, seed=42)

  train_ds = splitted["train"]
  test_ds = splitted["test"]

  model, snippet_acc = train_xgboost(
    train_ds=train_ds,
    test_ds=test_ds,
    label_fn=emotion_label,
    n_snippets=10,
    trial_length_secs=Fraction(3, 1),
    n_estimators=100,
    max_depth=6,
  )

  voting = voting_accuracy(
    model=model,
    test_ds=test_ds,
    label_fn=emotion_label,
    n_snippets=10,
    trial_length_secs=Fraction(3, 1),
    seed=42,
  )

  print(
    {
      "snippet_accuracy": snippet_acc,
      "recording_voting_accuracy": voting.accuracy,
    }
  )

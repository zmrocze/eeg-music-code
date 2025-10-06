"""Utilities for parsing emotion information from music filenames."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionCode:
  """Emotion code as an integer (1-9)."""

  code: int


def parse_calibration_emotion(filename: str) -> Optional[int]:
  """Parse calibration filename like 'hvla3.wav' -> emotion code.

  Mapping source: BaseBCMILoader.emotional_states in bcmi.py (lines 132-158)
  This follows the standard 9-point valence-arousal grid used in the BCMI datasets,
  based on Russell's circumplex model of affect.

  Filename format: {valence}{arousal}a{variant}.wav
  - valence: hv (high), nv (neutral), lv (low)
  - arousal: ha (high), na (neutral), la (low)

  Returns emotion code (1-9) or None if parsing fails.

  Reference:
  - BCMI-MIdAS dataset (OpenNeuro ds002722 for calibration)
  - Russell, J. A. (1980). A circumplex model of affect. Journal of Personality
    and Social Psychology, 39(6), 1161.
  """
  # Map prefix to emotion code based on 9-point valence-arousal grid
  # AUTHORITATIVE SOURCE: BIDS events.json in bcmi-calibration dataset
  # See: datasets/bcmi/bcmi-calibration/sub-*/eeg/*_events.json
  prefix_map = {
    "lvla": 1,  # Low valence, Low arousal (Sad/Depressed)
    "nvla": 2,  # Neutral valence, Low arousal (Calm/Relaxed)
    "hvla": 3,  # High valence, Low arousal (Peaceful/Content)
    "lvna": 4,  # Low valence, Neutral arousal (Negative/Unpleasant)
    "nvna": 5,  # Neutral valence, Neutral arousal (Neutral/Balanced)
    "hvna": 6,  # High valence, Neutral arousal (Positive/Pleasant)
    "lvha": 7,  # Low valence, High arousal (Angry/Agitated)
    "nvha": 8,  # Neutral valence, High arousal (Alert/Activated)
    "hvha": 9,  # High valence, High arousal (Excited/Happy)
  }

  # Match patterns like hvla3.wav, hvlaa3.wav, hvla12.wav
  match = re.match(r"^([hlnv]{2}[hln]a)a?\d+\.wav$", filename)
  if match:
    prefix = match.group(1)
    return prefix_map.get(prefix)
  return None


def parse_training_emotion(filename: str) -> Optional[int]:
  """Parse training filename like '1-6_3_first.wav' or '1-6_3_second.wav.npz'.

  For first half: returns emotion_code_1 (the first number)
  For second half: returns emotion_code_2 (the second number)

  Returns emotion code or None if parsing fails.
  """
  # Handle both .wav and .wav.npz extensions (for preprocessed files)
  match = re.match(r"^(\d)-(\d)_\d+_(first|second)\.wav(\.npz)?$", filename)
  if match:
    code1, code2, which_half = int(match.group(1)), int(match.group(2)), match.group(3)
    return code1 if which_half == "first" else code2
  return None


def parse_music_emotion(filename: str, dataset: str) -> Optional[int]:
  """Parse emotion code from music filename based on dataset type.

  Returns:
      Integer emotion code (1-9), or None if parsing fails
  """
  if "calibration" in dataset.lower():
    return parse_calibration_emotion(filename)
  elif "training" in dataset.lower():
    return parse_training_emotion(filename)
  return None

"""Conversion functions from TrialData with audio to TrialData with NoteOnsets."""

from eeg_music.data import WavRAW, MelRaw, NoteOnsets, TrialData, EegData, MusicData
from eeg_music.onset_markers import detect_onsets_wavraw, detect_onsets_melraw


def trial_wavraw_to_noteonsets(
  trial: TrialData[EegData, MusicData],
  method: str = "hfc",
  frame_size: int = 1024,
  hop_size: int = 512,
) -> TrialData[EegData, NoteOnsets]:
  """Convert a trial with WavRAW music to a trial with NoteOnsets.

  Args:
      trial: TrialData with WavRAW music data
      method: onset detection method ('hfc', 'complex', 'flux', 'melflux', 'rms', etc.)
      frame_size: size of analysis frame in samples
      hop_size: hop size between frames in samples

  Returns:
      TrialData with NoteOnsets music data
  """
  music = trial.music_data.get_music()
  if not isinstance(music, WavRAW):
    raise TypeError(f"Expected WavRAW, got {type(music).__name__}")

  onset_times = detect_onsets_wavraw(
    music, method=method, frame_size=frame_size, hop_size=hop_size
  )

  note_onsets = NoteOnsets(
    onset_times=onset_times,
    sample_rate=music.sample_rate,
    duration_seconds=music.length_seconds(),
  )

  return TrialData(
    dataset=trial.dataset,
    subject=trial.subject,
    session=trial.session,
    run=trial.run,
    trial_id=trial.trial_id,
    music_filename=trial.music_filename,
    eeg_data=trial.eeg_data,
    music_data=note_onsets,
  )


def trial_melraw_to_noteonsets(
  trial: TrialData[EegData, MusicData],
) -> TrialData[EegData, NoteOnsets]:
  """Convert a trial with MelRaw music to a trial with NoteOnsets.

  Args:
      trial: TrialData with MelRaw music data

  Returns:
      TrialData with NoteOnsets music data
  """
  music = trial.music_data.get_music()
  if not isinstance(music, MelRaw):
    raise TypeError(f"Expected MelRaw, got {type(music).__name__}")

  onset_times = detect_onsets_melraw(music)

  note_onsets = NoteOnsets(
    onset_times=onset_times,
    sample_rate=music.sample_rate,
    duration_seconds=music.length_seconds(),
  )

  return TrialData(
    dataset=trial.dataset,
    subject=trial.subject,
    session=trial.session,
    run=trial.run,
    trial_id=trial.trial_id,
    music_filename=trial.music_filename,
    eeg_data=trial.eeg_data,
    music_data=note_onsets,
  )

"""Onset detection functions for music analysis."""

import numpy as np
from numpy.typing import NDArray
import essentia
from essentia.standard import (
  OnsetDetection,
  Windowing,
  FFT,
  CartesianToPolar,
  FrameGenerator,
)
from typing import Tuple, TypeVar
from eeg_music.data import WavRAW, MelRaw, TrialData, EegData


def compute_odf(
  wav: WavRAW,
  method: str = "hfc",
  frame_size: int = 1024,
  hop_size: int = 512,
) -> Tuple[NDArray[np.floating], int, int]:
  """Compute Onset Detection Function from WavRAW.

  Args:
      wav: WavRAW object containing audio data
      method: Onset detection method (default: "hfc" - high frequency content)
      frame_size: Frame size in samples (default: 1024)
      hop_size: Hop size in samples (default: 512)

  Returns:
      Tuple of (odf_array, sample_rate, hop_size)
      - odf_array: 1D array of ODF values
      - sample_rate: Original audio sample rate
      - hop_size: Hop size used for computation
  """
  # Convert to mono if needed
  audio_data = wav.raw_data if wav.raw_data.ndim == 1 else np.mean(wav.raw_data, axis=1)

  # Initialize Essentia algorithms
  od = OnsetDetection(method=method, sampleRate=float(wav.sample_rate))
  w = Windowing(type="hann", size=frame_size, normalized=True, zeroPhase=True)
  fft = FFT(size=frame_size)
  c2p = CartesianToPolar()

  # Compute ODF frame by frame
  pool = essentia.Pool()
  for frame in FrameGenerator(audio_data, frameSize=frame_size, hopSize=hop_size):
    windowed = w(frame)
    spectrum = fft(windowed)
    magnitude, phase = c2p(spectrum)
    odf_value = od(magnitude, phase)
    pool.add("odf", odf_value)

  odf_array = np.array(pool["odf"])
  return odf_array, wav.sample_rate, hop_size


E = TypeVar("E", bound=EegData)


def trial_to_odf(
  trial: TrialData[E, WavRAW],
  method: str = "hfc",
  frame_size: int = 1024,
  hop_size: int = 512,
) -> TrialData[E, MelRaw]:
  """Convert trial with WavRAW to trial with MelRaw (ODF).

  Args:
      trial: TrialData object with music_data as WavRAW
      method: Onset detection method (default: "hfc")
      frame_size: Frame size in samples (default: 1024)
      hop_size: Hop size in samples (default: 512)

  Returns:
      New TrialData object with music_data as MelRaw containing ODF
  """
  # Get the actual WavRAW data
  music_data = trial.music_data.get_music()

  if not isinstance(music_data, WavRAW):
    raise TypeError(f"Expected WavRAW, got {type(music_data)}")

  # Compute ODF
  odf_array, sample_rate, hop_length = compute_odf(
    music_data,
    method=method,
    frame_size=frame_size,
    hop_size=hop_size,
  )

  # Create MelRaw object (repurposed for ODF)
  # ODF is 1D, so we reshape to (1, n_frames) to match mel format
  odf_melraw = MelRaw(
    mel=odf_array.reshape(1, -1),  # (1, n_frames)
    sample_rate=sample_rate,
    hop_length=hop_length,
    fmin=0.0,  # Not applicable for ODF
    fmax=None,  # Not applicable for ODF
    to_db=False,  # ODF values are not in dB
  )

  # Create new trial with ODF data
  return TrialData(
    dataset=trial.dataset,
    subject=trial.subject,
    session=trial.session,
    run=trial.run,
    trial_id=trial.trial_id,
    music_filename=trial.music_filename,
    eeg_data=trial.eeg_data,
    music_data=odf_melraw,
  )

"""Onset detection functions using Essentia library."""

import numpy as np
from essentia.standard import (  # pyright: ignore
  OnsetDetection,  # pyright: ignore
  Onsets,  # pyright: ignore
  Windowing,  # pyright: ignore
  FFT,  # pyright: ignore
  CartesianToPolar,  # pyright: ignore
  FrameGenerator,  # pyright: ignore
)
import essentia

from eeg_music.data import WavRAW, MelRaw


def detect_onsets_wavraw(
  wav: WavRAW,
  method: str = "hfc",
  frame_size: int = 1024,
  hop_size: int = 512,
) -> np.ndarray:
  """Detect onset times from WavRAW audio data.

  Args:
      wav: WavRAW audio data
      method: onset detection method ('hfc', 'complex', 'flux', 'melflux', 'rms', etc.)
      frame_size: size of analysis frame in samples
      hop_size: hop size between frames in samples

  Returns:
      numpy array of onset times in seconds
  """

  # ensure 1d wav array, mean matches plots of wavraw_to_melspectrogram
  wav.raw_data = (
    wav.raw_data if wav.raw_data.ndim == 1 else np.mean(wav.raw_data, axis=1)
  )

  # Initialize onset detection algorithm
  # Convert sample_rate to float for Essentia compatibility
  od = OnsetDetection(method=method, sampleRate=float(wav.sample_rate))

  # Auxiliary algorithms for spectral analysis
  w = Windowing(type="hann", size=frame_size, normalized=True, zeroPhase=True)
  fft = FFT(size=frame_size)  # Pass size for optimization
  c2p = CartesianToPolar()

  # Compute ODF frame by frame
  pool = essentia.Pool()
  for frame in FrameGenerator(wav.raw_data, frameSize=frame_size, hopSize=hop_size):
    magnitude, phase = c2p(fft(w(frame)))
    pool.add("odf", od(magnitude, phase))

  # Detect onset locations from ODF
  # frameRate = frames per second = sample_rate / hop_size
  frame_rate = float(wav.sample_rate) / float(hop_size)
  onsets_algo = Onsets(frameRate=frame_rate)
  onset_times = onsets_algo(
    essentia.array([pool["odf"]]),
    [1],  # weight (doesn't matter for single ODF)
  )

  return onset_times


def detect_onsets_melraw(
  mel: MelRaw,
) -> np.ndarray:
  """Detect onset times from MelRaw spectrogram data.

  Note: This function reconstructs approximate audio from mel spectrogram
  using Griffin-Lim algorithm, then performs onset detection.
  For more accurate onset detection, use detect_onsets_wavraw on original audio.

  Args:
      mel: MelRaw spectrogram data
      method: onset detection method ('hfc', 'complex', 'flux', 'melflux', 'rms', etc.)
      frame_size: size of analysis frame in samples
      hop_size: hop size between frames in samples

  Returns:
      numpy array of onset times in seconds
  """

  raise NotImplementedError("detect_onsets_melraw is not implemented")

  # For mel spectrograms, we need to work with the spectrogram directly
  # or reconstruct audio. Since Essentia's OnsetDetection works on magnitude/phase,
  # we'll use a simpler approach: detect onsets from spectral flux in mel domain

  # Alternative: Use OnsetDetectionGlobal which can work on features
  # For now, we compute a simple spectral flux from mel frames

  mel_data = mel.mel  # shape: (n_mels, n_frames)
  n_frames = mel_data.shape[1]

  # Compute onset detection function manually from mel spectrogram
  # Simple spectral flux: sum of positive differences between consecutive frames
  odf = np.zeros(n_frames)
  for i in range(1, n_frames):
    diff = mel_data[:, i] - mel_data[:, i - 1]
    odf[i] = np.sum(np.maximum(diff, 0))

  # Detect onsets from ODF using the mel's actual sample rate and hop length
  # frameRate = frames per second = sample_rate / hop_length
  frame_rate = float(mel.sample_rate) / float(mel.hop_length)
  onsets_algo = Onsets(frameRate=frame_rate)
  onset_times = onsets_algo(essentia.array([odf]), [1])

  return onset_times

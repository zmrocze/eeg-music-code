"""Frequency band power feature extraction for EEG data."""

from dataclasses import dataclass
from typing import List, Tuple, Callable, TypeVar
import numpy as np
from numpy.typing import NDArray

from eeg_music.data import EegData, TrialData, ArrayEeg, MusicData
from scipy import signal

# Define type variable for MusicData subclass
MusicType = TypeVar("MusicType", bound=MusicData)


@dataclass
class BandPowerParams:
  """Parameters for band power calculation.

  Args:
      frequency_bands: List of (low_freq, high_freq) tuples in Hz
      window_width: Width of the window for power calculation in samples
      hop_length: Distance between consecutive windows in samples
  """

  frequency_bands: List[Tuple[float, float]]
  window_width: int
  hop_length: int


def bandpass_filter(
  data: NDArray[np.floating],
  low_freq: float,
  high_freq: float,
  sfreq: float,
  order: int = 5,
) -> NDArray[np.floating]:
  """
  Apply a bandpass filter to the input data using a Butterworth filter.

  Parameters
  ----------
  data : NDArray[np.floating]
      Input signal array. Can be 1D (single channel) or 2D (channels x samples).
  low_freq : float
  high_freq : float
  sfreq : float
      Sampling frequency in Hz.
  order : int, optional
      Filter order (default is 5).

  Returns
  -------
  NDArray[np.floating]
      Filtered signal with the same shape as input.

  """
  # Design Butterworth bandpass filter in SOS format
  sos = signal.butter(
    N=order, Wn=[low_freq, high_freq], btype="bandpass", fs=sfreq, output="sos"
  )

  # Apply zero-phase filtering
  filtered_data = signal.sosfiltfilt(sos, data, axis=-1)

  return filtered_data


def calculate_windowed_power(
  data: NDArray[np.floating], window_width: int, hop_length: int
) -> NDArray[np.floating]:
  """Calculate power (mean of squared signal) in sliding windows.

  Args:
      data: Input signal, shape (n_channels, n_samples)
      window_width: Width of each window in samples
      hop_length: Distance between consecutive window starts in samples

  Returns:
      Power values, shape (n_channels, n_windows)
  """
  n_channels, n_samples = data.shape
  n_windows = (n_samples - window_width) // hop_length + 1

  power = np.zeros((n_channels, n_windows), dtype=data.dtype)

  for i in range(n_windows):
    start = i * hop_length
    end = start + window_width
    window = data[:, start:end]
    power[:, i] = np.mean(window**2, axis=1)

  return power


def eeg_to_band_power(
  eeg_data: NDArray[np.floating], sfreq: float, params: BandPowerParams
) -> NDArray[np.floating]:
  """Transform EEG data to band power features.

  For each frequency band:
  1. Apply bandpass filter to get filtered signal
  2. Calculate windowed power (mean of squared signal)

  Args:
      eeg_data: EEG signal, shape (n_channels, n_samples)
      sfreq: Sampling frequency in Hz
      params: Band power calculation parameters

  Returns:
      Band power features, shape (n_channels, n_bands, n_windows)
  """
  n_channels = eeg_data.shape[0]
  n_bands = len(params.frequency_bands)

  # Calculate power for first band to get n_windows
  low, high = params.frequency_bands[0]
  filtered = bandpass_filter(eeg_data, low, high, sfreq)
  power_first = calculate_windowed_power(
    filtered, params.window_width, params.hop_length
  )
  n_windows = power_first.shape[1]

  # Allocate output array
  band_power = np.zeros((n_channels, n_bands, n_windows), dtype=eeg_data.dtype)
  band_power[:, 0, :] = power_first

  # Process remaining bands
  for band_idx, (low, high) in enumerate(params.frequency_bands[1:], start=1):
    filtered = bandpass_filter(eeg_data, low, high, sfreq)
    band_power[:, band_idx, :] = calculate_windowed_power(
      filtered, params.window_width, params.hop_length
    )

  return band_power


def trial_to_band_power(
  params: BandPowerParams,
) -> Callable[[TrialData[EegData, MusicType]], TrialData[ArrayEeg, MusicType]]:
  """Create a transform function that converts TrialData to band power features.

  The returned function extracts EEG data from a trial, computes band power features,
  and returns a new trial with the transformed EEG data as ArrayEeg.

  Args:
      params: Band power calculation parameters

  Returns:
      Transform function with signature: TrialData[EegData, MusicType] -> TrialData[ArrayEeg, MusicType]
  """

  def transform(trial: TrialData[EegData, MusicType]) -> TrialData[ArrayEeg, MusicType]:
    """Transform trial EEG data to band power features.

    Args:
        trial: Input trial with any EegData type

    Returns:
        New trial with ArrayEeg containing band power features.
        Shape: (n_channels, n_bands, n_windows)
    """
    # Get EEG as array
    eeg_data = trial.eeg_data
    if hasattr(eeg_data, "get_array"):
      array_eeg = getattr(eeg_data, "get_array")()  # type: ignore
    else:
      # Fall back to RawEeg
      raw_eeg = eeg_data.get_eeg()
      array_eeg = ArrayEeg(
        data=raw_eeg.raw_eeg.get_data().astype(np.float32),  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        ch_names=raw_eeg.raw_eeg.ch_names,
        sfreq=raw_eeg.raw_eeg.info["sfreq"],
      )

    # Calculate band power
    band_power = eeg_to_band_power(array_eeg.data, array_eeg.sfreq, params)

    # Create new channel names for band power features
    n_bands = len(params.frequency_bands)
    new_ch_names = [
      f"{ch}_band{band_idx}" for ch in array_eeg.ch_names for band_idx in range(n_bands)
    ]

    # Reshape to (n_channels * n_bands, n_windows)
    n_channels, n_bands, n_windows = band_power.shape
    band_power_2d = band_power.reshape(n_channels * n_bands, n_windows)

    # Create new ArrayEeg with band power features
    new_eeg = ArrayEeg(
      data=band_power_2d.astype(np.float32),
      ch_names=new_ch_names,
      sfreq=array_eeg.sfreq
      / params.hop_length,  # Effective sampling rate after windowing
    )

    # Return new trial with transformed EEG
    return TrialData(
      dataset=trial.dataset,
      subject=trial.subject,
      session=trial.session,
      run=trial.run,
      trial_id=trial.trial_id,
      music_filename=trial.music_filename,
      eeg_data=new_eeg,
      music_data=trial.music_data,
    )

  return transform

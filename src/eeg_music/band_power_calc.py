import mne
import numpy as np
from mne.time_frequency import psd_array_multitaper, psd_array_welch
from numpy.typing import NDArray


def calculate_band_power(
  raw: mne.io.Raw,
  bands: list[tuple[str, float, float]],
  method: str = "welch",
  **kwargs,
) -> dict[str, NDArray[np.float64]]:
  """
  Calculate band power for given EEG signal using MNE-Python.

  Parameters
  ----------
  raw : mne.io.Raw
      MNE Raw object containing EEG data
  bands : list[tuple[str, float, float]]
      List of frequency bands as (name, fmin, fmax) tuples.
      Example: [('delta', 0.5, 4), ('theta', 4, 8), ('alpha', 8, 13)]
  method : str, default='welch'
      Method for computing PSD. Options: 'welch', 'multitaper'
  **kwargs
      Additional arguments passed to mne.time_frequency.psd_array_welch or
      mne.time_frequency.psd_array_multitaper

  Returns
  -------
  dict[str, NDArray[np.float64]]
      Dictionary mapping band names to power arrays of shape (n_channels,)
  """
  data = raw.get_data()
  sfreq = raw.info["sfreq"]

  if method == "welch":
    psds, freqs = psd_array_welch(data, sfreq=sfreq, **kwargs)
  elif method == "multitaper":
    result = psd_array_multitaper(data, sfreq=sfreq, **kwargs)
    psds, freqs = result[0], result[1]  # Handle 2 or 3 element tuple
  else:
    raise ValueError(f"Unknown method: {method}. Use 'welch' or 'multitaper'")

  band_powers = {}
  for name, fmin, fmax in bands:
    freq_mask = (freqs >= fmin) & (freqs < fmax)
    band_powers[name] = np.mean(psds[:, freq_mask], axis=1)

  return band_powers

from dataclasses import replace
from typing import Literal

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from mne_icalabel import label_components
from numpy.typing import NDArray

from eeg_music.data import ArrayEeg, TrialData

Normalization = Literal["minmax", "std"] | None


def _normalize_band_power(
  bp: NDArray[np.float64], method: Normalization
) -> NDArray[np.float64]:
  """Normalize band power array of shape (num_bands, n_components, num_windows) per band."""
  if method is None:
    return bp
  for bi in range(bp.shape[0]):
    bmin = bp[bi].min()
    if method == "minmax":
      bmax = bp[bi].max()
      bp[bi] = (bp[bi] - bmin) / (bmax - bmin) if bmax > bmin else 0.0
    elif method == "std":
      mean, std = bp[bi].mean(), bp[bi].std()
      bp[bi] = (bp[bi] - mean) / std if std > 0 else 0.0
  return bp


def apply_ica(
  raw: mne.io.BaseRaw, n_components: int | None = None, random_state: int = 42
) -> tuple[ICA, mne.io.BaseRaw]:
  """Fit ICA on raw EEG and return (ica object, raw of ICA source activations)."""
  ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
  ica.fit(raw)
  sources = ica.get_sources(raw)
  assert isinstance(sources, mne.io.BaseRaw)
  return ica, sources


def clean_ica_artifacts(
  ica: ICA,
  raw: mne.io.BaseRaw,
  keep_labels: set[str] = {"brain", "other"},
) -> mne.io.BaseRaw:
  """Label ICA components with ICLabel, exclude non-brain artifacts, and return cleaned raw.

  Parameters
  ----------
  ica : fitted ICA object
  raw : the raw EEG used for ICA fitting
  keep_labels : set of ICLabel categories to keep (everything else is excluded)

  Returns
  -------
  cleaned : Raw with artifact components removed
  """
  labels = label_components(raw, ica, method="iclabel")
  component_labels = labels["labels"]

  exclude = [i for i, lbl in enumerate(component_labels) if lbl not in keep_labels]
  #   excluded_labels = [component_labels[i] for i in exclude]

  ica.exclude = exclude
  return ica.apply(raw.copy())


def windowed_band_power(
  raw: mne.io.BaseRaw,
  bands: list[tuple[float, float]],
  window_sec: float = 1.0,
  hop_sec: float = 0.5,
) -> tuple[NDArray[np.float64], list[str]]:
  """Compute band power in sliding windows.

  Returns
  -------
  result : ndarray, shape (num_bands, num_channels, num_windows)
  band_names : list of str describing each band
  """
  sfreq = raw.info["sfreq"]
  data = raw.get_data()  # type: ignore[reportAttributeAccessIssue]
  win_samples = int(window_sec * sfreq)
  hop_samples = int(hop_sec * sfreq)
  n_channels, n_times = data.shape  # type: ignore[reportAttributeAccessIssue]
  starts = np.arange(0, n_times - win_samples + 1, hop_samples)

  band_names = [f"{lo}-{hi} Hz" for lo, hi in bands]
  result = np.empty((len(bands), n_channels, len(starts)))

  for wi, s in enumerate(starts):
    segment = data[:, s : s + win_samples]  # type: ignore[reportCallIssue, reportArgumentType]
    psds, freqs = psd_array_welch(segment, sfreq=sfreq, verbose=False)
    assert isinstance(psds, np.ndarray) and isinstance(freqs, np.ndarray)
    for bi, (lo, hi) in enumerate(bands):
      mask = (freqs >= lo) & (freqs < hi)
      result[bi, :, wi] = np.mean(psds[:, mask], axis=1)

  return result, band_names


def ica_band_power_trial(
  trial: TrialData,
  n_components: int = 20,
  bands: list[tuple[float, float]] | None = None,
  window_sec: float = 2.0,
  hop_sec: float = 1.0,
  l_freq: float = 1.0,
  h_freq: float = 50.0,
  keep_labels: set[str] = {"brain", "other"},
  apply_normalization: Normalization = "std",
) -> TrialData:
  """Apply ICA artifact cleaning + band power to a trial, returning ArrayEeg output.

  Pipeline: filter → set montage → ICA → ICLabel cleanup → band power on cleaned sources.

  The resulting ArrayEeg has shape (num_bands * n_kept_components, num_windows)
  with channel names like "delta_IC0", "delta_IC1", ..., "gamma_IC19".

  Parameters
  ----------
  trial : TrialData with RawEeg or loadable eeg_data
  n_components : number of ICA components
  bands : frequency band edges, defaults to delta/theta/alpha/beta/gamma
  window_sec : band power window width in seconds
  hop_sec : band power hop in seconds
  l_freq, h_freq : bandpass filter edges
  keep_labels : ICLabel categories to keep

  Returns
  -------
  TrialData with ArrayEeg eeg_data, music_data unchanged
  """
  if bands is None:
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
  band_names_short = ["delta", "theta", "alpha", "beta", "gamma"]
  if len(band_names_short) != len(bands):
    band_names_short = [f"{lo}-{hi}" for lo, hi in bands]

  raw = trial.eeg_data.get_eeg().raw_eeg.copy()
  raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
  if "E129" in raw.ch_names:
    raw.rename_channels({"E129": "Cz"})
  raw.set_montage(
    mne.channels.make_standard_montage("GSN-HydroCel-129"), on_missing="warn"
  )

  ica, _ = apply_ica(raw, n_components=n_components)
  cleaned = clean_ica_artifacts(ica, raw, keep_labels=keep_labels)
  cleaned_sources = ica.get_sources(cleaned)
  assert isinstance(cleaned_sources, mne.io.BaseRaw)

  bp, _ = windowed_band_power(
    cleaned_sources, bands=bands, window_sec=window_sec, hop_sec=hop_sec
  )
  # bp shape: (num_bands, n_kept_components, num_windows)
  _normalize_band_power(bp, apply_normalization)

  num_bands, n_comp, num_windows = bp.shape
  flat = bp.reshape(num_bands * n_comp, num_windows).astype(np.float32)

  ch_names = [f"{bname}_IC{ic}" for bname in band_names_short for ic in range(n_comp)]

  return replace(
    trial,
    eeg_data=ArrayEeg(data=flat, ch_names=ch_names, sfreq=1.0 / hop_sec),
  )


_NON_EEG_CHANNELS = frozenset({"GSR", "ECG", "VA1", "VA2", "VAtarg"})


def _prepare_raw_1020(
  raw: mne.io.BaseRaw, l_freq: float, h_freq: float
) -> mne.io.BaseRaw:
  """Filter, drop non-EEG channels, fix casing, and set standard_1020 montage."""
  raw = raw.copy()
  raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
  raw.drop_channels([ch for ch in raw.ch_names if ch in _NON_EEG_CHANNELS])
  # BCMI training uses FP1/FPz but standard_1020 expects Fp1/Fpz
  montage = mne.channels.make_standard_montage("standard_1020")
  montage_upper = {name.upper(): name for name in montage.ch_names}
  raw.rename_channels(
    {
      ch: montage_upper[ch.upper()]
      for ch in raw.ch_names
      if ch.upper() in montage_upper and ch != montage_upper[ch.upper()]
    }
  )
  raw.set_montage(montage, on_missing="ignore")
  return raw


def ica_band_power_trial_1020(
  trial: TrialData,
  n_components: int = 20,
  bands: list[tuple[float, float]] | None = None,
  window_sec: float = 2.0,
  hop_sec: float = 1.0,
  l_freq: float = 1.0,
  h_freq: float = 50.0,
  keep_labels: set[str] = {"brain", "other"},
  apply_normalization: Normalization = "std",
) -> TrialData:
  """Like ica_band_power_trial but for datasets using standard 10-20 channel names.

  Handles BCMI training data: drops non-EEG channels (GSR, ECG, VA*),
  fixes casing (FP1→Fp1), and uses standard_1020 montage.
  """
  if bands is None:
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
  band_names_short = (
    ["delta", "theta", "alpha", "beta", "gamma"]
    if len(bands) == 5
    else [f"{lo}-{hi}" for lo, hi in bands]
  )

  raw = _prepare_raw_1020(trial.eeg_data.get_eeg().raw_eeg, l_freq, h_freq)

  ica, _ = apply_ica(raw, n_components=n_components)
  cleaned = clean_ica_artifacts(ica, raw, keep_labels=keep_labels)
  cleaned_sources = ica.get_sources(cleaned)
  assert isinstance(cleaned_sources, mne.io.BaseRaw)

  bp, _ = windowed_band_power(
    cleaned_sources, bands=bands, window_sec=window_sec, hop_sec=hop_sec
  )
  _normalize_band_power(bp, apply_normalization)

  num_bands, n_comp, num_windows = bp.shape
  flat = bp.reshape(num_bands * n_comp, num_windows).astype(np.float32)
  ch_names = [f"{bname}_IC{ic}" for bname in band_names_short for ic in range(n_comp)]

  return replace(
    trial,
    eeg_data=ArrayEeg(data=flat, ch_names=ch_names, sfreq=1.0 / hop_sec),
  )

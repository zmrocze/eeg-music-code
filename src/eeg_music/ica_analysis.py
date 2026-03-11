from dataclasses import replace
from typing import Literal

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from mne_icalabel import label_components
from numpy.typing import NDArray

from eeg_music.data import ArrayEeg, RawEeg, TrialData

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


def label_ica_components(
  ica: ICA,
  raw: mne.io.BaseRaw,
  keep_labels: set[str] = {"brain", "other"},
) -> list[int]:
  """Label ICA components with ICLabel and return indices to exclude.

  Parameters
  ----------
  ica : fitted ICA object
  raw : the raw EEG used for ICA fitting
  keep_labels : set of ICLabel categories to keep (everything else is excluded)

  Returns
  -------
  exclude : list of component indices classified as artifacts
  """
  component_labels = label_components(raw, ica, method="iclabel")["labels"]
  return [i for i, lbl in enumerate(component_labels) if lbl not in keep_labels]


def clean_ica_artifacts(
  ica: ICA,
  raw: mne.io.BaseRaw,
  keep_labels: set[str] = {"brain", "other"},
) -> mne.io.BaseRaw:
  """Label ICA components with ICLabel, exclude non-brain artifacts, and return cleaned raw."""
  ica.exclude = label_ica_components(ica, raw, keep_labels)
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


def ica_clean_trial(
  trial: TrialData,
  raw: mne.io.BaseRaw,
  return_n_components: int,
  keep_labels: set[str],
) -> TrialData:
  """Fit ICA on all channels, label via ICLabel, return top non-artifact sources.

  Fits ICA with n_components = number of EEG channels (full decomposition).
  Then walks components in explained-variance order, skipping artifacts,
  collecting the first ``return_n_components`` non-artifact sources.
  If fewer non-artifact components exist, remaining channels are zero-filled.

  Returns a TrialData whose eeg_data is a RawEeg with exactly
  ``return_n_components`` channels.
  """
  ica, sources = apply_ica(raw, n_components=return_n_components + 10)
  exclude = set(label_ica_components(ica, raw, keep_labels=keep_labels))

  kept: list[int] = []
  for i in range(ica.n_components_):
    if i not in exclude:
      kept.append(i)
    if len(kept) == return_n_components:
      break

  picked = sources.copy().pick([sources.ch_names[i] for i in kept])

  n_missing = return_n_components - len(kept)
  if n_missing > 0:
    zero_data = np.zeros((n_missing, picked.n_times))
    zero_names = [f"zero{i}" for i in range(n_missing)]
    zero_info = mne.create_info(zero_names, sfreq=picked.info["sfreq"], ch_types="misc")
    zero_raw = mne.io.RawArray(zero_data, zero_info, verbose=False)
    picked = picked.add_channels([zero_raw], force_update_info=True)

  assert isinstance(picked, mne.io.BaseRaw)
  return replace(trial, eeg_data=RawEeg(picked))


_DEFAULT_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _default_band_names(bands: list[tuple[float, float]]) -> list[str]:
  return (
    ["delta", "theta", "alpha", "beta", "gamma"]
    if len(bands) == 5
    else [f"{lo}-{hi}" for lo, hi in bands]
  )


def band_power_trial(
  trial: TrialData,
  bands: list[tuple[float, float]] | None = None,
  window_sec: float = 2.0,
  hop_sec: float = 1.0,
  apply_normalization: Normalization = "std",
) -> TrialData:
  """Compute band power on a trial's raw EEG, returning ArrayEeg output.

  Reads channel names from the raw object to construct output names like
  "delta_ICA000", "alpha_ICA003", preserving original channel identity.
  """
  if bands is None:
    bands = _DEFAULT_BANDS
  band_names_short = _default_band_names(bands)

  raw = trial.eeg_data.get_eeg().raw_eeg
  bp, _ = windowed_band_power(raw, bands=bands, window_sec=window_sec, hop_sec=hop_sec)
  _normalize_band_power(bp, apply_normalization)

  num_bands, n_comp, num_windows = bp.shape
  flat = bp.reshape(num_bands * n_comp, num_windows).astype(np.float32)
  ch_names = [f"{bname}_{ch}" for bname in band_names_short for ch in raw.ch_names]

  return replace(
    trial, eeg_data=ArrayEeg(data=flat, ch_names=ch_names, sfreq=1.0 / hop_sec)
  )


# -- Montage-specific preparation ---------------------------------------------------


def _prepare_raw_hydrocel129(
  raw: mne.io.BaseRaw, l_freq: float | None, h_freq: float | None
) -> mne.io.BaseRaw:
  """Filter, rename E129→Cz, and set GSN-HydroCel-129 montage."""
  raw = raw.copy()
  if l_freq is not None or h_freq is not None:
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
  if "E129" in raw.ch_names:
    raw.rename_channels({"E129": "Cz"})
  raw.set_montage(
    mne.channels.make_standard_montage("GSN-HydroCel-129"), on_missing="warn"
  )
  return raw


_NON_EEG_CHANNELS = frozenset({"GSR", "ECG", "VA1", "VA2", "VAtarg"})


def _prepare_raw_1020(
  raw: mne.io.BaseRaw, l_freq: float | None, h_freq: float | None
) -> mne.io.BaseRaw:
  """Filter, drop non-EEG channels, fix casing, and set standard_1020 montage."""
  raw = raw.copy()
  if l_freq is not None or h_freq is not None:
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


# -- Montage-specific ICA cleaning --------------------------------------------------


def ica_clean_trial_hydrocel129(
  trial: TrialData,
  return_n_components: int = 20,
  l_freq: float | None = 1.0,
  h_freq: float | None = 50.0,
  keep_labels: set[str] = {"brain", "other"},
) -> TrialData:
  """Prepare raw with GSN-HydroCel-129 montage, then ICA-clean."""
  raw = _prepare_raw_hydrocel129(trial.eeg_data.get_eeg().raw_eeg, l_freq, h_freq)
  return ica_clean_trial(trial, raw, return_n_components, keep_labels)


def ica_clean_trial_1020(
  trial: TrialData,
  return_n_components: int = 20,
  l_freq: float | None = 1.0,
  h_freq: float | None = 50.0,
  keep_labels: set[str] = {"brain", "other"},
) -> TrialData:
  """Prepare raw with standard 10-20 montage, then ICA-clean."""
  raw = _prepare_raw_1020(trial.eeg_data.get_eeg().raw_eeg, l_freq, h_freq)
  return ica_clean_trial(trial, raw, return_n_components, keep_labels)


# -- Top-level entrypoints ----------------------------------------------------------


def ica_band_power_trial(
  trial: TrialData,
  n_components: int = 20,
  bands: list[tuple[float, float]] | None = None,
  window_sec: float = 2.0,
  hop_sec: float = 0.1,
  l_freq: float | None = 1.0,
  h_freq: float | None = 50.0,
  keep_labels: set[str] = {"brain", "other"},
  apply_normalization: Normalization = "std",
) -> TrialData:
  """ICA artifact cleaning + band power for GSN-HydroCel-129 montage data.

  Pipeline: prepare raw → ICA clean → band power.
  """
  cleaned = ica_clean_trial_hydrocel129(
    trial, n_components, l_freq, h_freq, keep_labels
  )
  return band_power_trial(
    cleaned,
    bands=bands,
    window_sec=window_sec,
    hop_sec=hop_sec,
    apply_normalization=apply_normalization,
  )


def ica_band_power_trial_1020(
  trial: TrialData,
  n_components: int = 20,
  bands: list[tuple[float, float]] | None = None,
  window_sec: float = 2.0,
  hop_sec: float = 1.0,
  l_freq: float | None = 1.0,
  h_freq: float | None = 50.0,
  keep_labels: set[str] = {"brain", "other"},
  apply_normalization: Normalization = "std",
) -> TrialData:
  """ICA artifact cleaning + band power for standard 10-20 montage data.

  Handles BCMI training data: drops non-EEG channels (GSR, ECG, VA*),
  fixes casing (FP1→Fp1), and uses standard_1020 montage.
  """
  cleaned = ica_clean_trial_1020(trial, n_components, l_freq, h_freq, keep_labels)
  return band_power_trial(
    cleaned,
    bands=bands,
    window_sec=window_sec,
    hop_sec=hop_sec,
    apply_normalization=apply_normalization,
  )

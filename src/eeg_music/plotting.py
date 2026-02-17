"""Plotting utilities for EEG and music data visualization."""

from typing import Union, Dict, Any
import matplotlib.figure as mfig
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .data import (
  TrialData,
  RawEeg,
  WavRAW,
  MelRaw,
  NoteOnsets,
  ArrayEeg,
  melspectrogram_figure,
  mkplot_melspectrogram,
)


@dataclass
class TrialPlots:
  """Container for trial visualization plots and metadata."""

  eeg_plot: mfig.Figure
  spectrogram_plot: mfig.Figure
  metadata: Dict[str, Any]


def plot_trial_data(trial_data: TrialData[RawEeg, Union[WavRAW, MelRaw]]) -> TrialPlots:
  """
  Create comprehensive plots for trial data including EEG and music spectrogram.

  Args:
      trial_data: TrialData containing RawEeg and either WavRAW or MelRaw music data

  Returns:
      TrialPlots containing EEG plot, spectrogram plot, and metadata
  """
  # Extract EEG data and create plot
  eeg_raw = trial_data.eeg_data.get_eeg().raw_eeg
  eeg_fig = eeg_raw.plot(show=False, title=f"EEG - {trial_data.trial_id}")
  music = trial_data.music_data.get_music()

  # Create spectrogram plot based on music data type
  spectrogram_fig = None
  match music:
    case WavRAW() as wav:
      spectrogram_fig = mkplot_melspectrogram(
        wav,
        title=f"Mel Spectrogram - {trial_data.music_filename.filename}",
        fmax=10240.0,
      )
    case MelRaw() as mel:
      # Use existing mel spectrogram
      spectrogram_fig = melspectrogram_figure(
        mel=mel, title=f"Mel Spectrogram - {trial_data.music_filename.filename}"
      )

  # Collect metadata
  metadata = {
    "dataset": trial_data.dataset,
    "subject": trial_data.subject,
    "session": trial_data.session,
    "run": trial_data.run,
    "trial_id": trial_data.trial_id,
    "music_filename": trial_data.music_filename.filename,
    "eeg_channels": eeg_raw.ch_names,
    "eeg_sample_rate": eeg_raw.info["sfreq"],
    "eeg_duration_seconds": eeg_raw.times[-1] if len(eeg_raw.times) > 0 else 0,
    "music_sample_rate": music.sample_rate,
    "music_duration_seconds": music.length_seconds(),
  }

  return TrialPlots(
    eeg_plot=eeg_fig, spectrogram_plot=spectrogram_fig, metadata=metadata
  )


def plot_band_power_with_onsets(
  trial_data: TrialData[ArrayEeg, Union[WavRAW, MelRaw, NoteOnsets]],
  title: str = "Band Power Analysis",
) -> mfig.Figure:
  """
  Create a 2D image plot of band power data from EEG with optional note onsets overlaid as dotted lines.

  Args:
      trial_data: TrialData containing ArrayEeg (band power data) and any music data type
      title: Title for the plot

  Returns:
      Matplotlib figure containing the band power plot with onset lines (if music data is NoteOnsets)
  """
  # Extract EEG data
  eeg = trial_data.eeg_data
  band_power_data = eeg.data  # Shape: (channels * bands, time windows)
  sfreq = eeg.sfreq  # Effective sampling rate after windowing
  channel_names = eeg.ch_names

  # Get music data
  music = trial_data.music_data.get_music()

  # Create figure and axis
  fig, ax = plt.subplots(figsize=(12, 8))

  # Create time vector for x-axis (in seconds)
  n_windows = band_power_data.shape[1]
  duration = n_windows / sfreq

  # Plot band power as 2D image
  im = ax.imshow(
    band_power_data,
    aspect="auto",
    extent=(0.0, float(duration), float(len(channel_names) - 0.5), -0.5),
    cmap="viridis",
    interpolation="nearest",
  )

  # Add colorbar with value range
  cbar = plt.colorbar(im)
  cbar.set_label("Power (a.u.)")

  # Set y-axis ticks to channel names
  ax.set_yticks(np.arange(len(channel_names)))
  ax.set_yticklabels(channel_names)

  # Set axis labels and title
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Channel / Band")
  ax.set_title(title if title else f"Band Power - {trial_data.trial_id}")

  # Overlay note onsets as dotted lines (only if music data is NoteOnsets)
  if isinstance(music, NoteOnsets):
    for onset_time in music.onset_times:
      if 0 <= onset_time <= duration:
        ax.axvline(x=onset_time, color="white", linestyle=":", alpha=0.5, linewidth=1)

  # Adjust layout to prevent label cutoff
  plt.tight_layout()

  return fig

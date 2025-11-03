"""Tests for plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from eeg_music.data import TrialData, ArrayEeg, NoteOnsets, MusicFilename
from eeg_music.plotting import plot_band_power_with_onsets


def test_plot_band_power_with_onsets():
  """Test that band power plotting function creates a figure without errors."""
  # Create mock trial data with band power results
  n_channels = 3
  n_bands = 2
  n_windows = 100
  sfreq = (
    4.0  # Effective sampling rate (e.g., after windowing with hop_length=64 at 256 Hz)
  )

  # Create synthetic band power data
  band_power_data = np.random.rand(n_channels * n_bands, n_windows).astype(np.float32)
  channel_names = [
    f"Ch{ch}_band{b}" for ch in range(n_channels) for b in range(n_bands)
  ]

  eeg_data = ArrayEeg(data=band_power_data, ch_names=channel_names, sfreq=sfreq)

  # Create note onsets (in seconds)
  onset_times = np.array([2.0, 5.0, 10.0])
  music_data = NoteOnsets(
    onset_times=onset_times, sample_rate=44100, duration_seconds=25.0
  )

  trial = TrialData(
    dataset="test_dataset",
    subject="S01",
    session="ses1",
    run="run1",
    trial_id="trial1",
    music_filename=MusicFilename("music1.wav"),
    eeg_data=eeg_data,
    music_data=music_data,
  )

  # Create plot
  fig = plot_band_power_with_onsets(trial, title="Test Band Power Plot")

  # Check that figure was created
  assert isinstance(fig, Figure)

  # Check basic properties of the plot
  axes = fig.get_axes()
  assert len(axes) == 2  # Should have main axis and colorbar axis
  ax = axes[0]

  # Check that image was plotted
  assert len(ax.get_images()) == 1

  # Check y-axis labels (channels)
  yticks = ax.get_yticklabels()
  assert len(yticks) == len(channel_names)
  for label, expected in zip(yticks, channel_names):
    assert label.get_text() == expected

  # Check x-axis (time in seconds)
  assert ax.get_xlabel() == "Time (s)"

  # Check for colorbar
  cbar_ax = axes[1]
  assert cbar_ax.get_ylabel() == "Power (a.u.)"

  # Check for onset lines (count vertical lines)
  onset_lines = [
    line
    for line in ax.get_lines()
    if line.get_linestyle() == ":" and line.get_alpha() == 0.5
  ]
  assert len(onset_lines) == len(
    onset_times
  )  # Should have lines for each onset within duration

  # Clean up
  plt.close(fig)


def test_plot_band_power_with_onsets_empty_onsets():
  """Test plotting with no onsets."""
  n_channels = 2
  n_bands = 2
  n_windows = 50
  sfreq = 5.0

  band_power_data = np.random.rand(n_channels * n_bands, n_windows).astype(np.float32)
  channel_names = [
    f"Ch{ch}_band{b}" for ch in range(n_channels) for b in range(n_bands)
  ]

  eeg_data = ArrayEeg(data=band_power_data, ch_names=channel_names, sfreq=sfreq)

  music_data = NoteOnsets(
    onset_times=np.array([]), sample_rate=44100, duration_seconds=10.0
  )

  trial = TrialData(
    dataset="test_dataset",
    subject="S01",
    session="ses1",
    run="run1",
    trial_id="trial2",
    music_filename=MusicFilename("music2.wav"),
    eeg_data=eeg_data,
    music_data=music_data,
  )

  fig = plot_band_power_with_onsets(trial)
  assert isinstance(fig, Figure)

  ax = fig.get_axes()[0]
  onset_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
  assert len(onset_lines) == 0  # No onset lines should be drawn

  plt.close(fig)


def test_plot_band_power_with_onsets_out_of_range():
  """Test plotting with onsets outside the data duration."""
  n_channels = 2
  n_bands = 2
  n_windows = 40
  sfreq = 4.0  # 10 seconds total duration

  band_power_data = np.random.rand(n_channels * n_bands, n_windows).astype(np.float32)
  channel_names = [
    f"Ch{ch}_band{b}" for ch in range(n_channels) for b in range(n_bands)
  ]

  eeg_data = ArrayEeg(data=band_power_data, ch_names=channel_names, sfreq=sfreq)

  # Onsets outside the data range
  onset_times = np.array([-1.0, 15.0, 20.0])
  music_data = NoteOnsets(
    onset_times=onset_times, sample_rate=44100, duration_seconds=10.0
  )

  trial = TrialData(
    dataset="test_dataset",
    subject="S01",
    session="ses1",
    run="run1",
    trial_id="trial3",
    music_filename=MusicFilename("music3.wav"),
    eeg_data=eeg_data,
    music_data=music_data,
  )

  fig = plot_band_power_with_onsets(trial)
  assert isinstance(fig, Figure)

  ax = fig.get_axes()[0]
  onset_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
  assert len(onset_lines) == 0  # No lines should be drawn since onsets are out of range

  plt.close(fig)

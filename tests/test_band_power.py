"""Tests for band power feature extraction."""

import numpy as np

from eeg_music.band_power import (
  BandPowerParams,
  bandpass_filter,
  calculate_windowed_power,
  eeg_to_band_power,
  trial_to_band_power,
)
from eeg_music.data import ArrayEeg, TrialData, MusicFilename, WavRAW


def test_bandpass_filter_shape():
  """Test that bandpass filter preserves shape."""
  data = np.random.randn(4, 1000).astype(np.float32)
  filtered = bandpass_filter(data, 8.0, 12.0, 256.0)
  assert filtered.shape == data.shape


def test_bandpass_filter_frequency_response():
  """Test that bandpass filter attenuates out-of-band frequencies."""
  # Create signal with multiple frequency components
  sfreq = 256.0
  duration = 4.0
  t = np.arange(0, duration, 1 / sfreq)

  # Signal with 5Hz (should be removed), 10Hz (should pass), 50Hz (should be removed)
  signal = (
    np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
  )
  data = signal.reshape(1, -1)

  # Filter to keep only 8-12 Hz
  filtered = bandpass_filter(data, 8.0, 12.0, sfreq)

  # Check power in different frequency bands using FFT
  fft_orig = np.fft.rfft(data[0])
  fft_filt = np.fft.rfft(filtered[0])
  freqs = np.fft.rfftfreq(len(t), 1 / sfreq)

  # Power at 10 Hz should be relatively preserved
  idx_10hz = np.argmin(np.abs(freqs - 10))
  assert np.abs(fft_filt[idx_10hz]) > 0.5 * np.abs(fft_orig[idx_10hz])

  # Power at 5 Hz and 50 Hz should be significantly reduced
  idx_5hz = np.argmin(np.abs(freqs - 5))
  idx_50hz = np.argmin(np.abs(freqs - 50))
  assert np.abs(fft_filt[idx_5hz]) < 0.1 * np.abs(fft_orig[idx_5hz])
  assert np.abs(fft_filt[idx_50hz]) < 0.1 * np.abs(fft_orig[idx_50hz])


def test_bandpass_filter_multichannel():
  """Test that bandpass filter works independently on each channel."""
  # Create multi-channel data with different frequencies
  sfreq = 256.0
  duration = 2.0
  t = np.arange(0, duration, 1 / sfreq)

  ch1 = np.sin(2 * np.pi * 10 * t)  # 10 Hz
  ch2 = np.sin(2 * np.pi * 20 * t)  # 20 Hz
  data = np.stack([ch1, ch2])

  # Filter to keep only 8-12 Hz
  filtered = bandpass_filter(data, 8.0, 12.0, sfreq)

  # Channel 1 should have high power, channel 2 should be attenuated
  power_ch1 = np.mean(filtered[0] ** 2)
  power_ch2 = np.mean(filtered[1] ** 2)
  assert power_ch1 > 10 * power_ch2


def test_calculate_windowed_power_shape():
  """Test that windowed power calculation produces correct shape."""
  data = np.random.randn(3, 1000).astype(np.float32)
  window_width = 100
  hop_length = 50

  power = calculate_windowed_power(data, window_width, hop_length)

  n_windows = (1000 - window_width) // hop_length + 1
  assert power.shape == (3, n_windows)


def test_calculate_windowed_power_values():
  """Test that windowed power calculation is correct."""
  # Create simple signal where power is easy to verify
  data = np.ones((1, 200), dtype=np.float32) * 2.0  # Constant signal = 2
  window_width = 50
  hop_length = 50

  power = calculate_windowed_power(data, window_width, hop_length)

  # Power should be 4.0 (2^2) for all windows
  np.testing.assert_allclose(power, 4.0, rtol=1e-6)


def test_calculate_windowed_power_varying_signal():
  """Test windowed power with varying signal strength."""
  # Create signal with two regions: low and high amplitude
  low_amp = np.ones(100) * 1.0
  high_amp = np.ones(100) * 3.0
  data = np.concatenate([low_amp, high_amp]).reshape(1, -1)

  window_width = 50
  hop_length = 50

  power = calculate_windowed_power(data, window_width, hop_length)

  # First windows should have power ~1, later windows should have power ~9
  assert power[0, 0] < 2.0  # Low power region
  assert power[0, -1] > 7.0  # High power region


def test_eeg_to_band_power_shape():
  """Test that band power extraction produces correct shape."""
  n_channels = 4
  n_samples = 1000
  data = np.random.randn(n_channels, n_samples).astype(np.float32)

  params = BandPowerParams(
    frequency_bands=[(4, 8), (8, 12), (12, 30)], window_width=128, hop_length=64
  )

  band_power = eeg_to_band_power(data, 256.0, params)

  n_windows = (n_samples - params.window_width) // params.hop_length + 1
  assert band_power.shape == (n_channels, 3, n_windows)


def test_eeg_to_band_power_values_plausible():
  """Test that band power values are non-negative and plausible."""
  n_channels = 2
  n_samples = 1000
  data = np.random.randn(n_channels, n_samples).astype(np.float32)

  params = BandPowerParams(
    frequency_bands=[(4, 8), (8, 12)], window_width=128, hop_length=64
  )

  band_power = eeg_to_band_power(data, 256.0, params)

  # Power should be non-negative
  assert np.all(band_power >= 0)

  # Power should be in a reasonable range (not too large or too small)
  assert np.all(band_power < 1000)


def test_eeg_to_band_power_different_bands_different_power():
  """Test that different frequency bands capture different power."""
  # Create signal dominated by alpha band (8-12 Hz)
  sfreq = 256.0
  duration = 4.0
  t = np.arange(0, duration, 1 / sfreq)

  # Strong 10 Hz component (alpha), weak 5 Hz component (theta)
  signal = np.sin(2 * np.pi * 10 * t) * 3.0 + np.sin(2 * np.pi * 5 * t) * 0.5
  data = signal.reshape(1, -1)

  params = BandPowerParams(
    frequency_bands=[(4, 7), (8, 12)],  # Theta, Alpha
    window_width=256,
    hop_length=128,
  )

  band_power = eeg_to_band_power(data, sfreq, params)

  # Alpha band should have higher power than theta band
  theta_power = np.mean(band_power[0, 0, :])
  alpha_power = np.mean(band_power[0, 1, :])
  assert alpha_power > 3 * theta_power


def test_trial_to_band_power_transform():
  """Test that trial transform works correctly."""
  # Create mock trial with ArrayEeg
  n_channels = 3
  n_samples = 1000
  sfreq = 256.0

  eeg_data = ArrayEeg(
    data=np.random.randn(n_channels, n_samples).astype(np.float32),
    ch_names=["Ch1", "Ch2", "Ch3"],
    sfreq=sfreq,
  )

  music_data = WavRAW(
    raw_data=np.random.randn(n_samples).astype(np.float32), sample_rate=44100
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

  params = BandPowerParams(
    frequency_bands=[(4, 8), (8, 12)], window_width=128, hop_length=64
  )

  transform = trial_to_band_power(params)
  transformed_trial = transform(trial)

  # Check that metadata is preserved
  assert transformed_trial.dataset == trial.dataset
  assert transformed_trial.subject == trial.subject
  assert transformed_trial.session == trial.session
  assert transformed_trial.run == trial.run
  assert transformed_trial.trial_id == trial.trial_id
  assert transformed_trial.music_filename == trial.music_filename
  assert transformed_trial.music_data == trial.music_data

  # Check that EEG is transformed correctly
  assert isinstance(transformed_trial.eeg_data, ArrayEeg)
  n_bands = len(params.frequency_bands)
  n_windows = (n_samples - params.window_width) // params.hop_length + 1
  assert transformed_trial.eeg_data.data.shape == (n_channels * n_bands, n_windows)

  # Check channel names
  assert len(transformed_trial.eeg_data.ch_names) == n_channels * n_bands
  assert "Ch1_band0" in transformed_trial.eeg_data.ch_names
  assert "Ch3_band1" in transformed_trial.eeg_data.ch_names


def test_trial_to_band_power_effective_sfreq():
  """Test that effective sampling frequency is updated correctly."""
  n_channels = 2
  n_samples = 1000
  sfreq = 256.0
  hop_length = 64

  eeg_data = ArrayEeg(
    data=np.random.randn(n_channels, n_samples).astype(np.float32),
    ch_names=["Ch1", "Ch2"],
    sfreq=sfreq,
  )

  music_data = WavRAW(
    raw_data=np.random.randn(n_samples).astype(np.float32), sample_rate=44100
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

  params = BandPowerParams(
    frequency_bands=[(8, 12)], window_width=128, hop_length=hop_length
  )

  transform = trial_to_band_power(params)
  transformed_trial = transform(trial)

  # Effective sampling rate should be reduced by hop_length
  expected_sfreq = sfreq / hop_length
  assert transformed_trial.eeg_data.sfreq == expected_sfreq


def test_trial_to_band_power_edge_cases():
  """Test edge cases: single channel, single band, small windows."""
  n_samples = 300
  sfreq = 256.0

  eeg_data = ArrayEeg(
    data=np.random.randn(1, n_samples).astype(np.float32),
    ch_names=["Single"],
    sfreq=sfreq,
  )

  music_data = WavRAW(
    raw_data=np.random.randn(n_samples).astype(np.float32), sample_rate=44100
  )

  trial = TrialData(
    dataset="test",
    subject="S01",
    session="s1",
    run="r1",
    trial_id="t1",
    music_filename=MusicFilename("m.wav"),
    eeg_data=eeg_data,
    music_data=music_data,
  )

  params = BandPowerParams(frequency_bands=[(8, 12)], window_width=50, hop_length=25)

  transform = trial_to_band_power(params)
  transformed_trial = transform(trial)

  # Should work without errors
  assert isinstance(transformed_trial.eeg_data, ArrayEeg)
  assert transformed_trial.eeg_data.data.shape[0] == 1  # 1 channel * 1 band


def test_band_power_params_dataclass():
  """Test BandPowerParams dataclass creation."""
  params = BandPowerParams(
    frequency_bands=[(4, 8), (8, 12), (12, 30)], window_width=256, hop_length=128
  )

  assert len(params.frequency_bands) == 3
  assert params.window_width == 256
  assert params.hop_length == 128
  assert params.frequency_bands[0] == (4, 8)


def test_bandpass_filter_nyquist_limit():
  """Test that bandpass filter handles frequencies near Nyquist correctly."""
  sfreq = 256.0
  data = np.random.randn(2, 1000).astype(np.float32)

  # High frequency band near Nyquist
  filtered = bandpass_filter(data, 80.0, 120.0, sfreq)

  assert filtered.shape == data.shape
  assert not np.isnan(filtered).any()
  assert not np.isinf(filtered).any()


def test_calculate_windowed_power_non_overlapping():
  """Test windowed power with non-overlapping windows."""
  data = np.random.randn(2, 400).astype(np.float32)
  window_width = 100
  hop_length = 100  # Non-overlapping

  power = calculate_windowed_power(data, window_width, hop_length)

  # Should have exactly 4 windows (400 / 100)
  assert power.shape[1] == 4


def test_calculate_windowed_power_overlapping():
  """Test windowed power with overlapping windows."""
  data = np.random.randn(2, 500).astype(np.float32)
  window_width = 100
  hop_length = 50  # 50% overlap

  power = calculate_windowed_power(data, window_width, hop_length)

  # Calculate expected number of windows
  expected_windows = (500 - 100) // 50 + 1
  assert power.shape[1] == expected_windows

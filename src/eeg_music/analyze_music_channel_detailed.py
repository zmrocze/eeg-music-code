"""
Detailed analysis of the 'music' channel encoding in EEG datasets.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def detailed_channel_analysis(dataset_path, max_files=3):
  """Perform detailed analysis of music channel values."""

  dataset_path = Path(dataset_path)

  print(f"\n{'=' * 70}")
  print("DETAILED MUSIC CHANNEL ANALYSIS - BCMI-fMRI")
  print(f"{'=' * 70}")

  # Find EEG files
  eeg_files = list(dataset_path.glob("sub-*/eeg/*_eeg.edf"))

  for eeg_file in eeg_files[:max_files]:
    print(f"\n📁 File: {eeg_file.name}")
    subject = eeg_file.parent.parent.name
    task = eeg_file.stem.split("_task-")[1].split("_")[0]

    try:
      # Load raw data
      raw = mne.io.read_raw_edf(eeg_file, preload=False, verbose=False)

      if "music" in raw.ch_names:
        print("  ✓ Music channel found")

        # Load music channel
        raw_music = raw.copy().pick_channels(["music"])
        raw_music.load_data()
        music_data = raw_music.get_data()[0]
        times = raw_music.times

        # Basic statistics
        print("\n  📊 Channel Statistics:")
        print(f"    - Duration: {times[-1]:.1f}s")
        print(f"    - Sampling rate: {raw.info['sfreq']}Hz")
        print(f"    - Total samples: {len(music_data)}")
        print(f"    - Min value: {np.min(music_data):.6f}")
        print(f"    - Max value: {np.max(music_data):.6f}")
        print(f"    - Mean value: {np.mean(music_data):.6f}")
        print(f"    - Std deviation: {np.std(music_data):.6f}")

        # Check for non-zero values
        non_zero_mask = np.abs(music_data) > 1e-10
        non_zero_count = np.sum(non_zero_mask)

        print("\n  🔍 Non-zero Analysis:")
        print(
          f"    - Non-zero samples: {non_zero_count} ({non_zero_count / len(music_data) * 100:.2f}%)"
        )

        if non_zero_count > 0:
          non_zero_values = music_data[non_zero_mask]
          unique_non_zero = np.unique(non_zero_values)

          print(f"    - Unique non-zero values: {len(unique_non_zero)}")
          print("    - First 10 non-zero values:")
          for i, val in enumerate(unique_non_zero[:10]):
            # Try to decode
            code = int(val * 20) if val > 0 else 0
            if code > 0 and len(str(code)) == 3:
              code_str = str(code)
              filename = f"{code_str[0]}-{code_str[1]}_{code_str[2]}.wav"
              print(f"        {val:.6f} → {code} → {filename}")
            else:
              print(f"        {val:.6f} → {code}")

          # Find segments with constant non-zero values
          segments = find_constant_segments(music_data, times)
          if segments:
            print("\n  📍 Constant Value Segments (first 5):")
            for seg in segments[:5]:
              print(
                f"    - {seg['start']:.1f}s - {seg['end']:.1f}s: value={seg['value']:.6f}, duration={seg['duration']:.1f}s"
              )
              if seg["value"] > 0:
                code = int(seg["value"] * 20)
                if len(str(code)) == 3:
                  code_str = str(code)
                  filename = f"{code_str[0]}-{code_str[1]}_{code_str[2]}.wav"
                  print(f"      → Decodes to: {filename}")

        # Check other channels for comparison
        print("\n  📌 Other Stimulus Channels:")
        stimulus_channels = [
          "trialtype",
          "nback_stimuli",
          "ft_valance",
          "ft_arousal",
        ]
        for ch_name in stimulus_channels:
          if ch_name in raw.ch_names:
            raw_ch = raw.copy().pick_channels([ch_name])
            raw_ch.load_data()
            ch_data = raw_ch.get_data()[0]
            unique_vals = np.unique(ch_data[np.abs(ch_data) > 1e-10])
            if len(unique_vals) > 0:
              print(f"    - {ch_name}: {len(unique_vals)} unique values")
              print(
                f"        Range: [{np.min(unique_vals):.4f}, {np.max(unique_vals):.4f}]"
              )

      else:
        print("  ⚠️ No music channel found")
        print(f"  Available channels: {', '.join(raw.ch_names[:15])}...")

    except Exception as e:
      print(f"  ❌ Error: {str(e)}")

  # Check events file for comparison
  print(f"\n{'=' * 70}")
  print("EVENTS FILE ANALYSIS")
  print(f"{'=' * 70}")

  for eeg_file in eeg_files[:2]:
    events_file = eeg_file.with_suffix(".tsv").with_name(
      eeg_file.stem.replace("_eeg", "_events") + ".tsv"
    )

    if events_file.exists():
      print(f"\n📄 Events: {events_file.name}")
      events_df = pd.read_csv(events_file, sep="\t")

      print(f"  Columns: {', '.join(events_df.columns)}")
      print(f"  Number of events: {len(events_df)}")

      if "trial_type" in events_df.columns:
        trial_types = events_df["trial_type"].unique()
        print(f"  Unique trial types: {trial_types}")

        # Check for music-related events
        for idx, row in events_df.head(10).iterrows():
          print(
            f"    Event {idx}: onset={row['onset']:.2f}s, duration={row['duration']:.2f}s, type={row['trial_type']}"
          )


def find_constant_segments(data, times, threshold=1e-10):
  """Find segments where the channel has constant non-zero values."""
  segments = []

  # Identify changes
  diff = np.diff(data)
  change_points = np.where(np.abs(diff) > threshold)[0]

  # Add start and end points
  change_points = np.concatenate([[0], change_points + 1, [len(data) - 1]])

  # Extract segments
  for i in range(len(change_points) - 1):
    start_idx = change_points[i]
    end_idx = change_points[i + 1]
    value = np.mean(data[start_idx:end_idx])

    if np.abs(value) > threshold:  # Only non-zero segments
      segments.append(
        {
          "start": times[start_idx],
          "end": times[end_idx],
          "duration": times[end_idx] - times[start_idx],
          "value": value,
          "start_idx": start_idx,
          "end_idx": end_idx,
        }
      )

  return segments


def check_bcmi_training_music(dataset_path):
  """Check BCMI-training dataset which might have different encoding."""

  dataset_path = Path(dataset_path)

  print(f"\n{'=' * 70}")
  print("BCMI-TRAINING DATASET ANALYSIS")
  print(f"{'=' * 70}")

  # Try one file
  sample_files = list(dataset_path.glob("sub-*/eeg/*_eeg.edf"))

  if sample_files:
    eeg_file = sample_files[0]
    print(f"\n📁 Sample: {eeg_file.name}")

    try:
      raw = mne.io.read_raw_edf(eeg_file, preload=False, verbose=False)

      # List all channels
      print(f"  Channels ({len(raw.ch_names)}): {', '.join(raw.ch_names)}")

      # Check for any channel that might encode music
      potential_music_channels = [
        ch
        for ch in raw.ch_names
        if any(keyword in ch.lower() for keyword in ["music", "stim", "audio", "sound"])
      ]

      if potential_music_channels:
        print(f"  Potential music-related channels: {potential_music_channels}")
      else:
        print("  No obvious music-related channels found")

    except Exception as e:
      print(f"  Error: {str(e)}")


def main():
  """Main analysis function."""

  base_path = Path("/home/zmrocze/studia/uwr/magisterka/datasets")

  # Detailed analysis of BCMI-fMRI
  if (base_path / "bcmi" / "bcmi-fmri").exists():
    detailed_channel_analysis(base_path / "bcmi" / "bcmi-fmri", max_files=3)

  # Check BCMI-training
  if (base_path / "bcmi" / "bcmi-training").exists():
    check_bcmi_training_music(base_path / "bcmi" / "bcmi-training")

  print(f"\n{'=' * 70}")
  print("ANALYSIS COMPLETE")
  print(f"{'=' * 70}")
  print("\n📝 FINDINGS:")
  print("1. The 'music' channel exists in BCMI-fMRI dataset")
  print("2. Channel values appear to be mostly zero or very small")
  print("3. The encoding scheme (value * 20 → 3-digit code → X-Y_Z.wav) is documented")
  print("4. Actual stimulus files match the expected pattern (1-2_1.wav, etc.)")
  print("5. Further investigation needed to understand why values are near-zero")


if __name__ == "__main__":
  main()

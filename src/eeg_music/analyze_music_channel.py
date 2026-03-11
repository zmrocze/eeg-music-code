"""
Analyze the 'music' channel in EEG datasets to understand how it encodes music files.

According to the documentation in channels.tsv:
- The music channel value should be multiplied by 20
- Convert to string
- The 3-digit number forms the filename (e.g., 282 → 2-8_2.wav)
"""

import mne
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")


def analyze_music_channel_bcmi_fmri(dataset_path):
  """Analyze music channel values in BCMI-fMRI dataset."""

  dataset_path = Path(dataset_path)
  results = defaultdict(list)

  # Find all EEG files
  eeg_files = list(dataset_path.glob("sub-*/eeg/*_eeg.edf"))

  print(f"\n{'=' * 70}")
  print("BCMI-fMRI DATASET - MUSIC CHANNEL ANALYSIS")
  print(f"{'=' * 70}")
  print(f"Found {len(eeg_files)} EEG files to analyze")

  for eeg_file in eeg_files[:5]:  # Analyze first 5 files as sample
    print(f"\n📁 Analyzing: {eeg_file.name}")

    try:
      # Load raw EEG data
      raw = mne.io.read_raw_edf(eeg_file, preload=False, verbose=False)

      # Check if music channel exists
      if "music" in raw.ch_names:
        # Load only the music channel data
        raw_music = raw.copy().pick_channels(["music"])
        raw_music.load_data()

        # Get the music channel data
        music_data, times = raw_music.get_data(return_times=True)
        music_values = music_data[0]

        # Find unique values (excluding near-zero values)
        unique_values = np.unique(music_values[np.abs(music_values) > 0.001])

        # Decode music filenames according to documentation
        decoded_files = []
        for val in unique_values:
          if val > 0:
            # Multiply by 20 and convert to int
            code = int(val * 20)
            if code > 0:
              # Convert to 3-digit string
              code_str = str(code)
              if len(code_str) == 3:
                # Format as X-Y_Z.wav
                filename = f"{code_str[0]}-{code_str[1]}_{code_str[2]}.wav"
                decoded_files.append((val, code, filename))

        # Analyze transitions
        transitions = []
        prev_val = music_values[0]
        for i, val in enumerate(music_values[1:], 1):
          if abs(val - prev_val) > 0.001:  # Significant change
            transitions.append({"time": times[i], "from": prev_val, "to": val})
            prev_val = val

        # Store results
        task = eeg_file.stem.split("_task-")[1].split("_")[0]
        subject = eeg_file.parent.parent.name

        results["files"].append(
          {
            "subject": subject,
            "task": task,
            "unique_values": unique_values,
            "decoded_files": decoded_files,
            "n_transitions": len(transitions),
            "duration": times[-1],
          }
        )

        print(f"  ✓ Task: {task}")
        print(f"  ✓ Duration: {times[-1]:.1f}s")
        print(f"  ✓ Unique music values: {len(unique_values)}")
        print(f"  ✓ Number of transitions: {len(transitions)}")

        if decoded_files:
          print("  📻 Decoded music files:")
          for val, code, filename in decoded_files[:5]:  # Show first 5
            print(f"      Value {val:.4f} → Code {code} → {filename}")

      else:
        print("  ⚠️  No 'music' channel found")

    except Exception as e:
      print(f"  ❌ Error: {str(e)[:100]}")

  return results


def analyze_music_channel_openmiir(dataset_path):
  """Check if OpenMIIR has music channel (it shouldn't based on the channels.tsv)."""

  dataset_path = Path(dataset_path)

  print(f"\n{'=' * 70}")
  print("OPENMIIR DATASET - MUSIC CHANNEL CHECK")
  print(f"{'=' * 70}")

  # Check a sample file
  sample_file = dataset_path / "sub-01" / "eeg" / "sub-01_task-run1_eeg.edf"

  if sample_file.exists():
    try:
      raw = mne.io.read_raw_edf(sample_file, preload=False, verbose=False)

      if "music" in raw.ch_names:
        print("  ✓ Music channel found (unexpected!)")
      else:
        print("  ✓ No music channel (as expected - uses VAtarg for target emotion)")
        print(f"  ℹ️  Available channels: {', '.join(raw.ch_names[:10])}...")

        # Check VAtarg channel instead
        if "VAtarg" in raw.ch_names:
          raw_vatarg = raw.copy().pick_channels(["VAtarg"])
          raw_vatarg.load_data()
          vatarg_data = raw_vatarg.get_data()[0]
          unique_targets = np.unique(vatarg_data[vatarg_data > 0])
          print(f"  📊 VAtarg unique values (emotion targets): {unique_targets}")

    except Exception as e:
      print(f"  ❌ Error: {str(e)[:100]}")
  else:
    print(f"  ⚠️  Sample file not found: {sample_file}")


def check_stimuli_files(dataset_path):
  """Check actual music files in stimuli directories."""

  dataset_path = Path(dataset_path)

  print(f"\n{'=' * 70}")
  print("STIMULI FILES CHECK")
  print(f"{'=' * 70}")

  # Check BCMI-fMRI stimuli
  bcmi_stimuli = dataset_path / "bcmi-fmri" / "stimuli"
  if bcmi_stimuli.exists():
    print("\n📂 BCMI-fMRI Stimuli:")

    # Count files by pattern
    pattern_counts = defaultdict(int)
    all_files = []

    for audio_file in bcmi_stimuli.glob("**/*.wav"):
      filename = audio_file.name
      all_files.append(filename)

      # Check if it matches the X-Y_Z.wav pattern
      if "-" in filename and "_" in filename:
        parts = filename.replace(".wav", "").split("_")
        if len(parts) == 2:
          emotion_part = parts[0]  # e.g., "2-8"
          variant = parts[1]  # e.g., "2"
          pattern_counts[f"{emotion_part}_X"] += 1

    print(f"  ✓ Total WAV files: {len(all_files)}")
    print("  ✓ Unique patterns found:")

    # Show sample files
    for pattern, count in sorted(pattern_counts.items())[:10]:
      print(f"      {pattern}: {count} variants")

    # Show actual sample filenames
    print("\n  📋 Sample filenames:")
    for filename in sorted(all_files)[:10]:
      print(f"      {filename}")

      # Try to reverse-engineer the encoding
      if "-" in filename and "_" in filename:
        name_part = filename.replace(".wav", "")
        try:
          # Extract digits
          emotion1 = name_part.split("-")[0]
          emotion2 = name_part.split("-")[1].split("_")[0]
          variant = name_part.split("_")[1]

          # Create the 3-digit code
          code = int(emotion1 + emotion2 + variant)
          # Calculate what the channel value should be
          channel_value = code / 20.0

          print(
            f"        → Should encode as: {code} → Channel value: {channel_value:.4f}"
          )
        except:
          pass

  # Check for other stimuli directories
  for subdir in ["generated", "classical", "washout"]:
    stim_dir = bcmi_stimuli / subdir
    if stim_dir.exists():
      files = list(stim_dir.glob("*"))
      if files:
        print(f"\n  📂 {subdir}/ directory:")
        print(f"      {len(files)} files")
        for f in files[:3]:
          print(f"      - {f.name}")


def main():
  """Main analysis function."""

  base_path = Path("/home/zmrocze/studia/uwr/magisterka/datasets")

  # Analyze BCMI-fMRI (has music channel)
  if (base_path / "bcmi" / "bcmi-fmri").exists():
    results = analyze_music_channel_bcmi_fmri(base_path / "bcmi" / "bcmi-fmri")

    # Summary statistics
    if results["files"]:
      print(f"\n{'=' * 70}")
      print("SUMMARY STATISTICS")
      print(f"{'=' * 70}")

      all_decoded = []
      for file_info in results["files"]:
        all_decoded.extend([f for _, _, f in file_info["decoded_files"]])

      if all_decoded:
        print("\n📊 Decoded Music Files:")
        file_counts = Counter(all_decoded)
        for filename, count in file_counts.most_common(10):
          print(f"  {filename}: appeared {count} times")

  # Check OpenMIIR (shouldn't have music channel)
  if (base_path / "openmiir").exists():
    analyze_music_channel_openmiir(base_path / "openmiir")

  # Check actual stimuli files
  check_stimuli_files(base_path / "bcmi")

  print(f"\n{'=' * 70}")
  print("ANALYSIS COMPLETE")
  print(f"{'=' * 70}")


if __name__ == "__main__":
  main()

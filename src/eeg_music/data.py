"""Data types"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import (
  Dict,
  Optional,
  Union,
  Callable,
  TypeVar,
  Generic,
  List,
  TypedDict,
  Tuple,
  cast,  # added
)
import numpy as np
from numpy.typing import NDArray
from mne.io import BaseRaw
import pandas as pd
from scipy.io import wavfile
import mne
from pandas import DataFrame, Index
import json
import shutil
import torch.utils.data as torchdata
from speechbrain.dataio.batch import PaddedBatch
import librosa
import librosa.display as lbd
import matplotlib.pyplot as plt


class MusicData(ABC):
  """Abstract base class for music data."""

  @abstractmethod
  def get_music(self) -> "WavRAW | MelRaw | NoteOnsets":
    """Get the music as WavRAW, MelRaw, or NoteOnsets data."""
    pass

  @abstractmethod
  def save(self, filepath: Path) -> None:
    """Save the music data to a file."""
    pass


class EegData(ABC):
  """Abstract base class for EEG data."""

  @abstractmethod
  def get_eeg(self) -> "RawEeg":
    """Get the EEG data as RawEeg."""
    pass

  @abstractmethod
  def save(self, filepath: Path) -> None:
    """Save the EEG data to a file."""
    pass


class MusicID(ABC):
  """Abstract base class for within-dataset music identifiers."""

  @abstractmethod
  def to_filename(self) -> str:
    """Convert the music ID to a filename string."""
    pass


wav_filenames_ordered_calibration = [
  "hvha1.wav",
  "hvha10.wav",
  "hvha11.wav",
  "hvha12.wav",
  "hvha2.wav",
  "hvha3.wav",
  "hvha4.wav",
  "hvha5.wav",
  "hvha6.wav",
  "hvha7.wav",
  "hvha8.wav",
  "hvha9.wav",
  "hvla1.wav",
  "hvla10.wav",
  "hvla11.wav",
  "hvla12.wav",
  "hvla2.wav",
  "hvla3.wav",
  "hvla4.wav",
  "hvla5.wav",
  "hvla6.wav",
  "hvla7.wav",
  "hvla8.wav",
  "hvla9.wav",
  "hvna1.wav",
  "hvna10.wav",
  "hvna11.wav",
  "hvna12.wav",
  "hvna2.wav",
  "hvna3.wav",
  "hvna4.wav",
  "hvna5.wav",
  "hvna6.wav",
  "hvna7.wav",
  "hvna8.wav",
  "hvna9.wav",
  "lvha1.wav",
  "lvha10.wav",
  "lvha11.wav",
  "lvha12.wav",
  "lvha2.wav",
  "lvha3.wav",
  "lvha4.wav",
  "lvha5.wav",
  "lvha6.wav",
  "lvha7.wav",
  "lvha8.wav",
  "lvha9.wav",
  "lvla1.wav",
  "lvla10.wav",
  "lvla11.wav",
  "lvla12.wav",
  "lvla2.wav",
  "lvla3.wav",
  "lvla4.wav",
  "lvla5.wav",
  "lvla6.wav",
  "lvla7.wav",
  "lvla8.wav",
  "lvla9.wav",
  "lvna1.wav",
  "lvna10.wav",
  "lvna11.wav",
  "lvna12.wav",
  "lvna2.wav",
  "lvna3.wav",
  "lvna4.wav",
  "lvna5.wav",
  "lvna6.wav",
  "lvna7.wav",
  "lvna8.wav",
  "lvna9.wav",
  "nvha1.wav",
  "nvha10.wav",
  "nvha11.wav",
  "nvha12.wav",
  "nvha2.wav",
  "nvha3.wav",
  "nvha4.wav",
  "nvha5.wav",
  "nvha6.wav",
  "nvha7.wav",
  "nvha8.wav",
  "nvha9.wav",
  "nvla1.wav",
  "nvla10.wav",
  "nvla11.wav",
  "nvla12.wav",
  "nvla2.wav",
  "nvla3.wav",
  "nvla4.wav",
  "nvla5.wav",
  "nvla6.wav",
  "nvla7.wav",
  "nvla8.wav",
  "nvla9.wav",
  "nvna1.wav",
  "nvna10.wav",
  "nvna11.wav",
  "nvna12.wav",
  "nvna2.wav",
  "nvna3.wav",
  "nvna4.wav",
  "nvna5.wav",
  "nvna6.wav",
  "nvna7.wav",
  "nvna8.wav",
  "nvna9.wav",
]

# !!!!! These files are 19s long, not 21s !!!!!


@dataclass
class CalibrationMusicId(MusicID):
  """Music ID for calibration data."""

  number: int

  def to_filename(self) -> str:
    """Convert calibration music ID to filename."""
    return wav_filenames_ordered_calibration[self.number]


@dataclass
class TrainingMusicId(MusicID):
  """Music ID for training data."""

  emotion_code_1: int
  emotion_code_2: int
  session: Union[int, str]
  which_half: bool  # False: first half, True: second half of the music file

  def to_filename(self) -> str:
    """Convert training music ID to filename."""
    return f"{self.emotion_code_1}-{self.emotion_code_2}_{self.session}_{'second' if self.which_half else 'first'}.wav"


@dataclass
class WavRAW(MusicData):
  """Data class containing raw WAV data and its rate."""

  raw_data: NDArray[np.float32]  # Audio data as numpy array of float values
  sample_rate: int  # Sample rate in Hz

  def is_not_empty(self) -> bool:
    """Check if the WAV data is not empty."""
    return self.raw_data.size > 0

  def length_seconds(self) -> float:
    """Get the length of the WAV data in seconds."""
    return self.raw_data.shape[0] / self.sample_rate

  def length_samples(self) -> int:
    """Get the length of the WAV data."""
    return self.raw_data.shape[0]

  def get_music(self) -> "WavRAW":
    """Get the music as WavRAW data."""
    return self

  def save(self, filepath: Path) -> None:
    """Save the WAV data to a file."""
    wavfile.write(
      filepath if filepath.suffix else filepath.with_suffix(".wav"),
      self.sample_rate,
      np.clip(self.raw_data, -1, 1),  # saved as float32
    )

  def resampled(self, new_sr: int) -> "WavRAW":
    """Return a new WavRAW instance with the audio resampled to new_sr."""
    resampled_data = librosa.resample(
      self.raw_data, orig_sr=self.sample_rate, target_sr=new_sr, res_type="kaiser_best"
    )
    return WavRAW(raw_data=resampled_data, sample_rate=new_sr)


@dataclass
class MelRaw(MusicData):
  mel: NDArray[np.floating]  # (n_mels, n_frames)
  sample_rate: int  # original audio sample rate
  hop_length: int  # hop used to create mel
  fmin: float
  fmax: Optional[float]
  to_db: bool

  def length_seconds(self) -> float:
    return self.mel.shape[1] * self.hop_length / self.sample_rate

  def save(self, filepath: Path):
    # Ensure .npz extension since np.savez_compressed adds it automatically
    if filepath.suffix != ".npz":
      filepath = filepath.with_suffix(filepath.suffix + ".npz")
    np.savez_compressed(
      filepath,
      mel=self.mel,
      sample_rate=self.sample_rate,
      hop_length=self.hop_length,
      fmin=self.fmin,
      to_db=self.to_db,
      allow_pickle=True,
      **({"fmax": self.fmax} if self.fmax is not None else {}),
    )

  def get_music(self) -> "MelRaw":
    return self


@dataclass
class NoteOnsets(MusicData):
  """Data class containing note onset times detected from audio."""

  onset_times: NDArray[np.floating]  # array of onset times in seconds
  sample_rate: int  # for consistency with other music types
  duration_seconds: float  # total duration of the music

  def length_seconds(self) -> float:
    return self.duration_seconds

  def save(self, filepath: Path) -> None:
    """Save onset times to a numpy file."""
    # Ensure .npz extension
    if filepath.suffix != ".npz":
      filepath = filepath.with_suffix(filepath.suffix + ".npz")
    np.savez_compressed(
      filepath,
      onset_times=self.onset_times,
      sample_rate=self.sample_rate,
      duration_seconds=self.duration_seconds,
    )

  def get_music(self) -> "NoteOnsets":
    return self

  def filter_onsets_in_time_range(
    self, start_time: float, end_time: float
  ) -> "NoteOnsets":
    """Return a new NoteOnsets with only onsets within the time range [start_time, end_time)."""
    mask = (self.onset_times >= start_time) & (self.onset_times < end_time)
    # Shift onset times to be relative to start_time
    filtered_onsets = self.onset_times[mask] - start_time
    new_duration = end_time - start_time
    return NoteOnsets(
      onset_times=filtered_onsets,
      sample_rate=self.sample_rate,
      duration_seconds=new_duration,
    )


# MelOrWav = MelRaw | WavRAW  # type alias for external use

# helper functions (optional convenience)
# def mel_or_wav_length(x: MelOrWav) -> float: return x.length_seconds()


@dataclass
class OnDiskMusic(MusicData):
  """Music data backed by a file on disk."""

  filepath: Path

  def get_music(self) -> WavRAW:
    """Load and return the music as WavRAW data."""
    sample_rate, raw_data = wavfile.read(self.filepath)
    match raw_data.dtype:
      case np.int16:
        scale = 32768.0
      case np.int32:
        scale = 2147483648.0
      case np.float32:
        scale = 1.0
      case np.float64:
        scale = 1.0
      case _:
        raise ValueError(f"Unsupported WAV data type: {raw_data.dtype}")

    raw_data = raw_data.astype(np.float32) / scale
    return WavRAW(raw_data=raw_data, sample_rate=sample_rate)

  def save(self, filepath: Path) -> None:
    """Save the music data by copying the file."""
    shutil.copy2(self.filepath, filepath)


@dataclass
class OnDiskMel(MusicData):
  """Music mel data backed by a .npz file on disk.

  Expected archive keys: mel, sample_rate, hop_length.
  """

  filepath: Path

  def get_music(self) -> MelRaw:
    d = np.load(self.filepath)
    fmax = float(d["fmax"]) if "fmax" in d else None
    return MelRaw(
      mel=d["mel"],
      sample_rate=int(d["sample_rate"]),
      hop_length=int(d["hop_length"]),
      fmin=float(d["fmin"]),
      fmax=fmax,
      to_db=bool(d["to_db"]),
    )

  def save(self, filepath: Path) -> None:
    # Ensure .npz extension for consistency with MelRaw.save()
    if filepath.suffix != ".npz":
      filepath = filepath.with_suffix(filepath.suffix + ".npz")
    shutil.copy2(self.filepath, filepath)


@dataclass
class OnDiskOnsets(MusicData):
  """Note onsets data backed by a .npz file on disk.

  Expected archive keys: onset_times, sample_rate, duration_seconds.
  """

  filepath: Path

  def get_music(self) -> NoteOnsets:
    d = np.load(self.filepath)
    return NoteOnsets(
      onset_times=d["onset_times"],
      sample_rate=int(d["sample_rate"]),
      duration_seconds=float(d["duration_seconds"]),
    )

  def save(self, filepath: Path) -> None:
    # Ensure .npz extension for consistency with NoteOnsets.save()
    if filepath.suffix != ".npz":
      filepath = filepath.with_suffix(filepath.suffix + ".npz")
    shutil.copy2(self.filepath, filepath)


# @dataclass
class RawEeg(EegData):
  """EEG data stored in memory."""

  def __init__(self, raw_eeg: BaseRaw):
    self.raw_eeg = raw_eeg
    self.raw_eeg.load_data()

  def get_eeg(self) -> "RawEeg":
    """Get the EEG data."""
    # self.raw_eeg.load_data()
    return self

  def length_seconds(self) -> float:
    """Get the length of the EEG data in seconds."""
    sfreq = float(self.raw_eeg.info["sfreq"])
    return self.raw_eeg.n_times / sfreq

  def save(self, filepath: Path) -> None:
    """Save the EEG data to a file."""
    # mne.export.export_raw expects a Raw object, not a RawEeg wrapper
    mne.export.export_raw(filepath, self.raw_eeg, fmt="edf", overwrite=True)


@dataclass
class OnDiskEeg(EegData):
  """EEG data backed by a file on disk."""

  filepath: Path

  def get_eeg(self) -> "RawEeg":
    """Load and return the EEG data as RawEeg."""
    # Note: we could go on with preload=False here, but then we'd need another
    # differentiating type for RawEEG but actually certainly loaded.
    # Turns out methods like filter don't load when needed but error out.
    return RawEeg(raw_eeg=mne.io.read_raw(self.filepath, preload=True, verbose="error"))

  def save(self, filepath: Path) -> None:
    """Save the EEG data by copying the file."""
    shutil.copy2(self.filepath, filepath)


# Type variables for generic Trial class
M = TypeVar("M", bound=MusicData)
E = TypeVar("E", bound=EegData)


@dataclass(frozen=True)
class MusicFilename:
  """Reference to music data in the music collection."""

  filename: str

  @classmethod
  def from_musicid(cls, music_id: MusicID) -> "MusicFilename":
    """Create a MusicFilename from a MusicID."""
    return cls(filename=music_id.to_filename())


@dataclass
class TrialData(Generic[E, M]):
  """Data class containing music and EEG data."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str
  music_filename: MusicFilename

  eeg_data: E
  music_data: M

  def load_to_mem(self) -> "TrialData[RawEeg, WavRAW | MelRaw | NoteOnsets]":
    """Load any on-disk data into memory, returning a new TrialData instance."""
    return TrialData(
      dataset=self.dataset,
      subject=self.subject,
      session=self.session,
      run=self.run,
      trial_id=self.trial_id,
      music_filename=self.music_filename,
      eeg_data=self.eeg_data.get_eeg(),
      music_data=self.music_data.get_music(),
    )

  def _music_brief(self) -> str:
    m = self.music_data
    match m:
      case WavRAW(raw_data=raw, sample_rate=sr):
        return f"WavRAW(sr={sr}, secs={len(raw) / sr:.3f}, samples={raw.shape[0]})"
      case MelRaw(mel=mel, sample_rate=sr, hop_length=hop, fmin=_, fmax=_, to_db=_):
        return f"MelRaw(sr={sr}, hop={hop}, mel_shape={mel.shape}, secs={mel.shape[1] * hop / sr:.3f})"
      case _:
        return type(m).__name__

  def _eeg_brief(self) -> str:
    e = self.eeg_data
    if isinstance(e, RawEeg):
      sf = float(e.raw_eeg.info["sfreq"])
      return f"RawEeg(sfreq={int(sf)}, chans={len(e.raw_eeg.ch_names)}, secs={e.raw_eeg.n_times / sf:.3f}, samples={e.raw_eeg.n_times})"
    if isinstance(e, OnDiskEeg):
      return f"OnDiskEeg(path='{e.filepath.name}')"
    return type(e).__name__

  def pretty(self) -> str:
    return (
      f"TrialData(\n"
      f"  dataset={self.dataset}, subject={self.subject}, session={self.session}, run={self.run}, trial_id={self.trial_id},\n"
      f"  music_filename={self.music_filename.filename},\n"
      f"  eeg={self._eeg_brief()},\n"
      f"  music={self._music_brief()}\n"
      f")"
    )

  def __str__(self) -> str:
    return self.pretty()


class TrialMetadataRecord(TypedDict):
  """Typed dict for trial metadata record in JSON."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str
  music_filename: str


@dataclass
class DatasetMetadata:
  """Metadata for a saved EEG music dataset: metadata.json."""

  trials: List[TrialMetadataRecord]
  stimuli: Dict[str, List[str]]  # Mapping from dataset name to list of music filenames
  num_trials: int

  def to_dict(self) -> dict:
    """Convert to dictionary for JSON serialization."""
    return {
      "trials": self.trials,
      "stimuli": self.stimuli,
      "num_trials": self.num_trials,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "DatasetMetadata":
    """Create from dictionary loaded from JSON."""
    return cls(
      trials=data["trials"], stimuli=data["stimuli"], num_trials=data["num_trials"]
    )

  def save_json(self, filepath: Path) -> None:
    """Save metadata to JSON file."""
    with open(filepath, "w") as f:
      json.dump(self.to_dict(), f, indent=2)

  @classmethod
  def load_json(cls, filepath: Path) -> "DatasetMetadata":
    """Load metadata from JSON file."""
    with open(filepath, "r") as f:
      data = json.load(f)
    return cls.from_dict(data)


@dataclass
class TrialRow(Generic[E]):
  """Data class containing music ID, raw EEG data, and emotion code."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str
  eeg_data: E
  music_filename: MusicFilename


def make_eeg_path(
  base_dir: Path, dataset: str, subject: str, session: str, run: str, trial_id: str
) -> Path:
  """Construct the EEG file path following dataset/subject/session/run/trial_id/eeg.edf structure."""
  return base_dir / dataset / subject / session / run / trial_id / "eeg.edf"


def copy_from_dataloader_into_dir(loader, base_dir: Path):
  """
  Iterates over dataset loader trials and music collection,
  saving these into a specified directory.

  The directory can already contain a saved dataset.
  """

  base_dir.mkdir(parents=True, exist_ok=True)
  stimuli_dataset_dir = base_dir / "stimuli" / loader.dataset_name
  eeg_dir = base_dir / "eeg"
  stimuli_dataset_dir.mkdir(parents=True, exist_ok=True)
  eeg_dir.mkdir(exist_ok=True)

  metadata_path = base_dir / "metadata.json"
  existing_metadata = (
    DatasetMetadata.load_json(metadata_path)
    if metadata_path.exists()
    else DatasetMetadata(trials=[], stimuli={}, num_trials=0)
  )

  # Save music files
  for music_ref, music_data in loader.music_iterator():
    stimuli_file = stimuli_dataset_dir / music_ref.filename
    if not stimuli_file.exists():
      music_data.save(stimuli_file)
      if loader.dataset_name not in existing_metadata.stimuli:
        existing_metadata.stimuli[loader.dataset_name] = []
      if (
        music_ref.filename not in existing_metadata.stimuli[loader.dataset_name]
      ):  # O(n) search!
        existing_metadata.stimuli[loader.dataset_name].append(music_ref.filename)

  # Save trials
  for trial in loader.trial_iterator():
    trial_record: TrialMetadataRecord = {
      "dataset": trial.dataset,
      "subject": trial.subject,
      "session": trial.session,
      "run": trial.run,
      "trial_id": trial.trial_id,
      "music_filename": trial.music_filename.filename,
    }

    eeg_path = make_eeg_path(
      eeg_dir, trial.dataset, trial.subject, trial.session, trial.run, trial.trial_id
    )
    eeg_path.parent.mkdir(parents=True, exist_ok=True)

    # Save EEG data - all loaders now return EegData objects
    trial.eeg_data.save(eeg_path)

    existing_metadata.trials.append(trial_record)
    existing_metadata.num_trials += 1

  existing_metadata.save_json(metadata_path)


@dataclass(frozen=True)
class MusicRef:
  """Reference to music data in the full multi-dataset collection."""

  filename: MusicFilename
  dataset: str


class EEGMusicDataset(torchdata.Dataset):
  """
  Dataset containing EEG trials with metadata.

  Can be stored into a directory with structure:
  base_dir/
    metadata.json
    stimuli/
      dataset_name/
        music files...
    eeg/
      dataset_name/
        subject/
          session/
            run/
              trial_id/
                eeg.edf
  """

  def __init__(self):
    """
    Trial ids with eeg data (raw or on-disk) and music,
    pointed by reference into a music collection dict (which stores music raw or on-disk).
    That's because music files are often reused between trials.
    """
    self.df = pd.DataFrame(
      columns=Index(
        [
          "dataset",
          "subject",
          "session",
          "run",
          "trial_id",
          "music_filename",
          "eeg_data",
        ]
      )
    )

    self.music_collection: Dict[MusicRef, MusicData] = {}

  @property
  def df(self) -> pd.DataFrame:
    """Get the dataframe."""
    return self._df

  @df.setter
  def df(self, value: pd.DataFrame) -> None:
    """Set the dataframe with proper indexing."""
    cols = ["dataset", "subject", "session", "run", "trial_id"]
    indexed = value.reindex(columns=cols + ["music_filename", "eeg_data"])
    indexed = indexed.set_index(
      cols,
      drop=False,
      verify_integrity=True,
    )
    self._df = indexed

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int) -> TrialData[EegData, MusicData]:
    row = self.df.iloc[idx]
    music_ref = MusicRef(filename=row.music_filename, dataset=row.dataset)
    music_data = self.music_collection[music_ref]
    return TrialData(
      dataset=row.dataset,
      subject=row.subject,
      session=row.session,
      run=row.run,
      trial_id=row.trial_id,
      music_filename=row.music_filename,
      eeg_data=row.eeg_data,
      music_data=music_data,
    )

  def merge(self, other: "EEGMusicDataset") -> "EEGMusicDataset":
    """Merge this dataset with another dataset."""
    merged_dataset = EEGMusicDataset()
    merged_dataset.df = pd.concat([self.df, other.df], ignore_index=True)
    # this is enough because music refs are unique outside of dataset as well
    merged_dataset.music_collection = {
      **self.music_collection,
      **other.music_collection,
    }
    return merged_dataset

  def map_df(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> "EEGMusicDataset":
    """Map underlying dataframe."""
    mapped_dataset = EEGMusicDataset()
    df = func(self.df.copy())
    mapped_dataset.df = df
    return mapped_dataset

  def subject_wise_split(
    self, p_train: float, p_val: float, seed: int = 42
  ) -> Tuple["EEGMusicDataset", "EEGMusicDataset", "EEGMusicDataset"]:
    """Split subjects into train/val/test using two proportions.

    p_train: fraction of subjects for train
    p_val: fraction of subjects for val (after train); must satisfy p_train>0, p_val>=0, p_train+p_val<1
    Remainder subjects form test. Deterministic via seed.
    """
    if not (0 < p_train < 1):
      raise ValueError("p_train in (0,1)")
    if not (0 <= p_val < 1):
      raise ValueError("p_val in [0,1)")
    if p_train + p_val >= 1:
      raise ValueError("p_train + p_val < 1 required")
    np.random.seed(seed)
    subj = np.array(self.df["subject"].unique())
    np.random.shuffle(subj)
    n = len(subj)
    n_tr = int(n * p_train)
    n_va = int(n * p_val)

    def mk(s: np.ndarray) -> "EEGMusicDataset":
      ds = EEGMusicDataset()
      df = self.df[self.df["subject"].isin(s.tolist())]
      ds.df = cast(DataFrame, df.reset_index(drop=True))
      ds.music_collection = self.music_collection
      return ds

    return mk(subj[:n_tr]), mk(subj[n_tr : n_tr + n_va]), mk(subj[n_tr + n_va :])

  def save(self, base_dir: Path) -> None:
    """
    Save dataset to directory with metadata and trial data.

    Filters out unused music from music_collection.
    Relies on data from __getitem__!
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = base_dir / "metadata.json"

    if metadata_path.exists():
      print(f"Overwriting existing metadata at {metadata_path}")

    # filtered and with potential mapping (in __getitem__) applied, see MappedDataset
    new_music_collection = {}

    # Save EEG data
    eeg_dir = base_dir / "eeg"
    for trial in self:
      eeg_path = make_eeg_path(
        eeg_dir, trial.dataset, trial.subject, trial.session, trial.run, trial.trial_id
      )
      eeg_path.parent.mkdir(parents=True, exist_ok=True)
      trial.eeg_data.save(eeg_path)

      new_music_collection[
        MusicRef(filename=trial.music_filename, dataset=trial.dataset)
      ] = trial.music_data

    stimuli_dir = base_dir / "stimuli"
    for music_ref, music_data in new_music_collection.items():
      music_dir = stimuli_dir / music_ref.dataset
      music_dir.mkdir(parents=True, exist_ok=True)
      music_data.save(music_dir / music_ref.filename.filename)

    # Save metadata
    stimuli_by_dataset = {}
    for music_ref in new_music_collection.keys():
      if music_ref.dataset not in stimuli_by_dataset:
        stimuli_by_dataset[music_ref.dataset] = []
      stimuli_by_dataset[music_ref.dataset].append(music_ref.filename.filename)

    metadata = DatasetMetadata(
      trials=[
        {
          "dataset": row.dataset,
          "subject": row.subject,
          "session": row.session,
          "run": row.run,
          "trial_id": row.trial_id,
          "music_filename": row.music_filename.filename,
        }
        for _, row in self.df.iterrows()
        # ^ assuming MappedDataset's map over __getitem__ doesn't change trial metadata, only data
      ],
      stimuli=stimuli_by_dataset,
      num_trials=len(self.df),
    )
    metadata.save_json(metadata_path)

  @classmethod
  def load_ondisk(cls, base_dir: Path) -> "EEGMusicDataset":
    """Load dataset from directory, using music and EEG data on-disk representations."""
    dataset = cls()

    # Read metadata
    metadata = DatasetMetadata.load_json(base_dir / "metadata.json")

    # Create music collection from metadata stimuli
    stimuli_dir = base_dir / "stimuli"
    for dataset_name, music_filenames in metadata.stimuli.items():
      for music_filename in music_filenames:
        music_ref = MusicRef(
          filename=MusicFilename(filename=music_filename), dataset=dataset_name
        )
        expected_path = stimuli_dir / dataset_name / music_filename
        if expected_path.suffix == ".wav" and not expected_path.exists():
          # Try .wav.npz version
          npz_path = expected_path.with_suffix(".wav.npz")
          if npz_path.exists():
            # Check if it's mel or onsets by inspecting keys
            with np.load(npz_path) as data:
              if "onset_times" in data:
                dataset.music_collection[music_ref] = OnDiskOnsets(filepath=npz_path)
              else:
                dataset.music_collection[music_ref] = OnDiskMel(filepath=npz_path)
          else:
            dataset.music_collection[music_ref] = OnDiskMusic(filepath=expected_path)
        elif expected_path.suffix == ".npz":
          # Check if it's mel or onsets by inspecting keys
          with np.load(expected_path) as data:
            if "onset_times" in data:
              dataset.music_collection[music_ref] = OnDiskOnsets(filepath=expected_path)
            else:
              dataset.music_collection[music_ref] = OnDiskMel(filepath=expected_path)
        else:
          dataset.music_collection[music_ref] = OnDiskMusic(filepath=expected_path)

    # Create dataframe from trial metadata
    eeg_dir = base_dir / "eeg"
    rows = []
    for trial_record in metadata.trials:
      eeg_path = make_eeg_path(
        eeg_dir,
        trial_record["dataset"],
        trial_record["subject"],
        trial_record["session"],
        trial_record["run"],
        trial_record["trial_id"],
      )
      rows.append(
        {
          "dataset": trial_record["dataset"],
          "subject": trial_record["subject"],
          "session": trial_record["session"],
          "run": trial_record["run"],
          "trial_id": trial_record["trial_id"],
          "music_filename": MusicFilename(filename=trial_record["music_filename"]),
          "eeg_data": OnDiskEeg(filepath=eeg_path),
        }
      )

    dataset.df = pd.DataFrame(rows)
    return dataset

  def load_to_mem(self):
    """Load all eeg and music data into memory."""
    # Convert all MusicData to WavRAW in the music collection
    for music_ref, music_data in self.music_collection.items():
      self.music_collection[music_ref] = music_data.get_music()

    # Convert all EegData to RawEeg in the dataframe
    for idx, row in self.df.iterrows():
      self.df.at[idx, "eeg_data"] = row.eeg_data.get_eeg()

  def remove_short_trials(self, min_trial_length_seconds: float) -> "EEGMusicDataset":
    """Return a new dataset with trials shorter than the threshold removed.

    A trial is kept iff both:
    - EEG duration in seconds >= min_trial_length_seconds
    - Music duration in seconds >= min_trial_length_seconds
    """
    to_keep: List[int] = []
    for i in range(len(self)):
      trial = self[i]
      eeg_duration_sec = trial.eeg_data.get_eeg().length_seconds()
      if (
        eeg_duration_sec >= min_trial_length_seconds
        and trial.music_data.get_music().length_seconds() >= min_trial_length_seconds
      ):
        to_keep.append(i)
    filtered = EEGMusicDataset()
    filtered.df = self.df.iloc[to_keep].reset_index(drop=True)
    filtered.music_collection = self.music_collection
    return filtered


def example_collate_fn(trials: List[TrialData[EegData, MusicData]]):
  # todo: preload before collate_fn? matters with pin_memory and all that?
  eegs = [t.eeg_data.get_eeg() for t in trials]
  music = [t.music_data.get_music() for t in trials]
  return PaddedBatch(eegs), PaddedBatch(music)


@dataclass(frozen=True)
class MelParams:
  n_mels: int = 128
  n_fft: int = 2048
  hop_length: int = 512
  fmin: float = 0.0
  fmax: Optional[float] = None
  center: bool = True
  power: float = 2.0
  to_db: bool = True

  def as_kwargs(self) -> dict:
    return {
      "n_mels": self.n_mels,
      "n_fft": self.n_fft,
      "hop_length": self.hop_length,
      "fmin": self.fmin,
      "center": self.center,
      "power": self.power,
      "to_db": self.to_db,
      "fmax": self.fmax,
    }


def prepare_trial(
  trial: TrialData[EegData, MusicData],
  eeg_resample: Optional[int] = 256,
  eeg_l_freq: Optional[float] = None,
  eeg_h_freq: Optional[float] = None,
  wav_resample: Optional[int] = None,
  apply_mel: Optional[MelParams] = None,
  # remove_channels: Optional[List[str]] = None,
  pick_channels: Optional[List[str]] = None,
) -> TrialData[RawEeg, WavRAW | MelRaw | NoteOnsets]:
  """Set common length between music and eeg, resample eeg and filter eeg, transform music to mel spectrogram.

  Optional music resampling, applied before mel transform if any.
  apply_mel: None -> keep music type; dict -> parameters for mel transform (see helper.wavraw_to_melspectrogram args).
  Supports WavRAW, MelRaw, and NoteOnsets music types.
  """

  eeg: BaseRaw = trial.eeg_data.get_eeg().raw_eeg
  eeg = eeg.copy()
  music = trial.music_data.get_music()
  m_len = music.length_seconds()
  e_len = eeg.n_times / eeg.info["sfreq"]
  min_len = min(m_len, e_len)

  match music:
    case WavRAW(raw_data=raw, sample_rate=sr) as wav:
      # let's do resampling first, then cropping. we dont cut length a lot here either way (for any speed gains)
      if wav_resample is not None and wav_resample != sr:
        wav = wav.resampled(new_sr=wav_resample)
        raw, sr = wav.raw_data, wav.sample_rate
      max_samples = int(min_len * sr)
      music_cropped: MusicData = WavRAW(raw[:max_samples], sr)
      # (optional) apply mel transform could go here if apply_mel is not None
      if apply_mel is not None:
        music_cropped = wavraw_to_melspectrogram(music_cropped, **apply_mel.as_kwargs())
    case MelRaw(
      mel=mel, sample_rate=sr, hop_length=hop, fmin=fmin, fmax=fmax, to_db=to_db
    ):
      assert apply_mel is None, (
        "Can't apply_mel if the input is already a mel spectrogram"
      )
      max_frames = int(min_len * sr / hop)
      music_cropped = MelRaw(
        mel[:, :max_frames], sr, hop, fmin=fmin, fmax=fmax, to_db=to_db
      )
    case NoteOnsets(onset_times=_, sample_rate=sr, duration_seconds=_):
      assert apply_mel is None, "Can't apply_mel if the input is NoteOnsets"
      # Filter onsets to keep only those within [0, min_len)
      music_cropped = music.filter_onsets_in_time_range(0.0, min_len)

  if eeg_l_freq is not None or eeg_h_freq is not None:
    eeg: BaseRaw = cast(BaseRaw, eeg.filter(l_freq=eeg_l_freq, h_freq=eeg_h_freq))
  eeg: BaseRaw = cast(
    BaseRaw, eeg if eeg_resample is None else eeg.resample(eeg_resample)
  )
  eeg = eeg.crop(
    tmax=(min(min_len, eeg.times[-1]))
  )  # when l=e_len then eeg_times[-1] is that 1s/sample_rate early to l which errors

  if pick_channels:
    eeg = eeg.pick(pick_channels)

  return TrialData(
    dataset=trial.dataset,
    subject=trial.subject,
    session=trial.session,
    run=trial.run,
    trial_id=trial.trial_id,
    music_filename=trial.music_filename,
    eeg_data=RawEeg(raw_eeg=eeg),
    music_data=music_cropped,
  )


def rereference_trial(
  trial: TrialData[EegData, MusicData],
) -> TrialData[EegData, MusicData]:
  """Rereference the EEG data in a trial."""
  eeg = trial.eeg_data.get_eeg().raw_eeg.copy()
  mne.set_eeg_reference(eeg, ref_channels="average", verbose="error")
  return TrialData(
    dataset=trial.dataset,
    subject=trial.subject,
    session=trial.session,
    run=trial.run,
    trial_id=trial.trial_id,
    music_filename=trial.music_filename,
    music_data=trial.music_data,
    eeg_data=RawEeg(eeg),
  )


class MappedDataset(EEGMusicDataset):
  """Dataset with a mapping function applied to each trial on access."""

  def __init__(
    self,
    base_dataset: EEGMusicDataset,
    ### map_fn is assumed to only change the held data, but keep the ids (dataset, subject, ...) constant!
    map_fn: Callable[[TrialData[EegData, MusicData]], TrialData[E, M]],
  ):
    self.ds = base_dataset
    self.map_fn = map_fn  # type: ignore[assignment]

  @property
  def df(self) -> pd.DataFrame:
    """Get the dataframe from the base dataset."""
    return self.ds.df

  @df.setter
  def df(self, value: pd.DataFrame) -> None:
    """Set the dataframe."""
    self.ds.df = value

  @property
  def music_collection(self) -> Dict[MusicRef, MusicData]:  # type: ignore[reportIncompatibleVariableOverride]
    """Get the music collection from the base dataset."""
    return self.ds.music_collection

  @music_collection.setter
  def music_collection(self, value: Dict[MusicRef, MusicData]):  # type: ignore[reportIncompatibleVariableOverride]
    """Set the music collection."""
    self.ds.music_collection = value

  def __getitem__(self, idx: int) -> TrialData[EegData, MusicData]:
    trial = self.ds.__getitem__(idx)
    return cast(TrialData[EegData, MusicData], self.map_fn(trial))


def int_or_err(x: Fraction) -> int:
  if x.denominator != 1:
    raise ValueError(f"Value {x} is not integer")
  return x.numerator


class StratifiedSamplingDataset(EEGMusicDataset):
  """
  Wrapper over ds, basically ds x n_strata.
  Indexing returns (trials, stratum_index).

  Useful for stratified sampling in DataLoader.
  """

  def __init__(
    self,
    base_dataset: EEGMusicDataset,
    n_strata: int,
    trial_length_secs: Fraction,
  ):
    """n_strata should be picked so that"""
    # super().__init__()
    self.ds = base_dataset
    self.n_strata = n_strata  # type: ignore[assignment]
    self.trial_length_secs: Fraction = trial_length_secs

  @property
  def df(self) -> pd.DataFrame:
    """Get the dataframe from the base dataset."""
    return self.ds.df

  @df.setter
  def df(self, value: pd.DataFrame) -> None:
    """Set the dataframe."""
    self.ds.df = value

  @property
  def music_collection(self) -> Dict[MusicRef, MusicData]:  # type: ignore[reportIncompatibleVariableOverride]
    """Get the music collection from the base dataset."""
    return self.ds.music_collection

  @music_collection.setter
  def music_collection(self, value: Dict[MusicRef, MusicData]):  # type: ignore[reportIncompatibleVariableOverride]
    """Set the music collection."""
    self.ds.music_collection = value

  def __len__(self) -> int:
    return len(self.ds) * self.n_strata

  def __getitem__(self, idx: int) -> TrialData[EegData, MusicData]:
    #  -> TrialData[RawEeg, WavRAW | MelRaw]:
    """
    Here we return a portion of a trial, starting at a random index, within a stratum (for balancing).
    """
    trial_index = idx // self.n_strata
    trial: TrialData[EegData, MusicData] = self.ds.__getitem__(trial_index)
    stratum_index = idx % self.n_strata

    music_obj = trial.music_data.get_music()
    eeg_obj = trial.eeg_data.get_eeg()
    m_len: float = music_obj.length_seconds()
    e_len: float = eeg_obj.length_seconds()
    eeg_raw = eeg_obj.raw_eeg
    length = min(m_len, e_len)

    n_starts = int((length - self.trial_length_secs) * eeg_raw.info["sfreq"])
    new_length_samples: int = int_or_err(
      self.trial_length_secs * Fraction(eeg_raw.info["sfreq"])
    )
    n_starts_exact = int(eeg_raw.n_times) - new_length_samples + 1
    s_start = (n_starts * stratum_index) // self.n_strata
    s_end = (n_starts * (stratum_index + 1)) // self.n_strata  # exclusive
    random_start = np.random.randint(s_start, min(s_end, n_starts_exact))
    data, _times = eeg_raw[:, random_start : random_start + new_length_samples]
    eeg_raw = mne.io.RawArray(
      data=data,
      info=eeg_raw.info,
      first_samp=eeg_raw.first_samp + random_start,
      verbose="error",
    )

    # some notes, maybe irrelevant now:
    # Note: doesnt work for 44100 / 256 !!!
    # Note: would be good to assume that:
    #  either sample_rate is divisible by eeg_raw.info["sfreq"]
    #  or the other way round (i.e. for mel)
    #  (which can be forced by resampling music, which likely is sensible anyway)
    # Q: do we strictly need this?
    # the max misalignment is going to be sth like: 1/sample_rate + 1/eeg_raw.info["sfreq"]
    #  which is not more than few milliseconds

    match music_obj:
      case WavRAW(raw_data, sample_rate):
        new_length_samples: int = int_or_err(self.trial_length_secs * sample_rate)
        tot_m = music_obj.length_samples()

        random_start_music = round((tot_m * random_start) / n_starts_exact)
        random_start_music = min(random_start_music, tot_m - new_length_samples)
        return_music = WavRAW(
          raw_data=raw_data[
            random_start_music : random_start_music + new_length_samples
          ],
          sample_rate=sample_rate,
        )

      case MelRaw(mel, sample_rate, hop_length, fmin, fmax, to_db):
        new_length_samples: int = int_or_err(
          self.trial_length_secs * sample_rate / hop_length
        )
        tot_m = music_obj.mel.shape[-1]
        random_start_music = round((tot_m * random_start) / n_starts_exact)
        random_start_music = min(random_start_music, tot_m - new_length_samples)

        return_music = MelRaw(
          mel=mel[:, random_start_music : random_start_music + new_length_samples],
          sample_rate=sample_rate,
          hop_length=hop_length,
          fmin=fmin,
          fmax=fmax,
          to_db=to_db,
        )

      case NoteOnsets(_, _, _):
        # Calculate start and end times for this slice
        start_time_sec = random_start / eeg_raw.info["sfreq"]
        end_time_sec = start_time_sec + float(self.trial_length_secs)
        return_music = music_obj.filter_onsets_in_time_range(
          start_time_sec, end_time_sec
        )

    trial: TrialData[EegData, MusicData] = TrialData(
      dataset=trial.dataset,
      subject=trial.subject,
      session=trial.session,
      run=trial.run,
      trial_id=trial.trial_id,
      music_filename=trial.music_filename,
      eeg_data=RawEeg(raw_eeg=eeg_raw),
      music_data=return_music,
    )
    return trial


class RepeatedDataset(torchdata.Dataset):
  def __init__(self, dataset, num_repeats):
    self.dataset = dataset
    self.num_repeats = num_repeats

  def __len__(self):
    return len(self.dataset) * self.num_repeats

  def __getitem__(self, idx):
    return self.dataset[idx % len(self.dataset)]


def onset_secs_to_samples(onset_secs, sfreq):
  return round(onset_secs * sfreq)


def wavraw_to_melspectrogram(
  wav: WavRAW,
  n_mels: int = 128,
  n_fft: int = 2048,
  hop_length: int = 512,
  fmin: float = 0.0,
  fmax: float | None = None,
  center: bool = True,
  power: float = 2.0,
  to_db: bool = True,
) -> MelRaw:
  """Return MelRaw (mel-spectrogram + sr + hop_length) for a WavRAW.

  Defaults: 128 mels, n_fft=2048, hop=512, power spectrogram, dB-scaled.
  """
  y = wav.raw_data if wav.raw_data.ndim == 1 else np.mean(wav.raw_data, axis=1)
  y = y.astype(np.float32)
  m = np.max(np.abs(y))
  y = y / (m + 1e-12) if m > 1.0 else y
  S = librosa.feature.melspectrogram(
    y=y,
    sr=wav.sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax,
    center=center,
    power=power,
    norm="slaney",
    htk=False,
  )
  if to_db:
    S = librosa.power_to_db(S, ref=np.max)
  return MelRaw(
    mel=S,
    sample_rate=wav.sample_rate,
    hop_length=hop_length,
    fmin=fmin,
    fmax=fmax,
    to_db=to_db,
  )


def melspectrogram_figure(
  mel: MelRaw,
  cmap: str = "magma",
  title: str = "Mel-spectrogram",
  onset_times: Optional[np.ndarray] = None,
):
  """Build and return a matplotlib Figure with the mel-spectrogram plot.

  Args:
    mel: MelRaw spectrogram data
    cmap: colormap for the spectrogram
    title: plot title
    onset_times: optional array of sample indices marking note beginnings
  """
  S = mel.mel
  fig, ax = plt.subplots(figsize=(8, 3))
  img = lbd.specshow(
    S,
    x_axis="time",
    y_axis="mel",
    sr=mel.sample_rate,
    fmin=mel.fmin,
    fmax=mel.fmax,
    cmap=cmap,
    ax=ax,
  )
  ax.set(title=title + (" (dB)" if mel.to_db else ""))
  cbar = fig.colorbar(img, ax=ax)
  cbar.set_label("dB" if mel.to_db else "power")

  # Mark onset timestamps if provided
  if onset_times is not None and len(onset_times) > 0:
    # Draw vertical lines at onset times
    for onset_time in onset_times:
      ax.axvline(x=onset_time, color="white", alpha=0.4, linewidth=0.8, linestyle="--")

  fig.tight_layout()
  return fig


def mkplot_melspectrogram(
  wav: WavRAW, cmap="magma", title="Mel-spectrogram", onset_times=None, **kwargs
):
  """Plot the mel-spectrogram and show it. Returns the created Figure."""
  mel = wavraw_to_melspectrogram(wav, **kwargs)
  fig = melspectrogram_figure(
    mel,
    cmap=cmap,
    title=title,
    onset_times=onset_times,
  )
  # plt.show()
  return fig

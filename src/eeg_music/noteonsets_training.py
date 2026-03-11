from pathlib import Path
from fractions import Fraction
import random
from eeg_music.training import NoteOnsetsTrainingConfig
from eeg_music.eegnet import EEGNetConfig
from eeg_music.eegpt import USING_CHANNELS

config = NoteOnsetsTrainingConfig(
  # Model config
  model_config=EEGNetConfig(
    model_type="eegnet",
    chunk_width=128,  # 256Hz * 1/2s
    num_channels=len(USING_CHANNELS),  # 28 channels
    eeg_sample_rate=256,
    window_start=32,
    window_end=32 + 64,
    lr_config=1e-4,
    pos_weight=None,  # Can be tuned for class imbalance
  ),
  # Checkpoint path (if available, otherwise train from scratch)
  checkpoint_path=None,
  # Data settings
  data_path=Path("./datasets/bcmi_preprocessed/bcmi_combined_noteonsets_28ch"),
  data_loader_num_workers=4,
  prefetch_factor=2,
  batch_size=32,
  # Training settings
  num_epochs=100,
  save_model_per_epochs=5,
  val_every_n_epoch=1,
  # Dataset split settings
  ds_p_train=0.85,
  ds_p_val=0.0,
  ds_split_seed=42,
  ds_use_test_for_val=True,
  ds_train_repeated_mul=1,
  ds_val_repeated_mul=1,
  ds_test_repeated_mul=10,
  ds_chunk_width=Fraction(1, 2),
  # Wandb logging
  wandb_log_model="all",
  project_name="neural-noteonsets-decoding",
  run_name="eegnet-onset-detection",
  run_extra_name="0",
  randint=random.randint(0, 1000),
  save_path="eegnet-onset-detection-ckpt",
  # Learning rate finder
  use_learning_rate_finder=False,
  # Dataloader settings
  include_info=False,
)

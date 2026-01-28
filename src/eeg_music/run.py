from pathlib import Path

# from eeg_music.training import config
from eeg_music.eegpt import LRCosine, LRStepLR
from eeg_music.emotion_eegnet import (
  EmotionEEGNetTrainingConfig,
  EmotionEEGNetModelConfig,
  BinaryEmotionEEGNetTraining,
)
from eeg_music.eegnet import ATCNetConfig, EEGNetConfig, FBCNetConfig, TSCeptionConfig
from eeg_music.dataloader import TrialWiseSplit
from fractions import Fraction
from eeg_music.eegpt import UseAdamW

import os
import shutil


# assuming mean ~ median here
mean_num_onsets = {
  18.5: 36.09252738654147,
  4: 7.803789705198156,
  8: 15.607579410396312,
  12: 23.411369115594468,
  16: 31.215158820792624,
}


def create_config(
  model_config=None,
  ds_split_type=None,
  use_subject_specific=False,
  optimizer=None,
  lr_config: float | LRStepLR | LRCosine = 1e-5,
  num_epochs=200,
  batch_size=64,
  trial_length_secs=4,
):
  if model_config is None:
    model_config = TSCeptionConfig()
  if optimizer is None:
    optimizer = UseAdamW()
  if ds_split_type is None:
    ds_split_type = TrialWiseSplit()

  config = EmotionEEGNetTrainingConfig(
    model_config=EmotionEEGNetModelConfig(
      model_config=model_config,
      chunk_width=250 * trial_length_secs,  # 256Hz * 4s
      num_channels=18,
      eeg_sample_rate=250,
      num_classes=1,
      lr_config=lr_config,
      use_subject_specific=use_subject_specific,
      optimizer=optimizer,
      median_num_noteonsets=int(mean_num_onsets[trial_length_secs]),
    ),
    # data_path = Path("./onesubject_bcmi_37ch"),
    # data_path=Path("./datasets/onesubject_bcmi_combined_subject10_18ch"),
    data_path=Path("./datasets/onesubject_bcmi_combined_18ch_18s"),
    # data_path=Path("./datasets/bcmi_preprocessed/bcmi_combined_18ch_18s_onsets/"),
    # data_path=Path("./datasets/bcmi_combined_18ch"),
    batch_size=batch_size,
    data_loader_num_workers=2,
    prefetch_factor=2,
    ds_p_train=0.75,
    pin_memory=True,
    ds_split_seed=13,
    num_epochs=num_epochs,
    # ds_test_repeated_mul = 10,
    ds_test_repeated_mul=2,
    ds_train_repeated_mul=2,
    ds_chunk_width=Fraction(trial_length_secs, 1),
    ds_split_type=ds_split_type,
    run_name="eegnet-emotion-binary",
    save_path="eegnet-emotion-binary-ckpt",
  )
  return config


all_configs = [
  # create_config(model_config=EEGNetConfig(), lr_config=5e-4, num_epochs=50),
  # create_config(model_config=ATCNetConfig(), lr_config=1e-3, num_epochs=400),
  # create_config(model_config=TSCeptionConfig(hid_channels=24, num_T=10, num_S=10), lr_config=LRStepLR(initial_lr=1e-3, step_size=30, gamma=0.9), num_epochs=600, batch_size=512),
  # create_config(model_config=FBCNetConfig(num_bands=1), lr_config=LRStepLR(initial_lr=5e-4, step_size=1, gamma=0.9), num_epochs=40, batch_size=4096),
  # create_config(
  #   model_config=TSCeptionConfig(),
  #   lr_config=LRStepLR(initial_lr=1e-3, step_size=10, gamma=0.9),
  #   num_epochs=1000,
  #   batch_size=1024,
  # ),
  create_config(
    model_config=FBCNetConfig(),
    lr_config=LRStepLR(initial_lr=2e-4, step_size=10, gamma=0.9),
    num_epochs=30,
    batch_size=256,
    trial_length_secs=12,
    # use_subject_specific=True,
    optimizer=UseAdamW(),
  ),
  create_config(
    model_config=ATCNetConfig(),
    lr_config=LRStepLR(initial_lr=2e-5, step_size=10, gamma=0.9),
    num_epochs=30,
    batch_size=256,
    trial_length_secs=16,
    # use_subject_specific=True,
    optimizer=UseAdamW(weight_decay=0.1),
  ),
  # create_config(
  #   model_config=TSCeptionConfig(),
  #   lr_config=LRStepLR(initial_lr=2e-5, step_size=10, gamma=0.9),
  #   num_epochs=100,
  #   batch_size=256,
  #   trial_length_secs=8,
  #   # use_subject_specific=True,
  #   optimizer=UseAdamW(weight_decay=0.1),
  # ),
  # create_config(
  #   model_config=TSCeptionConfig(),
  #   lr_config=LRStepLR(initial_lr=2e-4, step_size=10, gamma=0.9),
  #   num_epochs=100,
  #   batch_size=256,
  #   trial_length_secs=16,
  # ),
  # create_config(
  #   model_config=TSCeptionConfig(),
  #   lr_config=LRStepLR(initial_lr=1e-3, step_size=10, gamma=0.9),
  #   num_epochs=1000,
  #   batch_size=512,
  #   trial_length_secs=18,
  # ),
  create_config(
    model_config=ATCNetConfig(),
    lr_config=LRStepLR(initial_lr=1e-4, step_size=10, gamma=0.9),
    num_epochs=1000,
    batch_size=512,
  ),
  create_config(
    model_config=EEGNetConfig(),
    lr_config=LRStepLR(initial_lr=1e-4, step_size=10, gamma=0.9),
    num_epochs=1000,
    batch_size=512,
  ),
  # create_config(model_config=EEGNetConfig(), lr_config=LRStepLR(initial_lr=1e-3, step_size=3, gamma=0.9), num_epochs=100, batch_size=2048),
  # create_config(model_config=ATCNetConfig(), lr_config=LRStepLR(initial_lr=1e-3, step_size=3, gamma=0.9), num_epochs=100, batch_size=2048),
  # create_config(model_config=ATCNetConfig(tcn_depth=3, F1=8, num_windows=6, D=1), lr_config=LRStepLR(initial_lr=1e-3, step_size=10, gamma=0.9), num_epochs=120, batch_size=4096),
  # create_config(model_config=TSCeptionConfig(), lr_config=LRStepLR(initial_lr=5e-4, step_size=4, gamma=0.9), num_epochs=100, batch_size=1024),
  # create_config(model_config=TSCeptionConfig(), lr_config=LRStepLR(initial_lr=1e-3, step_size=30, gamma=0.9), num_epochs=600, batch_size=128),
  # create_config(model_config=TSCeptionConfig(), lr_config=LRStepLR(initial_lr=1e-3, step_size=30, gamma=0.9), num_epochs=600, batch_size=64),
  # create_config(model_config=ATCNetConfig(), lr_config=LRStepLR(initial_lr=1e-3, step_size=30, gamma=0.9), num_epochs=600, batch_size=64),
  # create_config(model_config=TSCeptionConfig(hid_channels=24, num_T=10, num_S=10), lr_config=LRStepLR(initial_lr=1e-3, step_size=30, gamma=0.9), num_epochs=600, batch_size=64),
  # create_config(model_config=TSCeptionConfig(hid_channels=24, num_T=10, num_S=10), lr_config=1e-3, num_epochs=400),
  # create_config(lr_config=1e-3, num_epochs=50),
  # create_config(lr_config=1e-4),
  # create_config(),
]

i = 0
for config in all_configs:
  directory_to_remove = "eegnet-emotion-binary-ckpt"
  # if os.path.exists(directory_to_remove) and i > 0:
  if os.path.exists(directory_to_remove):
    shutil.rmtree(directory_to_remove)
    print(f"Directory '{directory_to_remove}' removed successfully.")
  else:
    print(f"Directory '{directory_to_remove}' does not exist.")

  training = BinaryEmotionEEGNetTraining(config)
  # training = EmotionEEGNetTraining(config)
  model, trainer, dataloaders = training.run()
  i += 1

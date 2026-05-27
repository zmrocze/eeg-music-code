"""Train CNNClassifier to predict labels from EEG (MusingMusicIdData datasets)."""

from pathlib import Path
from fractions import Fraction
from eeg_music.data import (
  EEGMusicDataset,
  ArrayStratifiedSamplingDataset,
)
from eeg_music.eegpt import UseAdamW
from eeg_music.mel_training import (
  CNNClassifierConfig,
  ClassifierModelConfig,
  ClassifierTrainingConfig,
  ClassifierTraining,
)


if __name__ == "__main__":
  trial_length_secs = Fraction(1, 1)

  ds = EEGMusicDataset.load_ondisk(
    # Path("./datasets/musing_preprocessed/musing_pre_60ch/")
    # Path("./datasets/bcmi_preprocessed/bcmi_notes_60ch")
    # Path("./datasets/musing_preprocessed/musing_basic_id_60ch")
    Path("./datasets/musing_preprocessed/musing_pre_60ch")
    # Path("./datasets/musing_preprocessed/musing_basic_id_60ch")
    # Path("./datasets/musing_preprocessed/musing_mel64_60ch")
    #### Path("./datasets/bcmi_preprocessed/bcmi_ids_60ch")
    # Path("./datasets/bcmi_preprocessed/bcmi_emotion_60ch/")
    # Path("./datasets/musing_preprocessed/musing_basic_id_129ch/")
  )

  # filtered_ds = EEGMusicDataset()
  # filtered_ds.df = pd.DataFrame(
  #   ds.df[(ds.df["subject"] == "10")].reset_index(drop=True)
  # )
  # filtered_ds.music_collection = ds.music_collection
  # ds = filtered_ds

  # train_ds, test_ds = temporal_train_test_split(ds, length_sec=Fraction(20, 1))
  # val_ds, test_ds = temporal_train_test_split(ds, length_sec=Fraction(20, 1))
  splitted = ds.trial_wise_split(p_train=0.6, p_val=0.2)

  # train_ds = prepare_ds(splitted["train"], trial_length_secs)
  # test_ds = prepare_ds(splitted["test"], trial_length_secs)

  # splitted = ds.subject_wise_split(p_train=0.6, p_val=0.2)

  # train_ds = ArrayStratifiedSamplingDataset(
  #   train_ds, 10, trial_length_secs=Fraction(1, 1)
  # )
  # val_ds = ArrayStratifiedSamplingDataset(
  #   val_ds, 10, trial_length_secs=Fraction(1, 1)
  # )
  # test_ds = ArrayStratifiedSamplingDataset(
  #   test_ds, 10, trial_length_secs=Fraction(1, 1)
  # )

  train_ds = ArrayStratifiedSamplingDataset(
    splitted["train"],  # type: ignore
    10,
    trial_length_secs=trial_length_secs,
  )
  val_ds = ArrayStratifiedSamplingDataset(
    splitted["val"],  # type: ignore
    10,
    trial_length_secs=trial_length_secs,
  )
  test_ds = ArrayStratifiedSamplingDataset(
    splitted["test"].merge(splitted["val"]),  # type: ignore
    10,
    trial_length_secs=trial_length_secs,
  )

  # --- Cross-entropy variant (multi-class) ---
  config = ClassifierTrainingConfig(
    model_config=ClassifierModelConfig(
      # model_config=CNNClassifierConfig(in_channels=1, dropout=0.25, channels=16), # 0.1510416716337204
      # model_config=CNNClassifierConfig(in_channels=1, dropout=0.25, channels=32), # 0.16562500596046448
      # model_config=CNNClassifierConfig(in_channels=1, dropout=0.25, channels=64), # 0.16354165971279144
      model_config=CNNClassifierConfig(
        in_channels=1, dropout=0.25, channels=128
      ),  # 0.18020834028720856
      # model_config=CNNClassifierConfig(in_channels=1, dropout=0.25, channels=256), # 0.16458334028720856
      #  model_config=CNNClassifierRawConfig(in_channels=1, dropout=0.25),
      num_classes=12,
      # num_classes=3456,
      loss="ce",
      # lr_config=LRCosine(max_lr=1e-5, T_0=10, T_mult=2),
      lr_config=1e-5,
      # lr_config=1e-5,
      optimizer=UseAdamW(),
    ),
    batch_size=120,
    num_epochs=4000,
    project_name="cnn-classifier-musing",
    run_name="cnn-classifier-musing",
    save_path="cnn-classifier-musing",
  )

  training = ClassifierTraining(config, train_ds, val_ds, test_ds)
  model, trainer, dataloaders = training.run()

  # --- Binary cross-entropy variant (uncomment to use) ---
  # config_bce = ClassifierTrainingConfig(
  #   model_config=ClassifierModelConfig(
  #     model_config=CNNClassifierConfig(in_channels=1, dropout=0.25),
  #     num_classes=2,
  #     loss="bce",
  #     lr_config=LRCosine(max_lr=3e-4, T_0=10, T_mult=2),
  #     optimizer=UseAdamW(),
  #   ),
  #   batch_size=64,
  #   num_epochs=200,
  #   project_name="cnn-classifier-bce",
  #   run_name="cnn-classifier-bce",
  #   save_path="cnn-classifier-bce-ckpt",
  # )
  #
  # training_bce = ClassifierTraining(config_bce, train_ds, val_ds, test_ds)
  # model_bce, trainer_bce, dataloaders_bce = training_bce.run()

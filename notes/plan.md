
# Magisterka

## tytuł

optymistycznie:
"Improving neural decoding of music from the EEG"
pesymistycznie:
"Neural decoding of music from the EEG with diffusion"

## Plan

Publikacja ["Neural decoding of music from the EEG"](https://www.nature.com/articles/s41598-022-27361-x) dekoduje muzykę, której słucha uczestnik badania, na podstawie mierzonego sygnału EEG.
Zbiór danych (kilka różnych, razem kilkanaście godzin nagrań) z równoczesną muzyką i sygnałem fal mózgowych EEG jest dostępny w internecie.
Autorzy używają sprytnego preprocessingu i sieci bi-LSTM z ~5M parametrów.

Moje pytanie badawcze: Czy da się poprawić wynik używając technik znanych np. z zadań audio generation albo audio source separation?

Konkretnie myślę o architekturze typu: 

 - model A buduje reprezentacje operując na spektrogramie sygnału EEG
 - drugi model B działa jak vocoder który generuje kilkaset milisekund fali dźwiękowej na podstawie odpowiedniego fragmentu outputu modelu A

gdzie model A to może być open-sourcowy model [EEGPT](https://github.com/BINE022/EEGPT) po finetuningu (tj. "foundation" model typu transformer uczony self-supervised, ~10M parametrów).
Z kolei model B to może być model na bazie architektury DiffWave/UNet, który zaczyna od szumu i w procesie dyfuzji generuje fale dźwiękową na podstawie ("conditioned") reprezentacji otrzymanej z modelu A, która powinna kodować spektralne charakterystyki dekodowanej muzyki.

Wątpliwość: brakuje mi doświadczenia by pewnie oszacować czy posiadam wystarczająco danych i mocy obliczeniowej by osiągnąć coś lepszego niż oryginalny LSTM.
Posiadam +100$ do wydania w chmurze dzięki Github Student Pack, a w razie faktycznego otrzymania niezłych wyników i potrzeby na więcej chciałbym się zgłosić o przyznanie czasu komputera w [WCSS](https://wcss.pl/uslugi/25/przetwarzanie-danych-na-superkomputerze/) (czy posiadam do tego prawo jako magistrant?).

### PM

1. Datasets

 - [ ] access nature's dataset
 - [ ] access nmed datasets
 - [ ] access other available datasets (list below)
 - [ ] summarize metadata

2. Pretrained models

 - [ ] access neuro-gpt
 - [ ] access eegpt 1
 - [ ] access eegpt 2
 - [ ] summarize differences



## trening

np. ostatnia warstwa eegpt trenowana.

eeg, music <- batch
A <- eegpt eeg
B <- vocoder A [eeg] ## diffusion loop
loss music B

## plan w punktach

1. Indexable dataset.
 - [ ] combined training and calibration
 - [ ] Random per subject test-val-train split.
 - [ ] common preprocessing
  * [ ] 
 - [ ] Add fmri, scores etc.
2. Preprocessing
3. Basic model: UNet/Diffwave on 0.1-1s chunks training.
4. model: EEGpt finetuned to generate spectrograms
5. Checkpointing. Evalution by ssim.


## plan update

1. Eksperyment ze zmienianą długością próbki: noteonsets.

2. Eksperyment ze zmienianą długością próbki: emotion.

3. 


## plan if sleep

1. experiment window length

2. experiment models

both? "use of neural nets in eeg classification"


1. neural nets

 - eegnet
 - attention
 - tsception

2. eeg
 
 - physics
 - history of use
 - datasets
  * sleep edf
  * music

3. experiment
 - eeg music


## plan svm

przetestowane:
 - [x] ✅ musing prefix-suffix-split single-subject songid-prediction train_musing_baseline.ipynb
 - [x] ✅ bcmi-training prefix-suffix-split single-subject songid-prediction train_bcmi_baseline.ipynb
 - [x] ❌ bcmi-training trial-split single-subject noteonsets-prediction train_bcmi_notonsets.ipynb
 - [x] ❌ bcmi-training trial-split single-subject emotion-prediction train_bcmi_emotion.ipynb
 - [x] ✅❌ Negatywny eksperyment DTW: musing prefix-infix-suffix-split single-subject songid-prediction
 - [x] ✅ 16% acc musing subject-split songid-prediction

## todo 

 - balanced (emotion wise) trial split
 - neural nets on musing subject-split songid-prediction: preliminary got 0.16500000655651093
 - calculate f1 score in multi class

## tytuły

"Classification/Prediction of listened music tempo from EEG signal"
"Classification of listened music from eeg signal"
"Prediction of listened music features from eeg signal"

"Prediction tasks with EEG signal"

""

## plan magisterki

### wstep o eeg

argumenty medyczno fizjologiczne dlaczego to słaby sygnał. co da sie wyłuskać a co trudno.

#### jak zbierane

#### do czego przysłużyło się w badaniach, historia

#### bcmi - opis datasetu

#### musing - opis datasetu

#### ...

### metody

#### xgboost, knn, svm

#### neural nets

#### dtw

### zadania

#### predykcja piosenki

##### musing

###### sukces 16%

###### dtw

###### bcmi

#### noteonsets

##### single subj 53%
reszta fail

#### emotion codes

❌

### podsumowanie







do dodania do pracy:
 - stratified sampling ablation
 - reconstruction related
 - reconstruction fails:
  * cnn
  * do promotora ?: eegpt
 - emotion pred fails
 - methods:
  * CNNClassifier
  * CNNReconstruction




a co z:
 - pretrained eeg
 - [ ] test, validation, repeating - czy mam dobrze?
 - dokładne accuracy czy przybliżone
 - hydra-net a subject-specific
 - w pdf jest jedynie wynik onesubj-bcmi-songid co nie pozwala mówić o generalizacji

pytanie:
 - jak ustalić że wynik jest statystycznie istotny. jak ustalić że jest failem


eksperymenty:
 - liczba ica components
 - ica vs raw eeg
 - [ ] stratified sampling usefulness
 - [ ] varying length, discussion vs pandey
 - ablation: band power ICA vs raw EEG
 - loudness?

faile:
 - 

## results:

### raw eeg ablatian

porównanie na cnn? ale wtedy inne architektury

### stratified ablation

doesn't improve results, but we keep using it anyway

### musing all subjects subject-wise split 3s

train_musing_baseline_fullset
XGBoost Accuracy: 0.1956
SVM Accuracy:     0.1177
KNN Accuracy:     0.1737

# 1s

jako część stratified ablation:
(10, 'XGBoost'): 0.19895833333333332,
(10, 'SVM'): 0.12229166666666667,
(10, 'KNN'): 0.1675,
NN: 0.1740 (?) jaka


### musing temporal-split 1s

XGBoost Accuracy: 0.9118
SVM Accuracy:     0.4222
KNN Accuracy:     0.7352

### bcmi temporal-split 1s
single-subject
NN      & ---    \\

### bcmi noteonsets

NN      & - \\

### bcmi emotions

XGBoost Accuracy: 0.1089
SVM Accuracy:     0.1123
KNN Accuracy:     0.1117
NN      & 0.1130 \\
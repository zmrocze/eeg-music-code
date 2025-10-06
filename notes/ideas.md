
1. are songs stereo, mono? data augmentation?
2. how about a task that lets ai listen to first seconds of a song and need to fill rest based on eeg?
 - how to do it: diffusion, self-supervised with missing patches (joint eeg, audio)?
3. how to pass raw wav to the vocoder? 
 - preprocessing the signal by taking consecutive differences? reasearch: differential entropy
4. Q: eegpt operates on 250ms patches. how much sense does it make to task it to generate finer grained spectrogram?
5. give subject+dataset embedding to the model.
6. try lstm
7. try reconstructor + different prediction task (ideas?)
8. emotion classification?
9. less often auroc
10. parametric main (in a form of a class?)
11. 
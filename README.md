# LINGUISTIC CONTEXT AIDED SPEECH EMOTION RECOGNITION IN CONVERSATION WITH HUBERT-ASR
Achieves SOTA results on [IEMOCAP](https://sail.usc.edu/iemocap/) Speech Emotion Recognition for speaker-independent setting (compared to the best models listed in these [papers](https://paperswithcode.com/sota/speech-emotion-recognition-on-iemocap))

*Under Submission on ICASSP 2024

## Abstract
This paper proposes a novel speech emotion recognition system in a speech conversation setting, which is an ongoing research problem. Traditional LSTM and GRU-based methods suffer from vanishing gradients in longer sequences that hinder the learning of long-term dependency. Meanwhile, for HuBERT and transformer/attention-based methods, a longer temporal view of speech features requires higher computational memory which is often infeasible. Developing a robust speech emotion recognition system with context understanding from past utterances remains a challenge. To mitigate this challenge, we aim to leverage linguistic context from past utterances provided through Automatic Speech Recognition (ASR) system and RoBERTa language model. Furthermore, we use a co-attention layer to fuse these linguistic context features with the audio features of the current utterance extracted from fine-tuned HuBERT model. Implementation of this method shows that we achieve higher weighted accuracy (WA) on the IEMOCAP corpus compared to the prevailing state-of-the-art (SoTA) model.


## Model
<img src="https://github.com/bellagodiva/Context-Aided-Speech-Emotion-Recognition/blob/973407b9e191da28237a47159ada4d9feeeceb0c/models.png" width=820>

## Results
<img src="https://github.com/bellagodiva/Context-Aided-Speech-Emotion-Recognition/blob/bbfbca1a1e8d45f6365a0e4bb7b9dc175577fc3a/results.png" width=420>

## Code
main_transcription.py for transcription context

main_acoustic_transcription.py for acoustic+transcription

input default value for num past utterances (num_past_utterance=N) in src/at_dataset.py or dataset.py to include N past utterances as context

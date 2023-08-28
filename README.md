# Context-Aided Speech Emotion Recognition
Achieves SOTA results on [IEMOCAP](https://sail.usc.edu/iemocap/) Speech Emotion Recognition for speaker independent setting (compared to the best models listed in these [papers](https://paperswithcode.com/sota/speech-emotion-recognition-on-iemocap))
Under Submission on ICASSP 2024

## Abstract
This paper proposes a novel speech emotion recognition system in a conversation setting, which has not been well addressed in the previous literature. Since longer temporal views of speech features requires high computational mem- ory which is often infeasible and suffers from vanishing and exploding gradient problems that hinder the learning of long-term dependency, it is challenging to develop a speech emotion recognition system that takes into account past utter- ances. To mitigate this challenge, we try to leverage linguistic context from past utterances provided through HuBERT Au- tomatic Speech Recognition (ASR) system and RoBERTa language model. Furthermore, we integrate emotional speech features extracted from fine-tuned HuBERT model which are absent in linguistic features to complement one another. Implementation of this method shows that we achieve higher weighted accuracy (WA) on the IEMOCAP corpus compared to the prevailing state-of-the-art (SoTA) model.


## Model
<img src="https://github.com/bellagodiva/Context-Aided-Speech-Emotion-Recognition/blob/682a0128c55d509e139893dd9cfe1ff87ad45d20/model.png" width=520>

## Results
<img src="https://github.com/bellagodiva/Context-Aided-Speech-Emotion-Recognition/blob/161a4457faa3ad5e803d5c4303d4ba96ca4cb8c8/results.png" width=920>

## Code
main_transcription.py for transcription context

main_acoustic_transcription.py for acoustic+transcription

input default value for num past utterances (num_past_utterance=N) in src/at_dataset.py or dataset.py to include N past utterances as context

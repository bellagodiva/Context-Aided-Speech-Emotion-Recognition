# Context-Aided Speech Emotion Recognition
Achieves SOTA results on [IEMOCAP](https://sail.usc.edu/iemocap/) Speech Emotion Recognition for speaker independent setting (compared to the best models listed in these [papers](https://paperswithcode.com/sota/speech-emotion-recognition-on-iemocap))

## Model
<img src="https://github.com/bellagodiva/Context-Aided-Speech-Emotion-Recognition/blob/682a0128c55d509e139893dd9cfe1ff87ad45d20/model.png" width=520>

## Results
<img src="https://github.com/bellagodiva/Context-Aided-Speech-Emotion-Recognition/blob/161a4457faa3ad5e803d5c4303d4ba96ca4cb8c8/results.png" width=920>

## Code
main_transcription.py for transcription context

main_acoustic_transcription.py for acoustic+transcription

input default value for num past utterances (num_past_utterance=N) in src/at_dataset.py or dataset.py to include N past utterances as context

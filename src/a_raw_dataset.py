"""utility and helper functions / classes."""
import json
import logging
import os
import random
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
import os.path
from os import path
import torch.nn.functional as F
from PIL import Image
import scipy.io
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_num_classes(DATASET: str) -> int:
    """Get the number of classes to be classified by dataset."""
    if DATASET == "meld":
        NUM_CLASSES = 7
    elif DATASET == "IEMOCAP":
        NUM_CLASSES = 4
    elif DATASET == "MELD_IEMOCAP":
        NUM_CLASSES = 7
    else:
        raise ValueError

    return NUM_CLASSES


def compute_metrics(eval_predictions) -> dict:
    """Return f1_weighted, f1_micro, and f1_macro scores."""
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(label_ids, preds, average="weighted")
    f1_micro = f1_score(label_ids, preds, average="micro")
    f1_macro = f1_score(label_ids, preds, average="macro")

    return {"f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}


def set_seed(seed: int) -> None:
    """Set random seed to a fixed value.

    Set everything to be deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_emotion2id(DATASET: str) -> Tuple[dict, dict]:
    """Get a dict that converts string class to numbers."""

    if DATASET == "MELD":
        # MELD has 7 classes
        emotions = [
            "neutral",
            "joy",
            "surprise",
            "anger",
            "sadness",
            "disgust",
            "fear",
        ]
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}

    elif DATASET == "IEMOCAP":
        # IEMOCAP originally has 11 classes but we'll only use 4 of them.
        
        emotions = [
            "neutral",
            "sad",
            "angry",
            "happy",
        ]
        '''
        emotions = [
            "neutral",
            "frustration",
            "sadness",
            "anger",
            "excited",
            "happiness",
        ]
        '''
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}

    elif DATASET == "iemocap-6":
        # IEMOCAP originally has 11 classes but we'll only use 6 of them.
        emotions = [
            "neutral",
            "frustration",
            "sadness",
            "anger",
            "excited",
            "happiness",
        ]
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}

    return emotion2id, id2emotion


class Multimodal_Datasets(torch.utils.data.Dataset):
    def __init__(
        self,
        DATASET="IEMOCAP",
        SPLIT="train",
        speaker_mode="upper",
        num_past_utterances=0,
        num_future_utterances=0,
        model_checkpoint="roberta-large",
        ROOT_DIR="multimodal-datasets/",
        SEED=0,
    ):
        """Initialize emotion recognition in conversation text modality dataset class."""
        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.model_checkpoint = model_checkpoint
        self.emotion2id, self.id2emotion = get_emotion2id(self.DATASET)
        self.SEED = SEED

        """Initialize emotion recognition in conversation audio vision modality dataset class."""
        #self.audio_root = path.join(ROOT_DIR, self.DATASET, 'norm_aufeatures.npy')
        #self.video_root = path.join(ROOT_DIR, self.DATASET, 'visual')
        #norm_aufeature = np.load(self.audio_root, allow_pickle=True)
        #self.norm_aufeatures= norm_aufeature.item()
        
        self.utterance_transcriptions = np.load('/mnt/hard2/bella/IEMOCAP/audio_transcription/transcription_'+SPLIT+'.npy', allow_pickle=True)
        self._load_utterance_ordered()
        self._string2tokens()

    def _load_utterance_ordered(self):
        """Load the ids of the utterances in order."""
        if self.DATASET in ["MELD", "IEMOCAP"]:
            path = os.path.join(self.ROOT_DIR, 'IEMOCAP', "utterance-ordered.json")
        elif self.DATASET == "MELD_IEMOCAP":
            path = "./utterance-ordered-MELD_IEMOCAP.json"

        with open(path, "r") as stream:
            self.utterance_ordered = json.load(stream)[self.SPLIT]

    def __len__(self):
        return len(self._inputs)

    def _load_utterance_speaker(self, uttid, speaker_mode) -> dict:
        """Load an speaker-name prepended utterance and emotion label"""

        if self.DATASET in ["MELD", "IEMOCAP"]:
            text_path = os.path.join(
                self.ROOT_DIR, 'IEMOCAP', "raw-texts", self.SPLIT, uttid + ".json"
            )
        elif self.DATASET == "MELD_IEMOCAP":
            assert len(uttid.split("/")) == 4
            d_, s_, d__, u_ = uttid.split("/")
            text_path = os.path.join(self.ROOT_DIR, d_, "raw-texts", s_, u_ + ".json")

        with open(text_path, "r") as stream:
            text = json.load(stream)

        #transcriptions
        ##if uttid in self.utterance_transcriptions.item():
            ##utterance = self.utterance_transcriptions.item()[uttid].strip()
        ##else:
            ##utterance='BLANK'
        #raw
        utterance = text["Utterance"].strip()
        emotion = text["Emotion"] #del this later

        if self.DATASET == "MELD":
            speaker = text["Speaker"]
        elif self.DATASET == "IEMOCAP":
            sessid = text["SessionID"]
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {
                "Ses01": {"Female": "Mary", "Male": "James"},
                "Ses02": {"Female": "Patricia", "Male": "John"},
                "Ses03": {"Female": "Jennifer", "Male": "Robert"},
                "Ses04": {"Female": "Linda", "Male": "Michael"},
                "Ses05": {"Female": "Elizabeth", "Male": "William"},
            }[sessid][text["Speaker"]]
        elif self.DATASET == "MELD_IEMOCAP":
            speaker = ""
        else:
            raise ValueError("{self.DATASET} not supported!!!!!!")

        if speaker_mode is not None and speaker_mode.lower() == "upper":
            utterance = speaker.upper() + ": " + utterance
        elif speaker_mode is not None and speaker_mode.lower() == "title":
            utterance = speaker.title() + ": " + utterance

        return {"Utterance": utterance, "Emotion": emotion}
    
    def _create_input(
        self, diaids, speaker_mode, num_past_utterances, num_future_utterances
    ):
        """Create an input which will be an input to RoBERTa."""

        args = {
            "diaids": diaids,
            "speaker_mode": speaker_mode,
            "num_past_utterances": num_past_utterances,
            "num_future_utterances": num_future_utterances,
        }

        logging.debug("arguments given: {args}")
        tokenizer = AutoTokenizer.from_pretrained('roberta-large', use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes["roberta-large"]
        num_truncated = 0

        inputs = []
        f = open(path.join(self.ROOT_DIR, self.DATASET, 'emotion_label.json'))
        emotion_labels = json.load(f)
       
        for diaid in tqdm(diaids):
            ues=[]
            uttids=[]
            for uttid in self.utterance_ordered[diaid]:
                if uttid in emotion_labels:
                    #print(os.path.isfile('/mnt/hard2/bella/IEMOCAP/audio_raw/'+uttid+'.wav'))
                    ues.append(self._load_utterance_speaker(uttid, speaker_mode))
                    uttids.append(uttid)
            num_tokens = [len(tokenizer(ue["Utterance"])["input_ids"]) for ue in ues]

            for idx, ue in enumerate(ues):
                #print(path.join(self.audio_root, ids[idx]+'.mat'), ids[idx],path.join(self.video_root, ids[idx]))
                uttid = uttids[idx]
                filename = '/mnt/hard2/bella/IEMOCAP/audio_raw/'+uttid+'.wav'
                y, sr = librosa.load(filename, sr=16000)
                audio_raw = np.array(y)
                label = self.emotion2id[emotion_labels[uttid]]
                label = [label]
                #temp_label=torch.zeros(4)
                #temp_label[label]=1
                #label= temp_label #one hot encoded

                indexes = [idx]
                indexes_past = [
                    i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
                ]
                indexes_future = [
                    i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
                ]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future) - len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if (
                            sum([num_tokens[idx_] for idx_ in indexes])
                            > max_model_input_size
                        ):
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if (
                            sum([num_tokens[idx_] for idx_ in indexes])
                            > max_model_input_size
                        ):
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_]["Utterance"] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[-1]
                    else:
                        final_utterance = (
                            " ".join(utterances[:-1]) + "</s></s>" + utterances[-1]
                        )

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            utterances[0] + "</s></s>" + " ".join(utterances[1:])
                        )

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            " ".join(utterances[:offset])
                            + "</s></s>"
                            + utterances[offset]
                            + "</s></s>"
                            + " ".join(utterances[offset + 1 :])
                        )
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask["input_ids"]
                attention_mask = input_ids_attention_mask["attention_mask"]

                text_input = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                #inputs.append(input_)
                inputs.append({"text": text_input, "audio_raw":audio_raw, "label":label})
                #inputs.append({"text": text_input, "audio": aus[idx], "visual": vis[idx], "label":label})

        logging.info("number of truncated utterances: " + str(num_truncated))
        return inputs

    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        logging.info("converting utterances into tokens ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(self.SEED)
        random.shuffle(diaids)

        logging.info("creating input utterance data ... ")
        self._inputs= self._create_input(
            diaids=diaids,
            speaker_mode=self.speaker_mode,
            num_past_utterances=self.num_past_utterances,
            num_future_utterances=self.num_future_utterances,
        )

    def __getitem__(self, index):
        # X = (self._inputs[index]['text'], self._inputs[index]['audio'], self._inputs[index]['visual'])
        # Y = self._inputs[index]['label']
        # return X, Y 
        return {
            'source': {
                'text_inputids': torch.LongTensor(self._inputs[index]['text']['input_ids']),
                # 'text_mask': torch.BoolTensor(self._inputs[index]['text']['attention_mask']),
                'audio_raw': torch.Tensor(self._inputs[index]['audio_raw']),
                #'visual': torch.Tensor(self._inputs[index]['visual']),
            },
            'target': self._inputs[index]['label']
        }

    @staticmethod
    def lengths_to_mask(lengths, max_len=None, negative_mask=False):
        if max_len is None:
            max_len = max(lengths)

        if len(lengths.shape) == 1:
            lengths = lengths.unsqueeze(1)

        batch_size = lengths.size(0)
        # batch_size, max_len
        mask = torch.arange(max_len).expand(batch_size, max_len).type_as(lengths) < lengths

        if negative_mask:
            mask = ~mask

        return mask

    def collater_modal(self, audios, audio_sizes, pad_value=0.0):

        audio_size = max(audio_sizes)

        ### 1st option
        collated_audios = torch.stack([F.pad(a, (*(0, 0)*(len(audios[0].shape)-1), 0, audio_size-len(a)), "constant", pad_value) for a in audios])
        mask = self.lengths_to_mask(audio_sizes)
        ### 1st option

        # ### 2nd option
        # audio_feat_shape = list(audios[0].shape[1:])
        # collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        # mask = torch.BoolTensor(len(audios), audio_size).fill_(True)
        # for i, audio in enumerate(audios):
        #     diff = len(audio) - audio_size
        #     if diff == 0:
        #         collated_audios[i] = audio
        #     elif diff < 0:
        #         collated_audios[i] = torch.cat([audio, audio.new_full([-diff]+audio_feat_shape, pad_value)])
        #         mask[i, diff:] = False
        # ### 2nd option

        return collated_audios, mask


    def collater(self, samples):
        
        audio_source = [s['source']['audio_raw'] for s in samples]
        audio_sizes = torch.LongTensor([len(s) for s in audio_source])
        audio_raw, audio_mask = self.collater_modal(audio_source, audio_sizes)
        '''
        visual_source = [s['source']['visual'] for s in samples]
        visual_sizes = torch.LongTensor([len(s) for s in visual_source])
        visual, visual_mask = self.collater_modal(visual_source, visual_sizes)
        '''
        text_source = [s['source']['text_inputids'] for s in samples]
        text_sizes = torch.LongTensor([len(s) for s in text_source])
        text, text_mask = self.collater_modal(text_source, text_sizes, pad_value=0)
        
        return {
            'source': {
                'text_inputids': text,
                'text_mask': text_mask,
                'audio_raw': audio_raw,
                'audio_mask': audio_mask,
                #'visual': visual,
                #'visual_mask': visual_mask,
            },
            'target': torch.Tensor([s['target'] for s in samples])
        }

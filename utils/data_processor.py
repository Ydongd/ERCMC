import pandas as pd
from utils.data_utils import InputExample
import os
import json

class DataProcessor:
    def __init__(self, data_path, specific):
        if specific == "iemocap":
            self.all_labels = ["happy", "sad", "angry", "excited", "frustrated", "neutral"]
        elif specific == "meld":
            self.all_labels = ["anger", "disgust", "sadness", "joy", "surprise", "fear", "neutral"]
        elif specific == "emory":
            self.all_labels = ["Sad", "Mad", "Peaceful", "Powerful", "Joyful", "Neutral", "Scared"]
        elif specific == "dailydialog":
            self.all_labels = ['anger', 'disgust', 'fear', 'happiness', 'no_emotion', 'sadness', 'surprise']
        self.label2idx = {tag: idx for idx, tag in enumerate(self.all_labels)}
        self.idx2label = {idx: tag for idx, tag in enumerate(self.all_labels)}
        self.data_path = data_path
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.json'.format(split))
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        for i, dialog in enumerate(data):
            original_utterances = []
            generated_utterances = []
            speakers = []
            labels = []
            for utterance in dialog:
                original_utterances.append(utterance['text'])
                generated_utterances.append(utterance['generated'])
                speakers.append(utterance['speaker'])
                labels.append(utterance['label'])
            example = InputExample(guid=str(i), original_utterances=original_utterances, generated_utterances=generated_utterances, speakers=speakers, labels=labels)
            examples.append(example)
        return examples
        

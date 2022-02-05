import re

import pandas as pd
from torch.utils.data import Dataset

from Constants import *


class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, regex_transformations={}):
        
        
        
        '''
        df = pd.read_csv(dataset_file_path,encoding="gbk")
        df = df[['Emotion','Utterance']]
        emotion_labels = []
        for each in df["Emotion"]:
            if each =="surprise":
                emotion_labels.append(0)
            elif each =="fear":
                emotion_labels.append(1)
            elif each =="disgust":
                emotion_labels.append(2)
            elif each =="joy":
                emotion_labels.append(3)
            elif each =="sadness":
                emotion_labels.append(4)
            elif each =="anger":
                emotion_labels.append(5)
            elif each =="neutral":
                emotion_labels.append(6)
            else:
                print(each)
                #exit()
        df["emotion_labels"] = emotion_labels
        df = df.drop(['Emotion'], axis=1)
        self.headlines = df.values
        '''
        lines = open(dataset_file_path,encoding="utf8").readlines()
        labels = [each.split()[0] for each in lines]
        lines = [" ".join(each.split()[1:]) for each in lines]
        self.headlines = []
        for a,b in zip(lines,labels):
            self.headlines.append([a,int(b)])
            
        #print(self.headlines)
        # Regex Transformations can be used for data cleansing.
        # e.g. replace
        #   '\n' -> ' ',
        #   'wasn't -> was not
        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        headline,emotion_label,  = self.headlines[index]
        #print(headline,emotion_label)
        for regex, value_to_replace_with in self.regex_transformations.items():
            headline = re.sub(regex, value_to_replace_with, headline)
        # Convert input string into tokens with the special BERT Tokenizer which can handle out-of-vocabulary words using subgrams
        
        
        tokens = self.tokenizer.tokenize(headline)

        
        tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        
        segment_ids = [0] * len(input_ids)

        
        input_mask = [1] * len(input_ids)

        
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        input_mask = input_mask + [0] * padding_length
        segment_ids = segment_ids + [0] * padding_length

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH

        return torch.tensor(input_ids, dtype=torch.long, device=DEVICE), \
               torch.tensor(segment_ids, dtype=torch.long, device=DEVICE), \
               torch.tensor(input_mask, device=DEVICE), \
               torch.tensor(emotion_label, dtype=torch.long, device=DEVICE)

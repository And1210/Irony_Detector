import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg
from transformers import BertTokenizer
import re
from keras.preprocessing.sequence import pad_sequences

def preprocess(tweet):
    # Convert www.* or https?://* to 'link'
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','link',tweet)
    # Separate hashtags into separate words
    tweet = ' '.join(re.split(r'(?=[A-Z])', tweet))
    # Convert @username to 'username'
    tweet = re.sub('@[^\s]+','username',tweet)
    # Convert to lower case
    tweet = tweet.lower()
    #Convert '#' to 'hash'
    tweet = tweet.replace('#','hash ')
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    return tweet

class TweetsDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        # self._data = pd.read_csv(os.path.join(configuration["dataset_path"], "{}.csv".format(self._stage)), sep="\t")
        self.dataset_path = configuration["dataset_path"]
        train_filename = os.path.join(self.dataset_path, "{}_dataset.txt".format(self._stage))
        self._data = pd.read_csv(train_filename, delimiter='\t', header=None, names=['label', 'tweet'], skiprows=1,usecols=[1,2])

        self.tweets = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        for i in range(self._data['tweet'].count()):
            processed_tweet = preprocess(self._data['tweet'][i])
            encoded_tweet = tokenizer.encode(processed_tweet, add_special_tokens=True)
            self.tweets.append(encoded_tweet)
        self.max_len = max([len(sen) for sen in self.tweets])
        self.tweets = pad_sequences(self.tweets, maxlen=self.max_len, dtype="long", value=0, truncating="post", padding="post")
        self.attn_mask = [[int(token_id > 0) for token_id in tweet] for tweet in self.tweets]
        self.labels = self._data.label.values

        self._transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):

        return np.array(self.tweets[index]), np.array(self.labels[index]), np.array(self.attn_mask[index])

    def __len__(self):
        # return the size of the dataset
        return self._data['tweet'].count()

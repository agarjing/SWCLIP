import json

import nltk
import numpy as np
import torch

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
stop_words = set(stopwords.words('english'))
from transformers import BertModel, BertTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    captions = []
    with open(f'/home/ZJ/data/coco/precomp/testall_caps.txt', 'rb') as f:
        for line in f:
            captions.append(line.decode().strip())
    print(len(captions))
    text_weight_info = []
    for i, caption in enumerate(captions):
        # weight of caption
        tokens = word_tokenize(caption)
        pos_tags = nltk.pos_tag(tokens)
        weight = np.ones(77, dtype=np.int32)
        for k, tag in enumerate(pos_tags):
            if k >= 76:
                break
            if 'NN' in tag[1]:
                weight[k + 1] = weight[k + 1] + 70
            elif 'JJ' in tag[1]:
                weight[k + 1] = weight[k + 1] + 30
            elif 'VB' in tag[1]:
                weight[k + 1] = weight[k + 1] + 50
        weights = weight.tolist()

        text_weight = {
            "caption": caption,
            "weight": weights
        }

        text_weight_info.append(text_weight)
        print(i)

    with open("coco_testall_weight.json", "w") as json_file:
        json.dump(text_weight_info, json_file)
import json
import os

import nltk
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
# Image.LOAD_TRUNCATED_IMAGES = True
from timm.data import create_transform

from clip import clip

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import CLIPProcessor

stop_words = set(stopwords.words('english'))

"""
def get_vocab():
    model_name = '/home/ZJ/data/clip-vit-base-patch32'
    tokenizer = CLIPProcessor.from_pretrained(model_name)
    vocab = tokenizer.tokenizer.vocab
    return vocab, tokenizer
"""

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def swin_transform():
    transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment=None,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
    )
    return transform


# Flickr数据集的数据处理
class FlickrDataset(data.Dataset):
    def __int__(self, data_path, data_split, tokenizer, args=None, target_size=None, fextractor=None):
        self.data_path = data_path
        self.data_split = data_split
        self.tokenizer = tokenizer
        self.args = args
        self.swin_transform = swin_transform()
        self.fextractor = fextractor

        # load captions
        self.captions = []
        self.groups = []
        group = []
        with open(f'{self.data_path}/precomp/{self.data_split}_caps.txt', 'rb') as f:
            for line in f:
                self.captions.append(line.decode().strip())
                if len(group) < 5:
                    group.append(line.decode().strip())
                else:
                    self.groups.append(group)
                    group = [line.decode().strip()]
            self.groups.append(group)

        # load image_id
        with open(f'{self.data_path}/precomp/{self.data_split}_ids.txt', 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        # load id_path file
        with open(self.data_path + '/id_mapping.json', 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        self.ca_length = len(self.captions)
        num_images = len(self.images)
        if self.ca_length != num_images:
            self.im_div = 5
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # caption
        caption = self.captions[index]
        text = caption
        caption = self.tokenizer(str(caption).lower()).data
        caption = caption['input_ids']
        caption = torch.tensor(caption)

        # image
        img_index = index // self.im_div
        image_id = self.images[img_index]
        image_name = self.id_to_path['images'][image_id]
        image_name = image_name['filename']
        image_path = f'{self.data_path}/images/{image_name}'
        img = Image.open(image_path).convert('RGB')
        image = self.fextractor(img)
        img = self.swin_transform(img)

        target = []

        return image, caption, target, index, img_index, text, img

    def __len__(self):
        return self.ca_length


class CocoDataset(data.Dataset):
    def __init__(self, data_path, data_split, args=None, transform=None):
        self.data_path = data_path
        self.data_split = data_split
        self.args = args
        self.transform = transform

        # load captions
        self.captions = []
        with open(f'/home/hdu/data_ZJ/coco/precomp/{self.data_split}_caps.txt', 'rb') as f:
            for line in f:
                self.captions.append(line.decode().strip())
        print('cap')

        # load image
        with open(f'/home/hdu/data_ZJ/coco/precomp/{self.data_split}_ids.txt', 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]
        print('image')

        # load id_path file
        with open(self.data_path + '/id_mapping.json', 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)
        print('idtopath')

        with open(f'/home/hdu/data_ZJ/coco/preprocess/coco_{self.data_split}_weight.json', 'r') as f_weight:
            self.weight = json.load(f_weight)
        print('weight')

        with open(f'/home/hdu/data_ZJ/coco/preprocess/coco_train_detect_bert.json', 'r') as f_detect:
            self.train_detect = json.load(f_detect)
        print('train_detect')

        with open(f'/home/hdu/data_ZJ/coco/preprocess/coco_dev_detect_bert.json', 'r') as ff_detect:
            self.dev_detect = json.load(ff_detect)
        print('dev_detect')

        self.ca_length = len(self.captions)
        num_images = len(self.images)

        if self.ca_length != num_images:
            self.im_div = 5
        else:
            self.im_div = 1

    def __getitem__(self, index):

        ids = index

        # text
        caption = self.captions[index]

        text = clip.tokenize(caption)

        # weight of caption
        weights = []
        weight_infos = filter(lambda a: a['caption'] == caption, self.weight)
        for weight_info in weight_infos:
            weights = weight_info['weight']

        if len(weights) < 77:
            print(caption)
            print(index)
        weights = torch.tensor(weights)

        # caption list to get the high TFIDF words
        tfidfs = []

        # image
        img_index = index // self.im_div
        image_id = self.images[img_index]
        images_id2p = self.id_to_path['images']
        image_info = filter(lambda a: a.get('cocoid') == image_id, images_id2p)
        for name in image_info:
            image_name = name['filename']
            image_file = name['filepath']
        image_path = f'{self.data_path}/images/{image_file}/{image_name}'
        img = Image.open(image_path).convert('RGB')
        image = self.transform(img)

        # detection
        detect = []
        if image_file == 'train2014':
            detect_infos = filter(lambda d: d['file_name'] == image_name, self.train_detect)
        else:
            detect_infos = filter(lambda d: d['file_name'] == image_name, self.dev_detect)
        for detect_info in detect_infos:
            detect = detect_info['detect_bert']

        if len(detect) < 1536:
            print(index)
            print(image_path)
            print(image_id)
        detects = torch.tensor(detect)

        return text, tfidfs, weights, image, detects, ids

    def __len__(self):
        return self.ca_length


def collate_unsorted_fn(datas: object) -> object:
    texts, tfidfs, weights, images, detects, ids = zip(*datas)

    # Merge sentences
    txt_lengths = torch.LongTensor([len(cap) for cap in texts])

    return texts, tfidfs, weights, images, detects, txt_lengths, ids


def get_dataloader(dapath, data_split, args, batch_size,
                   shuffle=True, number_workers=0, swin_model=None):
    _, transform = clip.load("ViT-B/32", device=args.device)
    if args.data_name == 'f30k':
        target_size = 224
        dataset = FlickrDataset(dapath, data_split, args, target_size, transform)
    else:
        dataset = CocoDataset(dapath, data_split, args, transform)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, pin_memory=True,
                                  num_workers=number_workers,
                                  collate_fn=collate_unsorted_fn)

    return data_loader


def get_loaders(data_name, batch_size, workers, args, return_test=False):
    dapath = os.path.join(args.data_path, data_name)
    train_data, val_data = None, None
    if not return_test:
        train_data = get_dataloader(dapath, 'train', args, batch_size,
                                    True, workers)
        val_data = get_dataloader(dapath, 'dev', args, batch_size,
                                  False, workers)
    test_data = get_dataloader(dapath, 'testall', args, batch_size, False, workers)

    return train_data, val_data, test_data

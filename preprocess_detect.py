import os
import torch
from PIL import Image
import json
from argparse import ArgumentParser
from transformers import BertModel, BertTokenizer
import torch
import nltk
import numpy as np
import json
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
stop_words = set(stopwords.words('english'))
from transformers import BertModel, BertTokenizer
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
device = "cuda:3" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    """
    parser = ArgumentParser()
    parser.add_argument('--img', default='1254659.jpg', help='Image file')
    parser.add_argument('--config',
                        default='configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                        help='Config file')
    parser.add_argument('--checkpoint', default='mask_rcnn_swin_tiny_patch4_window7.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:3', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    image_file = '/home/ZJ/data/coco/images/val2014'
    img_names = os.listdir(image_file)

    image_info_list = []

    for i, img_name in enumerate(img_names):
        image_path = os.path.join(image_file, img_name)

        # test a single image
        result = inference_detector(model, image_path)
        # show the results
        labels_text, scores_final = show_result_pyplot(model, image_path, result, score_thr=0.6)

        image_info = {
            "file_name": img_name,
            "detect_res": labels_text,
            # 可以根据需要添加其他图片信息
        }

        # 将当前图片的信息字典添加到列表中
        image_info_list.append(image_info)
        print(i)

    with open("coco_test_detect.json", "w") as json_file:
        json.dump(image_info_list, json_file)
    """

    bert_tokenizer = BertTokenizer.from_pretrained('/home/ZJ/data/bert-base-uncased')
    bert_model = BertModel.from_pretrained('/home/ZJ/data/bert-base-uncased')

    with open("coco_test_detect.json", "rb") as f:
        detect_json = json.load(f)
    image_detect_info_list = []
    for i, detect in enumerate(detect_json):
        detect_res = []
        if len(detect['detect_res']) == 0:
            for j in range(2):
                detect_res.append('None')
        elif len(detect['detect_res']) == 1:
            for j in range(2):
                detect_res.append(detect['detect_res'][0])
        else:
            for j in range(2):
                detect_res.append(detect['detect_res'][j])

        x2 = torch.tensor([]).to(device)
        for word in detect_res:
            word = "[CLS] " + word + " [SEP]"
            tokenized_text = bert_tokenizer.tokenize(word)
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            with torch.no_grad():
                last_hidden_states = bert_model(tokens_tensor)[0]
            embedding = last_hidden_states.squeeze(0)
            word_embedding = embedding[1].to(device)
            x2 = torch.cat([x2, word_embedding], dim=0)

        words_bert = x2.tolist()

        image_detect = {
            "file_name": detect['file_name'],
            "detect_bert": words_bert
        }

        image_detect_info_list.append(image_detect)
        print(i)

    with open("coco_test_detect_bert.json", "w") as json_file:
        json.dump(image_detect_info_list, json_file)

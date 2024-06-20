import json
import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    model, preprocess = clip.load('ViT-B/32', jit=False)
    model.load_state_dict(torch.load('/home/hdu/code_ZJ/STCLIP-detect-weight/runs/B32/loss_B32_detectVision_weight_optstepLR_1e6to1e8.pth'))
    with open(f'/home/hdu/data_ZJ/coco/preprocess/coco_train_weight.json', 'r') as f_weight:
        weight = json.load(f_weight)
    with open(f'/home/hdu/data_ZJ/coco/preprocess/coco_train_detect_bert.json', 'r') as f_detect:
        train_detect = json.load(f_detect)

    caption = 'An elephant plays with a ball in an enclosed area .'
    text = clip.tokenize(caption).to(device)
    image_name = 'COCO_train2014_000000004956.jpg'
    image = preprocess(Image.open(image_name)).unsqueeze(0).to(device)

    weights = []
    weight_infos = filter(lambda a: a['caption'] == caption, weight)
    for weight_info in weight_infos:
        weights = weight_info['weight']
    weights = torch.tensor(weights).unsqueeze(0)


    tfidf = []

    detect = []
    detect_infos = filter(lambda d: d['file_name'] == image_name, train_detect)
    for detect_info in detect_infos:
        detect = detect_info['detect_bert']
    detects = torch.tensor(detect).unsqueeze(0)

    text = text.squeeze(dim=1).cuda()
    image = image.to(device)
    detects = detects.to(device)
    weights = weights.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image, detects)
        text_features = model.encode_text(text, tfidf, weights)

        # 归一化特征
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity = (image_features @ text_features.T).item()
    print(similarity)

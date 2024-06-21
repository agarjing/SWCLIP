import argparse
import os
import time
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import BertModel, BertTokenizer
from arguments import get_args
from logger import get_logger
from dataload import get_loaders
from clip import clip
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from evaluation import i2t, t2i


def rank_recall(d):
    i2t_r = i2t(d)
    t2i_r = t2i(d)
    rsum = sum(i2t_r) + sum(t2i_r)
    return i2t_r, t2i_r, rsum


def evaluate(model, epoch, iterator, logger, args, fold5=False):
    epoch_start = time.time()
    model.eval()
    img_embs, cap_embs, cap_lens = None, None, None
    with torch.no_grad():
        for i, data in enumerate(iterator):
            text, tfidf, weights, image, detect, txt_lengths, ids = data
            text = torch.stack(text, dim=0)
            image = torch.stack(image, dim=0)
            detect = torch.stack(detect, dim=0)
            weights = torch.stack(weights, dim=0)
            max_seq_length = max(txt_lengths)
            ids = np.array(ids)
            if torch.cuda.is_available():
                text = text.squeeze(dim=1).cuda()
                image = image.to(args.device)
                detect = detect.to(args.device)
                weights = weights.to(args.device)

            txt_emb = model.encode_text(text, tfidf, weights)
            txt_emb = txt_emb.to(torch.float32)
            img_emb = model.encode_image(image, detect)
            img_emb = img_emb.to(torch.float32)

            if img_embs is None:
                if img_emb.dim() == 3:
                    img_embs = torch.zeros((len(iterator.dataset),
                                            img_emb.size(1),
                                            img_emb.size(2)))
                    cap_embs = torch.zeros((len(iterator.dataset),
                                            max_seq_length,
                                            txt_emb.size(2)))
                else:
                    img_embs = torch.zeros((len(iterator.dataset),
                                            img_emb.size(1)))
                    cap_embs = torch.zeros((len(iterator.dataset),
                                            txt_emb.size(1)))
                cap_lens = [0] * len(iterator.dataset)

            # cache embeddings
            img_embs[ids] = img_emb.data.cpu()
            if cap_embs.dim() == 3:
                cap_embs[ids, :max(txt_lengths), :] = txt_emb.data.cpu()
            else:
                cap_embs[ids, :] = txt_emb.data.cpu()

        if not fold5:
            img_embs = torch.cat([img_embs[i].unsqueeze(0) for i in range(0, len(img_embs), 5)])

            cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

            d = img_embs.mm(cap_embs.t()).detach().data.numpy()
            i2t_r, t2i_r, rsum = rank_recall(d)
        else:
            res = []
            for i in range(5):
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]

                img_embs_shard = img_embs_shard / img_embs_shard.norm(dim=-1, keepdim=True)
                cap_embs_shard = cap_embs_shard / cap_embs_shard.norm(dim=-1, keepdim=True)

                d = img_embs_shard.mm(cap_embs_shard.t()).detach().data.numpy()
                i2t_r, t2i_r, rsum = rank_recall(d)
                res.append([i2t_r[0], i2t_r[1], i2t_r[2], t2i_r[0], t2i_r[1], t2i_r[2], rsum])
            mean_res = np.array(res).mean(axis=0).flatten()
            i2t_r, t2i_r, rsum = mean_res[0:3], mean_res[3:6], mean_res[-1]

    epoch_time = time.time() - epoch_start
    mins, secs = int(epoch_time // 60), int(epoch_time % 60)

    logger.info('Val_Rsum: {0:.1f} | Val_Time: {1}m {2}s'.format(rsum, mins, secs))
    logger.info('Text-Image Retrieval | R1:{0:.1f}, R5:{1:.1f}, R10:{2:.1f}'.format(*t2i_r))
    logger.info('Image-Text Retrieval | R1:{0:.1f}, R5:{1:.1f}, R10:{2:.1f}'.format(*i2t_r))
    return i2t_r, t2i_r, rsum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='coco', type=str)
    parser.add_argument('--txt_enc_type', default='test1K', type=str)
    parser.add_argument('--img_enc_type', default='finetune-ViTB32', type=str)
    parser.add_argument('--log_path', default='./runs/', type=str)
    args = parser.parse_args([])

    args.log_name = f'test_{args.txt_enc_type}_{args.img_enc_type}'
    logger = get_logger(args, args.log_name)
    logger.info(f'Initialize logger file.')

    # use clip
    store_matrixs = []
    args = get_args()
    print(torch.cuda.device_count())

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load precomputed vocabulary
    model, _ = clip.load('ViT-B/32', args.device)
    # model.load_state_dict(torch.load('/home/ZJ/code/SWCLIP/runs/*.pth'))
    args.vacab_size = model.vocab_size
    model = model.to(args.device)

    logger.info(f'Model Structure: ')
    logger.info(f'Model Structure: {model}')

    if args.data_name == 'coco':
        _, val_data, test_data = get_loaders(args.data_name, args.batch_size,
                                             args.num_workers, args, return_test=True)
        logger.info('For MS-COCO 5-fold 1K:')
        _, _, d = evaluate(model, 0, test_data, logger, args, True)
        logger.info('For MS-COCO 5K:')
        _, _, d = evaluate(model, 0, test_data, logger, args, False)
        store_matrixs.append(d)

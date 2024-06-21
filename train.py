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
import multiprocessing
from mmdet.apis import init_detector
from argparse import ArgumentParser


def train(model, epoch, iterator, optimizer, loss_img, loss_txt, logger, length, args):
    epoch_start = time.time()  # 返回当前的时间戳（以秒为单位）
    epoch_loss = 0.0

    torch.cuda.empty_cache()
    model.train()
    for step, data in enumerate(iterator):
        text, tfidf, weight, image, detect, _, _ = data
        text = torch.stack(text, dim=0)
        image = torch.stack(image, dim=0)
        detect = torch.stack(detect, dim=0)
        weight = torch.stack(weight, dim=0)
        if torch.cuda.is_available():
            text = text.squeeze(dim=1).cuda()
            image = image.to(args.device)
            detect = detect.to(args.device)
            weight = weight.to(args.device)

        # calculate loss and optimize
        optimizer.zero_grad()

        logits_per_image, logits_per_text = model(text, image, tfidf, weight, detect)
        ground_truth = torch.arange(len(image), dtype=torch.long, device=args.device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()

        epoch_loss += total_loss.item()

        optimizer.step()

        if (step + 1) % args.log_step == 0:
            step_time = time.time() - epoch_start
            mins, secs = int(step_time // 60), int(step_time % 60)
            logger.info('Step: {0}/{1} | Loss: {2:.4f} | Time: {3}m {4}s'.format(step + 1, length,
                                                                                 epoch_loss / (step + 1),
                                                                                 mins, secs))

    epoch_time = time.time() - epoch_start
    mins, secs = int(step_time // 60), int(step_time % 60)
    logger.info('Train_loss: {0:.4f} | Train_time： {1}m {2}s'.format(epoch_loss / length,
                                                                     mins, secs))

    return epoch_loss / length


def evaluate(model, epoch, iterator, logger, args):
    epoch_start = time.time()
    model.eval()
    img_embs, cap_embs, cap_lens = None, None, None
    with torch.no_grad():
        for i, data in enumerate(iterator):
            text, tfidf, weight, image, detect, txt_lengths, ids = data
            text = torch.stack(text, dim=0)
            image = torch.stack(image, dim=0)
            detect = torch.stack(detect, dim=0)
            weight = torch.stack(weight, dim=0)
            max_seq_length = max(txt_lengths)
            ids = np.array(ids)
            if torch.cuda.is_available():
                text = text.squeeze(dim=1).cuda()
                image = image.to(args.device)
                detect = detect.to(args.device)
                weight = weight.to(args.device)

            txt_emb = model.encode_text(text, tfidf, weight)
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

        img_embs = torch.cat([img_embs[i].unsqueeze(0) for i in range(0, len(img_embs), 5)])

        cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

        d = img_embs.mm(cap_embs.t()).detach().data.numpy()

    i2t_r = i2t(d)
    t2i_r = t2i(d)
    rsum = sum(i2t_r) + sum(t2i_r)

    epoch_time = time.time() - epoch_start
    mins, secs = int(epoch_time // 60), int(epoch_time % 60)

    logger.info('Val_Rsum: {0:.1f} | Val_Time: {1}m {2}s'.format(rsum, mins, secs))
    logger.info('Text-Image Retrieval | R1:{0:.1f}, R5:{1:.1f}, R10:{2:.1f}'.format(*t2i_r))
    logger.info('Image-Text Retrieval | R1:{0:.1f}, R5:{1:.1f}, R10:{2:.1f}'.format(*i2t_r))
    return i2t_r, t2i_r, rsum


if __name__ == '__main__':
    # load init arguments
    print(torch.cuda.is_available())
    args = get_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True

    # Initialize logger
    logger = get_logger(args, args.log_name)
    logger.info('Using devices:"{0}'.format(args.device))
    logger.info('Using txt:"{0}" and img:"{1}" for retrieval in dataset:"{2}"'.format(args.txt_enc,
                                                                                      args.img_enc,
                                                                                      args.data_name))

    # load precomputed vocabulary
    model, _ = clip.load('ViT-B/32', args.device, jit=False)
    args.vacab_size = model.vocab_size
    model = model.to(args.device)

    # swin_model = init_detector(args.swin_config, args.swin_checkpoint, device=args.device)

    train_data, val_data, test_data = get_loaders(args.data_name, args.batch_size,
                                                  args.num_workers, args)

    _, _, rsum = evaluate(model, 0, test_data, logger, args)

    logger.info('The iterator number of train/val/test is {0}/{1}/{2}.'.format(len(train_data),
                                                                               len(val_data),
                                                                               len(test_data)))
    loss_img = nn.CrossEntropyLoss().to(args.device)
    loss_txt = nn.CrossEntropyLoss().to(args.device)

    add_params = list(map(id, model.visual.final_embedding.parameters()))
    base_params = filter(lambda p: id(p) not in add_params, model.parameters())
    optimizer = optim.Adam([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': model.visual.final_embedding.parameters(), 'lr': args.add_lr}
    ])
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-3)

    best_rsum = 0.0
    best_i2t_r = 0.0
    best_loss = 100
    for epoch in tqdm(range(args.epochs)):
        logger.info('Epoch: {0}'.format(epoch))
        logger.info('lr: {0}'.format(args.learning_rate))
        logger.info('add_lr: {0}'.format(args.add_lr))

        train_loss = train(model, epoch, train_data, optimizer, loss_img, loss_txt, logger, len(train_data), args)

        if train_loss < best_loss:
            best_loss = train_loss
            filename = args.log_path + 'B32-loss_detectVision_weight_optstepLR_1e6_1e3.pth'
            torch.save(model.state_dict(), filename)

        # evaluate
        _, _, rsum = evaluate(model, epoch, val_data, logger, args)

    torch.save(model.state_dict(), './runs/B32-latest_model_stepLR_optstepLR_1e6_1e3.pth')


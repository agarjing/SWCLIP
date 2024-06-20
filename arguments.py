import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', default='B32_detectVision_weight_optstepLR_1e6_1e4', type=str,
                        help='log file name')
    '''params for data'''
    parser.add_argument('--data_path', default='/home/hdu/data_ZJ',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco',
                        help='coco,f30k')
    parser.add_argument('--vocab_path', default='/home/hdu/data_ZJ/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--log_path', default='./runs/',
                        help='Path to saved logger files.')
    parser.add_argument('--precomp_enc_type', default="backbone",
                        help='basic|backbone')
    parser.add_argument('--input_scale_factor', type=float, default=1,
                        help='The factor for scaling the input image')

    '''hyper params'''
    parser.add_argument('--epochs', default=18, type=int,
                        help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--best_loss', default=10, type=float,
                        help='Train loss compare.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        type=str)
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr_update', default=15, type=int)
    parser.add_argument('--log_step', default=200, type=int,
                        help='basic|weight_norm')
    parser.add_argument('--tau', default=0.05, type=float,
                        help='param for contrastive loss')
    parser.add_argument('--warmup_epochs', default=1, type=int,
                        help='The number of epochs for warmup operation.')
    parser.add_argument('--learning_rate', default=1e-6, type=float,
                        help='Initial learning rate')
    parser.add_argument('--add_lr', default=1e-4, type=float,
                        help='Initial learning rate')
    parser.add_argument('--loss_type', default='acc', type=str,
                        help='')

    # params for text
    parser.add_argument('--txt_enc', default='clip', type=str,
                        help='the type of textual encoder | rnn/bert')
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', type=str,
                        help='the type of bert | bert-base-uncased/ ')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='the type of bert | bert-base-uncased/ ')
    parser.add_argument('--vocab_size', default=30000, type=int,
                        help='the total number of words')
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--emb_size', default=1024, type=int)
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--use_bigru', action='store_true',
                        help='Use bidirectional GRU.')
    # params for image
    parser.add_argument('--img_enc', default='clip', type=str,
                        help='the type of visiual encoder |')
    parser.add_argument('--vit_type', default='google/vit-base-patch16-224', type=str,
                        help='the type of vit | google/vit-base-patch16-224')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--swin_config',
                        default='configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                        help='Config file')
    parser.add_argument('--swin_checkpoint', default='mask_rcnn_swin_tiny_patch4_window7.pth',
                        help='Checkpoint file')

    args = parser.parse_args()
    return args
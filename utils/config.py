import argparse
import torch
import logging
import os
import datetime
import json


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt='%Y/%d/%m %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
# Dataset and Other Parameters
parser.add_argument('--data_dir', type=str, default='data/ProSLU')
parser.add_argument('--save_dir', type=str, default='save/')
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument('--gpu', action='store_true', help='use gpu', required=False)
parser.add_argument("--use_info", help='use info', action='store_true', required=False)
parser.add_argument('--use_pretrained', action='store_true', help='use pretrained models', required=False)
parser.add_argument('--model_type', type=str, default="ELECTRA")
parser.add_argument('--max_length', type=int, help='max length for KG', default=512)
parser.add_argument('--early_stop', action='store_true', required=False)
parser.add_argument('--patience', type=int, default=10)

# Training parameters.
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--l2_penalty', type=float, default=1e-6)
parser.add_argument("--learning_rate", type=float, default=8e-4)
parser.add_argument("--bert_learning_rate", type=float, default=4e-5)
parser.add_argument('--dropout_rate', type=float, default=0.4)
parser.add_argument('--bert_dropout_rate', type=float, default=0.4)
parser.add_argument("--use_crf", action="store_true", default=False)
parser.add_argument("--intent_slot_coef", type=float, default=0.5)
parser.add_argument("--i2s", action="store_true", default=False)
parser.add_argument("--s2i", action="store_true", default=False)
parser.add_argument("--up", action="store_true", default=False)
parser.add_argument("--ca", action="store_true", default=False)
parser.add_argument("--kg", action="store_true", default=False)

# Model parameters.
parser.add_argument('--word_embedding_dim', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', type=int, default=128)
parser.add_argument('--attention_hidden_dim', type=int, default=256)
parser.add_argument('--attention_output_dim', type=int, default=128)
parser.add_argument('--d_a', type=int, default=128)
parser.add_argument('--d_c', type=int, default=256)
parser.add_argument('--intent_emb', type=int, default=128)
parser.add_argument('--info_embedding_dim', type=int, default=128)

args = parser.parse_args()
args.up_keys = ['音视频应用偏好', '出行交通工具偏好', '长途交通工具偏好', '是否有车']
args.ca_keys = ['移动状态', '姿态识别', '地理围栏', '户外围栏']
args.gpu = args.gpu and torch.cuda.is_available()


timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
if args.use_pretrained:
    prefix = 'BERTSLU'
else:
    prefix = 'SLU'
    args.model_type = 'LSTM'
args.use_info = args.up or args.ca or args.kg
if args.use_info:
    prefix += '++'
args.save_dir = os.path.join(args.save_dir, prefix, '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                                                args.model_type,
                                                                args.random_seed,
                                                                args.intent_slot_coef,
                                                                args.i2s,
                                                                args.s2i,
                                                                args.up,
                                                                args.ca,
                                                                args.kg
                                                                ))
os.makedirs(args.save_dir, exist_ok=True)
log_path = os.path.join(args.save_dir, "config.json")
with open(log_path, "w", encoding="utf8") as fw:
    fw.write(json.dumps(args.__dict__, indent=True))

mylogger = get_logger(os.path.join(args.save_dir, 'log.txt'), name='SLU')
mylogger.info(str(vars(args)))

# Model Dict
if args.model_type != 'LSTM':
    model_type = {
        'RoBERTa': "hfl/chinese-roberta-wwm-ext",
        'BERT': "hfl/chinese-bert-wwm-ext",
        'XLNet': "hfl/chinese-xlnet-base",
        'ELECTRA': "hfl/chinese-electra-180g-base-discriminator"
    }

    args.model_type_path = model_type[args.model_type]

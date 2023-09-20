import random
import re
import numpy as np
from pathlib import Path
from transformers import SwinForImageClassification

from parameters import parse_args
from model import Model
from data_utils import read_images, read_behaviors, Build_Lmdb_Dataset, Build_Id_Dataset, LMDB_Image, \
    eval_model, get_itemId_embeddings, get_itemLMDB_embeddings
from data_utils.utils import *
import torchvision.models as models
from torch import nn

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_, constant_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(args, use_modal, local_rank):
    if use_modal:
        # read cv encoder
        if 'resnet50' in args.CV_model_load:
            Log_file.info('load resnet model...')
            cv_model_load = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            if '18' in cv_model_load:
                cv_model = models.resnet18(pretrained=False)
            elif '34' in cv_model_load:
                cv_model = models.resnet34(pretrained=False)
            elif '50' in cv_model_load:
                cv_model = models.resnet50(pretrained=False)
            elif '101' in cv_model_load:
                cv_model = models.resnet101(pretrained=False)
            elif '152' in cv_model_load:
                cv_model = models.resnet152(pretrained=False)
            else:
                cv_model = None
            cv_model.load_state_dict(torch.load(cv_model_load))
            num_fc_ftr = cv_model.fc.in_features
            cv_model.fc = nn.Linear(num_fc_ftr, args.embedding_dim)
            xavier_normal_(cv_model.fc.weight.data)
            if cv_model.fc.bias is not None:
                constant_(cv_model.fc.bias.data, 0)

        elif 'swin' in args.CV_model_load:
            if 'swin_tiny' in args.CV_model_load:
                Log_file.info('load swin_tiny model...')
                cv_model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
            if 'swin_small' in args.CV_model_load:
                Log_file.info('load swin_small model...')
                cv_model = SwinForImageClassification.from_pretrained('microsoft/swin-small-patch4-window7-224')
            elif 'swin_base' in args.CV_model_load:
                Log_file.info('load swin_base model...')
                cv_model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224-in22k')
            num_fc_ftr = cv_model.classifier.in_features
            cv_model.classifier = nn.Linear(num_fc_ftr, args.embedding_dim)
            xavier_normal_(cv_model.classifier.weight.data)
            if cv_model.classifier.bias is not None:
                constant_(cv_model.classifier.bias.data, 0)
        else:
            cv_model = None

        for index, (name, param) in enumerate(cv_model.named_parameters()):
            if index < args.freeze_paras_before:
                param.requires_grad = False
    else:
        cv_model = None
        args.CV_model_load = 'None'

    Log_file.info('read images...')
    before_item_id_to_keys, before_item_name_to_id = read_images(
        os.path.join(args.root_data_dir, args.dataset, args.images))

    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_keys,
                       before_item_name_to_id, args.max_seq_len, args.min_seq_len, Log_file)

    Log_file.info('build model...')
    if True:  # test
        show_running_details('test', args.item_tower, args.model_tower, args.dataset, args.behaviors, Log_file)
        # load checkpoint, partial fine tune
        Log_file.info(f'load {args.model_tower} ckpt for transfer...')
        ckpt_path = get_checkpoint(args.model_path, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')

        # initialize model
        model = Model(args, item_num, use_modal, cv_model).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        # Log_file.info(model)

        # update weights
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path}")

        # set paras
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        is_early_stop = True

    Log_file.info('model.cuda()...')
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_num = sum(p.numel() for p in model.module.parameters())
    Log_file.info("##### total_num {} #####".format(total_num))
    if 'test' in args.mode:
        run_eval_test(model, item_id_to_keys, users_history_for_test, users_test, 256, item_num,
                      use_modal, args.mode, local_rank)


def run_eval_test(model, item_id_to_keys, user_history, users_eval, batch_size, item_num, use_modal, mode, local_rank):
    eval_start_time = time.time()
    Log_file.info('test...')
    if use_modal:
        item_embeddings = get_itemLMDB_embeddings(model, item_num, item_id_to_keys, batch_size, args, local_rank)
    else:
        item_embeddings = get_itemId_embeddings(model, item_num, batch_size, args, local_rank)

    eval_model(model, user_history, users_eval, item_embeddings, batch_size, args, item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(123456)

    if 'img' in args.item_tower:
        is_use_modal = True
        dir_label = str(args.model_tower) + '_img_' + \
            str(args.CV_model_load) + '_' + f'{args.freeze_paras_before}' + \
            '_' + str(args.dataset)[9:-1]  # e.g. logs_sasrec_img_vit_0_Bilibili_25w_test

    else:  # id
        is_use_modal = False
        dir_label = str(args.model_tower) + '_id_' + \
            str(args.dataset)[9:-1]

    log_paras = f'bs_{args.batch_size}_gpu_{args.gpu_device}_ed_{args.embedding_dim}_tb_{args.block_num}'\
                f'_lr_{args.lr}_Flr_{args.fine_tune_lr}_dp_{args.drop_rate}_L2r_{args.l2_weight_R}_L2m_{args.l2_weight_M}'

    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())
    Log_file.info(args)

    test(args, is_use_modal, local_rank)

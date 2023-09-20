import random
import re
import numpy as np
from pathlib import Path
from transformers import BertModel, BertTokenizer, BertConfig

from parameters import parse_args
from model import ModelCPC
from data_utils import read_text, read_text_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_item_embeddings
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
        if 'bert' in args.NLP_model_load:
            if 'bert_base_uncased' in args.NLP_model_load:
                Log_file.info('load bert_base_uncased model...')
                bert_model_load = 'https://huggingface.co/bert-base-uncased'
            if 'chinese_bert_wwm' in args.NLP_model_load:
                Log_file.info('load chinese-bert-wwm model...')
                bert_model_load = 'https://huggingface.co/hfl/chinese-bert-wwm-ext'
            tokenizer = BertTokenizer.from_pretrained(bert_model_load)
            config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            nlp_model = BertModel.from_pretrained(bert_model_load, config=config)

            if 'tiny' in args.NLP_model_load:
                pooler_para = [37, 38]
                args.word_embedding_dim = 128
            if 'mini' in args.NLP_model_load:
                pooler_para = [69, 70]
                args.word_embedding_dim = 256
            if 'medium' in args.NLP_model_load:
                pooler_para = [133, 134]
                args.word_embedding_dim = 512
            if 'base' or 'chinese-bert-wwm' in args.NLP_model_load:
                pooler_para = [197, 198]
                args.word_embedding_dim = 768
            if 'large' in args.NLP_model_load:
                pooler_para = [389, 390]
                args.word_embedding_dim = 1024

        for index, (name, param) in enumerate(nlp_model.named_parameters()):
            if index < args.freeze_paras_before or index in pooler_para:
                param.requires_grad = False

        Log_file.info('read texts...')
        before_item_id_to_dic, before_item_name_to_id = read_text_bert(
            os.path.join(args.root_data_dir, args.dataset, args.texts), args, tokenizer, args.which_language)

        Log_file.info('read behaviors...')
        item_num, item_id_to_dic, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
            read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                           before_item_id_to_dic, before_item_name_to_id,
                           args.max_seq_len, args.min_seq_len, Log_file)

        Log_file.info('combine texts information...')
        news_title, news_title_attmask, \
        news_abstract, news_abstract_attmask, \
        news_body, news_body_attmask = get_doc_input_bert(item_id_to_dic, args)

        item_content = np.concatenate([
            x for x in
            [news_title, news_title_attmask,
             news_abstract, news_abstract_attmask,
             news_body, news_body_attmask]
            if x is not None], axis=1)

    else:
        before_item_id_to_dic, before_item_name_to_id = read_text(
            os.path.join(args.root_data_dir, args.dataset, args.texts), args.which_language)

        Log_file.info('read behaviors...')
        item_num, item_id_to_dic, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
            read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                           before_item_id_to_dic, before_item_name_to_id,
                           args.max_seq_len, args.min_seq_len, Log_file)
        item_content = np.arange(item_num + 1)
        nlp_model = None

    Log_file.info('build model...')
    if True:  # test
        ShowRunningDetails('test', args.item_tower, args.model_tower, args.dataset, args.behaviors)
        # load checkpoint, partial fine tune
        Log_file.info(f'load {args.model_tower} ckpt for transfer...')
        ckpt_path = get_checkpoint(args.model_path, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')

        # initialize model
        model = ModelCPC(args, item_num, use_modal, nlp_model).to(local_rank)
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
        run_eval_test(model, item_content, users_history_for_test, users_test, 256, item_num,
                      use_modal, args.mode, local_rank)


def run_eval_test(model, item_content, user_history, users_eval, batch_size, item_num, use_modal, mode, local_rank):
    eval_start_time = time.time()
    Log_file.info('test...')

    item_embeddings = get_item_embeddings(model, item_content, batch_size, args, use_modal, local_rank)

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


def ShowRunningDetails(mode, item_tower, model_tower, dataset, behaviors):
    boundary = max([len(mode), len(item_tower), len(model_tower), len(dataset)])
    Log_file.info('=' * 50 + f' {mode} ' + ' ' * (boundary - len(mode)) + '=' * 50)
    Log_file.info('=' * 50 + f' {item_tower} ' + ' ' * (boundary - len(item_tower)) + '=' * 50)
    Log_file.info('=' * 50 + f' {model_tower} ' + ' ' * (boundary - len(model_tower)) + '=' * 50)
    Log_file.info('=' * 50 + f' {dataset} ' + ' ' * (boundary - len(dataset)) + '=' * 50)
    Log_file.info(behaviors)
    return None


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(123456)

    if 'txt' in args.item_tower:
        is_use_modal = True
        dir_label = str(args.model_tower) + '_txt_' + \
            str(args.NLP_model_load) + '_' + f'{args.freeze_paras_before}' + \
            '_' + str(args.dataset)[9:-1]  # e.g. logs_sasrec_txt_bert_0_Bilibili_25w_test
        log_paras = f'bs_{args.batch_size}_gpu_{args.gpu_device}_ed_{args.embedding_dim}_tb_{args.block_num}'\
                    f'_lr_{args.lr}_Flr_{args.fine_tune_lr}_dp_{args.drop_rate}_L2r_{args.l2_weight_R}_L2m_{args.l2_weight_M}'

    else:  # id
        is_use_modal = False
        dir_label = str(args.model_tower) + '_id_' + \
            str(args.dataset)[9:-1]
        log_paras = f'bs_{args.batch_size}_gpu_{args.gpu_device}_ed_{args.embedding_dim}_tb_{args.block_num}'\
                    f'_lr_{args.lr}_Flr_{args.fine_tune_lr}_dp_{args.drop_rate}_L2_{args.l2_weight_R}'

    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())
    Log_file.info(args)

    test(args, is_use_modal, local_rank)

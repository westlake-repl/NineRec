from xml.sax import parseString
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os
import math
from .dataset import BuildEvalDataset, SequentialDistributedSampler, Build_Lmdb_Eval_Dataset, \
                    Build_Id_Eval_Dataset, Build_Text_Eval_Dataset, Build_MMEncoder_Eval_Dataset


def item_collate_fn(arr):
    arr = torch.LongTensor(np.array(arr))
    return arr


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t + "_results   {}".format('\t'.join(["{:0.5f}".format(i * 100) for i in x])))


def get_mean(arr):
    return [i.mean() for i in arr]

def distributed_concat(tensor, num_total_examples):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    # output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset)) \
            .to(torch.device("cpu")).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result

def scoring_concat(scoring, test_sampler):
    scoring = distributed_concat(scoring, len(test_sampler.dataset))
    return scoring


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return eval_ra

# text-only (distributed)
def get_text_only_scoring(model, item_content, test_batch_size, args, local_rank):
    
    model.eval()
    item_dataset = Build_Text_Eval_Dataset(item_content)
    test_sampler = SequentialDistributedSampler(item_dataset, batch_size=test_batch_size)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=item_collate_fn, sampler=test_sampler)

    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            # text
            input_ids = input_ids.to(local_rank)
            hidden_states_text = model.module.text_encoder(input_ids)
            # mask
            text_mask = torch.narrow(input_ids, 1, args.num_words_title, args.num_words_title)
            
            text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float().to(local_rank)       
            hidden_states_text = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9) # mean
            item_scoring.extend(hidden_states_text)

        item_scoring = torch.stack(tensors=item_scoring, dim=0).contiguous()
        item_scoring = scoring_concat(item_scoring, test_sampler)

    return item_scoring.to(torch.device("cpu")).detach()



# id(no distributed because it is fast enough in single GPU)
def get_itemId_scoring(model, item_num, test_batch_size, args, local_rank):
    model.eval()

    item_dataset = Build_Id_Eval_Dataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,pin_memory=True, collate_fn=item_collate_fn)
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.id_encoder(input_ids)
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0).to(torch.device("cpu")).detach()

# CV-only (distributed)
def get_LMDB_only_scoring(model, item_num, item_id_to_keys, test_batch_size, args, local_rank):
    model.eval()

    item_dataset = Build_Lmdb_Eval_Dataset(data=np.arange(item_num + 1), item_id_to_keys=item_id_to_keys,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                           resize=args.CV_resize)
    test_sampler = SequentialDistributedSampler(item_dataset, batch_size=test_batch_size)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, sampler=test_sampler)

    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            hidden_states_CV = model.module.cv_encoder(input_ids)
            if 'resnet' in args.CV_model_load or 'RN50' in args.CV_model_load:
                item_emb = hidden_states_CV  # resnet
            else:
                item_emb = torch.mean(hidden_states_CV, dim=1)  
            item_scoring.extend(item_emb)

        item_scoring = torch.stack(tensors=item_scoring, dim=0).contiguous()
        item_scoring = scoring_concat(item_scoring, test_sampler)

    return item_scoring.to(torch.device("cpu")).detach()



# Multi-modal (distributed)
def get_MMEncoder_scoring(model, item_content, item_num, item_id_to_keys, test_batch_size, args, local_rank):
    model.eval()

    item_dataset = Build_MMEncoder_Eval_Dataset(data_text=item_content, # text的参数
                                                #cv的参数
                                                data_cv=np.arange(item_num + 1),
                                                item_id_to_keys=item_id_to_keys,
                                                db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                                resize=args.CV_resize)
    
    test_sampler = SequentialDistributedSampler(item_dataset, batch_size=test_batch_size)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, sampler=test_sampler)

    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids_text, input_ids_cv = input_ids
            input_ids_text = input_ids_text.to(local_rank)
            input_ids_cv = input_ids_cv.to(local_rank)
            
            # text mask
            text_mask_scoring = torch.narrow(input_ids_text, 1, args.num_words_title, args.num_words_title)
            
            # get text and cv scoring
            item_scoring_text = model.module.text_encoder(input_ids_text)
            item_scoring_CV = model.module.cv_encoder(input_ids_cv)

            if args.fusion_method in ['sum', 'concat', 'film', 'gated', 'sum_dnn', 'concat_dnn', 'film_dnn', 'gated_dnn']:
                text_mask_expanded = text_mask_scoring.unsqueeze(-1).expand(item_scoring_text.size()).float().to(local_rank)
                item_scoring_text = torch.sum(item_scoring_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9)  # mean
                if 'resnet' in args.CV_model_load or 'RN50' in args.CV_model_load:
                    item_scoring_CV = item_scoring_CV  # resnet
                else:
                    item_scoring_CV = torch.mean(item_scoring_CV, dim=1)  # mean
                item_emb = model.module.fusion_module(item_scoring_text, item_scoring_CV)
                item_scoring.extend(item_emb)

            elif args.fusion_method in ['co_att', 'merge_attn']:
                CV_mask = torch.ones(item_scoring_CV.size()[0], item_scoring_CV.size()[1]).to(local_rank)
                item_emb = model.module.fusion_module(item_scoring_text, text_mask_scoring, item_scoring_CV, CV_mask, local_rank)
                item_scoring.extend(item_emb)

        item_scoring = torch.stack(tensors=item_scoring, dim=0).contiguous()
        item_scoring = scoring_concat(item_scoring, test_sampler)

    return item_scoring.to(torch.device("cpu")).detach()


def eval_model(model, user_history, eval_seq, item_scoring, test_batch_size, args, item_num, Log_file, v_or_t,
               local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_scoring,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank), \
                log_mask.to(local_rank), labels.to(local_rank).detach()
            
            if "sasrec" in args.benchmark:
                prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            elif  "grurec" in args.benchmark:
                prec_emb = model.module.user_encoder(input_embs)[:, -1].detach()
            elif  "nextit" in args.benchmark:
                prec_emb = model.module.user_encoder(input_embs)[:, -1].detach()
                
            # prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1]

import torch
from torch import nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from .text_encoders import TextEncoder
from .img_encoders import VisionEncoder
from .user_encoders import User_Encoder_NextItNet, User_Encoder_GRU4Rec, User_Encoder_SASRec

from .fushion_module import SumFusion, ConcatFusion, FiLM, GatedFusion

# from lxmert, two attention fusion methods of co- and merge-.
from .modeling import CoAttention, MergedAttention



class Model(torch.nn.Module):
    def __init__(self, args, item_num, bert_model, image_net):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len + 1 #修正
        
        # various benchmark
        if "sasrec" in args.benchmark:
            self.user_encoder = User_Encoder_SASRec(item_num=item_num, args=args)
        elif "nextit" in args.benchmark:
            self.user_encoder = User_Encoder_NextItNet(args=args)
        elif "grurec"  in args.benchmark:
            self.user_encoder = User_Encoder_GRU4Rec(args=args)

        # various encoders
        if "CV" in args.item_tower or "modal" in args.item_tower:
            self.cv_encoder = VisionEncoder(args=args, image_net=image_net)

        if "text" in args.item_tower or "modal" in args.item_tower:
            self.text_encoder = TextEncoder(args=args, bert_model=bert_model)

        if  "ID" in args.item_tower:
            self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_encoder.weight.data)

        if "modal" in args.item_tower:
            # various fusion methods
            if args.fusion_method == 'sum':
                self.fusion_module = SumFusion(args=args)
            elif args.fusion_method == 'concat':
                self.fusion_module = ConcatFusion(args=args)
            elif args.fusion_method == 'film':
                self.fusion_module = FiLM(args=args, x_film=True)
            elif args.fusion_method == 'gated':
                self.fusion_module = GatedFusion(args=args, x_gate=True)
            elif args.fusion_method == 'co_att' :
                cofig_path = "/home/xihu/lyh/MMRS/TextEncoders/bert-base-uncased"
                self.fusion_module = CoAttention.from_pretrained(cofig_path, args=args)
                # self.fusion_module = CoAttention.from_pretrained("bert-base-uncased", args=args)
            elif args.fusion_method == 'merge_attn':
                cofig_path = "/home/xihu/lyh/MMRS/TextEncoders/bert-base-uncased"
                self.fusion_module = MergedAttention.from_pretrained(cofig_path, args=args)
                # self.fusion_module = MergedAttention.from_pretrained("bert-base-uncased",args=args)

        # loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items_id, sample_items_text, sample_items_CV, log_mask, local_rank, args):

        if "modal" in args.item_tower:
            batch_size, num_words = sample_items_text.shape
            num_words = num_words // 2
            text_mask = torch.narrow(sample_items_text, 1, num_words, num_words)

            hidden_states_text = self.text_encoder(sample_items_text.long())
            hidden_states_CV = self.cv_encoder(sample_items_CV)

            if args.fusion_method in ['sum', 'concat', 'film', 'gated']:
                text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float()
                hidden_states_text = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9)
                hidden_states_CV = torch.mean(hidden_states_CV, dim=1)  # mean
                input_embs = self.fusion_module(hidden_states_text, hidden_states_CV)
            if args.fusion_method in ['co_att', 'merge_attn']:
                CV_mask = torch.ones(hidden_states_CV.size()[0], hidden_states_CV.size()[1]).to(local_rank)
                input_embs = self.fusion_module(hidden_states_text, text_mask, hidden_states_CV, CV_mask,local_rank)


        if "text-only" in args.item_tower:
            batch_size, num_words = sample_items_text.shape
            num_words = num_words // 2
            text_mask = torch.narrow(sample_items_text, 1, num_words, num_words)
            hidden_states_text = self.text_encoder(sample_items_text.long())
            text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float().to(local_rank)       
            input_embs = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9) # mean

        if "CV-only" in args.item_tower:
            hidden_states_CV = self.cv_encoder(sample_items_CV)
            input_embs = torch.mean(hidden_states_CV, dim=1)  

        if "ID" in args.item_tower:
            input_embs = self.id_encoder(sample_items_id)

        # print("input_embs", input_embs.size())
        input_embs = input_embs.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        # print("input_embs", input_embs.size())
        # exit()

        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # various benchmark
        if "sasrec" in args.benchmark:
             prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        elif "nextit" in args.benchmark:
            prec_vec = self.user_encoder(input_logs_embs)
        elif "grurec"  in args.benchmark:
            prec_vec = self.user_encoder(input_logs_embs)

        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)

        loss_1 = self.criterion(pos_score[indices], pos_labels[indices]) 
        loss_2 = self.criterion(neg_score[indices], neg_labels[indices])
        loss = loss_1 + loss_2 

        return loss

import torch
from torch import nn
from .encoders import Bert_Encoder, UserEncoder_bert4rec
from torch.nn.init import xavier_normal_


class Model(torch.nn.Module):
    def __init__(self, args, item_num, use_modal, nlp_model):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len

        self.user_encoder = UserEncoder_bert4rec(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if self.use_modal:
            self.nlp_encoder = Bert_Encoder(args=args, nlp_model=nlp_model)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items, log_mask, mask_index, local_rank, batch_index):
        # if batch_index % 1000 == 0:
        #     print(f'======================================== batch {batch_index} ========================================')

        if self.use_modal:
            input_embs_all = self.nlp_encoder(sample_items)
        else:
            input_embs_all = self.id_embedding(sample_items)

        input_embs = input_embs_all.view(-1, self.max_seq_len, 3, self.args.embedding_dim)
        input_logs_embs = input_embs[:, :, 0]
        pos_items_embs = input_embs[:, :, 1]
        neg_items_embs = input_embs[:, :, 2]

        # user encoder
        prec_vec = self.user_encoder(input_logs_embs, log_mask)
        pos_score = (prec_vec * pos_items_embs).sum(-1)
        neg_score = (prec_vec * neg_items_embs).sum(-1)

        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        indices = torch.where(mask_index != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
            self.criterion(neg_score[indices], neg_labels[indices])
        return loss

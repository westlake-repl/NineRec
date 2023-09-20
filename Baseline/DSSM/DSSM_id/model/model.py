import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from .encoders import Bert_Encoder, MLP_Encoder


class Model(torch.nn.Module):
    def __init__(self, args, user_num, item_num, use_modal, bert_model):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.dnn_layers = args.dnn_layers
        self.embedding_dim = args.embedding_dim
        self.l2_weight = args.l2_weight
        self.neg_num = args.neg_num

        self.user_embedding = nn.Embedding(user_num + 1, self.embedding_dim, padding_idx=0)
        self.user_encoder = MLP_Encoder(embedding_dim=self.embedding_dim,
                                        dnn_layers=args.dnn_layers,
                                        drop_rate=args.drop_rate)
        if self.use_modal:
            self.bert_encoder = Bert_Encoder(args=args,
                                             bert_model=bert_model)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, self.embedding_dim, padding_idx=0)
            self.id_encoder = MLP_Encoder(embedding_dim=self.embedding_dim,
                                          dnn_layers=args.dnn_layers,
                                          drop_rate=args.drop_rate)
        self.apply(self._init_weights)
        self.criterion = nn.BCEWithLogitsLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_user, sample_items, bce_label):
        user_embedding = self.user_embedding(input_user)
        user_feature = self.user_encoder(user_embedding)
        if self.use_modal:
            item_embedding = None
            item_feature = self.bert_encoder(sample_items)
        else:
            item_embedding = self.id_embedding(sample_items)
            item_feature = self.id_encoder(item_embedding)
        item_feature = item_feature.view(-1, 1 + self.neg_num, self.embedding_dim)
        score = torch.bmm(item_feature, user_feature.unsqueeze(-1)).squeeze(dim=-1)
        loss = self.criterion(score.view(-1), bce_label.view(-1))
        return loss

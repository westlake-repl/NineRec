import torch
import torch.nn as nn
from torch.nn.init import uniform_,xavier_uniform_,  xavier_normal_, constant_, normal_
from .modules import TransformerEncoder
from .modules import ResidualBlock_a, ResidualBlock_b
import numpy as np


# ========================================================== image encoder =============================================
class Swin_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Swin_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        hidden_states = self.image_net(item_content)[0]
        return self.activate(hidden_states)


class Resnet_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Resnet_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content))


# ========================================================== user encoder ==============================================
class UserEncoder_sasrec(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(UserEncoder_sasrec, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)


class UserEncoder_nextitnet(nn.Module):
    def __init__(self, args, item_num):  # config, dataset
        super(UserEncoder_nextitnet, self).__init__()  # config, dataset

        # load parameters info
        self.embedding_size = args.embedding_dim  # config['embedding_size']
        self.residual_channels = args.embedding_dim  # config['embedding_size']
        self.block_num = args.block_num  # config['block_num']
        self.dilations = args.dilations * self.block_num  # config['dilations'] * self.block_num
        self.kernel_size = args.kernel_size  # config['kernel_size']
        self.output_dim = item_num
        self.pad_token = args.pad_token
        self.all_time = 0

        # define layers and loss
        # self.item_embedding = nn.Embedding(self.output_dim+1, self.embedding_size, padding_idx=self.pad_token)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        # self.final_layer = nn.Linear(self.residual_channels, self.output_dim+1)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / (self.output_dim+1))
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, input_embs):  # pos, neg
        # print("--------", item_seq.max())
        # item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_outputs = self.residual_blocks(input_embs)
        # hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        # seq_output = self.final_layer(dilate_outputs)  # [batch_size, embedding_size]hidden
        return dilate_outputs  # pos_logit, neg_logit


class UserEncoder_gru4rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__()

        self.embedding_size = args.embedding_dim
        self.n_layers = args.block_num
        self.hidden_size = args.embedding_dim
        self.dropout = args.drop_rate

        # define layers
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.emb_dropout = nn.Dropout(self.dropout)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, input_embs):
        item_seq_emb_dropout = self.emb_dropout(input_embs)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        return gru_output


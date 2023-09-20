import torch
import torch.nn as nn
from torch.nn.init import uniform_, xavier_normal_, constant_, normal_
from .modules import TransformerEncoder
from .modules import ResidualBlock_a, ResidualBlock_b
import numpy as np


# ========================================================== image encoder =============================================
class Vit_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Vit_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        hidden_states = self.image_net(item_content)[0]
        return self.activate(hidden_states)


class CLIP_vit_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(CLIP_vit_Encoder, self).__init__()
        self.image_net = image_net
        self.fc = nn.Linear(768, 256)
        self.activate = nn.GELU()

    def forward(self, item_content):
        hidden_states = self.image_net(item_content)[0]
        cls = self.fc(hidden_states[:, 0])  # 拿[cls]的output
        return self.activate(cls)


class BLIP_vit_Encoder(torch.nn.Module):
    def __init__(self, image_net, enc_token_id):
        super(BLIP_vit_Encoder, self).__init__()
        self.image_net = image_net
        self.enc_token_id = enc_token_id
        self.fc = nn.Linear(768, 256)
        self.activate = nn.GELU()

    def forward(self, item_content):
        hidden_states = self.image_net(item_content,
                                       'fake_text_ids',  # 看了源码，不会用caption的，也不用走tokenizer了 28/07/2022
                                       'fake_text_attmask',
                                       mode='image',
                                       enc_token_id=self.enc_token_id)
        cls = self.fc(hidden_states[:, 0])
        return self.activate(cls)


class DeiT_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(DeiT_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        hidden_states = self.image_net(item_content)[0]
        return self.activate(hidden_states)


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


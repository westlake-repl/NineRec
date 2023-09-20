import torch
import torch.nn as nn
from torch.nn.init import uniform_, xavier_uniform_, xavier_normal_, constant_, normal_
from .modules import TransformerEncoder
from .modules import ResidualBlock_a, ResidualBlock_b
import numpy as np


# ========================================================== texts encodr =============================================
class Text_Encoder(torch.nn.Module):
    def __init__(self,
                 nlp_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder, self).__init__()
        self.nlp_model = nlp_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.nlp_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        cls = self.fc(hidden_states[:, 0])
        return self.activate(cls)

class Text_Encoder_mean(torch.nn.Module):
    def __init__(self,
                 nlp_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder_mean, self).__init__()
        self.nlp_model = nlp_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words) 
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.nlp_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        input_mask_expanded = text_attmask.unsqueeze(-1).expand(hidden_states.size()).float()
        mean_output = torch.sum(hidden_states * input_mask_expanded, 1) \
                      / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_output = self.fc(mean_output)
        return self.activate(mean_output)


class Bert_Encoder(torch.nn.Module):
    def __init__(self, args, nlp_model):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 2,
            'abstract': args.num_words_abstract * 2,
            'body': args.num_words_body * 2
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)]
            )
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract', 'body']
        if 'opt' in args.NLP_model_load:
            self.text_encoders = nn.ModuleDict({
                'title': Text_Encoder_mean(nlp_model, args.embedding_dim, args.word_embedding_dim)
            })
        else:
            self.text_encoders = nn.ModuleDict({
                'title': Text_Encoder(nlp_model, args.embedding_dim, args.word_embedding_dim)
            })

        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]

    def forward(self, text):
        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(text, 1, self.attributes2start[name], self.attributes2length[name]))
            for name in self.newsname
        ]
        if len(text_vectors) == 1:
            final_news_vector = text_vectors[0]
        else:
            final_news_vector = torch.mean(torch.stack(text_vectors, dim=1), dim=1)
        return final_news_vector


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
        Regarding the innovation of this article, we can only achieve the data augmentation mentioned
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


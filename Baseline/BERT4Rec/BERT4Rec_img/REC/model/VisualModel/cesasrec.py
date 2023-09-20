import torch
from torch import nn

from REC.model.layers import TransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.load import load_model
from REC.model.basemodel import BaseModel
from REC.data.dataset import BatchDataset
from torch.utils.data import DataLoader


class CESASRec(BaseModel):
    input_type = InputType.SEQ
    
    def __init__(self, config, dataload):
        super(CESASRec, self).__init__()
        self.config = config
        self.dataload = dataload
        # load parameters info
        self.pretrain_weights = config['pretrain_path']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.embedding_size = config['embedding_size']  # same as embedding_size
        self.inner_size = config['inner_size']* self.embedding_size # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_num = dataload.item_num
        # define layers and loss
    
        self.visual_encoder = load_model(config=config)
        if self.pretrain_weights:
            self.load_weights(self.pretrain_weights)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.embedding_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.item_feature = None
        self.loss_func = nn.CrossEntropyLoss()
        self.position_embedding.weight.data.normal_(mean=0.0, std=self.initializer_range)
        self.trm_encoder.apply(self._init_weights)
        self.LayerNorm.bias.data.zero_()
        self.LayerNorm.weight.data.fill_(1.0)
        self.pred = nn.Linear(self.embedding_size, self.item_num)
        self.pred.apply(self._init_weights)

    def _init_weights(self, module):      
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



    def forward(self, inputs):
        item_seqs, item_modal = inputs
        item_seq = item_seqs[:, :-1]
        item_target = item_seqs[:, 1:]
       
        input_emb = self.visual_encoder(item_modal.flatten(0,1)).view(-1, self.max_seq_length,self.embedding_size)        #[batch, max_seq_len, dim]
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq,bidirectional=False)

        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
        output_embs = output_embs[-1]  ##[batch, max_seq_len,dim]
        output_logits = self.pred(output_embs) #[batch, max_seq_len,n_item]
        indices = torch.where(item_seq != 0)
        logits = output_logits[indices]   #[N, n_item]
        target = item_target[indices]
        loss = self.loss_func(logits, target)
        return loss

    

    @torch.no_grad()
    def predict(self, item_seq, item_feature):
       
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = item_feature[item_seq]
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq,bidirectional=False)

        output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)
        output_embs = output[-1]
        seq_output = output_embs[:, -1]
               
        scores = torch.matmul(seq_output, item_feature.t())  # [B n_items]
        return scores


    def compute_item(self, item):
        return self.visual_encoder(item)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask

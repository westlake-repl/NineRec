
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class TextEncoder(torch.nn.Module):
    def __init__(self, bert_model, args):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model

        self.text_proj = nn.Linear(args.word_embedding_dim, args.embedding_dim)
        xavier_normal_(self.text_proj.weight.data)
        if self.text_proj.bias is not None:
            constant_(self.text_proj.bias.data, 0)
        


    def forward(self, text):

        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)

        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]

        

        hidden_states = self.text_proj(hidden_states)
        return hidden_states


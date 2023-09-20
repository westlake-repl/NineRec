import torchvision.models as models
import clip
from transformers import CLIPVisionModel
from REC.model.layers import ItemEncoder, FIXItemEncoder
from transformers import CLIPVisionModel,SwinModel,ViTMAEModel,SwinConfig
import torch
from REC.model.layers import *
from transformers import BertModel, BertTokenizer, BertConfig


class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        cls_after_fc = self.fc(hidden_states[:, 0])
        return self.activate(cls_after_fc)


def load_model(config):
    bert_model_load = 'hfl/chinese-bert-wwm'
    bconfig = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    nlp_model = BertModel.from_pretrained(bert_model_load, config=bconfig)
    pooler_para = [197, 198]    
    for index, (name, param) in enumerate(nlp_model.named_parameters()):
        if index < 0 or index in pooler_para:
            param.requires_grad = False
    model = TextEncoder(nlp_model, config['embedding_size'],768)
    return model





def load_weights(config):
    image_feature_path = config['v_feat_path']
    device = config['device']
    output_dim = config['embedding_size']
    activation = config['fine_tune_arg']['activation']
    dnn_layers = config['dnn_layers']
  
    model = FIXItemEncoder(weight_path=image_feature_path, device=device
    , output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)

    return model
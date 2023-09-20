import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers, BaseFactorizationMachine
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from logging import getLogger
from REC.model.load import load_model


class MOFM(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(MOFM, self).__init__()

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']

        self.device = config['device']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']

        self.item_num = dataload.item_num
        self.visual_encoder = load_model(config=config)

        self.fm = BaseFactorizationMachine(reduce_sum=True)
        
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
    
    def mask_emb(self, input_item_embs, mask):
       
        mask = mask.unsqueeze(-1).expand_as(input_item_embs)
        masked_token_seq_embedding = input_item_embs * mask
 
        return masked_token_seq_embedding

    
    
    def forward(self, inputs):  #[batch, 2, seq_len+2]              

        items_index,all_item_modal = inputs
        mask = items_index != 0    
        all_item_embs = self.visual_encoder(all_item_modal)
        input_item_embs = all_item_embs[items_index, :]
        
        inputs_embedding = self.mask_emb(input_item_embs, mask)
        scores = self.fm(inputs_embedding.flatten(0,1))  
        output = scores.view(-1,2) 

        # from logging import getLogger
        # import sys
        # import random
        # logger = getLogger()
        # logger.info(items_index.shape)
        # logger.info(all_item_modal.shape)
        # logger.info(f'pos : {items_index[0][0]}, \n neg : {items_index[0][1]}\n')
        # logger.info(items_index)
        # logger.info(all_item_embs.shape)
        # logger.info(input_item_embs.shape)
        # logger.info(input_item_embs)
        # logger.info(all_item_modal)
        # # logger.info(f'pos_emb : {self.mask_emb(inputs[0][0])}, \n neg_emb : {self.mask_emb(inputs[0][1])}\n')
        # # logger.info(f'pos_score : {self.fm(self.mask_emb(inputs[0][0]).unsqueeze(0))}, \n   \
        # # neg_score : {self.fm(self.mask_emb(inputs[0][1]).unsqueeze(0))}\n')
        
        # if random.random() < 0.5:
        #     sys.exit()
          


        
        #self.logger.info(output[0])

        # import sys
        # import random
        # if random.random() < 0.5:
        #     sys.exit()        
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
   

    @torch.no_grad()
    def predict(self,user_seq,item_feature):
        mask = user_seq != 0
        input_embs = item_feature[user_seq]                                                
        user_embedding = self.mask_emb(input_embs, mask)   
        user_embedding = torch.sum(user_embedding, dim=1)
        scores = torch.matmul(user_embedding,item_feature.t())
        return scores

    @torch.no_grad()    # [num_item, 64]
    def compute_item(self,item):
        return self.visual_encoder(item)
 
        #return torch.arange(0,self.n_items).to(self.device)





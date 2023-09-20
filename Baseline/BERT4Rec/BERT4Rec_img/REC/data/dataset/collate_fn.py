import torch
import numpy as np



def seq_eval_collate(batch):
    item_seq = []
    item_target = []
    #item_length = []

    history_i = []

    for item in batch:
        history_i.append(item[0])
        item_seq.append(item[1])
        item_target.append(item[2])
        #item_length.append(item[3])
        
        
    
    
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)
    
    item_seq = torch.tensor(item_seq)          #[batch, len]
    item_target = torch.tensor(item_target)    #[batch]
    #item_length = torch.tensor(item_length)    #[batch]    
    positive_u = torch.arange(item_seq.shape[0])   #[batch]


    #return item_seq, None, positive_u, item_target  
    return item_seq, (history_u, history_i), positive_u, item_target





def pair_eval_collate(batch):

    user = []
    history_i = []
    positive_i = []
    for item in batch:
        user.append(item[0])
        history_i.append(item[1])
        positive_i.append(item[2])
    
    user = torch.tensor(user)
    
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)

    positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_i)])            
    positive_i = torch.cat(positive_i)

    return user, (history_u, history_i), positive_u, positive_i





def candi_eval_collate(batch):
    item_seq = []
    item_target = []
    history_i = []

    for item in batch:
        history_i.append(item[0])
        item_seq.append(item[1])         #[n_items, len]
        item_target.append(item[2])
           
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)
    
    item_seq = torch.stack(item_seq)          #[batch, n_items, len]
    item_target = torch.tensor(item_target)    #[batch]
    positive_u = torch.arange(item_seq.shape[0])   #[batch]

    return item_seq, (history_u, history_i), positive_u, item_target




def sampletower_train_collate(batch):
    items = []
    for item_aug in batch:
        items.append(item_aug)
    return torch.cat(items)



def base_collate(batch):
    return batch[0]

def mosampletower_train_collate(batch):
    items_index = []
    items_modal = []
    items_bias = 0
    for patch in batch:
        index= patch[0] + items_bias
        mask = patch[1]
        modal = patch[2]
        index *= mask 
        items_index.append(index)   
        items_modal.append(modal)
        items_bias += modal.shape[0]
    return torch.cat(items_index), torch.cat(items_modal)
import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_, constant_


class VisionEncoder(torch.nn.Module):
    def __init__(self, args, image_net):
        super(VisionEncoder, self).__init__()

        self.image_net = image_net
        if "swin-base-patch4-window7-224"  in args.CV_model_load:
            self.cv_proj = nn.Linear(1024, args.embedding_dim)
        else:
            self.cv_proj = nn.Linear(args.word_embedding_dim, args.embedding_dim)

        xavier_normal_(self.cv_proj.weight.data)
        if self.cv_proj.bias is not None:
            constant_(self.cv_proj.bias.data, 0)

    def forward(self, item_content):
        
        last_hidden_state_CV = self.image_net(item_content)[0]
        last_hidden_state_CV = self.cv_proj(last_hidden_state_CV)

        return last_hidden_state_CV

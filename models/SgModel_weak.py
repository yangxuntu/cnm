# This file contains Supervised training of Scene Graph


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import misc.utils as utils

class SgModel(nn.Module):
    def __init__(self, opt):
        super(SgModel, self).__init__()
        #number of classes
        self.attr_num = opt.sg_attr_num

        self.drop_prob_lm = opt.drop_prob_lm
        self.att_feat_size = opt.att_feat_size

        self.train_attr = getattr(opt, 'train_attr', 0)
        self.train_rela = getattr(opt, 'train_rela', 0)

        self.small_net_attr = nn.Sequential(nn.Linear(self.att_feat_size, 1024),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm),
                                            nn.Linear(1024, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(self.drop_prob_lm)
                                            )

        self.attr_cls = nn.Linear(1024, self.attribute_size)
        self.attr_det = nn.Linear(1024, self.attribute_size)



    def forward(self, att_feats, att_masks):
        map_fc = self.small_net_attr(att_feats)
        fc_cls = self.attr_cls(map_fc) #batch_size* box_size *1024
        fc_det = self.attr_det(map_fc) #batch_size* box_size *1024

        nor_cls = F.softmax(fc_cls, dim = 2) #batch_size* box_size *attribute_size

        weight = F.softmax(fc_det, dim = 1) #batch_size* box_size *attribute_size
        # att_masks batch_size*box_size; att_masks_temp batch_size* box_size *attribute_size
        att_masks_temp = att_masks.unsqueeze(2).expand_as(weight)
        weight = weight * att_masks_temp.float() #batch_size* box_size *attribute_size
        nor_det = weight / weight.sum(1, keepdim=True).expand_as(weight)
        output_temp = nor_cls*nor_det #batch_size* box_size *attribute_size
        output = output_temp.sum(dim=1) #batch_size*attribute_size
        return output, output_temp
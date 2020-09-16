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
        self.cls_num = opt.sg_cls_num
        self.train_mode = opt.sg_train_mode

        self.drop_prob_lm = opt.drop_prob_lm
        self.att_feat_size = opt.att_feat_size

        if self.train_mode == 'rela':
            self.small_net = nn.Sequential(nn.Linear(self.att_feat_size*2, 1024),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(self.drop_prob_lm),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(self.drop_prob_lm)
                                                )
        else:
            self.small_net = nn.Sequential(nn.Linear(self.att_feat_size, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(self.drop_prob_lm),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(self.drop_prob_lm)
                                                )

        self.cls = nn.Linear(1024, self.cls_num)

    def forward(self, att_feats):
        map_fc = self.small_net(att_feats) #batch_size* att_max *1024
        fc_cls = self.cls(map_fc)  #batch_size* att_max *cls_num
        if self.train_mode != 'attr':
            logit = F.log_softmax(fc_cls, dim = 2) #batch_size* att_max *cls_num
        else:
            logit = F.sigmoid(fc_cls)
        return logit

#
# class SgModel(nn.Module):
#     def __init__(self, opt):
#         super(SgModel, self).__init__()
#         #number of classes
#         self.cls_num = opt.attr_num
#
#         self.drop_prob_lm = opt.drop_prob_lm
#         self.att_feat_size = opt.att_feat_size
#
#
#         self.small_net = nn.Sequential(nn.Linear(512*7*7, self.att_feat_size),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(self.drop_prob_lm),
#                                             nn.Linear(self.att_feat_size, self.att_feat_size),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(self.drop_prob_lm)
#                                             )
#
#         self.cls = nn.Linear(self.att_feat_size, self.cls_num)
#
#     def forward(self, rs_data):
#         map_fc = self.small_net(rs_data['attr_feats']) #batch_size* att_max *1024
#         fc_cls = self.cls(map_fc)  #batch_size* att_max *cls_num
#
#         logit = F.sigmoid(fc_cls)
#         rs_data['attr_logits'] = logit
#         return rs_data
#
#     def extract(self, rs_data):
#         map_fc = self.small_net(rs_data['attr_feats'])
#         rs_data['map_fc'] = map_fc
#         return rs_data
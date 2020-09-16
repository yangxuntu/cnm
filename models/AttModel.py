# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import copy

from .CaptionModel import CaptionModel

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img

        self.index_eval = getattr(opt, 'index_eval', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_attr_info = getattr(opt, 'use_attr_info', 1)

        #whether use relationship or not
        self.use_rela = getattr(opt, 'use_rela', 0)
        self.use_gru = getattr(opt, 'use_gru', 0)
        self.use_gfc = getattr(opt, 'use_gfc', 0)
        self.gru_t = getattr(opt, 'gru_t', 1)

        self.rbm_logit = getattr(opt, 'rbm_logit', 0)
        self.rbm_size = getattr(opt, 'rbm_size', 2000)

        #whether use sentence scene graph or not
        self.use_ssg = getattr(opt, 'use_ssg', 0)
        self.relu_mod = getattr(opt, 'relu_mod', 'relu')

        self.ss_prob = 0.0 # Schedule sampling probability
        if self.relu_mod == 'relu':
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        elif self.relu_mod == 'leaky_relu':
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
            self.att_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.LeakyReLU(0.1, inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        if self.use_rela:
            self.rela_dict_len = getattr(opt, 'rela_dict_size', 0)

            if self.use_gru:
                self.rela_embed = nn.Embedding(self.rela_dict_len, self.rnn_size)
                self.gru = nn.GRU(self.rnn_size * 2, self.rnn_size)
            if self.use_gfc:
                self.rela_embed = nn.Linear(self.rela_dict_len, self.rnn_size, bias=False)
                self.sbj_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
                self.obj_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
                self.rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
                self.attr_fc = nn.Sequential(nn.Linear(self.rnn_size*2, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
        if self.use_ssg:
            self.sbj_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 3, self.rnn_size),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm))
            self.obj_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size*3, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.obj_obj_fc = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.obj_attr_fc = nn.Sequential(nn.Linear(self.input_encoding_size*2, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size*3, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.attr_fc = nn.Sequential(nn.Linear(self.input_encoding_size*2, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))

        if self.rbm_logit == 1:
            self.logit = Log_Rbm(self.rnn_size, self.rbm_size, self.vocab_size + 1)
        else:
            self.logit_layers = getattr(opt, 'logit_layers', 1)
            if self.logit_layers == 1:
                self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
            else:
                self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
                self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        #att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)
        nzs = torch.nonzero(att_feats[0]).size(0)
        nt = att_feats[0].numel()
        print('nonzeros:{0}'.format((nzs+0.0)/nt))

        return fc_feats, att_feats, p_att_feats

    def graph_gru(self, rela_data, pre_hidden_state):
        """
        :param att_feats: roi features of each bounding box
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param pre_hidden_state: previous hidden state
        :return: hidden_state: current hidden state
        """
        att_feats = rela_data['att_feats']
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        neighbor = torch.zeros([N_img, att_feats_size[1], att_feats_size[2]])
        neighbor = neighbor.cuda()
        #hidden_state = torch.zeros(pre_hidden_state.size())
        hidden_state = att_feats.clone()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id*seq_per_img,:])
            N_rela = np.int32(N_rela)
            box_num = np.zeros([att_feats_size[1],])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id*seq_per_img,i,0]
                obj_id = rela_matrix[img_id*seq_per_img,i,1]
                rela_id = rela_matrix[img_id*seq_per_img,i,2]
                sub_id = np.int32(sub_id)
                obj_id = np.int32(obj_id)
                rela_embedding = self.rela_embed(rela_id.long())
                rela_embedding = torch.squeeze(rela_embedding)
                neighbor[img_id, sub_id] += rela_embedding * att_feats[img_id * seq_per_img, obj_id]
                box_num[sub_id] += 1
            for i in range(att_feats_size[1]):
                if box_num[i] != 0:
                    neighbor[img_id, i] /= box_num[i]
                    input = torch.cat((att_feats[img_id*seq_per_img, i], neighbor[img_id, i]))
                    input = torch.unsqueeze(input, 0)
                    input = torch.unsqueeze(input, 0)
                    hidden_state_temp = pre_hidden_state[img_id*seq_per_img, i]
                    hidden_state_temp = torch.unsqueeze(hidden_state_temp,0)
                    hidden_state_temp = torch.unsqueeze(hidden_state_temp,0)
                    hidden_state_temp, out_temp = self.gru(input, hidden_state_temp)
                    hidden_state[img_id*seq_per_img:(img_id+1)*seq_per_img, i] = torch.squeeze(hidden_state_temp)


        return hidden_state

    def graph_gfc(self, rela_data):
        """
        :param att_feats: roi features of each bounding box, [N_img*5, N_att_max, rnn_size]
        :param rela_feats: the embeddings of relationship, [N_img*5, N_rela_max, rnn_size]
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param att_masks: attention masks, [N_img*5, N_att_max].
                            For each row, the sum of that row is the total number
                            of roi poolings.
        :param attr_matrix: attribute matrix,[N_img*5, N_attr_max, N_attr_each_max]
                            N_img is the batch size, N_attr_max is the maximum number
                            of attributes of one mini-batch, N_attr_each_max is the
                            maximum number of attributes of each objects in that mini-batch
        :param attr_masks: attribute masks, [N_img*5, N_attr_max, N_attr_each_max]
                            the sum of attr_masks[img_id*5,:,0] is the number of objects
                            which own attributes, the sum of attr_masks[img_id*5, obj_id, :]
                            is the number of attribute that object has
        :return: att_feats_new: new roi features
                 rela_feats_new: new relationship embeddings
                 attr_feats_new: new attribute features
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_matrix = rela_data['rela_matrix']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_matrix = rela_data['attr_matrix']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        rela_feats_size = rela_feats.size()
        attr_masks_size = attr_masks.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        att_feats_new = att_feats.clone()
        rela_feats_new = rela_feats.clone()
        if self.use_attr_info == 1:
            attr_feats_new = torch.zeros([attr_masks_size[0], attr_masks_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_box = torch.sum(att_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            N_box = int(N_box)
            box_num = np.ones([N_box,])
            rela_num = np.ones([N_rela,])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id * seq_per_img, i, 0]
                sub_id = int(sub_id)
                box_num[sub_id] += 1.0
                obj_id = rela_matrix[img_id * seq_per_img, i, 1]
                obj_id = int(obj_id)
                box_num[obj_id] += 1.0
                rela_id = i
                rela_num[rela_id] += 1.0
                sub_feat_use = att_feats[img_id * seq_per_img, sub_id, :]
                obj_feat_use = att_feats[img_id * seq_per_img, obj_id, :]
                rela_feat_use = rela_feats[img_id * seq_per_img, rela_id, :]


                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, sub_id, :] += \
                    self.sbj_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, obj_id, :] += \
                    self.obj_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, rela_id, :] += \
                    self.rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))

            if self.use_attr_info == 1:
                N_obj_attr = torch.sum(attr_masks[img_id * seq_per_img, :, 0])
                N_obj_attr = int(N_obj_attr)
                for i in range(N_obj_attr):
                    attr_obj_id = int(attr_matrix[img_id * seq_per_img, i, 0])
                    obj_feat_use = att_feats[img_id * seq_per_img, int(attr_obj_id), :]
                    N_attr_each = torch.sum(attr_masks[img_id * seq_per_img, i, :])
                    for j in range(N_attr_each-1):
                        attr_index = attr_matrix[img_id * seq_per_img, i, j+1]
                        attr_one_hot = torch.zeros([self.rela_dict_len,])
                        attr_one_hot = attr_one_hot.scatter_(0,attr_index.cpu().long(),1).cuda()
                        attr_feat_use = self.rela_embed(attr_one_hot)
                        attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :] += \
                            self.attr_fc( torch.cat((attr_feat_use, obj_feat_use)) )
                    attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :] = \
                        attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :]/(float(N_attr_each)-1)


            for i in range(N_box):
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i] = \
                    att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i]/box_num[i]
            for i in range(N_rela):
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] = \
                    rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :]/rela_num[i]

        rela_data['att_feats'] = att_feats_new
        rela_data['rela_feats'] = rela_feats_new
        if self.use_attr_info == 1:
            rela_data['attr_feats'] = attr_feats_new
        return rela_data


    def prepare_rela_feats(self, rela_data):
        """
        Change relationship index (one-hot) to relationship features, or change relationship
        probability to relationship features.
        :param rela_matrix:
        :param rela_masks:
        :return: rela_features, [N_img*5, N_rela_max, rnn_size]
        """
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        rela_feats_size = rela_matrix.size()
        N_att = rela_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        rela_feats = torch.zeros([rela_feats_size[0], rela_feats_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            if N_rela>0:
                rela_index = rela_matrix[img_id*seq_per_img,:N_rela,2].cpu().long()
                rela_index = torch.unsqueeze(rela_index,1)
                rela_one_hot = torch.zeros([N_rela, self.rela_dict_len])
                rela_one_hot = rela_one_hot.scatter_(1, rela_index, 1).cuda()
                rela_feats_temp = self.rela_embed(rela_one_hot)
                rela_feats[img_id*seq_per_img:(img_id+1)*seq_per_img,:N_rela,:] = rela_feats_temp
        rela_data['rela_feats'] = rela_feats
        return rela_data

    def merge_rela_att(self, rela_data):
        """
        merge attention features (roi features) and relationship features together
        :param att_feats: [N_att, N_att_max, rnn_size]
        :param att_masks: [N_att, N_att_max]
        :param rela_feats: [N_att, N_rela_max, rnn_size]
        :param rela_masks: [N_att, N_rela_max]
        :return: att_feats_new: [N_att, N_att_new_max, rnn_size]
                 att_masks_new: [N_att, N_att_new_max]
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        if self.use_attr_info == 1:
            attr_feats = rela_data['attr_feats']
            attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        N_att_new_max = -1
        for img_id in range(int(N_img)):
            if self.use_attr_info != 0:
                N_att_new_max = \
                max(N_att_new_max,torch.sum(rela_masks[img_id * seq_per_img, :]) +
                    torch.sum(att_masks[img_id * seq_per_img, :]) + torch.sum(attr_masks[img_id * seq_per_img,:,0]))
            else:
                N_att_new_max = \
                    max(N_att_new_max, torch.sum(rela_masks[img_id * seq_per_img, :]) +
                        torch.sum(att_masks[img_id * seq_per_img, :]))
        att_masks_new = torch.zeros([N_att, int(N_att_new_max)]).cuda()
        att_feats_new = torch.zeros([N_att, int(N_att_new_max), self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = int(torch.sum(rela_masks[img_id * seq_per_img, :]))
            N_box = int(torch.sum(att_masks[img_id * seq_per_img, :]))
            if self.use_attr_info == 1:
                N_attr = int(torch.sum(attr_masks[img_id * seq_per_img,:,0]))
            else:
                N_attr = 0

            att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :] = \
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :]
            if N_rela > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela, :] = \
                    rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_rela, :]
            if N_attr > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela: N_box + N_rela + N_attr, :] = \
                    attr_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_attr, :]
            att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box] = 1
            if N_rela > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela] = 1
            if N_attr > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela:N_box + N_rela + N_attr] = 1

        return att_feats_new, att_masks_new

    def ssg_gfc(self, ssg_data):
        """
        use sentence scene graph's graph network to embed feats,
        :param ssg_data: one dict which contains the following data:
               ssg_data['ssg_rela_matrix']: relationship matrix for ssg data,
                    [N_att, N_rela_max, 3] array
               ssg_data['ssg_rela_masks']: relationship masks for ssg data,
                    [N_att, N_rela_max]
               ssg_data['ssg_obj']: obj index for ssg data, [N_att, N_obj_max]
               ssg_data['ssg_obj_masks']: obj masks, [N_att, N_obj_max]
               ssg_data['ssg_attr']: attribute indexes, [N_att, N_obj_max, N_attr_max]
               ssg_data['ssg_attr_masks']: attribute masks, [N_att, N_obj_max, N_attr_max]
        :return: ssg_data_new one dict which contains the following data:
                 ssg_data_new['ssg_rela_feats']: relationship embeddings, [N_att, N_rela_max, rnn_size]
                 ssg_data_new['ssg_rela_masks']: equal to ssg_data['ssg_rela_masks']
                 ssg_data_new['ssg_obj_feats']: obj embeddings, [N_att, N_obj_max, rnn_size]
                 ssg_data_new['ssg_obj_masks']: equal to ssg_data['ssg_obj_masks']
                 ssg_data_new['ssg_attr_feats']: attributes embeddings, [N_att, N_attr_max, rnn_size]
                 ssg_data_new['ssg_attr_masks']: equal to ssg_data['ssg_attr_masks']
        """
        ssg_data_new = {}
        ssg_data_new['ssg_rela_masks'] = ssg_data['ssg_rela_masks']
        ssg_data_new['ssg_obj_masks'] = ssg_data['ssg_obj_masks']
        ssg_data_new['ssg_attr_masks'] = ssg_data['ssg_attr_masks']

        ssg_obj = ssg_data['ssg_obj']
        ssg_obj_masks = ssg_data['ssg_obj_masks']
        ssg_attr = ssg_data['ssg_attr']
        ssg_attr_masks = ssg_data['ssg_attr_masks']
        ssg_rela_matrix = ssg_data['ssg_rela_matrix']
        ssg_rela_masks = ssg_data['ssg_rela_masks']

        ssg_obj_feats = torch.zeros([ssg_obj.size()[0], ssg_obj.size()[1], self.rnn_size]).cuda()
        ssg_rela_feats = torch.zeros([ssg_rela_matrix.size()[0], ssg_rela_matrix.size()[1], self.rnn_size]).cuda()
        ssg_attr_feats = torch.zeros([ssg_attr.size()[0], ssg_attr.size()[1], self.rnn_size]).cuda()
        ssg_attr_masks_new = torch.zeros(ssg_obj.size()).cuda()

        ssg_obj_size = ssg_obj.size()
        N_att = ssg_obj_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att/seq_per_img)

        for img_id in range(N_img):
            N_obj = int(torch.sum(ssg_obj_masks[img_id*seq_per_img,:]))
            if N_obj == 0:
                continue
            obj_feats_ori = self.embed(ssg_obj[img_id*seq_per_img,:N_obj].cuda().long())
            obj_feats_temp = self.obj_obj_fc(obj_feats_ori)
            obj_num = np.ones([N_obj,])

            N_rela = int(torch.sum(ssg_rela_masks[img_id*seq_per_img,:]))
            rela_feats_temp = torch.zeros([N_rela, self.rnn_size])
            for rela_id in range(N_rela):
                sbj_id = int(ssg_rela_matrix[img_id * seq_per_img, rela_id, 0])
                obj_id = int(ssg_rela_matrix[img_id * seq_per_img, rela_id, 1])
                rela_index = ssg_rela_matrix[img_id * seq_per_img, rela_id, 2]
                sbj_feat = obj_feats_ori[sbj_id]
                obj_feat = obj_feats_ori[obj_id]
                rela_feat = self.embed(rela_index.cuda().long())
                obj_feats_temp[sbj_id] = obj_feats_temp[sbj_id] + self.sbj_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
                obj_num[sbj_id] = obj_num[sbj_id] + 1.0
                obj_feats_temp[obj_id] = obj_feats_temp[obj_id] + self.obj_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
                obj_num[obj_id] = obj_num[obj_id] + 1.0
                rela_feats_temp[rela_id] = self.rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
            for obj_id in range(N_obj):
                obj_feats_temp[obj_id] = obj_feats_temp[obj_id]/obj_num[obj_id]

            attr_feats_temp = torch.zeros([N_obj, self.rnn_size]).cuda()
            obj_attr_ids = 0
            for obj_id in range(N_obj):
                N_attr = int(torch.sum(ssg_attr_masks[img_id*seq_per_img, obj_id,:]))
                if N_attr != 0:
                    attr_feat_ori = self.embed(ssg_attr[img_id * seq_per_img, obj_id, :N_attr].cuda().long())
                    for attr_id in range(N_attr):
                        attr_feats_temp[obj_attr_ids] = attr_feats_temp[obj_attr_ids] +\
                                                        self.attr_fc(torch.cat((obj_feats_ori[obj_id], attr_feat_ori[attr_id])))
                    attr_feats_temp[obj_attr_ids] = attr_feats_temp[obj_attr_ids]/(N_attr + 0.0)
                    obj_attr_ids += 1
            N_obj_attr = obj_attr_ids
            ssg_attr_masks_new[img_id*seq_per_img:(img_id+1)*seq_per_img, :N_obj_attr] = 1

            ssg_obj_feats[img_id * seq_per_img: (img_id+1) * seq_per_img, :N_obj, :] = obj_feats_temp
            if N_rela != 0:
               ssg_rela_feats[img_id * seq_per_img: (img_id+1) * seq_per_img, :N_rela, :] = rela_feats_temp
            if N_obj_attr != 0:
                ssg_attr_feats[img_id * seq_per_img: (img_id+1) * seq_per_img, :N_obj_attr, :] = attr_feats_temp[:N_obj_attr]


        ssg_data_new['ssg_obj_feats'] = ssg_obj_feats
        ssg_data_new['ssg_rela_feats'] = ssg_rela_feats
        ssg_data_new['ssg_attr_feats'] = ssg_attr_feats
        ssg_data_new['ssg_attr_masks'] = ssg_attr_masks_new
        return ssg_data_new

    def merge_ssg_att(self, ssg_data_new):
        """
        merge ssg_obj_feats, ssg_rela_feats, ssg_attr_feats together
        :param ssg_data_new:
        :return: att_feats: [N_att, N_att_max, rnn_size]
                 att_masks: [N_att, N_att_max]
        """
        ssg_obj_feats = ssg_data_new['ssg_obj_feats']
        ssg_rela_feats = ssg_data_new['ssg_rela_feats']
        ssg_attr_feats = ssg_data_new['ssg_attr_feats']
        ssg_rela_masks = ssg_data_new['ssg_rela_masks']
        ssg_obj_masks = ssg_data_new['ssg_obj_masks']
        ssg_attr_masks = ssg_data_new['ssg_attr_masks']

        ssg_obj_size = ssg_obj_feats.size()
        N_att = ssg_obj_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att / seq_per_img)

        N_att_max = -1
        for img_id in range(N_img):
            N_rela = int(torch.sum(ssg_rela_masks[img_id*seq_per_img,:]))
            N_obj = int(torch.sum(ssg_obj_masks[img_id*seq_per_img,:]))
            N_attr = int(torch.sum(ssg_attr_masks[img_id*seq_per_img,:]))
            N_att_max = max(N_att_max, N_rela + N_obj + N_attr)

        att_feats = torch.zeros([N_att, N_att_max, self.rnn_size]).cuda()
        att_masks = torch.zeros([N_att, N_att_max]).cuda()

        for img_id in range(N_img):
            N_rela = int(torch.sum(ssg_rela_masks[img_id * seq_per_img, :]))
            N_obj = int(torch.sum(ssg_obj_masks[img_id * seq_per_img, :]))
            N_attr = int(torch.sum(ssg_attr_masks[img_id * seq_per_img, :]))
            if N_obj != 0:
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_obj, :] = \
                    ssg_obj_feats[img_id * seq_per_img, :N_obj, :]
            if N_rela != 0:
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_obj:N_obj+N_rela, :] = \
                    ssg_rela_feats[img_id * seq_per_img, :N_rela, :]

            if N_attr != 0:
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_obj+N_rela:N_obj+N_attr+N_rela, :] = \
                    ssg_attr_feats[img_id * seq_per_img, :N_attr, :]
            att_masks[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_obj+N_rela+N_attr] = 1
        return att_feats, att_masks


    def _forward(self, fc_feats, att_feats, seq, att_masks=None, rela_data=None, ssg_data=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        if self.use_ssg == 0:
            fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
            if self.use_rela:
                rela_data['att_feats'] = att_feats

        #process feats
        if self.use_rela:
            if self.use_gru:
                pre_hidden_state = torch.zeros(att_feats.size()).cuda()
                for t in range(self.gru_t):
                    pre_hidden_state = self.graph_gru(att_feats, rela_data, pre_hidden_state.cuda())
                att_feats = pre_hidden_state.cuda()
                p_att_feats = self.ctx2att(att_feats)
            elif self.use_gfc:
                rela_data = self.prepare_rela_feats(rela_data)
                for t in range(self.gru_t):
                    rela_data = \
                        self.graph_gfc(rela_data)
                att_feats, att_masks = self.merge_rela_att(rela_data)
                p_att_feats = self.ctx2att(att_feats)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new =self.ssg_gfc(ssg_data)
            att_feats, att_masks = self.merge_ssg_att(ssg_data_new)
            p_att_feats = self.ctx2att(att_feats)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            outputs[:, i] = output
            # outputs.append(output)

        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def _extract_hs(self, fc_feats, att_feats, seq, att_masks=None, rela_data=None, ssg_data=None):
        """
        extract hidden states
        :param fc_feats:
        :param att_feats:
        :param seq:
        :param att_masks:
        :param rela_data:
        :param ssg_data:
        :return:
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.rnn_size)

        if self.use_ssg == 0:
            fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
            if self.use_rela:
                rela_data['att_feats'] = att_feats

        #process feats
        if self.use_rela:
            if self.use_gru:
                pre_hidden_state = torch.zeros(att_feats.size()).cuda()
                for t in range(self.gru_t):
                    pre_hidden_state = self.graph_gru(att_feats, rela_data, pre_hidden_state.cuda())
                att_feats = pre_hidden_state.cuda()
                p_att_feats = self.ctx2att(att_feats)
            elif self.use_gfc:
                rela_data = self.prepare_rela_feats(rela_data)
                for t in range(self.gru_t):
                    rela_data = \
                        self.graph_gfc(rela_data)
                att_feats, att_masks = self.merge_rela_att(rela_data)
                p_att_feats = self.ctx2att(att_feats)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new =self.ssg_gfc(ssg_data)
            att_feats, att_masks = self.merge_ssg_att(ssg_data_new)
            p_att_feats = self.ctx2att(att_feats)


        for i in range(seq.size(1) - 1):
            it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            outputs[:, i] = state[0][-1]
            # outputs.append(output)

        return outputs

    def _extract_e(self, fc_feats, att_feats, seq, att_masks=None, rela_data=None, ssg_data=None):
        """
        extract embeddings
        :param fc_feats:
        :param att_feats:
        :param seq:
        :param att_masks:
        :param rela_data:
        :param ssg_data:
        :return:
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.rnn_size)

        if self.use_ssg == 0:
            fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
            if self.use_rela:
                rela_data['att_feats'] = att_feats

        #process feats
        if self.use_rela:
            if self.use_gru:
                pre_hidden_state = torch.zeros(att_feats.size()).cuda()
                for t in range(self.gru_t):
                    pre_hidden_state = self.graph_gru(att_feats, rela_data, pre_hidden_state.cuda())
                att_feats = pre_hidden_state.cuda()
                p_att_feats = self.ctx2att(att_feats)
            elif self.use_gfc:
                rela_data = self.prepare_rela_feats(rela_data)
                for t in range(self.gru_t):
                    rela_data = \
                        self.graph_gfc(rela_data)
                att_feats, att_masks = self.merge_rela_att(rela_data)
                p_att_feats = self.ctx2att(att_feats)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new =self.ssg_gfc(ssg_data)
            att_feats, att_masks = self.merge_ssg_att(ssg_data_new)
            p_att_feats = self.ctx2att(att_feats)

        return att_feats, att_masks

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, rela_data = None, ssg_data=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        if self.use_ssg == 0:
            fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
            if self.use_rela:
                rela_data['att_feats'] = att_feats

        # process feats
        if self.use_rela:
            if self.use_gru:
                pre_hidden_state = torch.zeros(att_feats.size()).cuda()
                for t in range(self.gru_t):
                    pre_hidden_state = self.graph_gru(att_feats, rela_data, pre_hidden_state.cuda())
                att_feats = pre_hidden_state.cuda()
                p_att_feats = self.ctx2att(att_feats)
            elif self.use_gfc:
                rela_data = self.prepare_rela_feats(rela_data)
                for t in range(self.gru_t):
                    rela_data = \
                        self.graph_gfc(rela_data)
                att_feats, att_masks = self.merge_rela_att(rela_data)
                p_att_feats = self.ctx2att(att_feats)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new = self.ssg_gfc(ssg_data)
            att_feats, att_masks = self.merge_ssg_att(ssg_data_new)
            p_att_feats = self.ctx2att(att_feats)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, rela_data=None, ssg_data=None, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, rela_data, ssg_data, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        if self.use_ssg == 0:
            fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
            if self.use_rela:
                rela_data['att_feats'] = att_feats

        # process feats
        if self.use_rela:
            if self.use_gru:
                pre_hidden_state = torch.zeros(att_feats.size()).cuda()
                for t in range(self.gru_t):
                    pre_hidden_state = self.graph_gru(att_feats, rela_data, pre_hidden_state.cuda())
                att_feats = pre_hidden_state.cuda()
                p_att_feats = self.ctx2att(att_feats)
            elif self.use_gfc:
                rela_data = self.prepare_rela_feats(rela_data)
                for t in range(self.gru_t):
                    rela_data = \
                        self.graph_gfc(rela_data)
                att_feats, att_masks = self.merge_rela_att(rela_data)
                p_att_feats = self.ctx2att(att_feats)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new = self.ssg_gfc(ssg_data)
            att_feats, att_masks = self.merge_ssg_att(ssg_data_new)
            p_att_feats = self.ctx2att(att_feats)

        # seq = []
        # seqLogprobs = []
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                # seq.append(it) #seq[t] the input of t+2 time step

                # seqLogprobs.append(sampleLogprobs.view(-1))
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

        return seq, seqLogprobs
        # return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = F.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        t=0
        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)

        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:,:1], att_masks], 1) # assume one one at the first time step.
            PI = PI / PI.sum(1, keepdim=True)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.topdown_res = getattr(opt, 'topdown_res', 0)

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        #self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        #att_lstm_input = torch.cat([fc_feats, xt], 1)

        # state[0][0] means the hidden state h in first lstm
        # state[1][0] means the cell c in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        if self.topdown_res:
            h_lang = h_lang + h_att
            c_lang = c_lang + c_att

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class MTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(MTopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.mtopdown_res = getattr(opt, 'mtopdown_res', 0)
        self.mtopdown_num = opt.mtopdown_num
        self.topdown_layers = clones(TopDownCore(opt), self.mtopdown_num)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        for i in range(self.mtopdown_num):
            _, (state[0][2*i:2*(i+1)],state[1][2*i:2*(i+1)]) = \
                self.topdown_layers[i](xt, fc_feats, att_feats, p_att_feats, (state[0][2*i:2*(i+1)].clone(),state[1][2*i:2*(i+1)].clone()), att_masks=None)
            # state[0][2*i:2*(i+1)] = state_temp[0][0:2]
            # state[1][2*i:2*(i+1)] = state_temp[1][0:2]
            # xt_new = state_temp[0][1]
            if self.mtopdown_res:
                xt = xt + state[0][2*i+1].clone()
            else:
                xt = state[0][2*i+1].clone()

        output = F.dropout(xt, self.drop_prob_lm, self.training)
        return output, state


class GTssgCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(GTssgCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        #self.lstm1 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, h^2_t-1
        self.lstm1 = nn.LSTMCell(opt.input_encoding_size, opt.rnn_size) # we, h^2_t-1
        self.lstm2 = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        #lstm1_input = torch.cat([prev_h, xt], 1)
        lstm1_input = torch.cat([xt], 1)

        #state[0][0] means the hidden state c in first lstm
        #state[1][0] means the cell h in first lstm
        #state[0] means hidden state and state[1] means cell, state[0][i] means
        #the i-th layer's hidden state
        h_att, c_att = self.lstm1(lstm1_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lstm2_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lstm2(lstm2_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

from .FCModel import LSTMCore
class StackAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1,att_res_2],1), [state[0][2:3], state[1][2:3]])

        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class DenseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size*2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size*3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)),att_res_2],1), [state[0][2:3], state[1][2:3]])

        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        #att_index = torch.argmax(weight,dim=1)
        #print(att_index)
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class Att2inCore(Att2in2Core):
    def __init__(self, opt):
        super(Att2inCore, self).__init__(opt)
        del self.a2c
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

"""
Note this is my attempt to replicate att2all model in self-critical paper.
However, this is not a correct replication actually. Will fix it.
"""
class Att2all2Core(nn.Module):
    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class AdaAttModel(AttModel):
    def __init__(self, opt):
        super(AdaAttModel, self).__init__(opt)
        self.core = AdaAttCore(opt)

# AdaAtt with maxout lstm
class AdaAttMOModel(AttModel):
    def __init__(self, opt):
        super(AdaAttMOModel, self).__init__(opt)
        self.core = AdaAttCore(opt, True)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class Att2all2Model(AttModel):
    def __init__(self, opt):
        super(Att2all2Model, self).__init__(opt)
        self.core = Att2all2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)

class MTopDownModel(AttModel):
    def __init__(self, opt):
        super(MTopDownModel, self).__init__(opt)
        #self.num_layers = 2
        self.num_layers = 2*opt.mtopdown_num
        self.core = MTopDownCore(opt)

class GTssgModel(AttModel):
    def __init__(self, opt):
        super(GTssgModel, self).__init__(opt)
        self.num_layers = 2
        self.core = GTssgCore(opt)

class StackAttModel(AttModel):
    def __init__(self, opt):
        super(StackAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = StackAttCore(opt)

class DenseAttModel(AttModel):
    def __init__(self, opt):
        super(DenseAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = DenseAttCore(opt)

class Att2inModel(AttModel):
    def __init__(self, opt):
        super(Att2inModel, self).__init__(opt)
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)


class Log_Rbm(nn.Module):
    def __init__(self, D_in, R, D_out):
        super(Log_Rbm, self).__init__()
        self.D_in = D_in
        self.R = R
        self.D_out = D_out
        self.w = nn.Linear(D_in, R, bias=False)
        u_init = np.random.rand(self.R, D_out) / 1000
        u_init = np.float32(u_init)
        self.u = torch.from_numpy(u_init).cuda().requires_grad_()

    def forward(self, x):
        v = self.w(x) #x Batch*D_in, v Batch*R

        v = v.unsqueeze(2).expand(-1,-1,self.D_out) #v: Batch*R*D_out
        u = self.u.unsqueeze(0).expand(v.size(0),-1,-1) #u: Batch*R*D_out
        v = v+u
        v = torch.exp(v)
        v = v+1
        y = torch.log(v)
        y = torch.sum(v,1)
        return y
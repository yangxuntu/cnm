from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .BasicModel import BasicModel
from functools import reduce


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        masks_list = list(att_masks.data.long().sum(1))
    else:
        masks_list = [att_feats.size(1)] * att_feats.size(0)

    packed = pack_padded_sequence(att_feats, masks_list, batch_first=True)
    return pad_packed_sequence(PackedSequence(module(packed[0]), packed[1]), batch_first=True)[0]


# noinspection PyArgumentList
class VRConFrame(BasicModel):
    def __init__(self, opt):
        super(VRConFrame, self).__init__()
        self.vocab_size = opt.vocab_size
        self.word_embed_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.word_embed_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.ind_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.ind_proj = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = Variable(fc_feats.data.new(batch_size, seq.size(1) - 1, self.vocab_size + 1).zero_())

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        ind_feats = pack_wrapper(self.ind_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_ind_feats = self.ind_proj(ind_feats)
        pre_feats = fc_feats

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.word_embed(it)

            output, state, pre_feats = self.core(xt, fc_feats, ind_feats, p_ind_feats, pre_feats, state, att_masks)
            output = F.log_softmax(self.logit(output), dim=1)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, tmp_pre_feats, tmp_fc_feats, tmp_ind_feats, tmp_p_ind_feats,
                            tmp_att_masks, state):
        # 'it' is Variable containing a word index
        xt = self.word_embed(it)

        output, state, tmp_pre_feats = self.core(xt, tmp_fc_feats, tmp_ind_feats, tmp_p_ind_feats,
                                                 tmp_pre_feats, state, tmp_att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state, tmp_pre_feats

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        ind_feats = pack_wrapper(self.ind_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_ind_feats = self.ind_proj(ind_feats)
        pre_feats = fc_feats

        assert beam_size <= self.vocab_size + 1, 'otherwise this corner case causes a few headaches down the road.'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seq_logprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_ind_feats = ind_feats[k:k + 1].expand(*((beam_size,) + ind_feats.size()[1:])).contiguous()
            tmp_p_ind_feats = p_ind_feats[k:k + 1].expand(*((beam_size,) + p_ind_feats.size()[1:])).contiguous()
            tmp_pre_feats = pre_feats[k:k + 1].expand(*((beam_size,) + pre_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k + 1].expand(
                *((beam_size,) + att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.word_embed(Variable(it, requires_grad=False))

                output, state, tmp_pre_feats = self.core(xt, tmp_fc_feats, tmp_ind_feats, tmp_p_ind_feats,
                                                         tmp_pre_feats, state, tmp_att_masks)
                logprobs = F.log_softmax(self.logit(output), dim=1)

            # In beam_search inside, call the get_logprobs_state using input args
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_pre_feats, tmp_fc_feats, tmp_ind_feats,
                                                  tmp_p_ind_feats,
                                                   tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seq_logprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return Variable(seq.transpose(0, 1)), Variable(seq_logprobs.transpose(0, 1))

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        ind_feats = pack_wrapper(self.ind_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_ind_feats = self.ind_proj(ind_feats)
        pre_feats = fc_feats

        # seq = []
        # seq_logprobs = []
        seq = Variable(fc_feats.data.new(batch_size, self.seq_length).long().zero_())
        seq_logprobs = Variable(fc_feats.data.new(batch_size, self.seq_length).zero_())
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sample_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                # gather the logprobs at sampled positions
                sample_logprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                it = it.view(-1).long()  # and flatten indices for downstream processing

            xt = self.word_embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                # seq.append(it) #seq[t] the input of t+2 time step

                # seq_logprobs.append(sample_logprobs.view(-1))
                seq_logprobs[:, t - 1] = sample_logprobs.view(-1)

            output, state, pre_feats = self.core(xt, fc_feats, ind_feats,
                                                 p_ind_feats, pre_feats, state, att_masks)

            if decoding_constraint and t > 0:
                tmp = output.data.new(output.size(0), self.vocab_size + 1).zero_()
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = F.log_softmax(self.logit(output) + Variable(tmp), dim=1)
            else:
                logprobs = F.log_softmax(self.logit(output), dim=1)

        return seq, seq_logprobs


class VRConCore(nn.Module):
    def __init__(self, opt):
        super(VRConCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v

        self.ind_attention = IndividualAttention(opt)
        self.rlt_attention = RelationAttention(opt)
        self.all_attention = AllAttention(opt)

    def forward(self, xt, fc_feats, ind_feats, p_ind_feats, pre_feats, state, att_masks=None):
        # The first attention LSTM
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # Compute each part of attention features
        ind = self.ind_attention(h_att, ind_feats, p_ind_feats, att_masks)
        rlt = self.rlt_attention(h_att, ind_feats, pre_feats, att_masks)

        # and compute final features
        att = self.all_attention(h_att, torch.stack([ind, rlt], dim=1))

        # The second language LSTM
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state, att


class ComputeRelation(nn.Module):
    def __init__(self, opt):
        super(ComputeRelation, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.sbj_proj = nn.Linear(self.rnn_size, self.att_hid_size)
        self.obj_proj = nn.Linear(self.rnn_size, self.att_hid_size)

    def forward(self, sbj_feats, obj_feats, att_masks=None):
        """
        :param sbj_feats: batch_size x num_boxes x rnn_size
        :param obj_feats: batch_size x num_boxes x rnn_size
        :param att_masks: batch_size x num_boxes
        :return: batch_size x num_boxes x rnn_size
        """
        # sbj_feats = self.sbj_proj(sbj_feats)
        # obj_feats = self.obj_proj(obj_feats)

        return obj_feats - sbj_feats


class IndividualAttention(nn.Module):
    def __init__(self, opt):
        super(IndividualAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class RelationAttention(nn.Module):
    def __init__(self, opt):
        super(RelationAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.rlt2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.alpha_net = nn.Linear(opt.att_hid_size, 1)

        self.compute_relation = ComputeRelation(opt)

    def forward(self, h, ind_feat, pre_feats, att_masks=None):
        att_size = ind_feat.numel() // ind_feat.size(0) // self.rnn_size

        rlt_feats = self.compute_relation(ind_feat, pre_feats.unsqueeze(1).repeat(1, att_size, 1), att_masks)

        att_v = pack_wrapper(self.rlt2att, rlt_feats, att_masks)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att_v)  # batch * att_size * att_hid_size
        dot = att_v + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1

        att_res = torch.bmm(weight.unsqueeze(1), rlt_feats).squeeze(1)  # batch * att_feat_size

        return att_res


class AllAttention(nn.Module):
    def __init__(self, opt):
        super(AllAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.f2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, feats):
        # The p_att_feats here is already projected
        att_size = feats.numel() // feats.size(0) // self.rnn_size

        att = self.f2att(feats).view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size

        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        att_feats_ = feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class VRConModel(VRConFrame):
    def __init__(self, opt):
        super(VRConModel, self).__init__(opt)
        self.num_layers = 2
        self.core = VRConCore(opt)

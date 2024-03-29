from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import nltk

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out,imgToEval

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    word_num = np.zeros([loader.vocab_size,])
    tag_acc = np.zeros([4,])
    tag_num = np.zeros([4,])
    tag_acc_rate = np.zeros([5,])
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            tmp = [data['labels'], data['masks'], data['mods']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            labels, masks, mods = tmp

            tmp = [data['att_feats'], data['att_masks'], data['attr_feats'], data['attr_masks'], data['rela_feats'],
                   data['rela_masks']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            att_feats, att_masks, attr_feats, attr_masks, rela_feats, rela_masks = tmp

            rs_data = {}
            rs_data['att_feats'] = att_feats
            rs_data['att_masks'] = att_masks
            rs_data['attr_feats'] = attr_feats
            rs_data['attr_masks'] = attr_masks
            rs_data['rela_feats'] = rela_feats
            rs_data['rela_masks'] = rela_masks
            rs_data['cont_ver'] = 0

            # with torch.no_grad():
            #     loss = crit(model(rs_data, labels), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        tmp = [data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['attr_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['rela_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]
               ]
        tmp = [torch.from_numpy(_).cuda() for _ in tmp]
        att_feats, att_masks, attr_feats, attr_masks, rela_feats, rela_masks,\
            labels, masks = tmp
        rs_data = {}
        rs_data['att_feats'] = att_feats
        rs_data['att_masks'] = att_masks
        rs_data['attr_feats'] = attr_feats
        rs_data['attr_masks'] = attr_masks
        rs_data['rela_feats'] = rela_feats
        rs_data['rela_masks'] = rela_masks
        rs_data['cont_ver'] = 0

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_temp, _, tag_temp= model(rs_data, opt=eval_kwargs, mode='sample')
            seq = seq_temp.data
            tag = tag_temp.data
            for sent in seq:
                for token in sent:
                    if token != 0:
                        word_num[token] += 1

        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=0)
        sents_tag = utils.decode_sequence_tags(loader.get_vocab(), seq, use_ssg=0, tags=tag)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=0)

        lan_mod = ['CC', 'EX', 'FW', 'LS', 'MD', 'PRP', 'PRP$', 'SYM', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']
        obj_mod = ['DT', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS']
        rela_mod = ['IN', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        attr_mod = ['CD', 'JJ', 'JJS', 'JJR']
        module_label = {}
        for ttt in lan_mod:
            module_label[ttt] = 0
        for ttt in obj_mod:
            module_label[ttt] = 1
        for ttt in rela_mod:
            module_label[ttt] = 2
        for ttt in attr_mod:
            module_label[ttt] = 3

        i=0
        for s in sents:
            tags_tmp = nltk.pos_tag(nltk. word_tokenize(s))
            j=0
            for tt in tags_tmp:
                tag_label = int(module_label[tt[1]])
                tag_num[tag_label]+=1
                if tag[i,j]==tag_label:
                    tag_acc[tag_label]+=1
                    j+=1
            i+=1


        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'caption_tag':sents_tag[k],\
                     'image_path':data['infos'][k]['file_path']}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    for t in range(4):
        tag_acc_rate[t]=tag_acc[t]/(tag_num[t]+0.0)
        tag_acc_rate[4]=np.sum(tag_acc)/(np.sum(tag_num)+0.0)
    print(tag_acc_rate)
    print(tag_num/(np.sum(tag_num)+0.0))

    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap'+ eval_kwargs['model'][-10:-4] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()
            for i in range(len(predictions)):
                text_file = open('gen_cap/cap_tag' + eval_kwargs['model'][-10:-4] + '.txt', "aw")
                text_file.write('image %s, %s\n' % (predictions[i]['image_id'], predictions[i]['caption_tag']))
                text_file.close()

    # Switch back to training mode
    model.train()
    np.save('word_num/wn'+ eval_kwargs['model'][-10:-4],word_num)
    return loss_sum/loss_evals, predictions, lang_stats, scores_each

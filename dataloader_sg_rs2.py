from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from models.ass_fun import *

import torch
import torch.utils.data as data

import multiprocessing

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_rela_dict_size(self):
        return self.rela_dict_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        #input_ssg_dir: the path to scene graph info
        self.input_ssg_dir = self.opt.input_ssg_dir
        #input_att_dir: the path to attribute features, which is different from other versions of dataloader
        self.input_att_dir = self.opt.input_att_dir

        sg_data_info = np.load(self.input_ssg_dir)
        #keys = sg_data_info.keys()[0]
        #sg_data_info = sg_data_info[keys][()]
        sg_data_info = sg_data_info[()]
        self.rela_dict = sg_data_info['rela_dict']
        self.attr_dict = sg_data_info['attr_dict']
        self.obj_dict = sg_data_info['obj_dict']

        self.attr_size = len(self.attr_dict)
        self.obj_size = len(self.obj_dict)
        self.rela_size = len(self.rela_dict)

        self.sg_data = sg_data_info['sg_data_refine']

        self.num_images = len(self.sg_data)
        self.train_split_num = np.int32(self.num_images*0.8)

        print('read %d image features' %(self.num_images))

        self.split_ix = {'train': [], 'test': [] }
        for ix in range(self.num_images):
            if ix <= self.train_split_num:
                self.split_ix['train'].append(ix)
            else:
                self.split_ix['test'].append(ix)


        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)


    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        attr_feat_batch = []
        attr_label_batch = []
        wrapped = False

        for i in range(batch_size):
            tmp_rs_data, ix, tmp_wrapped = self._prefetch_process[split].get()
            attr_feat_batch.append(tmp_rs_data['attr_feat'])
            attr_label_batch.append(tmp_rs_data['attr_label'])
            if tmp_wrapped:
                wrapped = True
        
            # record associated info as well

        data = {}
        max_attr_len = max([_.shape[0] for _ in attr_feat_batch])

        #attribute features and masks
        data['attr_feats'] = np.zeros([len(attr_feat_batch), max_attr_len, attr_feat_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(attr_feat_batch)):
            data['attr_feats'][i, :attr_feat_batch[i].shape[0]] = attr_feat_batch[i]
        data['attr_masks'] = np.zeros(data['attr_feats'].shape[:2], dtype='float32')
        for i in range(len(attr_feat_batch)):
            data['attr_masks'][i, :attr_feat_batch[i].shape[0]] = 1

        #attribute labels
        data['attr_labels'] = np.zeros([batch_size, max_attr_len, self.attr_size], dtype='int32')
        for i in range(len(attr_label_batch)):
            data['attr_labels'][i, :attr_label_batch[i].shape[0], :] = attr_label_batch[i]

        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        if os.path.isfile(os.path.join(self.input_att_dir, str(self.sg_data[ix]['image_id']) + '.npz')):
            att_feat = np.load(os.path.join(self.input_att_dir, str(self.sg_data[ix]['image_id']) + '.npz'))['feat']
            att_feat = att_feat.reshape(att_feat.shape[0],-1)
            sg_use = self.sg_data[ix]
            rs_data = {}

            attr_id = sg_use['obj_attr_id']
            N_attr = len(attr_id)
            index = []
            for i in range(N_attr):
                attr_temp = attr_id[i]
                if len(attr_temp) == 0:
                    continue
                index_temp = 0
                for j in range(len(attr_temp)):
                    if attr_temp[j]>=0:
                        index_temp = 1
                if index_temp == 1:
                    index.append(i)

            rs_data['attr_feat'] = att_feat[index]

            N_attr = len(index)
            label = np.zeros([N_attr, self.attr_size])
            for i in range(N_attr):
                attr_temp = attr_id[index[i]]
                for j in range(len(attr_temp)):
                    if attr_temp[j] != -1:
                        label[i,np.int32(attr_temp[j])] = 1
            rs_data['attr_label'] = label

        return (rs_data, ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[1] == ix, "ix not equal"

        return tmp + [wrapped]
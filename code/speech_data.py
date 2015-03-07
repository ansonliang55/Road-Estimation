# Code by Navdeep Jaitly, 2013.
# Email: ndjaitly@gmail.com

# Provides access to data. Do not modify.

import scipy.io, sys
from numpy.random import permutation
from numpy import arange, zeros, eye, tile, concatenate

class speech_data(object):
    def __init__(self, file_name, num_frames_per_pt):
        self._num_frames_per_pt = num_frames_per_pt
        self.load(file_name)

    def load(self, file_name):
        d = scipy.io.loadmat(file_name)
        self._data = d['data']
        try:
            self._targets = d['targets'].reshape(-1)
        except KeyError:
            sys.stderr.write("targets not specified in file: " + file_name)

        self._utt_indices = d['utt_indices']
        self.label_dim = d['label_dim']
        self._num_utterances = self._utt_indices.shape[0]

        self.frame_dim = self._data.shape[0]
        self.data_dim = self.frame_dim * self._num_frames_per_pt

        self.num_pts = 0
        lst_indices = [] 
        for i in range(self._num_utterances):
            s, e = self._utt_indices[i,:]
            indices = arange(s, e-self._num_frames_per_pt+1)
            lst_indices.append(indices.copy())
            self.num_pts += indices.size

        self._indices = zeros(self.num_pts, dtype='int32')
        num_pts_so_far = 0
        for indices in lst_indices:
            self._indices[num_pts_so_far:(num_pts_so_far+indices.size)] = \
                                                                    indices
            num_pts_so_far += indices.size

        assert(num_pts_so_far == self.num_pts)
        sys.stderr.write("Loaded %d points\n"%num_pts_so_far)
        self._data_mean = self._data.mean(axis=1).reshape(-1,1)
        self._data_std = self._data.std(axis=1).reshape(-1,1)

    def get_num_utterances(self):
        return self._num_utterances

    def get_utterance_data(self, utterance_num, get_labels=True):
        s_f, e_f = self._utt_indices[utterance_num,:]
        data = self._data[:, s_f:e_f]
        left_context = int(self._num_frames_per_pt/2)
        right_context = self._num_frames_per_pt - left_context
        left = tile(data[:,0].reshape(-1,1), (1, left_context))
        right = tile(data[:,-1].reshape(-1,1), (1, right_context))

        data = concatenate((left, data, right), axis=1)
        data_stacked = zeros((self.data_dim, e_f-s_f))
        for i in range(self._num_frames_per_pt):
            s, e = i*self.frame_dim, (i+1)*self.frame_dim
            data_stacked[s:e,:] = data[:,i:(e_f-s_f+i)]

        if get_labels:
            eye_mat = eye(self.label_dim)
            label_mat = eye_mat[:, self._targets[s_f:e_f]]
            return data_stacked, label_mat
        else:
            return data_stacked

    def normalize_data(self):
        self._data = (self._data - self._data_mean)/self._data_std

    def get_data_dim(self):
        return self.data_dim 

    def get_target_dim(self):
        return self.label_dim 

    def get_iterator(self, batch_size):
        indices = self._indices[permutation(self.num_pts)]
        eye_mat = eye(self.label_dim)
        frame_offset = int(self._num_frames_per_pt/2)

        for start_index in range(0, self.num_pts, batch_size):
            data_start_indices = indices[start_index:(start_index+batch_size)]
            data_indices = data_start_indices.reshape(1,-1) + \
                              arange(self._num_frames_per_pt).reshape(-1,1)
            data_indices = data_indices.reshape(-1, order='F')
            data = self._data[:, data_indices].reshape((self.data_dim, -1), 
                                                                  order='F')
            labels = self._targets[data_start_indices + frame_offset]
            label_mat = eye_mat[:, labels]

            yield data, label_mat


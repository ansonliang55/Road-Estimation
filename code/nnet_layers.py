# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

from numpy import sqrt, isnan, Inf, dot, zeros, exp, log, sum
from numpy.random import randn
import numpy as np
SIGMOID_LAYER = 0
SOFTMAX_LAYER = 1

class layer_definition(object):
    def __init__(self, name, layer_type, input_dim, num_units, wt_sigma):
        self.name, self.layer_type, self.input_dim, self.num_units, \
          self.wt_sigma  =  name, layer_type, input_dim, num_units, wt_sigma
        #print ("input_dim %i")%input_dim
        #print ("num_units %i")%num_units
        #print ("name %s")%name


class layer(object):
    def __init__(self, name):
        self.name = name

    @property
    def shape(self):
        return self._wts.shape

    @property
    def num_hid(self):
        return self._wts.shape[1]

    @property
    def num_dims(self):
        return self._wts.shape[0]

    def create_params(self, layer_def):
        input_dim, output_dim, wt_sigma = layer_def.input_dim, \
                        layer_def.num_units, layer_def.wt_sigma

        self._wts = randn(input_dim, output_dim) * wt_sigma
        self._b = zeros((output_dim, 1))

        self._wts_grad = zeros(self._wts.shape)
        self._wts_inc = zeros(self._wts.shape)

        self._b_grad = zeros(self._b.shape)
        self._b_inc = zeros(self._b.shape)

        self.__num_params = input_dim*output_dim

    def add_params_to_dict(self, params_dict):
        params_dict[self.name + "_wts"] = self._wts.copy()
        params_dict[self.name + "_b"] = self._b.copy()

    def copy_params_from_dict(self, params_dict):
        self._wts = params_dict[self.name + "_wts"].copy()
        self._b = params_dict[self.name + "_b"].copy()
        self.__num_params = self._wts.shape[0] * self._wts.shape[1]
        self._wts_inc = zeros(self._wts.shape)
        self._b_inc = zeros(self._b.shape)

    def apply_gradients(self, momentum, eps, l2=.0001):
        """ COMPLETE IMPLEMENTATION 
        """
        self._wts_inc = self._wts_inc * momentum - eps * (self._wts_grad + l2 * self._wts)
        self._b_inc = self._b_inc * momentum - eps * (self._b_grad + l2 * self._b)
        self._wts = self._wts + self._wts_inc
        self._b = self._b + self._b_inc



    def back_prop(self, act_grad, data):
        ''' 
        COMPLETE
        Feel free to add member variables.
        back prop activation grad, and compute gradients. 
        '''
        # matrix multiplication
        self._wts_grad = np.matrix(data) * np.matrix(act_grad).T
        self._b_grad = np.matrix(act_grad.sum(axis=1)).T
        input_grad = np.array(np.matrix(self._wts) * np.matrix(act_grad))
        return input_grad
 

class sigmoid_layer(layer):
    pass 

    def fwd_prop(self, data):
        """ COMPLETE IMPLEMENTATION
        """
        if len(self._wts) != len(data):
            raise Exception, "input data and wts dimension mismatch"
        dat = np.lib.pad(data.T,((0,0),(0,1)), 'constant', constant_values=1)
        wts = np.append(self._wts.T,self._b,1)
        activation = np.matrix(wts) * np.matrix(dat).T
        probs = 1.0/(1.0 + np.exp(-1*activation.clip(min=-100)))
        probs = np.array(probs)
        return probs

    def compute_act_grad_from_output_grad(self, output, output_grad):
        """ COMPLETE IMPLEMENTATION 
        """
        # element wise multiplication
        act_grad = output_grad * output * (1-output)
        return act_grad

 
class softmax_layer(layer):
    pass

    def fwd_prop(self, data):
        """ COMPLETE IMPLEMENTATION
        """
        if len(self._wts) != len(data):
            raise Exception, "input data and wts dimension mismatch"
        dat = np.lib.pad(data.T,((0,0),(0,1)), 'constant', constant_values=1)
        wts = np.append(self._wts.T,self._b,1)
        activation = np.exp(np.matrix(wts) * np.matrix(dat).T)  
        probs = activation/(activation.sum(axis=0))
        probs = np.array(probs) + 1e-32 
        return probs

    def compute_act_gradients_from_targets(self, targets, output):
        """ COMPLETE IMPLEMENTATION
        """
        act_grad = output - targets
        return act_grad

    @staticmethod 
    def compute_accuraccy(probs, label_mat):
        num_correct = sum(probs.argmax(axis=0) == label_mat.argmax(axis=0))
        log_probs = sum(log(probs) * label_mat)   
        return num_correct, log_probs

def create_empty_nnet_layer(name, layer_type):
    if layer_type == SIGMOID_LAYER:
        layer = sigmoid_layer(name)
    elif layer_type == SOFTMAX_LAYER:
        layer = softmax_layer(name)
    else:
        raise Exception, "Unknown layer type"
    return layer

def create_nnet_layer(layer_def):
    layer = create_empty_nnet_layer(layer_def.name, layer_def.layer_type)
    layer.create_params(layer_def)
    return layer

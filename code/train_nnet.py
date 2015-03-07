# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

import numpy
numpy.random.seed(42) # This seed is meaningful. :).

import nnet_train, nnet_layers, sys, os, argparse, speech_data
import logging
import pylab

def plot_lr(train_history,valid_history):
    pylab.ioff()
    pylab.figure()
    pylab.plot(train_history,'b')
    pylab.plot(valid_history,'r')
    ax = pylab.gca()
    pylab.xlabel('# of Epochs')
    pylab.ylabel('Error rate (%)')
    ax.legend(['Train error','Validation error'],'upper right')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest='seed', type=int,
                    default=-1, help='Seed for random number generators')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='Maximum number of epcohs to train for')
parser.add_argument('train_db_path', help='Path to training database')
parser.add_argument('dev_db_path', help='Path to validation database')
parser.add_argument('output_fldr', type=str, help='output folder')

arguments = parser.parse_args()
if not os.path.exists(arguments.output_fldr):
    os.makedirs(arguments.output_fldr)
logPath = os.path.join(arguments.output_fldr, "log.txt")
if os.path.exists(logPath): os.remove(logPath)

# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))

# Number of contiguous frames to predict phoneme labels. Leave at 15
num_frames_per_pt = 15


# DECIDE ON MEANINGFUL VALUES FOR THESE VARIABLES
batch_size, eps, momentum, l2 = 200, 0.3, 0.002, 0.0001


train_src = speech_data.speech_data(arguments.train_db_path, num_frames_per_pt)
validation_src = speech_data.speech_data(arguments.dev_db_path, 
                                         num_frames_per_pt)

# 1-layer
layer1_def = nnet_layers.layer_definition("Layer1", nnet_layers.SIGMOID_LAYER,
                                         train_src.get_data_dim(), 100, 0.01)
layer2_def = nnet_layers.layer_definition("Layer2", nnet_layers.SOFTMAX_LAYER,
                                         100, train_src.get_target_dim(), 0.01)

# 3-layer
#layer1_def = nnet_layers.layer_definition("Layer1", nnet_layers.SIGMOID_LAYER,
#                                         train_src.get_data_dim(), 300, 0.01)
#layer2_def = nnet_layers.layer_definition("Layer2", nnet_layers.SIGMOID_LAYER,
#                                         300, 300, 0.01)
#layer3_def = nnet_layers.layer_definition("Layer3", nnet_layers.SOFTMAX_LAYER,
#                                         300, train_src.get_target_dim(), 0.01)
## FILL IN SOME PRE-PROCESS  INSTRUCTIONS HERE [1. normalize?]
#train_src.normalize_data()
#validation_src.normalize_data()

# Definition of multi layer neural network
lst_def = [layer1_def, layer2_def] 
#lst_def = [layer1_def, layer2_def, layer3_def] 

nn_train = nnet_train.nn()
nn_train.create_nnet_from_def(lst_def)

# where the model gets written to.
param_file = os.path.join(arguments.output_fldr, "model.mat")

train_history = []
valid_history = []
prev_err = 100
lowest_err = 100
for i in range(arguments.max_epochs):
    err_train = nn_train.train_for_one_epoch(train_src, eps, momentum, l2, batch_size)
    
    
    # CODE FOR CONTROLLING OVERFITTING...GOES SOMEWHERE AROUND HERE..
    acc_valid, logp = nn_train.test(validation_src)
    err_valid = 100-acc_valid
    train_history.append(err_train)
    valid_history.append(err_valid)

    if lowest_err > err_valid:
    		lowest_err = err_valid
    		nn_train.save(param_file)

    err_diff = prev_err - err_valid 
    if abs(err_diff) < 0.05 and i > 20:
    		break
    prev_err = err_valid

print ('Validation Error: %2.2f')%lowest_err
# Plot the training/validation accuracy of each run.
plot_lr(train_history,valid_history)
pylab.show()



# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

# Creates prediction file. No need to modify.
import numpy, scipy.io, nnet_train, sys, os, argparse, speech_data
from numpy import zeros

parser = argparse.ArgumentParser()
parser.add_argument('--num_frames_per_pt', type = int, 
                    default = 15, help='# of frames per pt')
parser.add_argument('db_path', help='Path to validation database')
parser.add_argument('model_file', type=str, 
                     help='file with neural network parameters')
parser.add_argument('predictions_file', type=str, help='output folder')

arguments = parser.parse_args()
data_src = speech_data.speech_data(arguments.db_path, 
                                   arguments.num_frames_per_pt)
data_src.normalize_data()
 
nnet = nnet_train.nn()
nnet.load(arguments.model_file)

pred_lst, num_output_frames = nnet.create_predictions(data_src)

predictions_mat = zeros((pred_lst[-1].shape[0], num_output_frames))
utt_indices = zeros((len(pred_lst),2), dtype='int32')

num_so_far = 0

for index, predictions in enumerate(pred_lst):
    predictions_mat[:, num_so_far:(num_so_far+predictions.shape[1])] = \
                                                              predictions
    utt_indices[index, 0] = num_so_far
    num_so_far += predictions.shape[1]
    utt_indices[index, 1] = num_so_far

scipy.io.savemat(arguments.predictions_file, {'predictions': predictions_mat,
                                              'utt_indices': utt_indices})

import sys
import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
from multiprocessing import Process
import pickle
import argparse

def pretext_train_test(root_folder, k, N, epoch_sec):
    # get all data indices
    all_index = sorted([int(path[6:12]) - 200000 for path in os.listdir(root_folder + 'shhs1')])
    
    # split into 
    pretext_index = np.random.choice(all_index, int(len(all_index) * 0.98), replace=False)
    train_index = np.random.choice(list(set(all_index) - set(pretext_index)), int(len(all_index) * 0.01), replace=False)
    test_index = list(set(all_index) - set(pretext_index) - set(train_index))

    print ('start pretext process!')
    sample_process(root_folder, k, N, epoch_sec, 'pretext', pretext_index)
    print ()
    
    print ('start train process!')
    sample_process(root_folder, k, N, epoch_sec, 'train', train_index)
    print ()
    
    print ('start test process!')
    sample_process(root_folder, k, N, epoch_sec, 'test', test_index)
    print ()


def sample_process(root_folder, k, N, epoch_sec, train_test_val, index):
    # process each EEG sample: further split the samples into window sizes and using multiprocess
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))

            # load the signal "X" part
            data = mne.io.read_raw_edf(root_folder + 'shhs1/' + 'shhs1-' + str(200000 + j) + '.edf')
            X = data.get_data()
            
            # some EEG signals have missing channels, we treat them separately
            if X.shape[0] == 16:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
            elif X.shape[0] == 15:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
            X = X[[2,7], :]

            # load the label "Y" part
            with open(root_folder + 'label/' + 'shhs1-' + str(200000 + j) + '-profusion.xml', 'r') as infile:
                text = infile.read()
                root = ET.fromstring(text)
                y = [i.text for i in root.find('SleepStages').findall('SleepStage')]

            # slice the EEG signals into non-overlapping windows, window size = sampling rate per second * second time = 125 * windowsize
            for slice_index in range(X.shape[1] // (125 * epoch_sec)):
                path = root_folder + 'processed/{}/'.format(train_test_val) + 'shhs1-' + str(200000 + j) + '-' + str(slice_index) + '.pkl'
                pickle.dump({'X': X[:, slice_index * 125 * epoch_sec: (slice_index+1) * 125 * epoch_sec], \
                    'y': int(y[slice_index])}, open(path, 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=8, help="How many processes to use")
    args = parser.parse_args()

    if not os.path.exists('./SHHS_data/processed/'):
        os.makedirs('./SHHS_data/processed/pretext')
        os.makedirs('./SHHS_data/processed/train')
        os.makedirs('./SHHS_data/processed/test')

    root_folder = './SHHS_data/SHHS/'

    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(target=pretext_train_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()


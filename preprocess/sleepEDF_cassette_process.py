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
    all_index = np.unique([path[:6] for path in os.listdir(root_folder)])
    
    pretext_index = np.random.choice(all_index, int(len(all_index) * 0.9), replace=False)
    train_index = np.random.choice(list(set(all_index) - set(pretext_index)), int(len(all_index) * 0.05), replace=False)
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
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))

            # load signal "X" part
            data = mne.io.read_raw_edf(root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('PSG' in x), os.listdir(root_folder)))[0])
            X = data.get_data()[:2, :]
            
            # load label "Y" part
            ann = mne.read_annotations(root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('Hypnogram' in x), os.listdir(root_folder)))[0])
            labels = []
            for dur, des in zip(ann.duration, ann.description):
                for i in range(int(dur) // 30):
                    labels.append(des[-1])

            # slice the EEG signals into non-overlapping windows, window size = sampling rate per second * second time = 100 * windowsize
            for slice_index in range(X.shape[1] // (100 * epoch_sec)):
                # ingore the no labels
                if labels[slice_index] == '?':
                    continue
                path = './SLEEP_data/cassette_processed/{}/'.format(train_test_val) + 'cassette-' + j + '-' + str(slice_index) + '.pkl'
                pickle.dump({'X': X[:, slice_index * 100 * epoch_sec: (slice_index+1) * 100 * epoch_sec], \
                    'y': labels[slice_index]}, open(path, 'wb'))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=8, help="How many processes to use")
    args = parser.parse_args()
    
    if not os.path.exists('./SLEEP_data/cassette_processed'):
        os.makedirs('./SLEEP_data/cassette_processed/pretext')
        os.makedirs('./SLEEP_data/cassette_processed/train')
        os.makedirs('./SLEEP_data/cassette_processed/test')

    root_folder = './SLEEP_data/sleep-edf-database-expanded-1.0.0/sleep-cassette'

    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(target=pretext_train_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()


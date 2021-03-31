import sys
import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
from multiprocessing import Process
import pickle

# def get_channel_data(path):
#     outpath = '../data/processed/signal/' + \
#             path.split('/')[-1].split('.')[0] + '_signal.npy'
#     if os.path.exists(outpath): return

#     data = mne.io.read_raw_edf(path)
#     # get_stage(stage_path)
#     raw_data = data.get_data()
#     np.save(open(outpath, 'wb'), raw_data)

#     """ visualize 14 channels
#     plt.figure(figsize=(20,10))
#     for i in range(raw_data.shape[0]):
#         plt.subplot(raw_data.shape[0], 1, i + 1)
#         plt.plot(raw_data[i, :100000])
#     plt.show()
#     """

# def channel_process(signal_path, k, l):
#     for i, j in enumerate(os.listdir(signal_path)):
#         if i % l == k:
#             print (i, j, 'finished')
#             get_channel_data(signal_path + j)

# def get_stage(path):
#     with open(path, 'r') as infile:
#         text = infile.read()
#         root = ET.fromstring(text)
#         stages = [i.text for i in root.find('SleepStages').findall('SleepStage')]

#     outpath = '../data/processed/label/' + \
#             path.split('/')[-1].split('.')[0][:-10] + '_stages.npy'
#     np.save(open(outpath, 'wb'), stages)

def train_val_test(root_folder, k, N, epoch_sec):
    all_index = sorted([int(path[6:12]) - 200000 for path in os.listdir(root_folder + 'shhs1')])
    
    train_index = np.random.choice(all_index, int(len(all_index) * 0.98), replace=False)
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.01), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    sample_package(root_folder, k, N, epoch_sec, 'pretext', train_index)
    sample_package(root_folder, k, N, epoch_sec, 'train', test_index)
    sample_package(root_folder, k, N, epoch_sec, 'test', val_index)


def sample_package(root_folder, k, N, epoch_sec, train_test_val, index):
    for i, j in enumerate(index):
        if i % N == k:
            print ('train', i, j, 'finished')

            # X load
            data = mne.io.read_raw_edf(root_folder + 'shhs1/' + 'shhs1-' + str(200000 + j) + '.edf')
            X = data.get_data()
            if X.shape[0] == 16:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
            elif X.shape[0] == 15:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
            X = X[[2,7], :]

            # y load
            with open(root_folder + 'label/' + 'shhs1-' + str(200000 + j) + '-profusion.xml', 'r') as infile:
                text = infile.read()
                root = ET.fromstring(text)
                y = [i.text for i in root.find('SleepStages').findall('SleepStage')]

            for slice_index in range(X.shape[1] // (125 * epoch_sec)):
                path = root_folder + 'processed/{}/'.format(train_test_val) + 'shhs1-' + str(200000 + j) + '-' + str(slice_index) + '.pkl'
                pickle.dump({'X': X[:, slice_index * 125 * epoch_sec: (slice_index+1) * 125 * epoch_sec], \
                    'y': int(y[slice_index])}, open(path, 'wb'))



if __name__ == '__main__':
    if not os.path.exists('./SHHS_data/processed/'):
        os.makedirs('./SHHS_data/processed/pretext')
        os.makedirs('./SHHS_data/processed/train')
        os.makedirs('./SHHS_data/processed/test')

    root_folder = './SHHS_data/SHHS/'

    N, epoch_sec = 30, 30
    p_list = []
    for k in range(N):
        process = Process(target=train_val_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()


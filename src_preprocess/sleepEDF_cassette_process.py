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
    # np.save(open(outpath, 'wb'), stages)

def train_val_test(root_folder, k, N, epoch_sec):
    all_index = np.unique([path[:6] for path in os.listdir(root_folder)])
    
    train_index = np.random.choice(all_index, int(len(all_index) * 0.9), replace=False)
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.05), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    sample_package(root_folder, k, N, epoch_sec, 'pretext', train_index)
    sample_package(root_folder, k, N, epoch_sec, 'train', test_index)
    sample_package(root_folder, k, N, epoch_sec, 'test', val_index)


def sample_package(root_folder, k, N, epoch_sec, train_test_val, index):
    for i, j in enumerate(index):
        if i % N == k:
            print ('train', i, j, 'finished')

            # X load
            data = mne.io.read_raw_edf(root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('PSG' in x), os.listdir(root_folder)))[0])
            X = data.get_data()[:2, :]
            ann = mne.read_annotations(root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('Hypnogram' in x), os.listdir(root_folder)))[0])
            labels = []
            for dur, des in zip(ann.duration, ann.description):
                for i in range(int(dur) // 30):
                    labels.append(des[-1])

            for slice_index in range(X.shape[1] // (100 * epoch_sec)):
                if labels[slice_index] == '?':
                    continue
                path = './SLEEP_data/cassette_processed/{}/'.format(train_test_val) + 'cassette-' + j + '-' + str(slice_index) + '.pkl'
                pickle.dump({'X': X[:, slice_index * 100 * epoch_sec: (slice_index+1) * 100 * epoch_sec], \
                    'y': labels[slice_index]}, open(path, 'wb'))



if __name__ == '__main__':
    if not os.path.exists('./SLEEP_data/cassette_processed'):
        os.makedirs('./SLEEP_data/cassette_processed/pretext')
        os.makedirs('./SLEEP_data/cassette_processed/train')
        os.makedirs('./SLEEP_data/cassette_processed/test')

    root_folder = './SLEEP_data/sleep-edf-database-expanded-1.0.0/sleep-cassette'

    N, epoch_sec = 8, 30
    p_list = []
    for k in range(N):
        process = Process(target=train_val_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()


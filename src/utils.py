"""
spectral data augmentation
- chaoqi Oct. 29
"""

import torch
import numpy as np
from scipy.signal import spectrogram
import pickle
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram


def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = - bound

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    out_ts[out_ts > bound] = bound
    out_ts[out_ts < -bound] = - bound
        
    return out_ts

class SHHSLoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=True):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = [0, 1, 2, 3, 4]
        self.bandpass1 = (1, 3)
        self.bandpass2 = (30, 60)
        self.n_length = 125 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 125
        self.bound = 0.000125

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x
    
    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass

        return x
    
    def crop(self, x):
        l = np.random.randint(1, 3749)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        # np.random.shuffle(x)
        t = np.random.rand()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:]
        return x
    
    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']
        
        # original y.unique = [0, 1, 2, 3, 5]
        if y == 4:
            y = 3
        elif y > 4:
            y = 4
        y = torch.LongTensor([y])

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            return torch.FloatTensor(X), y

class SLEEPCALoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=True):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x
    
    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass
        return x
    
    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:]
        return x

        return x
    
    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']
        
        # original y.unique = [0, 1, 2, 3, 5]
        if y == 'W':
            y = 0
        elif y == 'R':
            y = 4
        elif y in ['1', '2', '3']:
            y = int(y)
        elif y == '4':
            y = 3
        else:
            y = 0
        
        y = torch.LongTensor([y])

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            return torch.FloatTensor(X), y


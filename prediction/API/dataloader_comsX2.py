
import os
import gzip
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
import torch
import torch.utils.data as data
from torch.utils.data import Dataset

import PIL
import torchvision.transforms as transforms

tf = transforms.ToTensor()

def augmentation(x, n=2.0, forward=True):
    if forward:
        return 1 - ((1 - x**n) ** (1/n))
    else:
        return (1 - (1 - x)**n) ** (1/n)

def get_train_test(start, end, n_train, n_test, date_list):
    TRAIN, TEST = [], []
    
    for t in trange(start, end):
        cur_time = date_list[t]
        
        train, test = [t], []
        
        _t = t - 1
        while _t >= start:
            delta = (cur_time - date_list[_t]).total_seconds()
            if delta > (n_train - 1) * 3600: 
                break
            if delta % 3600 == 0:
                train.append(_t)
            _t = _t - 1
            
        _t = t + 1
        while _t < end:
            delta = (date_list[_t] - cur_time).total_seconds()
            if delta > n_test * 3600:
                break
            if delta % 3600 == 0:
                test.append(_t)
            _t = _t + 1
            
        if len(train) == n_train and len(test) == n_test:
            TRAIN.append(list(reversed(train)))
            TEST.append(test)

    TRAIN, TEST = torch.tensor(TRAIN), torch.tensor(TEST)
    
    return TRAIN, TEST

def load_comsX2(n_train=24, n_test=12):
    root = '/data/COMS_reduceX2'
    TESTINDEX = 153749
    
    with open(os.path.join(root, 'data_list_ir02.pkl'), 'rb') as f:
        data_list = pkl.load(f)
    with open(os.path.join(root, 'date_list_ir02.pkl'), 'rb') as f:
        date_list = pkl.load(f)
        
    #for i in trange(len(data_list)):
    #    for j in range(3):
    #        data_list[i][j] = data_list[i][j][:5] + data_list[i][j][6:]
        
    X_train, y_train = get_train_test(0, TESTINDEX, n_train, n_test, date_list)
    X_test, y_test = get_train_test(TESTINDEX, len(date_list), n_train, n_test, date_list)
    
    return (X_train, y_train), (X_test, y_test), data_list

class COMS(Dataset):
    def __init__(self, index, data_list, n):
        super(COMS, self).__init__()
        self.X, self.y = index
        self.data_list = data_list
        self.mean = 0
        self.std = 1
        self.n = n

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data, labels = [], []
        
        for i in self.X[index]:
            img = tf(PIL.Image.open(self.data_list[i]))
            data.append(img)
        
        for i in self.y[index]:
            img = tf(PIL.Image.open(self.data_list[i]))
            labels.append(img)
            
        data = torch.stack(data)
        labels = torch.stack(labels)
        
        data = data[:,:,:,:748]
        labels = labels[:,:,:,:748]
        # 24 x 1 x 300 x 372
        
        return data, labels
        
def load_data(
        batch_size, val_batch_size,
        data_root, num_workers, aug_n, n_train, n_test):

    train, test, data_list = load_comsX2(n_train, n_test)
    
    print(f'Number of train data:\t{train[0].shape[0]}')
    print(f'Number of test data:\t{test[0].shape[0]}')
    
    train_set = COMS(train, data_list, aug_n)
    test_set = COMS(test, data_list, aug_n)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, None, dataloader_test, 0, 1
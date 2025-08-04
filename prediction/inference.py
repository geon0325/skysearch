import os
import pickle as pkl
import argparse
from tqdm import tqdm, trange

import torch
import torch.utils.data as data
from torch.utils.data import Dataset

import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
tf = transforms.ToTensor()

import imageio
from PIL import Image
import numpy as np

import model

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

def load_coms(n_train=24, n_test=12):
    TESTINDEX = 153749
    
    with open(os.path.join(root, 'data_list_ir02.pkl'), 'rb') as f:
        data_list = pkl.load(f)
    with open(os.path.join(root, 'date_list_ir02.pkl'), 'rb') as f:
        date_list = pkl.load(f)
        
    X_train, y_train = get_train_test(0, TESTINDEX, n_train, n_test, date_list)
    X_test, y_test = get_train_test(TESTINDEX, len(date_list), n_train, n_test, date_list)
    
    return (X_train, y_train), (X_test, y_test), data_list


class COMS(Dataset):
    def __init__(self, index, data_list):
        super(COMS, self).__init__()
        self.X, self.y = index
        self.data_list = data_list
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data, labels = [], []

        for i in self.X[index]:
            img = tf(PIL.Image.open(os.path.join(root, self.data_list[i])))
            data.append(img[:,:600,:748])

        for i in self.y[index]:
            img = tf(PIL.Image.open(os.path.join(root, self.data_list[i])))
            labels.append(img[:,:600,:748])

        data = torch.stack(data)
        labels = torch.stack(labels)

        return data, labels

# SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
parser.add_argument('--data_path', default='/data', type=str, help='Path of the datasets')
parser.add_argument('--ckpt_path', default='ckpt_mse30_gen10_disc60_feat1_orth1_step80000.pth', type=str, help='Path of the checkpoint')
parser.add_argument('--output_path', default='outputs', type=str, help='Path of the outputs')
args = parser.parse_args()

# SETTINGS
root = '/data/COMS_reduceX2'
root = args.data_path
TESTINDEX = 153749

gpu = args.gpu
device = torch.device('cuda:{}'.format(gpu))

checkpoint = f'results/comsX2_trn12_tst12_lr0.0002_mse30_gen10_disc60_feat1.0_orth1.0/checkpoints/80000.pth'
checkpoint = args.ckpt_path
M = model.SimVP(tuple([12, 1, 600, 748]), 64, 256, 4, 8)
M.load_state_dict(torch.load(checkpoint))
M = M.to(device)
M.eval()

n_train = 12
n_test = 12

train, test, data_list = load_coms(n_train, n_test)
print(f'Number of train data:\t{train[0].shape[0]}')
print(f'Number of test data:\t{test[0].shape[0]}')

train_set = COMS(train, data_list)
test_set = COMS(test, data_list)

dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=3, shuffle=False, pin_memory=True, num_workers=8)
dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

os.makedirs(args.output_path, exist_ok=True)

test_pbar = tqdm(dataloader_test)

cnt = 0
for batch_x, batch_y in test_pbar:
    pred_y = M(batch_x.to(device)).detach().cpu()
    ans_y = batch_y.detach().cpu()
    
    for i in range(n_test):
        img = pred_y[0][i][0].detach().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f'{args.output_path}/{cnt}_{i}.png')
    
    cnt += 1

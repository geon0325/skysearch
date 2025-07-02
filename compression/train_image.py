import os
import time
import datetime
import random
import warnings
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange

import PIL
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

import utils
import model

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='GPU device ID to use for model training or inference')
parser.add_argument("--inputdata", default='default', type=str, help='Name of the training dataset')
parser.add_argument("--channel", default='ir', choices=["all", "ir", "sw", "wv"], help='Data channel to use for training')
parser.add_argument("--batch_size", default=128, type=int, help='Number of samples per batch during training')
parser.add_argument("--epochs", default=20, type=int, help='Number of training epochs')
parser.add_argument("--learning_rate", default=1e-6, type=float, help='Learning rate for optimizer during training')
parser.add_argument("--dim", default=256, type=int, help='Dimension of the embedding vectors')
parser.add_argument("--gamma", default=0.5, type=float, help='Hyperparameter controlling the margin for positive and negative pairs in the loss function')
parser.add_argument("--savedir", default='/data/', type=str, help='Directory where the trained model will be saved')
args = parser.parse_args()

########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')
print(args.epochs)

########## Random Seed ##########
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

########## Log Path ##########
config = 'imageencoder_ch_{}_dim_{}_bs_{}_lr_{}_gm_{}_ip_{}'.format(args.channel, args.dim, args.batch_size, args.learning_rate, args.gamma, args.inputdata)
    
print(config)
if os.path.isdir('./logs_image/') is False:
    os.makedirs('./logs_image/')
if os.path.isfile(os.path.join('logs_image', config + '.txt')):
    os.remove(os.path.join('logs_image', config + '.txt'))
            
########## Read Data ##########
chlist = []
if args.channel == "all":
    chlist = range(3)
elif args.channel == "ir":
    chlist = [0]
elif args.channel == "sw":
    chlist = [1]
elif args.channel == "wv":
    chlist = [2]
coms_train = utils.ImageTripletDataset(channel_list=chlist, postfix="{}_{}".format(args.inputdata, "train"))
coms_trainloader = torch.utils.data.DataLoader(
    dataset = coms_train,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers=8
)

########## Prepare Training ##########
net = model.model(dim=args.dim, num_channel=len(chlist), gamma=args.gamma).to(device)
optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-6)
print('Preparing the model done')

########## Train Model ##########
training_step = 0
net.train()
for epoch in range(1, args.epochs+1):
    print('\nEpoch:\t', epoch)
    
    start_train_time = time.time()
    net.train()
    epoch_loss = []
    for step, (anchor, pos, neg) in tqdm(enumerate(coms_trainloader)):
        training_step += 1
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        loss = net(anchor, pos, neg)
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        
    train_time = time.time() - start_train_time    
    epoch_loss = np.mean(epoch_loss)
    print('Epoch {}:\t{} ; Time : {}'.format(epoch, epoch_loss, train_time))
    
    log_dic = {'epoch': epoch, 'loss': epoch_loss.item(), 'time': train_time}
    utils.write_log(os.path.join('logs_image', config + '.txt'), log_dic)
    torch.save(net, os.path.join(args.savedir, config + '.pt'))

########## Test ##########
net.eval()
coms_test = utils.ImageTripletDataset(channel_list=chlist, postfix="{}_{}".format(args.inputdata, "test"))
coms_testloader = torch.utils.data.DataLoader(
    dataset = coms_test,
    batch_size = args.batch_size,
    shuffle = False,
    num_workers=8
)
test_loss = []
with torch.no_grad(): 
    for step, (anchor, pos, neg) in tqdm(enumerate(coms_testloader)):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        loss = net(anchor, pos, neg)
        test_loss.append(loss.item())
print("Test Loss", np.mean(test_loss))

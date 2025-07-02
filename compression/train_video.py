import os
import sys
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
import model_video

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='GPU device ID to use for model training or inference')
parser.add_argument("--inputdata", default='default', type=str, help='Name of the training dataset')
parser.add_argument("--inputemb", default="/data/image_embeddings_ir.pkl", type=str, help='Path of the image embedding')
parser.add_argument("--channel", default='ir', choices=["all", "ir", "sw", "wv"], help='Data channel to use for training')
parser.add_argument("--batch_size", default=128, type=int, help='Number of samples per batch during training')
parser.add_argument("--epochs", default=20, type=int, help='Number of training epochs')
parser.add_argument("--learning_rate", default=1e-6, type=float, help='Learning rate for optimizer during training')
parser.add_argument("--dim", default=256, type=int, help='Dimension of the embedding vectors')
parser.add_argument("--gamma", default=0.5, type=float, help='Hyperparameter controlling the margin for positive and negative pairs in the loss function')
parser.add_argument("--videolength", type=int, default=12, help="Length of input/output video")
parser.add_argument("--savedir", default='/data/', type=str, help='Directory where the trained model will be saved')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)
assert args.inputemb.endswith(".pkl")
inputemb_channel = args.inputemb[:-4].split("_")[-1]
assert inputemb_channel == args.channel

########## GPU Settings ##########
if torch.cuda.is_available() and int(args.gpu) >= 0:
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')

########## Random Seed ##########
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


########## Log Path ##########
tmp = args.inputemb.split("/")[-1]
image_emb_savename = tmp.split(".")[0]
config = 'videoencoder_ch_{}_dim_{}_bs_{}_lr_{}_gm_{}_ip_{}_ie_{}_vl_{}'.format(args.channel, args.dim, args.batch_size, args.learning_rate, args.gamma, args.inputdata, image_emb_savename, args.videolength)

print(config)
if os.path.isdir('./logs_video/') is False:
    os.makedirs('./logs_video/')
if os.path.isfile(os.path.join('logs_video', config + '.txt')):
    os.remove(os.path.join('logs_video', config + '.txt'))

########## Read Data ##########
videodata = utils.VideoTripletDataset(image_embs_path=args.inputemb, postfix="{}_{}".format(args.inputdata, "train"))
video_loader = torch.utils.data.DataLoader(
    dataset = videodata,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers=4
)

########## Prepare Training ##########
net = model_video.VideoModel(dim=args.dim, gamma=args.gamma, video_length=args.videolength)
net = net.to(device)
optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-6)
print('Preparing the model done')

########## Train Model ##########
net.train() 
for epoch in range(1, args.epochs+1):
    print('\nEpoch:\t', epoch)
    
    train_time = time.time()
    epoch_loss = []
    for step, (anchor, pos, neg) in tqdm(enumerate(video_loader)):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        loss = net(anchor, pos, neg)
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        
    epoch_loss = np.mean(epoch_loss)
    epoch_training_time = time.time() - train_time
    print('Epoch {}:\t{} ; Time : {}'.format(epoch, epoch_loss, epoch_training_time))
    
    log_dic = {'epoch': epoch, 'loss': epoch_loss.item(), 'time': epoch_training_time}
    utils.write_log(os.path.join('logs_video', config + '.txt'), log_dic)
    torch.save(net, os.path.join(args.savedir, config + '.pt'))
    
########## Test ##########
net.eval()
videodata_test = utils.VideoTripletDataset(image_embs_path=args.inputemb, postfix="{}_{}".format(args.inputdata, "test"))
video_testloader = torch.utils.data.DataLoader(
    dataset = videodata_test,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers=4
)
test_loss = []
with torch.no_grad(): 
    for step, (anchor, pos, neg) in tqdm(enumerate(video_loader)):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        
        loss = net(anchor, pos, neg)
        test_loss.append(loss.item())
print("Test Loss", np.mean(test_loss))

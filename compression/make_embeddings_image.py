import os
import torch
import random
import numpy as np
import pickle
import PIL
import argparse
from tqdm import tqdm, trange
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils

# MAIN ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='GPU device ID to use for model training or inference')
parser.add_argument("--modelpath", type=str, default="/data/imageencoder_ch_ir_dim_256_bs_128_lr_1e-6_gm_0.5_ip_default.pt", help="Path to the pre-trained model file")
parser.add_argument("--outputdir", type=str, default="/data/", help="Directory where embeddings will be saved")
parser.add_argument("--outputname", type=str, default="image_embeddings", help="Name of the output embedding file")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing data")
parser.add_argument("--channel", default='ir', choices=["all", "ir", "sw", "wv"], help='Data channel to use for embedding')
args = parser.parse_args()

# python make_embeddings_image.py --modelpath "/data/weather-support/weather_support/SEND_2024_2_backup3/models/imageencoder_ch_ir_dim_256_bs_128_lr_1e-06_gm_0.5_ip_default.pt" --outputdir "/data/weather-support/weather_support/SEND_2024_2_backup3/embeddings/"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:{}".format(args.gpu) if USE_CUDA else "cpu")

########## Load Model ##########
model = torch.load(args.modelpath, map_location=DEVICE)
model = model.cuda(DEVICE)
model.eval()

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
imgdata = utils.ImageDataset(chlist=chlist)
data_loader = torch.utils.data.DataLoader(
    dataset = imgdata,
    batch_size = args.batch_size,
    shuffle = False,
    num_workers=8
)

########## Make Embeddings ##########
with torch.no_grad():
    embeddings = []
    idx = 1
    
    for x in tqdm(data_loader):
        x = x.cuda(DEVICE)
        embs = model.embed_image(x)
        embeddings.append(embs.detach().cpu())
        idx += 1
                
    embeddings = torch.cat(embeddings)
print("Embeddings")
print(embeddings.shape)

########## Save Embeddings ##########
with open("./train_data/date_list_image.pkl", "rb") as f:
    date_list = pickle.load(f)
label_list = [d.strftime("%Y%m%d%H%M%S") for d in date_list]
with open("./train_data/data_list_image.pkl", "rb") as f:
    path_list = pickle.load(f)
if args.channel == "ir":
    path_list = [path[0] for path in path_list]
elif args.channel == "sw":
    path_list = [path[1] for path in path_list]
elif args.channel == "wv":
    path_list = [path[2] for path in path_list]
save_vectors = {'labels': label_list, 
                'imgpaths': path_list, 
                'vectors': embeddings}
with open("{}{}_{}.pkl".format(args.outputdir, args.outputname, args.channel), 'wb') as f:
    pickle.dump(save_vectors, f)

print(save_vectors['labels'][0])
print(save_vectors['imgpaths'][0])
print(save_vectors['vectors'][0])

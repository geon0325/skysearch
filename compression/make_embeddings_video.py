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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import VideoDataset

# MAIN ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='GPU device ID to use for model training or inference')
parser.add_argument("--modelpath", type=str, default="/data/videoencoder_ch_ir_dim_256_bs_128_lr_1e-06_gm_0.5_ip_default_ie_image_embeddings_ir_vl_12.pt", help="Path to the pre-trained model file")
parser.add_argument("--inputemb", default="/data/image_embeddings_ir.pkl", type=str, help='Path of the image embedding')
parser.add_argument("--outputdir", type=str, default="/data/", help="Directory where embeddings will be saved")
parser.add_argument("--outputname", type=str, default="video_embeddings", help="Name of the output embedding file")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing data")
parser.add_argument("--channel", default='ir', choices=["all", "ir", "sw", "wv"], help='Data channel to use for embedding')
args = parser.parse_args()

assert args.inputemb.endswith(".pkl")
inputemb_channel = args.inputemb[:-4].split("_")[-1]
assert inputemb_channel == args.channel

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:{}".format(args.gpu) if USE_CUDA else "cpu")

model = torch.load(args.modelpath, map_location='cpu')
model = model.to(DEVICE)
model.eval()

dataset_video = VideoDataset(image_embs_path=args.inputemb)
data_loader = torch.utils.data.DataLoader(
    dataset = dataset_video,
    batch_size = args.batch_size,
    shuffle = False,
    num_workers=8
)

embeddings = []
with torch.no_grad():
    for x in tqdm(data_loader):
        x = x.to(DEVICE)
        embs = model.embed_video(x)
        embeddings.append(embs.detach().cpu())
    embeddings = torch.cat(embeddings)
    
print("Embeddings")
print(embeddings.shape)
with open("./train_data/data_index_list_video.pkl", "rb") as f:
    data_index_list_video = pickle.load(f)
with open("./train_data/data_list_image.pkl", "rb") as f:
    data_list_image = pickle.load(f)
path_list_video = []
for video_indexes in data_index_list_video:
    video_paths = []
    for idx in video_indexes:
        if args.channel == "ir":
            image_path = data_list_image[idx][0]
        elif args.channel == "sw":
            image_path = data_list_image[idx][1]
        elif args.channel == "wv":
            image_path = data_list_image[idx][2]
        elif args.channel == "all":
            image_path = data_list_image[idx]
        video_paths.append(image_path)
    path_list_video.append(video_paths)
with open("./train_data/date_list_video.pkl", "rb") as f:
    date_list_video = pickle.load(f)
label_list_video = [d.strftime("%Y%m%d%H%M%S") for d in date_list_video]
save_vectors = {'labels': label_list_video, 
                'imgpaths': path_list_video, 
                'vectors': embeddings}
with open("{}{}_{}.pkl".format(args.outputdir, args.outputname, args.channel), 'wb') as f:
    pickle.dump(save_vectors, f)
        
print(save_vectors['labels'][0])
print(save_vectors['imgpaths'][0])
print(save_vectors['vectors'][0])
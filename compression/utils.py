import os
import torch
import random
import numpy as np
import pickle as pkl
import PIL
import pandas as pd
from tqdm import trange, tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
tf = transforms.ToTensor()

def file_to_date(filename):
    term = filename.split('/')[-1][:-4].split('_')[-1]
    year, month, day, hour, minute = term[0:4], term[4:6], term[6:8], term[8:10], term[10:12]
    return '{}-{}-{} {}:{}'.format(year, month, day, hour, minute)

def stringformat_to_datetime(str_time):
    datetime_object = datetime.strptime(str_time, '%Y-%m-%d %H:%M')
    return datetime_object

def write_log(log_path, log_dic):
    with open(log_path, 'a') as f:
        for _key in log_dic:
            f.write(_key + '\t' + str(log_dic[_key]) + '\n')
        f.write('\n')
        
        
############## IMAGE ################
class ImageTripletDataset(Dataset):
    def __init__(self, channel_list, input_dir="./train_data/", postfix="default_train"):
        self.input_dir = input_dir
        self.chlist = channel_list
        self.postfix = postfix
        with open('{}/data_list_image.pkl'.format(self.input_dir), 'rb') as f:
            self.data_list = pkl.load(f)
        with open('{}/date_list_image.pkl'.format(self.input_dir), 'rb') as f:
            self.date_list = pkl.load(f)
        with open('{}/triplets_image_{}.pkl'.format(self.input_dir, self.postfix), 'rb') as f:
            self.triplet_list = pkl.load(f)
    
    def __len__(self):
        return len(self.triplet_list)
    
    def __getitem__(self, index):
        anchor_index, pos_index, neg_index = self.triplet_list[index]
        
        anchor_img_list = []
        pos_img_list = []
        neg_img_list = []
        
        for ch in self.chlist:
            anchor_img = tf(PIL.Image.open(self.data_list[anchor_index][ch])).squeeze(0)
            anchor_img_list.append(anchor_img)
            pos_img = tf(PIL.Image.open(self.data_list[pos_index][ch])).squeeze(0)
            pos_img_list.append(pos_img)
            neg_img = tf(PIL.Image.open(self.data_list[neg_index][ch])).squeeze(0)
            neg_img_list.append(neg_img)
            
        anchor_img_list = torch.stack(anchor_img_list)
        pos_img_list = torch.stack(pos_img_list)
        neg_img_list = torch.stack(neg_img_list)
        
        return anchor_img_list, pos_img_list, neg_img_list

class ImageDataset(Dataset):
    def __init__(self, chlist, input_dir='./train_data/'):
        self.input_dir = input_dir
        self.channel = chlist
        with open(self.input_dir + 'data_list_image.pkl', 'rb') as f:
            self.data_list = pkl.load(f)
        with open(self.input_dir + 'date_list_image.pkl', 'rb') as f:
            self.date_list = pkl.load(f)        
    
    def __len__(self):
        return len(self.date_list)
    
    def __getitem__(self, index):
        img_list = []
        for ch in self.channel:
            path = self.data_list[index][ch]
            img = tf(PIL.Image.open(path)).squeeze(0)
            img_list.append(img)
            
        img = torch.stack(img_list)
        
        return img

############## Video ################
tf = transforms.ToTensor()
class VideoTripletDataset(Dataset):
    def __init__(self, image_embs_path='/data/image_embeddings_ir.pkl', postfix="default_train"):
        self.dir = "./train_data"
        with open('{}/data_index_list_video.pkl'.format(self.dir), 'rb') as f:
            self.data_list = pkl.load(f)
        with open('{}/date_list_video.pkl'.format(self.dir), 'rb') as f:
            self.date_list = pkl.load(f) 
            
        assert image_embs_path.endswith(".pkl")
        with open(image_embs_path, 'rb') as f:
            emb_dict = pkl.load(f)
            self.embs = emb_dict['vectors']
        with open('{}/triplets_video_{}.pkl'.format(self.dir, postfix), 'rb') as f:
            self.triplet_list = pkl.load(f)
            
    def __len__(self):
        return len(self.triplet_list)
    
    def __getitem__(self, index):
        # b_z, ts, c, h, w
        anchor_index, pos_index, neg_index = self.triplet_list[index]
        
        anchor_video, pos_video, neg_video = None, None, None
        for i, idx in enumerate(self.triplet_list[index]):
            video = self.data_list[idx]
            video_list = []
            for image_index in video:
                video_list.append(self.embs[image_index])
            video_torch = torch.stack(video_list)
            if i == 0:
                anchor_video = video_torch
            elif i == 1:
                pos_video = video_torch
            elif i == 2:
                neg_video = video_torch
        
        return anchor_video, pos_video, neg_video
    
    
tf = transforms.ToTensor()
class VideoDataset(Dataset):
    def __init__(self, image_embs_path='/data/image_embeddings_ir.pkl'):
        with open('./train_data/data_index_list_video.pkl', 'rb') as f:
            self.data_list = pkl.load(f)
        with open('./train_data/date_list_video.pkl', 'rb') as f:
            self.date_list = pkl.load(f)
        with open(image_embs_path, 'rb') as f:
            emb_dict = pkl.load(f)
            self.embs = emb_dict['vectors']
        
    def __len__(self):
        return len(self.date_list)
    
    def __getitem__(self, index):
        video = self.data_list[index]
        video_list = []
        for image_index in video:
            video_list.append(self.embs[image_index])
        video_torch = torch.stack(video_list)
         
        return video_torch


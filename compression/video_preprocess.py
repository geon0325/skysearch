import math
import argparse
import numpy as np
import datetime
from collections import defaultdict
import pickle
import torch
from tqdm import tqdm, trange
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import PIL
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, default=12, help="Length of input/output video")
parser.add_argument("--gap", type=int, default=60, help="Length of input/output video")
args = parser.parse_args()
    
# Load Image Informations
with open('./train_data/data_list_image.pkl', 'rb') as f:
    image_data_list = pickle.load(f)
with open('./train_data/date_list_image.pkl', 'rb') as f:
    image_date_list = pickle.load(f)
    
# Make Data
sec_video = 60 * args.gap
video_indexes_list = []
video_date_list = []
for t in range(len(image_date_list)):
    cur_time = image_date_list[t]
    if "coms" in image_data_list[t][0].lower():
        cur_satellite = "coms"
    elif "gk2a" in image_data_list[t][0].lower():
        cur_satellite = "gk2a"
    video = [t]
    _t = t + 1
    while _t < len(image_date_list):
        if cur_satellite not in image_data_list[_t][0].lower():
            _t += 1
            continue
        delta = (image_date_list[_t] - cur_time).total_seconds()
        if delta > (args.length - 1) * sec_video:
            break
        if delta % sec_video == 0:
            video.append(_t)
        _t += 1
    # Filter Full-length Video
    if len(video) == args.length:
        video_indexes_list.append(video)
        video_date_list.append(cur_time)
        
# Save Data
with open('./train_data/date_list_video.pkl', 'wb') as f:
    pickle.dump(video_date_list, f)
with open('./train_data/data_index_list_video.pkl', 'wb') as f:
    pickle.dump(video_indexes_list, f)

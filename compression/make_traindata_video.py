import os
import gc
import random
import torch
import PIL
import pickle as pkl
import datetime
import numpy as np
from tqdm import tqdm, trange
import torchvision.transforms as transforms
import argparse
from utils import file_to_date, stringformat_to_datetime


def sample_triplet_for_i(date_list, pos_range, neg_range, i):
    pos_a, pos_b, pos_c, pos_d = i, i, i, i
    neg_a, neg_b, neg_c, neg_d = 0, 0, len(date_list) - 1, len(date_list) - 1

    idx = i - 1
    while idx > 0:
        delta = (date_list[i] - date_list[idx]).total_seconds() / 3600 # hour

        if (pos_b == i) and (delta > pos_range[0]):
            pos_b = idx
        if (pos_a == i) and (delta > pos_range[1]):
            pos_a = idx + 1
        if (neg_b == 0) and (delta > neg_range[0]):
            neg_b = idx + 1
        if (neg_b != 0):
            if neg_range[1] == -1:
                neg_a = 0
                break
            elif (delta > neg_range[1]):
                neg_a = idx + 1
                break
        idx -= 1
    idx = i + 1
    while idx < len(date_list):
        delta = (date_list[idx] - date_list[i]).total_seconds() / 3600 # hour

        if (pos_c == i) and (delta > pos_range[0]):
            pos_c = idx
        if (pos_d == i) and (delta > pos_range[1]):
            pos_d = idx # - 1
        if (neg_c == len(date_list) - 1) and (delta > neg_range[0]):
            neg_c = idx
        if (neg_c != len(date_list) - 1):
            if neg_range[1] == -1:
                neg_d = len(date_list) - 1
                break
            elif (delta > neg_range[1]):
                neg_d = idx # - 1
                break
        idx += 1
    j_pool = [_j for _j in range(pos_a, pos_b)] + [_j for _j in range(pos_c, pos_d)]
    k_pool = [_k for _k in range(neg_a, neg_b)] + [_k for _k in range(neg_c, neg_d)]

    j_pool = [_j for _j in j_pool]
    k_pool = [_k for _k in k_pool]
    
    if len(j_pool) == 0:
        return i,i,i

    j = random.choice(j_pool)
    # check error
    delta = abs(date_list[j] - date_list[i]).total_seconds() / 3600
    if delta > pos_range[1]:
        print(pos_range)
        print(date_list[pos_a],date_list[pos_b], date_list[pos_c],date_list[pos_d])
        print(delta)
        print(pos_a, pos_b, pos_c, pos_d)
        print(j)
    assert delta <= pos_range[1]
    k = random.choice(k_pool)
    delta = abs(date_list[k] - date_list[i]).total_seconds() / 3600
    assert delta >= neg_range[0]
    
    return i,j,k

parser = argparse.ArgumentParser(description="Arguments for image preprocessing and triplet generation")
parser.add_argument("--repeat_num", default=2, type=int, help="Number of triplet sets generated per anchor image")
parser.add_argument("--posdiff_list", default=[4], nargs='+', type=int, help="List of maximum time differences allowed for positive image pairs")
parser.add_argument("--negdiff_list", default=[1,2], nargs='+', type=int, help="List of time difference multipliers required for negative image pairs compared to positive pairs")
parser.add_argument("--split_type", default="train", choices=["train", "test"], help="Select between [train, test]")
parser.add_argument("--savename", default="default", type=str, help="Specify a name for saving traindata")
parser.add_argument("--datasize", default=-1, type=int, help="Limit the number of training data samples to use")
args = parser.parse_args()

with open("./train_data/date_list_video.pkl", "rb") as f:
    date_list = pkl.load(f)
print("Total number of videos", len(date_list))
last_time = date_list[-1]
for split_index in range(len(date_list)-1, 0, -1):
    year_diff = (last_time - date_list[split_index]).total_seconds() // (60*60*24*365)
    if year_diff >= 1:
        break
split_index = max(split_index, args.datasize)
print("split", split_index)
if args.split_type == "train":
    sample_index = list(range(split_index))
    date_list = date_list[:split_index]
elif args.split_type == "test":
    sample_index = list(range(split_index, len(date_list)))
    
# Make triplet of videos: (anchor image, postivie image, negative image)
print(args.posdiff_list)
print(args.negdiff_list)
triplet_list = []
endflag = False
for _ in range(args.repeat_num): # repeat
    random.shuffle(sample_index)
    partion = len(sample_index) // (len(args.posdiff_list) * len(args.negdiff_list)) 
    for di, posdiff in enumerate(args.posdiff_list):
        for gi, negdiff in enumerate(args.negdiff_list):
            start_point = partion * (len(args.negdiff_list)*di+gi)
            end_point = min(partion * (len(args.negdiff_list)*di+gi+1), len(sample_index))
            print(start_point, end_point, partion)
            i_pool = sample_index[start_point:end_point]
            pos_range = [0, posdiff]
            neg_range = [posdiff * negdiff, -1]
            for anchor in tqdm(i_pool):
                i,j,k = sample_triplet_for_i(date_list, pos_range, neg_range, anchor)
                if j == i:
                    continue
                assert i >= 0 and i < len(date_list), str(i)
                assert j >= 0 and j < len(date_list), str(j)
                assert k >= 0 and k < len(date_list), str(k)
                triplet_list.append([i,j,k])
                
                if args.datasize > 0 and len(triplet_list) >= args.datasize:
                    endflag = True
                    break
            if endflag:
                break
        if endflag:
            break
    if endflag:
        break

triplets = torch.tensor(triplet_list)
print(triplets.shape)
with open('./train_data/triplets_video_{}_{}.pkl'.format(args.savename, args.split_type), 'wb') as f:
    pkl.dump(triplets, f)
    
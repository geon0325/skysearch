import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import random
from skimage.measure import block_reduce
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import pickle as pkl
import PIL
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import argparse

tf = transforms.ToTensor()
def adjust_read(image_path, size0, size1):
    flag = 0 # 0: coms / 1: gk2a-1300 / 2: gk2a-1332
    
    if "GK2A" in image_path and min(size0, size1) == 1300:
        flag = 1
    elif "GK2A" in image_path and min(size0, size1) == 1332:
        flag = 2
        
    if flag == 0:
        img = PIL.Image.open(image_path)
        width, height = img.size
        img = img.crop((0, 30, width, height - 20)) 
        img = img.resize((375, 300))
        img = transforms.ToPILImage()(tf(img).squeeze(0))
        
    elif flag == 1:
        img = PIL.Image.open(image_path)
        width, height = img.size
        img = img.crop((18, 62, width-5, height)) 
        img = img.resize((375, 300))
        img = transforms.ToPILImage()(tf(img).squeeze(0))
        
    elif flag == 2:
        img = PIL.Image.open(image_path)
        width, height = img.size
        img = img.crop((0, 80, width, height-10))
        img = img.resize((375, 300))
        img = transforms.ToPILImage()(tf(img).squeeze(0))
        
    return img

def rgb2gray(rgb):
    ratio = [0.2989, 0.5870, 0.1140]
    ret = np.expand_dims(np.dot(rgb[...,:3], ratio), axis=2)
    return ret


# Main ======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--rawdatapath', type=str, default="/database/", help="raw data directory for full resolution satelite image")
parser.add_argument('--outputpath', type=str, default="/lpips/raw/", help="preprocessed directory")
parser.add_argument('--smaller_outputdir', type=str, default="/lpips/", help="smaller-size image directory")
parser.add_argument('--rate', type=float, default=0.50, help="Reduction factor for resizing images further for efficient LPIPS computation")
args = parser.parse_args()

# Prepare Directories
assert os.path.isdir(args.rawdatapath)
if os.path.isdir(args.outputpath) is False:
    os.makedirs(args.outputpath)
if os.path.isdir(args.outputpath) is False:
    os.makedirs(args.outputpath)
if os.path.isdir(args.smaller_outputdir) is False:
    os.makedirs(args.smaller_outputdir)
smaller_outputpath = "%s%.2f/" % (args.smaller_outputdir, args.rate) # e.g., "/lpips/0.50/"
if os.path.isdir(smaller_outputpath) is False:
    os.makedirs(smaller_outputpath)

# Find Raw Images
validchannel = ["ir01_ea040ps", "ir105_ea020lc", "swir_ea040ps", "sw038_ea020lc", "wv_ea040ps", "wv069_ea020lc"]
rawdata_fullnames = []
rawdata_names = []
for root, dirs, files in os.walk(args.rawdatapath):
    for file in files:
        if file.endswith(".png"):
            check_channel = False
            for channelname in validchannel:
                if channelname in file:
                    check_channel = True
                    break
            if check_channel:
                rawdata_fullnames.append(os.path.join(root, file))
                rawdata_names.append(file)

# Preprocess Images
for rawfile_path, f in zip(rawdata_fullnames, rawdata_names):
    print(f)
    try:
        rawimage = plt.imread(rawfile_path)
    except:
        print("error ", f)
        continue
    rawimage_size = rawimage.shape

    processedimage = rgb2gray(rawimage[100:])
    processedimage = np.squeeze(processedimage)

    # Reduce Resolution by mean-pooling
    s = min(rawimage_size[0], rawimage_size[1])
    reduce_size = 1
    i = 1
    while True:
        if (s // i) < 300:
            reduce_size = i-1
            break
        i += 1
    reducedimage = block_reduce(processedimage, block_size=(reduce_size, reduce_size), func=np.mean)
    reducedimage_size = reducedimage.shape

    save_image = reducedimage * 255
    save_image = save_image.astype(np.uint8)
    imsave(args.outputpath + f, save_image)

    # Adjust Position
    adjusted_image = adjust_read(args.outputpath + f, rawimage_size[0], rawimage_size[1])
    adjusted_image.save(args.outputpath + f, "PNG")

    # Save Smaller
    reduce_h = int(375 * args.rate)
    reduce_w = int(300 * args.rate)
    smaller_image = adjusted_image.resize((reduce_h, reduce_w))
    smaller_image.save(smaller_outputpath + f, "PNG")


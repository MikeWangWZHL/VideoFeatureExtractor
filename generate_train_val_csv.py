import os
import decord
from decord import VideoReader
from glob import glob
import numpy as np
import json
import sys
from tqdm import tqdm
import pickle
''' generate caption.pickle that is required for UniVL '''
# video_id,feature_file
# Z8xhli297v8,Z8xhli297v8
# vOJkQwF2eno,vOJkQwF2eno
# b34VwqSkRE0,b34VwqSkRE0
# 6MBctYaMU8U,6MBctYaMU8U
# gswKIbddBHw,gswKIbddBHw
# cppB7IXFySk,cppB7IXFySk


''' config input (WebVid) '''
caption_pickle_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/UniVL_VideoFeatureExtractor/caption_pickles/webvid_1percent_caption.pickle'
output_csv = './train_val_csv_for_captioning_UniVL/webvid_1percent_caption.csv'

with open(caption_pickle_path,'rb') as f:
    data_dict = pickle.load(f)
print(len(data_dict))

lines = ['video_id,feature_file']
for key in data_dict.keys():
    lines.append(f'{key},{key}')

with open(output_csv, 'w') as out:
    for line in lines:
        out.write(line)
        out.write('\n')
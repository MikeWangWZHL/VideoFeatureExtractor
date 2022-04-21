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
# {
#     'video_id 1':{
#         'start': array([0.08, 7.37, 15.05, ...], dtype=object),
#         'end': array([9.96, 16.98, 27.9, ...], dtype=object),
#         'text': array(['sentence 1 placehodolder',
#                     'sentence 2 placehodolder',
#                     'sentence 3 placehodolder', ...], dtype=object)
#     },
#     ...
# }


''' config input (WebVid) '''
video_root = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/WebVid/train_subset_1_percent_video'
video_captions_path = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/WebVid/train_subset_1_percent_ann/video_2_text_original_train_subset_1_percent.json'
output_path = './caption_pickles/webvid_1percent_caption.pickle'

video_paths = sorted(glob(os.path.join(video_root, '*.mp4')))
video_captions = json.load(open(video_captions_path))

data_dict = {}
for video_path in tqdm(video_paths):
    video_id = os.path.basename(video_path)[:-4]
    try:
        vr = VideoReader(video_path)
        start = 0.01
        end = int(len(vr)/vr.get_avg_fps()) - 0.01
        # print(start, end)
        assert video_id in video_captions

        data_dict[video_id] = {
            'start':[],
            'end':[],
            'text':[],
            'transcript':[]
        }
        for cap in video_captions[video_id]:
            data_dict[video_id]['start'].append(start)            
            data_dict[video_id]['end'].append(end)            
            data_dict[video_id]['text'].append(cap)          
            data_dict[video_id]['transcript'].append('none')          

        data_dict[video_id]['start'] = np.array(data_dict[video_id]['start'])
        data_dict[video_id]['end'] = np.array(data_dict[video_id]['end'])
        data_dict[video_id]['text'] = np.array(data_dict[video_id]['text'])
        data_dict[video_id]['transcript'] = np.array(data_dict[video_id]['transcript'])
    except KeyboardInterrupt:
        sys.exit()
    except:
        print(f'cannot load video, skip {video_id}')
    # break

print(len(data_dict))
with open(output_path, 'wb') as f:
    pickle.dump(data_dict, f)

file_ = open(output_path,'rb')
object_file = pickle.load(file_)
print(object_file)

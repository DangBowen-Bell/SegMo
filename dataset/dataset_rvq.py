from torch.utils.data import Dataset
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

from utils import new_utils


class MotionDataset(Dataset):
    def __init__(self, 
                 dataname,
                 usage='train',
                 window_size=64, 
                 unit_length=4):
        self.dataname = dataname
        self.window_size = window_size
        self.unit_length = unit_length
                
        dataset_info = new_utils.DatasetInfo(dataname, usage=usage)
        self.data_root = dataset_info.data_root
        self.motion_dir = dataset_info.motion_dir
        self.text_dir = dataset_info.text_dir
        self.joints_num = dataset_info.joints_num
        self.max_motion_length = dataset_info.max_motion_length
        self.mean = dataset_info.mean
        self.std = dataset_info.std
        id_list = dataset_info.id_list
                
        self.data = []
        self.lengths = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        print("Total number of motions for {}ing VQ-VAE: {}".format(usage, len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx+self.window_size]

        motion = (motion - self.mean) / self.std

        return motion

import os

import cv2
import numpy as np
import pandas as pd
from dask.array import indices
from dask.array.overlap import boundaries

from  modules.simple_tokenizer import SimpleTokenizer
from utils.h5 import load_h5
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

class Dataset_img(Dataset):
    def __init__(self, root_path, data_path, label_dict, flag='train', caption_len=32):
        assert flag in ['train', 'val', 'test']
        self.train_ratio = 0.7
        self.valid_ratio = 0.2
        self.test_ratio = 0.1
        self.flag = flag
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.label_dict = label_dict
        self.tokenizer = SimpleTokenizer()
        self.caption_max_len = caption_len
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.__read_data__()

    def __read_data__(self):
        path = os.path.join(self.root_path, self.data_path)

        # 加载H5数据
        h5_file = load_h5(path)
        self.data = h5_file['data'][:]

        # 加载img
        self.all_pic = []
        pic_list = []
        pic_folder = os.path.join(self.root_path, "pic")
        for folder in os.listdir(pic_folder):
            folder_path = os.path.join(pic_folder, folder)
            pic_list.append(folder)
        pic_list.sort(key=lambda x: int(x.split('_')[1]))
        for pic in pic_list:
            all_device = []
            pic_path = os.path.join(pic_folder, pic)
            for file in os.listdir(pic_path):
                img_path = os.path.join(pic_path, file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array = np.array(img)
                all_device.append(img_array)

            self.all_pic.append(all_device)
        self.all_pic = np.array(self.all_pic)

        # 拆分数据集：train/validation/test
        total_size = len(self.data)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.valid_ratio * total_size)
        test_size = total_size - train_size - val_size
        boundary_left = [0, train_size, train_size+val_size]
        boundary_right = [train_size, train_size+val_size, total_size]
        left = boundary_left[self.type]
        right = boundary_right[self.type]


        # 读取标签CSV文件
        df_raw = pd.read_csv(self.label_dict)
        self.text = np.zeros((total_size, self.caption_max_len))
        self.mask = np.zeros((total_size, self.caption_max_len))
        labels = df_raw[['caption']].values
        index = 0
        for label in labels:
            caption = self.tokenizer.tokenize(str(label))
            caption = [self.SPECIAL_TOKEN['CLS_TOKEN']] + caption
            total_length_with_CLS = self.caption_max_len - 1
            if len(caption) > total_length_with_CLS:
                caption = caption[:total_length_with_CLS]
            caption = caption + [self.SPECIAL_TOKEN['SEP_TOKEN']]
            ids = self.tokenizer.convert_tokens_to_ids(caption)
            m = [1] * len(caption)
            while len(ids) < self.caption_max_len:
                ids.append(0)
                m.append(0)
            assert len(ids) == self.caption_max_len
            assert len(m) == self.caption_max_len

            self.text[index] = np.array(ids)
            self.mask[index] = np.array(m)
            index += 1

        self.data = self.data[left:right]
        self.text = self.text[left:right]
        self.mask = self.mask[left:right]
        self.pic = self.all_pic[left:right]

    def __getitem__(self, index):
        return self.data[index], self.text[index], self.mask[index], self.pic[index]

    def __len__(self):
        return len(self.data)



class Dataset_raw(Dataset):
    def __init__(self, root_path, data_path, label_path, flag='train', caption_len=20, use_patch=False, patch_len=None, stride=None):

        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.use_patch = use_patch
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.tokenizer = SimpleTokenizer()
        self.caption_max_len = caption_len
        self.patch_len = patch_len
        self.stride = stride
        self.__read_data__()

    def __read_data__(self):
        data_path = os.path.join(self.root_path, self.data_path)
        label_path = os.path.join(self.root_path, self.label_path)

        # 加载H5数据
        h5_file = load_h5(data_path)
        data = h5_file['data'][:]  # size: (N,5,1500,6)
        if self.use_patch:
            data = self.create_patches(data, self.patch_len, self.stride)

        # 加载label文件
        label = pd.read_csv(label_path)
        # print(len(label))
        # print(len(data))
        total_size = len(data)
        text = np.zeros((total_size, self.caption_max_len))
        mask = np.zeros((total_size, self.caption_max_len))
        labels = label[['caption']].values
        index = 0
        for label in labels:
            label = str(label)[2:-2]
            caption = self.tokenizer.encode_word(label) # list
            caption.insert(0,1)
            caption.append(2)
            m = [1] * len(caption)
            while len(caption) < self.caption_max_len:
                caption.append(0)
                m.append(0)
            assert len(caption) == self.caption_max_len
            assert len(m) == self.caption_max_len

            text[index] = np.array(caption)
            mask[index] = np.array(m)
            index += 1


        # 拆分数据集：train-val
        if self.flag == 'test':
            indices = [i for i in range(len(data))]
        else:
            if self.flag == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            else :
                indices = [i for i in range(len(data)) if i % 10 == 0]

        self.data = data[indices]  # RWA_DATA
        self.text = text[indices]  # CAPTION_GT
        self.mask = mask[indices]  # CAPTION_PAD_MASK


    def __getitem__(self, index):
        return self.data[index], self.text[index], self.mask[index]

    def __len__(self):
        return len(self.data)

    def create_patches(self, data, patch_len=300, stride=100):
        """
           data: np.Array of shape (N, D, T, F)
                 N: number of samples
                 D: number of devices per sample
                 T: time steps
                 F: features per time step
           returns:
               patches: shape (N, D, num_patches, patch_len, F)
           """
        N, D, T, F = data.shape
        num_patches = (T - patch_len) // stride + 1
        patch_data = np.zeros((N, D, num_patches, patch_len, F))

        for i in range(num_patches):
            start = i * stride
            end = start + patch_len
            patch_data[:, :, i, :, :] = data[:, :, start:end, :]

        return patch_data




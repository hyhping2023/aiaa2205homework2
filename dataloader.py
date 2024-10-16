from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import pandas as pd
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir:str = './video_frames_30fpv_320p', 
                 csv_file:str = './trainval.csv', 
                 transform=None,
                 frames_per_video:int = 30,
                 group_size:int = 30,
                 batch_size:int = 32):
        self.root_dir, self.csv_file, self.transform, self.frames_per_video, self.group_size, self.batch_size = root_dir, csv_file, transform, frames_per_video, group_size, batch_size
        self.df = pd.read_csv(csv_file, header=None, skiprows=1).sort_values(by=[0]).reset_index(drop=True)
        self.classes = sorted(self.df[1].unique())


    def __len__(self):
        return len(self.df)
    
if __name__ == "__main__":
    dataset = MyDataset(root_dir='/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p')
    print(len(dataset))
    print(dataset.df)
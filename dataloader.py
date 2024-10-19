from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import pandas as pd
import glob
from PIL import Image

class FatherDataset():
    '''
        This will return trainset and valset

    '''
    def __init__(self, root_dir:str = './video_frames_30fpv_320p', 
                 csv_file:str = './trainval.csv', 
                 transform:transforms.Compose | list = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                 frames_per_video:int = 30,
                 batch_size:int = 32,
                 frames:int = 30,
                 train_ratio:float = 0.8,
                 random_state = None,
                 offset = None, 
                 channel_first = False):
        if not isinstance(transform, list): # isinstance
            self.trans = [transform, transform]
        else:
            self.trans = transform
        self.root_dir, self.csv_file, self.frames_per_video, self.batch_size = root_dir, csv_file, frames_per_video, batch_size
        if train_ratio != 1:
            self.df = pd.read_csv(csv_file, header=None, skiprows=1).sample(frac = 1, random_state=random_state).reset_index(drop=True) # random state 可以让结果复现
        else:
            self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())
        if self.frames_per_video > frames:
            self.frames_per_video = frames
        self.framestep = frames // self.frames_per_video
        if train_ratio < 1:
            self.traindf = self.df[:int(len(self.df) * train_ratio)].reset_index(drop=True)
            self.valdf = self.df[int(len(self.df) * train_ratio):].reset_index(drop=True)
        elif train_ratio > 1:
            self.valdf = self.df[:int(len(self.df) * (train_ratio - 1))].reset_index(drop=True)
        self.train_ratio = train_ratio
        self.offset = offset
        self.channel_first = channel_first

    def __len__(self):
        return len(self.df)
    
    def get(self, ):
        if self.train_ratio == 1:
            return DataLoader(MyDataset(df=self.df, classes=self.classes, root_dir=self.root_dir, transform=self.trans[0], frames_per_video=self.frames_per_video, offset=self.offset, channel_first=self.channel_first), batch_size=self.batch_size, shuffle=True)
        elif self.train_ratio > 1:
            return DataLoader(MyDataset(df=self.df, classes=self.classes, root_dir=self.root_dir, transform=self.trans[0], frames_per_video=self.frames_per_video, offset=self.offset, channel_first=self.channel_first), batch_size=self.batch_size, shuffle=True), \
                DataLoader(MyDataset(df=self.valdf, classes=self.classes, root_dir=self.root_dir, transform=self.trans[1], frames_per_video=self.frames_per_video, offset=self.offset, channel_first=self.channel_first), batch_size=self.batch_size, shuffle=True)
        return DataLoader(MyDataset(df=self.traindf, classes=self.classes, root_dir=self.root_dir, transform=self.trans[0], frames_per_video=self.frames_per_video, offset=self.offset, channel_first=self.channel_first), batch_size=self.batch_size, shuffle=True), \
            DataLoader(MyDataset(df=self.valdf, classes=self.classes, root_dir=self.root_dir, transform=self.trans[1], frames_per_video=self.frames_per_video, offset=self.offset, channel_first=self.channel_first), batch_size=self.batch_size, shuffle=True)

class MyDataset(Dataset):
    '''
        The Range is used to divide the whole data into several groups, 
        then each groups willbe loaded to the Dataloader

        If channel_first is False(default), the return will be 
        BatchSize x Frames x Channels x Height x Width
        else
        BatchSize x Channels x Frames x Height x Width 

        for computer with RAM 32GB, the batch size should not larger than 128

    '''
    def __init__(self, df, classes, root_dir,
                 transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                 frames_per_video:int = 30,
                 frames = 30,
                 offset = None,
                 channel_first = False
                 ):
        self.trans, self.frames_per_video = transform, frames_per_video
        self.root_dir = root_dir
        self.df = df
        self.classes = classes
        self.framestep = frames // frames_per_video
        self.offset = offset
        self.channel_first = channel_first

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        '''
        This will load all the frames in the folder of the current slide 
        '''
        file, label = self.df[0][index], self.df[1][index]
        path = sorted(glob.glob(os.path.join(self.root_dir, file, 'H*.jpg')))
        if self.frames_per_video == 1:
            if self.offset is not None:
                frames = self.trans(Image.open(path[self.offset%30]).convert('RGB'))
            else:
                frames = self.trans(Image.open(path[len(path)//2]).convert('RGB')) 
        else:
            path = path[:self.frames_per_video*self.framestep:self.framestep]
            if not self.channel_first:
                frames = torch.stack([self.trans(Image.open(file).convert('RGB')) for file in path]) # channels x depth x height x width
            else:
                frames = torch.cat([self.trans(Image.open(file).convert('RGB')).unsqueeze(1) for file in path], dim=1) # Depth x channels x height x width
        return frames, label        

    
if __name__ == "__main__":
    paths = glob.glob('/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p/*')
    for path in paths:
        files = sorted(glob.glob(path + '/*'))
        if len(files) != 30:
            for idx in range(30-len(files)):
                new_file = files[-1]
                new_file = new_file[:-6] +str(int(new_file[-6:-5])+idx+1) + new_file[-4:]
                os.system(f'cp {files[-1]} {new_file}')
            print(path)

    trans = transform = transforms.Compose([
            transforms.Resize((225, 225)),
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
        ])
    dataset = FatherDataset(root_dir='/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p', csv_file='./trainval.csv', transform=trans, frames_per_video=30, batch_size=64)
    
    trainLoader, valLoader = dataset.get()

    for data, label in trainLoader:
        print(data.shape, label)
    for data, label in valLoader:
        print(data.shape, label)
            # for data, label in dataloader:
            #     print(data.shape, label)
   
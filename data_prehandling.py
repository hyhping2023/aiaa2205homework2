import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from PIL import Image
import os, glob

if __name__ == '__main__':
    folders = glob.glob('./video_frames_30fpv_320p/*')
    folders = sorted(folders)
    shape = []
    for folder in folders:
        final_img = None
        files = glob.glob(folder + '/*')
        print(folder)
        for file in files:
            if 'mean' in file:
                continue
            img = Image.open(file)
            img = img.convert('RGB')
            img = transforms.ToTensor()(img)
            img.unsqueeze_(0)
            if final_img is None:
                final_img = img
            else:
                final_img = torch.cat((final_img, img), 0)
        final_img = torch.mean(final_img, 0)
        print(final_img.shape)
        shape.append(final_img.shape)
    import collections
    print(collections.Counter(shape))
        # # print(final_img, type(final_img))
        # final_img = transforms.GaussianBlur(3, sigma=(1, 1))(final_img)
        # # print(final_img, type(final_img))
        # final_img = transforms.ToPILImage()(final_img)
        # # print(os.path.join(folder, 'mean.jpg'))
        # final_img.save(os.path.join(folder, 'mean.jpg'), 'JPEG', quality=100)

    
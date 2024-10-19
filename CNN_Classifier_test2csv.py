import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from models import GoogleNet
from torchvision import models
from dataloader import FatherDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.001),
])

test_dataset = FatherDataset("/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p", "./test_for_student.csv", transform, 
                        train_ratio=1, random_state=0, frames_per_video=15, batch_size=4)

test_loader = test_dataset.get()



net = models.resnet152().to(device)
net.fc = nn.Linear(2048, 10).to(device)
net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_best_swinbv2_4_94.63_91.97_93.20.pth'))

# net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
# net.fc = nn.Linear(2048, 10).to(device)
# net.load_state_dict(torch.load('/home/hyh/wslbackup/homework/hw2/results/model_last_inceptionv3_64.pth'))

# Evaluation

net.eval()

result = []
with torch.no_grad():
    # TODO: Evaluation result here ...
    for test_inputs, test_labels in test_loader:
        # TODO: Validation code ...
        batch_size = test_inputs.shape[0]
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_inputs = test_inputs.reshape(-1, test_inputs.shape[2], test_inputs.shape[3], test_inputs.shape[4])
        test_outputs= net(test_inputs)
        test_outputs = test_outputs.reshape(batch_size, -1, test_outputs.shape[1])
        test_outputs = torch.sum(test_outputs, dim=1)
        for i in range(len(test_outputs)):
            result.append(torch.argmax(test_outputs[i]).item())         

    video_ids = [test_dataset.df.iloc[i, 0] for i in range(len(test_dataset))]

with open('result.csv', "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(result):
        f.writelines("%s,%d\n" % (video_ids[i], pred_class))




# class MyDataset(Dataset):
#     def __init__(self, root, csv_file, transform=None):
#         self.root = root
#         self.transforms = transform
#         self.df = pd.read_csv(csv_file, header=None, skiprows=1)
#         self.classes = sorted(self.df[1].unique())

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index, ):
#         vid, label = self.df.iloc[index, :]
#         img_list = os.listdir(os.path.join(self.root, f"{vid}"))
#         img_list = sorted(img_list)
#         img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
#         # img_path = os.path.join(self.root, f"{vid}", 'mean.jpg')
#         img = Image.open(img_path).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
#         label = self.classes.index(label)
#         return img, label


# test_dataset = MyDataset("/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p", "./test_for_student.csv", transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
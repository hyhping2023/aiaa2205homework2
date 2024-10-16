import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from models import GoogleNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
 
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

transform = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor()
])

test_dataset = MyDataset("video_frames_30fpv_320p", "./test_for_student.csv", transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load Model
net = GoogleNet().to(device)
net.load_state_dict(torch.load('results/model_last_googlenet.pth'))

# Evaluation
net.eval()
result = []
with torch.no_grad():
    # TODO: Evaluation result here ...
    for test_inputs, test_labels in test_loader:
        # TODO: Validation code ...
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_outputs, aux1, aux2 = net(test_inputs)
        for i in range(len(test_outputs)):
            result.append(torch.argmax(test_outputs[i]).item())         

    video_ids = [test_dataset.df.iloc[i, 0] for i in range(len(test_dataset))]

with open('result.csv', "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(result):
        f.writelines("%s,%d\n" % (video_ids[i], pred_class))

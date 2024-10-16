import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from models import Net, GoogleNet, PreRNN, SwinTransformerV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index, ):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list.remove('mean.jpg')
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # img = torch.stack([self.transforms(Image.open(os.path.join(self.root, f"{vid}", img_list[i])).convert('RGB')) for i in range(0,len(img_list),15)])

        label = self.classes.index(label)
        return img, label

# You can add data augmentation here
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


trainval_dataset = MyDataset("./video_frames_30fpv_320p", "./trainval.csv", transform)
train_data, val_data = train_test_split(trainval_dataset, test_size=0.2, random_state=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# net = Net().to(device)
# net = PreRNN().to(device)
net = SwinTransformerV2(num_classes=10).cuda()
# net.load_state_dict(torch.load('results/model_last_googlenet.pth'))

# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

train, val, eval = [], [], []
tl, vl, el = [], [], []
for epoch in range(100):
    # TODO: Metrics variables ...
    start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # TODO: Training code ...
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # outputs, aux1, aux2 = net(inputs)
        # running_loss_train = criterion(outputs, labels)
        # aux_loss = 0.1*criterion(aux1, labels) + 0.3*criterion(aux2, labels)
        # loss = running_loss_train + aux_loss
        outputs = net(inputs)
        running_loss_train = criterion(outputs, labels)
        loss = running_loss_train
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            # TODO: Validation code ...
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs= net(val_inputs)
            net.eval()
            eval_outputs= net(val_inputs)
            eval_loss = criterion(eval_outputs, val_labels)
            net.train()
            running_loss_val = criterion(val_outputs, val_labels)

    correct_train = torch.sum(torch.argmax(outputs, dim=1) == labels).item()
    total_train = len(labels)
    correct_val = torch.sum(torch.argmax(val_outputs, dim=1) == val_labels).item()
    total_val = len(val_labels)
    correct_eval = torch.sum(torch.argmax(eval_outputs, dim=1) == val_labels).item()
    total_eval = len(val_labels)
        

    # # TODO: save best model

    # # save last model
    output_file = 'googlenet_mean'
    torch.save(net.state_dict(), f'model_last_{output_file}.pth')

    # print metrics log
    print('[Epoch %d] Loss (train/val/eval): %.6f/%.6f/%.6f' % (epoch + 1, running_loss_train, running_loss_val, eval_loss),
        ' Acc (train/val/eval): %.2f%%/%.2f%%/%.2f%%' % (100 * correct_train/total_train, 100 * correct_val/total_val, 100 * correct_eval/total_eval),
        ' Epoch Time: %.2fs' % (end_time - start_time))
    
    train.append(100 * correct_train/total_train)
    val.append(100 * correct_val/total_val)
    eval.append(100 * correct_eval/total_eval)

    tl.append(running_loss_train)
    vl.append(running_loss_val)
    el.append(eval_loss)


# use matplotlib to display
import matplotlib.pyplot as plt
plt.figure()
plt.plot(train, label='train')
plt.plot(val, label='val')
plt.plot(eval, label='eval')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.savefig(f'{output_file}.png')


plt.figure()
plt.plot(tl, label='train')
plt.plot(vl, label='val')
plt.plot(el, label='eval')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig(f'{output_file}_loss.png')
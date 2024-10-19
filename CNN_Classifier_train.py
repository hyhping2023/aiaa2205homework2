import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import time, os
from models import Net, GoogleNet
from model_swtr_2d import SwinTransformerV2
from dataloader import FatherDataset
import torchvision.models as models
from video_model import SwinTransformer3D
from drawer import Drawer
from resnet import MyResnetCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can add data augmentation here
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        ])

transformer_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
])

random_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation((-90, 90)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        ])

gooogle_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.001),
        ])


# net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
# net.fc = nn.Linear(2048, 10).to(device)
# net.load_state_dict(torch.load('/home/hyh/wslbackup/homework/hw2/results/model_best_inceptionv3_64.pth'))
# dataset = FatherDataset(root_dir='/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p', csv_file='./trainval.csv'
#                                          , transform=gooogle_transform, frames_per_video=1, batch_size=64, random_state=0)

net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device)
net.fc = nn.Linear(2048, 10).to(device)
net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_best_resnet152_default_rand_32_96.62_89.50_91.97_320.pth'))

# net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_best_resnet152_default_64_99.52_99.56_99.91.pth'))
# dataset = FatherDataset(root_dir='/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p', csv_file='./trainval.csv'
#                                          , transform=transformer_transform, frames_per_video=1, batch_size=64, random_state=1, train_ratio=1.2)


# net = models.vit_l_32(pretrained=True).to(device)
# net.heads = nn.Linear(1024, 10).to(device)
# # net.load_state_dict(torch.load('/home/hyh/wslbackup/homework/hw2/results/model_last_vitl32.pth'))
# dataset = FatherDataset(root_dir='/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p', csv_file='./trainval.csv'
#                                          , transform=transformer_transform, frames_per_video=1, batch_size=96, random_state=0)

# net = SwinTransformer3D(num_classes=10).cuda()
# net.load_state_dict(torch.load('./model_best_tr3d_12.pth'))
# dataset = FatherDataset("/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p", "./trainval.csv"
#                                  , transformer_transform, batch_size=12)

# net = SwinTransformerV2(num_classes=10).cuda()
# net.load_state_dict(torch.load('/home/hyh/wslbackup/homework/hw2/results/model_best_swtr_2d_96.pth'))
# dataset = FatherDataset("/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p", "./trainval.csv"
#                                  , transformer_transform, batch_size=96, frames_per_video=1, random_state=0)

# net = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(device)
# net.classifier = nn.Linear(1920, 10).to(device)
# net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_best_densenet201_96.pth'))

# net = MyResnetCNN().to(device)
# net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_last_myresnet152_96.pth'))

# net = models.swin_b(weights=models.Swin_B_Weights.DEFAULT).to(device)
# net.heads = nn.Linear(1536, 10).to(device)
# net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_best_swinb_32_96.36_90.11_91.53.pth'))

# net = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT).to(device)
# net.heads = nn.Linear(1536, 10).to(device)
# net.load_state_dict(torch.load('/home/hyh/wslbackup/aiaa2205homework2/model_best_swinbv2_16_97.53_90.38_91.26.pth'))

dataset = FatherDataset(root_dir='/home/hyh/wslbackup/homework/hw2/video_frames_30fpv_320p', csv_file='./trainval.csv'
                                        , transform=[random_transform, transform], frames_per_video=15, batch_size=4, random_state=0, offset=0)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
train_loader, val_loader = dataset.get()

bta, bva = 0, 0

drawer = Drawer()

for epoch in range(100):
    # TODO: Metrics variables ...
    start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # TODO: Training code ...
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]) # batch x depth
        optimizer.zero_grad()
        # outputs, aux1 = net(inputs)
        outputs = net(inputs)
        # print(outputs.shape)
        outputs = outputs.reshape(batch_size, -1, outputs.shape[1])
        outputs = torch.sum(outputs, dim=1)
        running_loss_train = criterion(outputs, labels)
        # aux_loss = 0.3*criterion(aux1, labels)
        # loss = running_loss_train + aux_loss
        loss = running_loss_train
        loss.backward()
        optimizer.step()
        correct_train = torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        drawer.add('train', correct_train, loss.item(), len(labels))

        
    end_time = time.time()
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            # TODO: Validation code ...
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            batch_size = val_inputs.shape[0]
            val_inputs = val_inputs.reshape(-1, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4])
            # val_outputs, aux1= net(val_inputs)
            val_outputs = net(val_inputs)
            val_outputs = val_outputs.reshape(batch_size, -1, val_outputs.shape[1])
            val_outputs = torch.sum(val_outputs, dim=1)
            net.eval()
            eval_outputs= net(val_inputs)
            eval_outputs = eval_outputs.reshape(batch_size, -1, eval_outputs.shape[1])
            eval_outputs = torch.sum(eval_outputs, dim=1)
            # eval_outputs = net(val_inputs)
            eval_loss = criterion(eval_outputs, val_labels)
            net.train()
            running_loss_val = criterion(val_outputs, val_labels)
            correct_val = torch.sum(torch.argmax(val_outputs, dim=1) == val_labels).item()
            total_val = len(val_labels)
            correct_eval = torch.sum(torch.argmax(eval_outputs, dim=1) == val_labels).item()

            drawer.add('val', correct_val, running_loss_val.item(), len(val_labels))
            drawer.add('eval', correct_eval, eval_loss.item(), len(val_labels))

    drawer.broadcast(end_time - start_time)

    output_file = 'swinbv2'+f'_{dataset.batch_size}'

    train_correctness, val_correctness, eval_correctness = drawer()
    if train_correctness >= bta and (val_correctness >= bva or eval_correctness >= bva):
        torch.save(net.state_dict(), f'model_best_{output_file}'+'_%.2f_%.2f_%.2f.pth'%(train_correctness, val_correctness, eval_correctness))
        # torch.save(net.state_dict(), f'model_best_{output_file}.pth')
        bta = train_correctness
        bva = max(val_correctness, eval_correctness)

    if train_correctness - val_correctness > 25:
        print('New round started!')
        break

    torch.save(net.state_dict(), f'model_last_{output_file}.pth')
            
drawer.draw(output_file)



# ----------------useless---------------------------

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

# use matplotlib to display
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(train, label='train')
# plt.plot(val, label='val')
# plt.plot(eval, label='eval')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()
# plt.savefig(f'{output_file}.png')


# plt.figure()
# plt.plot(train_loss, label='train')
# plt.plot(val_loss, label='val')
# plt.plot(eval_losses, label='eval')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
# plt.savefig(f'{output_file}_loss.png')
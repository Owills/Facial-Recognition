import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def use_gpu():
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                                         else torch.FloatTensor)
use_gpu()

import os


base_dir = 'mask_nomask_large'



train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')



# Directory with our training mask/nomask pictures
train_mask_dir = os.path.join(train_dir, 'mask')
train_nomask_dir = os.path.join(train_dir, 'nomask')


# Directory with our validation mask/nomask pictures
validation_mask_dir = os.path.join(validation_dir, 'mask')
validation_nomask_dir = os.path.join(validation_dir, 'nomask')

train_mask_fnames = os.listdir(train_mask_dir)
train_nomask_fnames = os.listdir(train_nomask_dir)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, pred = output.max(1)
            print(pred.sum().item())
            test_loss += criterion(output, target).item()
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)

    print('Average loss: {:.3f}, Test Acc: {:.3f} ({}/{})'.format(test_loss, 100.*correct/total, correct, total))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4,4), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        if torch.cuda.is_available():
           x = x.to(device="cuda")
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x
test_batch_size = 25
lr = 1e-4
criterion = nn.CrossEntropyLoss()

model = CNN()
model.load_state_dict(torch.load('OliverMaskDetection.ckpt'))
print('OliverMaskDetection.ckpt')

valset = datasets.ImageFolder(validation_dir, transform=transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()]))
val_loader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size, shuffle=False)


trainset = datasets.ImageFolder(train_dir,transform=transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=test_batch_size, shuffle=False)


test(model, train_loader)
print()
test(model, val_loader)
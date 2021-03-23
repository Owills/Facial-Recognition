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


base_dir = 'faces_large'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# Directory with our training mask/nomask pictures
train_Izzy_dir = os.path.join(train_dir, 'Izzy')
train_Josh_dir = os.path.join(train_dir, 'Josh')
train_Oliver_dir = os.path.join(train_dir, 'Oliver')
train_Sam_dir = os.path.join(train_dir, 'Sam')


# Directory with our validation mask/nomask pictures
validation_Izzy_dir = os.path.join(validation_dir, 'Izzy')
validation_Josh_dir = os.path.join(validation_dir, 'Josh')
validation_Oliver_dir = os.path.join(validation_dir, 'Oliver')
validation_Sam_dir = os.path.join(validation_dir, 'Sam')


print('total training Izzy images :', len(os.listdir(train_Izzy_dir)))
print('total training Josh images :', len(os.listdir(train_Josh_dir)))
print('total training Oliver images :', len(os.listdir(train_Oliver_dir)))
print('total training Sam images :', len(os.listdir(train_Sam_dir)))


print('total validation Izzy images :', len(os.listdir(validation_Izzy_dir)))
print('total validation Josh images :', len(os.listdir(validation_Josh_dir)))
print('total validation Oliver images :', len(os.listdir(validation_Oliver_dir)))
print('total validation Sam images :', len(os.listdir(validation_Sam_dir)))

#exit()
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}, Loss: {:.3f}'.format(epoch, loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, pred = output.max(1)
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
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2,2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 4)
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

batch_size = 100
test_batch_size = 75
epochs = 10
lr = 1e-4
criterion = nn.CrossEntropyLoss()



#resnet
#import torchvision.models as models
#resnet = models.resnet18(pretrained=True)
#print(resnet18)
#resnet.fc = nn.Linear(512,4)
#model = resnet

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=lr)
trainset = datasets.ImageFolder(train_dir,transform=transforms.Compose([transforms.CenterCrop(15), transforms.Resize((50, 50)),
                                                                        transforms.ColorJitter(brightness=0.1, contrast=0.1),
                                                                        transforms.ToTensor()]))
valset = datasets.ImageFolder(validation_dir, transform=transforms.Compose([transforms.CenterCrop(15), transforms.Resize((50, 50)), transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size, shuffle=False)

for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch)
    #test(model,train_loader)
    test(model, val_loader)

torch.save(model.state_dict(),'OliverFaceDetection.ckpt')
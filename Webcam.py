import cv2 #pip install opencv-python
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import Thread

criterion = nn.CrossEntropyLoss()

data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((50, 50)), transforms.ToTensor()])

def detect_image(img):
    # scale and pad image

    import os
    base_dir = 'mask_nomask_large'
    train_dir = os.path.join(base_dir, 'train')
    train_mask_dir = os.path.join(train_dir, 'mask')
    #img = os.path.join(train_mask_dir, "with_mask_300.jpg")

    #use transforms.CenterCrop(15), for OliverMaskDetection.ckpt
    img_transforms=transforms.Compose([transforms.CenterCrop(15), transforms.Resize((50,50)),
         transforms.ToTensor(),
         ])
    image_tensor = img_transforms(Image.open(img))
    image_tensor = image_tensor.unsqueeze_(0)


    model = CNN()
    model.load_state_dict(torch.load('OliverMaskDetection.ckpt'))

    #import torchvision.models as models
    #resnet = models.resnet18(pretrained=True)
    #resnet.fc = nn.Linear(512,2)
    #model = resnet
    #model.load_state_dict(torch.load('Resnet18MaskDetection.ckpt'))

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = output.max(1)
        if pred.sum().item() == 1:
            print("nomask")
        if pred.sum().item() == 0:
            print("mask")

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


cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
def make_420p():
    cam.set(3, 420)
    cam.set(4, 420)
make_420p()

def webcam():
    x = 0
    while(True):
        x+=1
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            pass# SPACE pressed
        if(x == 15):
            x=0
            img_name = "opencv_frame.jpg"
            cv2.imwrite(img_name, frame)
            t2 = Thread(target=detect_image(img_name))
            t2.start()

t1 = Thread(target=webcam())
t1.start()
cam.release()
cv2.destroyALLWindows()

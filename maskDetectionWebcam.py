import cv2 #pip install opencv-python
import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageGrab
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mss import mss
from threading import Thread
from facenet_pytorch import MTCNN

criterion = nn.CrossEntropyLoss()

data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((50, 50)), transforms.ToTensor()])

def detect_image(img):
    # scale and pad image

    import os
    base_dir = 'mask_nomask_large'
    train_dir = os.path.join(base_dir, 'train')
    train_mask_dir = os.path.join(train_dir, 'mask')
    #img = os.path.join(train_mask_dir, "with_mask_300.jpg")


    img_transforms=transforms.Compose([transforms.Resize((50,50)),
         transforms.ToTensor(),
         ])
    image_tensor = img_transforms(Image.open(img))
    image_tensor = image_tensor.unsqueeze_(0)


    model = CNN()
    model.load_state_dict(torch.load('OliverMaskDetection.ckpt'))
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


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                #cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture(0)
        x = 0
        while True:
            x += 1
            ret, frame = cap.read()
            try:
                # detect face box, probability and landmarks
                #landmarks true
                boxes, probs= self.mtcnn.detect(frame, landmarks=False)

                # draw on frame
                #self._draw(frame, boxes, probs, landmarks)

            except:
               pass
            try:
                if (x == 15):
                    x = 0
                    img_name = "opencv_frame.jpg"
                    max = 0
                    maxprob = probs[0]
                    for i  in range(0,len(probs)):
                        if probs[i] > maxprob:
                            maxprob = probs[i]
                            max = i
                    box = boxes[max]
                    im = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    cv2.imwrite(img_name, im)
                    t2 = Thread(target=detect_image(img_name))
                    t2.start()
            except:
               pass
            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the app
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()

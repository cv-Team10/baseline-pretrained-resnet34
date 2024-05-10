import torch
import torch.nn as nn
from torchvision import models


pretrained_resnet =  models.resnet34(pretrained = True)

pretrained_resnet.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 54), # 최총output: 54 (x1, y1, c1, ... , x18, y18, c18)
        nn.Sigmoid() #좌표, classificaion 모두 0~1 사이 값이므로
    )
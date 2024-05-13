import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models


pretrained_resnet =  models.resnet34(pretrained = True)

num_ftrs = pretrained_resnet.fc.in_features

pretrained_resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128), #layer 디자인 어떻게...??
        nn.ReLU()
    )

coordinate_head = nn.Linear(128, 34) #(x1, y1, ... , x17, y17)

classification_head = nn.Sequential(
    nn.Linear(128, 17), #(c1, ... , c17)
    nn.Sigmoid()
)


class MultiHeadResNet34(nn.Module):
    def __init__(self, pretrained_model, head1, head2):
        super(MultiHeadResNet34, self).__init__()
        self.pretrained_model = pretrained_model
        self.head1 = head1
        self.head2 = head2

    def forward(self, x):
        x = self.pretrained_model(x)
        output1 = self.head1(x)
        output2 = self.head2(x)
        return output1, output2

baseline_model = MultiHeadResNet34(pretrained_resnet, coordinate_head, classification_head)
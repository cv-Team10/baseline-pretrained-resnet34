import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from model import *


# 잘 작동되는지 보려고 샘플 트레이닝 데이터셋 만들어서 시험해봤음
# data_shape=(100,3,224,224)
# train_inputs = torch.rand(data_shape)
# coord_labels = torch.rand((100, 34))
# class_labels = torch.randint(0, 2, (100, 17)).float()

# dataset = TensorDataset(train_inputs, coord_labels, class_labels)

# batch_size = 20
# input_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 4)

# criterion_coord = nn.MSELoss()
# criterion_class = nn.BCELoss()
# optim = torch.optim.Adam(baseline_model.parameters(), lr = 0.0001)

# num_epochs = 5
# for epoch in range(num_epochs):
#   for inputs, coord_labels, class_labels in input_loader:
#     optim.zero_grad()
#     coord_outputs, class_outputs = baseline_model(inputs)

#     coord_loss = criterion_coord(coord_outputs, coord_labels)
#     class_loss = criterion_class(class_outputs, class_labels)
#     total_loss = coord_loss + class_loss

#     total_loss.backward()
#     optim.step()

#   print("epoch", epoch, "   Coord Loss:", coord_loss.item(), "  Class Loss:", class_loss.item())


torch.save(baseline_model.state_dict(), './baseline_resnet.pth')

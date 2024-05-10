import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from model import *


# 잘 작동되는지 보려고 샘플 트레이닝 데이터셋 만들어서 시험해봤음
# data_size = (100, 3, 224, 224) 
# label_size =(100, 54)
# train_datas = torch.rand(data_size)
# train_lables = torch.rand(label_size)

# class TestDataset(Dataset): 
#   def __init__(self, train_inputs, train_lables):
#     self.datas = train_inputs
#     self.lables = train_lables

#   def __len__(self): 
#     return len(self.datas)

#   def __getitem__(self, idx): 
#     x = torch.FloatTensor(self.datas[idx])
#     y = torch.FloatTensor(self.lables[idx])
#     return x, y

# dataset = TestDataset(train_datas, train_lables)
# batch_size = 20 
# train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4)



class MyLoss1(nn.Module):   
    def __init__(self):
        super(MyLoss1, self).__init__()

    def forward(self, predictions, lables):
        mse_loss = nn.MSELoss(reduction='mean')
        loss = 0

        predictions = predictions.view(-1, 18, 3) # output을 (batch_size, 18, 3) 형태로 재구성
        lables = lables.view(-1, 18, 3)

        for i in range(len(predictions)):
            for j in range(18):
                if predictions[i][j][2] > 0.5:  # classification score가 threshold를 넘는 경우
                    loss += mse_loss(predictions[i][j], lables[i][j])  # 모든 요소에 대한 MSE를 loss에 더함
                else:
                    loss += mse_loss(predictions[i][j][2], lables[i][j][2])  # classification에 대한 MSE만 loss에 더함
                
        return loss / len(predictions)
        

criterion1 = MyLoss1()  # classification이 0.5 이하면(keypoint가 사진에 없어야 한다고 판단하면) x좌표, y좌표 값에대한 error loss에 반영 안함(안더함)
criterion2 = nn.MSELoss() # keypoint가 없을 때 정답 label을 (0,0,0)으로 두는 경우 => predictin이 (0,0,0)이 되도록 학습하도록 MSE loss 사용
optim = torch.optim.Adam(pretrained_resnet.parameters(), lr = 0.0001)

# num_epoch = 10
# for epoch in range(num_epoch):
#   for i, data in enumerate(train_loader):
#     inputs, labels = data
#     optim.zero_grad()
#     predictions = pretrained_resnet(inputs)
#     loss = criterion1(predictions, labels)
#     loss.backward()
#     optim.step()
  
#   print("epoch: ", epoch, "loss: ", loss.item())


torch.save(pretrained_resnet.state_dict(), './baseline_resnet.pth')

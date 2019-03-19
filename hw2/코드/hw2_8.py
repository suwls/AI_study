# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        # 생성자에서 2개의 nn.linear모듈을 생성하고 멤버변수로 지정한다
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        # 순전파 함수에서는 입렵데이터를 받아서 출력데이터를 반환해야한다.
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size = 8)
model = TwoLayerNet(D_in, H, D_out)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss(size_average=False)

for epoch in range(20):
    for x_batch,y_batch in loader:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    plt.plot(epoch, loss.item(), "b.")
    plt.xlabel("$epoch$",fontsize = 18)
    plt.ylabel("$loss$",fontsize = 18)
        
plt.show()

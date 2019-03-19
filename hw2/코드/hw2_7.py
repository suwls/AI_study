# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

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
y = torch.randn(N, D_out, requires_grad = False)

model = TwoLayerNet(D_in, H, D_out)

# 손실함수와 optimizer를 만든다, SGD 생성자에서 model.parameters()를 호출하면
# 모델의 멤버인 2개의 nnLinear 모듈의 학습 가능한 매개변수들이 포함된다
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    # 순전파 단계 : 모델에 x를 전달하여 예상하는 y값을 계산한다.
    y_pred = model(x)
    
    # 손실을 계산한다.
    loss = criterion(y_pred, y)
    plt.plot(t, loss.data[0], "b.")
    plt.xlabel("$t$",fontsize = 18)
    plt.ylabel("$loss$",fontsize = 18)

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신한다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.show()


# (1) 매 t마다 y_pred, loss 변화를 화면 출력 확인 (plot)

# (2) 앞 문제의 코드와 비교

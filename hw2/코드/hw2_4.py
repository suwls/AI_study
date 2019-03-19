# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt

#N은 배치크기, D_in 은 입력의 차원
#H는 은닉 계층의 차원, D_out 은 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위로 입력과 출력 데이터를 생성한다
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 무작위로 가중치를 초기화한다.
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # 순전파 단계 : 예측값 y를 계산한다
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    # 손실을 계산하고 출력한다
    loss = (y_pred - y).pow(2).sum()
    
    plt.plot(t, loss, "b.")
    plt.xlabel("$t$",fontsize = 18)
    plt.ylabel("$loss$",fontsize = 18)
    
    print(t, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파 한다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 경사하강법을 사용하여 가중치를 갱신한다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
plt.show()
# (1) 매 t마다 y_pred에 따른 loss(accuracy) 변화를 화면 출력 확인 (plot)

# (2) 해당 학습이 적절히 진행되고 있는지 서술

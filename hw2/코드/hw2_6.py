# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

class MyReLU(torch.autograd.Function):
    @staticmethod
    #torch.autograd.Function를 상속받아 사용자 정의 autograd
    #tensor 연산을 하는 순전파와 역전파 단계를 구현함.
    def forward(ctx, x):
        # 순전파 단계에서는 입려을 갖는 tensor를 받아 출력 tensor를 반환해야한다.
        # ctx는 역전파 연산을 위한 정보를 저장하기 위해 사용하는 context object
        # ctx.save_for_backward method를 사용하여 역전파 단계에서 사용할 어떠한
        # 객체도 저장해 둘 수 있다.
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_y):
        # 역전파 단계에서는 출력에 대한 손실의 변화도를 갖는 tensor를 받고,
        # 입력에대한 손실의 변화도를 계산한다
        x, = ctx.saved_tensors
        grad_input =grad_y.clone()
        grad_input[x < 0] = 0
        return grad_input

def my_relu(x):
    return MyReLU.apply(x)

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    y_pred = my_relu(x.mm(w1)).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    plt.plot(t, loss.item(), "b.")
    plt.xlabel("$t$",fontsize = 18)
    plt.ylabel("$loss$",fontsize = 18)
    
    loss.backward()
    
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 가중치 갱신 후 수동으로 변화도를 0으로 만들어준다
        w1.grad.zero_()
        w2.grad.zero_()
plt.show()

# (1) 매 t마다 y_pred에 따른 loss(accyracy) 변화를 화면 출력 확인

# (2) 앞 문제의 코드와 비교






# -*- coding: utf -8 -*-

import torch

x = torch.randn(1, 10, requires_grad = True)
prev_h = torch.randn(1, 20, requires_grad = True)
w_h = torch.randn(20, 20, requires_grad = True)
w_x = torch.randn(20, 10, requires_grad = True)

i2h = torch.mm(w_x,x.t())
h2h = torch.mm(w_h, prev_h.t())
next_h = i2h + h2h
next_h = next_h.tanh()

# (1) 해당 신경망의 그래프 연산을 손으로 작성

loss = next_h.sum()
loss.backward()

# (2) loss 출력 확인

print("loss :"),
print(loss.item())


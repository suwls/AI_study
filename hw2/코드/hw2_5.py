# -*- coding: utf -8 -*-

import torch
import matplotlib.pyplot as plt

# N은 배치 크기이며, D_in은 입력의 차원이다
# H는 은닉 계층의 차원이며, D_out은 출력 차원이다
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 tensor를 생성한다.
# requires_grades=False로  설정하여 역전파 중에 이 값들에 대한 변화도를
# 계산할 필요가 없음을 나타낸다
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# requires_grades=True로 설정하여 역전파 중에 이 값들에 대한
# 변화도를 계산할 필요가 있음을 나타낸다.
w1 = torch.randn(D_in, H, requires_grad =True)
w2 = torch.randn(H, D_out, requires_grad =True)

# 원래 코드 학습률 (10e-6)을 1e-6으로 수정함
learning_rate = 1e-6
for t in range(500):
    # 순정파 단계 : y값을 예측한다
    # 이는 tensor를 사용한 순전파 단계와 완전히 동일하지만 역전파 단계를
    # 별도로 구현하지 않기위해 중간값들에 대한 참조를 갖고 있을 필요가 없다.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    # 손실을 계산한다.
    loss = (y_pred - y).pow(2).sum()

    plt.plot(t, loss.data[0], "b.")
    plt.xlabel("$t$",fontsize = 18)
    plt.ylabel("$loss$",fontsize = 18)

    # autograde를 사용하여 역전파 단계를 계산한다. 이것은 requires_grad = True를
    # 갖는 모든 값들에 대한 손실의 변화도를 계산한다. 이후 w1.grad와 w2.grad는
    # w1과 w2 각각에 대한 손실의 변화도를 갖게된다.
    loss.backward()

    # 경사하강법
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
plt.show()

# (1) 매 t마다 y_pred에 따른 loss(accuracy) 변화를 화면 출력 확인 (plot)

# (2) 앞 문제의 코드와 비교

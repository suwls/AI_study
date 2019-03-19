# -*- coding: utf -8 -*-

import numpy as np

# 0 근처의 난수를 생성
np.random.seed(0)

N, D = 3, 4

# 평균이 0 이고 분산이 1인 3행 4열짜리 난수를 생성
x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
c = np.sum(b)

# (1) 해당 연산망의 그래프 연산을 손으로 작성

# 행렬로 표현
grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a*y
grad_y = grad_a*grad_x

# (2) grad_c, grad_b, grad_a, grad_z, grad_x, grad_y 출력 확인

print("grad_c")
print(grad_c)
print("grad_b")
print(grad_b)
print("grad_a")
print(grad_a)
print("grad_z")
print(grad_z)
print("grad_x")
print(grad_x)
print("grad_y")
print(grad_y)

import torch
x = torch.randn(N, D, requires_grad = True)
y = torch.randn(N, D, requires_grad = True)
z = torch.randn(N, D)

a = x * y
b = a + z
c = torch.sum(b)
c.backward()
# (3) grad_x, grad_y 출력 확인
print("c.item")
print(c.item())
print("grad_x")
print(grad_x)
print("grad_y")
print(grad_y)

# 역전파를 통해 c에 대한 미분값을 연산한다.




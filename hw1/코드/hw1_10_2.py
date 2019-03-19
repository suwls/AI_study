# -*- coding: utf -8 -*-

eta = 0.1           # 학습률
n_iterations = 3    #반복 횟수

theta = [[1.0],[0.9]]

for iteration in range(n_iterations):
    a = theta[0][0]
    b = theta[1][0]
    gradients_x1 = 4*a + 3*b -4
    gradients_x2 = 3*a + 4*b +2
    theta[0][0] = theta[0][0] - eta * gradients_x1
    theta[1][0] = theta[1][0] - eta * gradients_x2
    print("x%d :"%(iteration+1)),
    print(theta)

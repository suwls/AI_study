# -*- coding: utf -8 -*-
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

# y가 예측값 , t 가 ans
def MSE(y, t):
    temp = 0
    for i in range(len(t)):
        temp += pow(y[i]-t[i] ,2)
    return 0.5 * temp

def cross_entropy_error(y,t):
    delta = 1e-7
    temp = 0
    for i in range(len(t)):
        temp += t[i] * np.log(y[i] + delta)
    return -temp

def loglikelihood(y, t):
    for i in range(len(t)):
        if t[i] == 1:
            break
    return -np.log2(y[i])

x = np.array([0.4, 2.0, 0.001, 0.32])

print("소프트맥스함수 적용 결과 : {}".format(softmax(x)))

prediction = [0.001, 0.9, 0.001, 0.098]
ans = [0, 0, 0, 1]

print("MSE")
print(MSE(prediction, ans))
print("교차엔트로피")
print(cross_entropy_error(prediction, ans))
print("로그우도")
print(loglikelihood(prediction,ans))




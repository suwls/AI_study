# -*- coding: utf -8 -*-

from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def logit(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# 도함수를 연산해주는 함수
def derivative(f, z, eps = 0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

z = np.linspace(-5, 5, 200)
plt.figure(figsize=(11,4))

plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth = 2, label = "step")
plt.plot(z, logit(z), "g--", linewidth = 2, label = "sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth = 2, label = "tanh")
plt.plot(z, relu(z), "m-", linewidth = 2, label = "ReLU")
plt.grid(True)
plt.legend(loc = "ecnter right", fontsize = 14)
plt.title("activation function: g(z)", fontsize = 14)
plt.axis([-5, 5, -1.2, 1.2])

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth = 2, label = "step")
plt.plot(0, 0, "ro", markersize = 5)
plt.plot(0, 0, "rx", markersize = 10)
plt.plot(z, derivative(logit,z), "g--", linewidth = 2, label = "sigmoid")
plt.plot(z, derivative(np.tanh,z), "b-", linewidth = 2, label = "tanh")
plt.plot(z, derivative(relu,z), "m-", linewidth = 2, label = "ReLU")
plt.grid(True)
plt.title("gradient: g'(z)", fontsize = 14)
plt.axis([5, 5, -0.2, 1.2])

plt.show()
# (1) 화면 출력 확인 및 각 활성함수의 특징을 비교 서술

# 로지스틱 시그모이드 , 탄젠트, 계딴, 렐류 그래프를 보고 차이를 기술하는 문제

# 렐류를 제외하고는 도함수가 0이 되서 오류를 감지못하는 경우가 있어 에러 검출을 위해 렐류를 많이 사용함


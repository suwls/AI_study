# -*- coding: utf -8 -*-
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

def cal_a(h,x):
    a = np.dot(W,h)
    a += np.dot(U,x)
    a += b
    return a

U = np.array([[0.1, 0.1],
              [0.0, 0.0],
              [0.0, -0.1]])
W = np.array([[0.1, 0.1, 0.0],
              [0.0, 0.0, 0.0],
              [0.2, -0.1, -0.1]])
V = np.array([[0.0, 0.1, 0.0],
              [-0.2, 0.0, 0.0]])
b = np.array([[0.0],
              [0.0],
              [0.2]])
c = np.array([[0.2],
              [0.1]])

h0 = np.array([[0],
               [0],
               [0]])

x = np.array([[[0.0],[1.0]],[[0.0],[0.1]], [[0.1],[-0.2]], [[0.5],[0.0]], [[0.1],[0.1]], [[0.1],[0.0]]])

# 각 과정별로 출력 추가 !

a1 =cal_a(h0, x[0])
h1 =np.tanh(a1)
y1 = softmax(np.dot(V,h1) +c)
print("a1")
print(a1)
print("h1")
print(h1)
print("y'(1)")
print(y1)
print("")

a2 = cal_a(h1,x[1])
h2 = np.tanh(a2)
y2 = softmax(np.dot(V,h2) +c)
print("a2")
print(a2)
print("h2")
print(h2)
print("y'(2)")
print(y2)
print("")

a3 = cal_a(h2,x[2])
h3 = np.tanh(a3)
y3 = softmax(np.dot(V,h3) +c)
print("a3")
print(a3)
print("h3")
print(h3)
print("y'(3)")
print(y3)
print("")

a4 = cal_a(h3,x[3])
h4 = np.tanh(a4)
y4 = softmax(np.dot(V,h4) +c)
print("a4")
print(a4)
print("h4")
print(h4)
print("y'(4)")
print(y4)
print("")

a5 = cal_a(h4,x[4])
h5 = np.tanh(a5)
y5 = softmax(np.dot(V,h5) +c)
print("y'(5)")
print(y5)
print("")

a6 = cal_a(h5,x[5])
h6 = np.tanh(a6)
y6 = softmax(np.dot(V,h6) +c)
print("y'(6)")
print(y6)





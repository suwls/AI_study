# -*- coding: utf -8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt

def logit(z):
    return 1/(1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


U1 =np.array([[0,0,0],[-0.3, 1.0, 1.2],[1.6, -1.0, -1.1]])
U2 =np.array([[0,0,0],[1.0, 1.0, -1.0],[0.7, 0.5, 1.0]])
U3 =np.array([[0,0,0],[0.5, -0.8,1.0],[-0.1, 0.3, 0.4]])
U4 =np.array([[1.0, 0.1, -0.2],[-0.2, 1.3, -0.4]])

x1 = np.array([1,1,0])

x2 = np.dot(U1, x1.T)

for i in range(3):
    #for j in range(3):
    x2[i] = logit(x2[i])

x3 = np.dot(U2, x2.T)
for i in range(3):
    # for j in range(3):
    x3[i] = logit(x3[i])

x4 = np.dot(U3, x3.T)
for i in range(3):
    #  for j in range(3):
    x4[i] = logit(x4[i])
x5 = np.dot(U4, x4.T)
for i in range(2):
    #for j in range(1):
    x5[i] = logit(x5[i])

print("(2) : {}".format(x5))
######################################3
x1 = np.array([1,1,0])

x2 = np.dot(U1, x1.T)

for i in range(3):
    #for j in range(3):
    x2[i] = relu(x2[i])

x3 = np.dot(U2, x2.T)
for i in range(3):
    # for j in range(3):
    x3[i] = relu(x3[i])

x4 = np.dot(U3, x3.T)
for i in range(3):
    #  for j in range(3):
    x4[i] = relu(x4[i])
x5 = np.dot(U4, x4.T)
for i in range(2):
    #for j in range(1):
    x5[i] = relu(x5[i])

print("(3) : {}".format(x5))


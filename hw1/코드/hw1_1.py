# -*- coding: utf -8 -*-
# 관련 라이브러리
import numpy as np
# %matplotlib inline

import matplotlib
import matplotlib.pyplot as plt


# 임의의 데이터 100개를 생성한다.
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)
plt.plot(X,y,"b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.axis([0, 2, 0, 15])
plt.show()

# (1) 화면 출력 확인
# 100개의 랜덤 선형 데이터셋을 생성한다.

### 정규 방정식을 사용한 선형회귀 접근 ###

# c_ 열추가, ones 1로 초기화된 데이터  100개 생성
# 샘플 X에 데이터 1 을 추가하여 X_b 데이터 셋을 만든다
X_b = np.c_[np.ones((100,1)), X]
# T => 전치,inv() => 역행렬, dot()=> 곱
# 정규 방정식에 대입
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# (2) theta_best 출력 확인
print("theta_best : ")
print (theta_best)
#print (theta_best)
# => theta_best[0] = 4, theta_best[1] = 3 을 기대했는데 노이즈 때문에
#    정확하게 원래 함수의 파라미터 재현 실패

# 구한 theta_best 와 새로운 X데이터로 y 모델 예측
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
# (3) y_predict 출력 확인
print("y_predict :")
print(y_predict)


plt.plot(X_new, y_predict, "r-", linewidth =2, label = "prediction")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.legend(loc = "upper left", fontsize = 14)
plt.axis([0, 2, 0, 15])
plt.show()
# (4) 화면 출력 확인

# 사이킷런을 이용한 선형회귀
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# (5) lin_reg.intercept_, lin_reg.coef_ 출력 확인
print ("\nlin_reg.intercept_ : %d, lin_reg.coef_ : %d\n"%(lin_reg.intercept_, lin_reg.coef_))

# (6) lin_reg.predict(X_new) 출력 확인
print("lin_reg.predict(X_new) :")
print(lin_reg.predict(X_new))

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond = 1e-6)
# (7) theta_best_svd 출력 확인


# 사이킷런을 이용한 예측값
print("\ntheta_best_svd :")
print(theta_best_svd)

# (8) np.linalg.pinv(X_b).dot(y) 출력 확인
# 유사 역행렬을 구하는 pinv() 함수를 이용하여 구한 예측값
print("\nnp.linalg.pinv(X_b).dot(y) :")
print(np.linalg.pinv(X_b).dot(y))



### 경사 하강법을 사용한 선형회귀 접근 ###
eta = 0.1    # 학습률
n_iterations = 1000         #반복 횟수
m = 100                     #입력받는 특징 수

theta = np.random.randn(2,1)  # theta를 무작위로 초기화한다



# 정해준 반복횟수 만큼 배치경사하강법을 수행한다
for iteration in range(n_iterations):
    gradients = 2.0/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

# (9) theta 출력 확인
print("\ntheta :")
print(theta)
# 정형방정식에서 찾은 값과 동일

# (10) X_new_b.dot(theta) 출력 확인
print("\nX_new_b.dot(theta)")
print(X_new_b.dot(theta))
# 정형방정식에서 찾은 값과 동일


theta_path_bgd = []

# 배치경사하강법
def plot_gradient_descent(theta, eta, theta_path = None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10 :
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        #손실함수의 편도함수 => 경사도를 구한다.
        gradients = 2.0/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize = 18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize = 16)

np.random.seed(42)
theta = np.random.randn(2,1)
plt.figure(figsize = (10, 4))
plt.subplot(131); plot_gradient_descent(theta, eta = 0.02)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.subplot(132); plot_gradient_descent(theta, eta = 0.1, theta_path = theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta = 0.5)
plt.show()
# (11) 화면 출력 확인


### 스토캐스틱(확률적) 경사 하강법을 사용한 선형회귀 접근 ###
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

n_epochs = 50
t0, t1 = 5.0, 50

def learning_schedule(t):
    return t0/(t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta)- yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.axis([0, 2, 0, 15])
plt.show()
# (12) 화면 출력 확인


# (13) theta 출력 확인
print("\ntheta :" )
print(theta)

# 사이킷런 SGD를 사용해 선형 회귀 모델 실행
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 50, penalty=None, eta0 = 0.1, random_state = 42)

# (14) sgd_reg.fit(X,y.ravel()) 출력 확인
print("\nsgd_reg.fit(X,y.ravel() :")
print (sgd_reg.fit(X,y.ravel()))

# (15) sgd_reg.intercept_, sgd_reg.coef_ 출력 확인
print("\nsgd_reg.intercept_ : %d, sgd_reg.coef_ : %d" %(sgd_reg.intercept_, sgd_reg.coef_))
# 정규 방정식이 리턴하는 값과 충분히 유사하다


### 미니배치 경사 하강법을 사용한 선형회귀 접근 ###
# sgd보다 덜 불규칙적이며 sgd보다는 좀 더 가까이 수렴한다.
# 그러나 좋지않은 지역적 최저점을 탈출하기 더 힘들어진다는 단점이 있다.

theta_path_mgd = []
n_iterations = 50
minibatch_size = 20
np.random.seed(42)
theta = np.random.randn(2,1)

t0, t1 = 200.0, 1000

def learning_schedule(t):
    return t0 / (t + t1)

t = 0

for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2.0/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# (16) theta 출력 확인
print("\ntheta :")
print(theta)
# 충분히 유사하다

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(7,4))
plt.plot(theta_path_sgd[:,0], theta_path_sgd[:,1], "r-s", linewidth=1, label = "SGD")
plt.plot(theta_path_mgd[:,0], theta_path_mgd[:,1], "g-+", linewidth =2, label = "MINI_BATCH")
plt.plot(theta_path_bgd[:,0], theta_path_bgd[:,1], "b-o", linewidth=3, label = "BATCH")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$     ",fontsize=20, rotation = 0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()
# (17) 화면 출력 확인

















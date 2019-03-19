# -*- coding: utf -8 -*-
# 관련 라이브러리
import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.pyplot as plt      

np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.axis([-3, 3, 0 , 10])
plt.show()
# (1) 화면 출력 확인
# 직선으로는 적절하게 데이터를 학습시킬 수 없을 것 같다


# 사이킷런의 PolynomialFeatures의 클래스를 사용해 학습 데이터 세트를 변환하고
# 새로운 특징값으로 학습 데이터 세트에 각각의 특징에 제곱을 추가한다.

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression     # pdf 이외 추가

poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly 는 기존 특징값에 그 특징값의 제곱값을 더한것
X_poly = poly_features.fit_transform(X)
# (2) X[0] 출력 확인
print ("\nX[0] :"),
print(X[0])

# (3) X_poly[0] 출력 확인
print("\nX_poly[0] :"),
print(X_poly[0])

# 이제 LinearRegression 학습 모델을 사용하여 학습시킨다.
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# (4) lin_reg.intercept_, lin_reg.coef_ 출력 확인
print ("\nlin_reg.intercept_ :"),
print(lin_reg.intercept_)
print("lin_reg.coef_ :"),
print(lin_reg.coef_)
# 잡음을 무시하면 얼추 비슷하다.

X_new = np.linspace(-3 , 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="prediction")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.legend(loc = "upper left", fontsize = 14)
plt.axis([-3, 3, 0, 10])
plt.show()
# (5) 화면 출력 확인

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+",2,1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
                    ("poly_features", polybig_features),
                    ("std_scaler",std_scaler),
                    ("lin_reg", lin_reg),
        ])
    polynomial_regression.fit(X,y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label = str(degree), linewidth=width)
plt.plot(X, y ,"b.", linewidth=3)
plt.legend(loc = "upper left")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$",rotation=0, fontsize=18)
plt.axis([-3, 3, 0 , 10])
plt.show()

# (6) 화면 출력 확인













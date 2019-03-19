# -*- coding: utf -8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 5)      # 1 * 6 * 5 * 5
        self.conv2 = nn.Conv2d(6, 16, 5)     # 6 * 16 * 5 * 5
        # fully connections 정의
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # 순전파 함수에 신경망의 구조를 정의
    def forward(self, x):
        # 컨볼루션 연산을 통해 특징 추출 후 풀링 연산하여 subsampling
        # 구조 conv1 - relu - pool - conv2 - relu -  pool - fc1 - relu - fc2 - relu - fc3
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x를 열의 개수가 self.num_flat_features(x)개이도록 재배열
        # -1 의 의미 : 행의 수를 알 수 없음
        # 1행 배열이 형성됨
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print("(1)")
print(net)
# (1) 화면 출력 확인 및 의미를 서술

# (2) 정의된 컨볼루션 신경망의 구조 설명 (위의 AlexNet 그림 참고)

params = list(net.parameters())
print("(3)")
print(len(params))
print(params[0].size())

## (3) 화면 출력 확인
#
input = torch.randn(1, 1, 32, 32)
out = net(input)
print("(4)")
print(out)
## (4) 화면 출력 확인

# 오류 역전파를 위해 가중치의 그래디언트 버퍼 초기화
net.zero_grad()
# 역전파
out.backward(torch.randn(1, 10))

# 손실 함수 정의 및 임의의 값들에 대해서 오차 결과 확인
# mse 손실함수 사용
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

# output = 예측값, target = 실제값
loss = criterion(output, target)
print("(5)")
print(loss)
## (5) 화면 출력 확인

net.zero_grad()

print("(6)")
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
# (6) 화면 출력 확인

loss.backward()

print("(7)")
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# (7) 화면 출력 확인

# 스토캐스틱 경사하강법(미래 가중치 = 현재가중치 - 학습률 * 그레이디언트)을 이용하여 가중치 갱신 코드
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# 오류역전파에서 최적화하는 방법
import torch.optim as optim

# torch.optim.SGD를 사용하여 가중치 갱신
optimizer = optim.SGD(net.parameters(), lr = 0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
# 최적화 과정을 수행한다.
optimizer.step()


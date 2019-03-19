# -*- coding: utf -8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

# torchvision 데이터셋의 출력(output)은 [0, 1] 범위를 갖는 PILImage 이미지이다.
# 이를 [-1, 1]의 범위로 정규화된 Tensor로 변환
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# torchvision을 이용해 훈련집합 적재
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)

# torchvision을 이용해 테스트집합 적재
testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# (1) 화면 출력 확인

import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수 정의
def imshow(img):
    img = img/ 2.0 + 0.5      #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()                # 커맨드라인에서 실행시 추가해주어야 함

# 훈련집합을 무작위로 가져온다
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 가져온 훈련집합을 보여준다
imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# (2) 화면 출력 확인

# 컨볼루션 신경망 정의
# 3채널 32 * 32 크기의 사진을 입력받고, 신경망을 통과해 10부류를 수정

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 커널 정의
        # 3채널을 입력받을 수 있도록 적의
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 풀링층 정의
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connections 정의
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # 순전파 함수에 신경망의 구조를 정의
    def forward(self, x):
        # 컨볼루션 연산을 통해 특징 추출 후 풀링 연산하여 subsampling
        # 구조 conv(conv1 - relu) - pool - conv(conv2 - relu) - pool - fc(fc1-relu) - fc(fc2-relu) - fc3
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim
import math

# 손실함수 정의. 교차 엔트로피와 SGD + momentum
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001, betas=(0.9, 0.999), eps =1e-3)

# 데이터를 반복해서 신경망에 입력으로 제공하고, 최적화(Optimize)
# 훈련집합을 이용하여 신경망 학슴시킴
for epoch in range(2):          # 데이터셋을 여러번 반복한다
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 데이터 입력
        inputs, labels = data
        # 그레이디언트 버퍼 초기화
        optimizer.zero_grad()
        
        # input에 대한 신경망의 예측값
        outputs = net(inputs)
        # 손실 계산
        loss = criterion(outputs, labels)
        # 오류 역전파
        loss.backward()
        # 가중치 갱신(최적화)
        optimizer.step()
        
        # 통계 출력
        running_loss += loss.item()
        if i % 1000 == 999:  # 모든 1000 미니배치들을 출력한다.
            print('[%d, %5d] loss : %.3f' %(epoch + 1, i + 1, running_loss/1000.0))
            running_loss = 0.0

print('Finished Training')
# (3) 화면 출력 확인 및 학습이 되고 있는지 서술

# 테스트 집합을 이용하여 학습시킨 신경망 성능 확인

# 테스트 집합을 무작위로 가져온다
dataiter = iter(testloader)
images, labels = dataiter.next()

# 가져온 테스트집합을 보여준다.
imshow(torchvision.utils.make_grid(images))

print('GroundTurth: '+''.join('%5s ' %classes[labels[j]] for j in range(4)))
# (4) 화면 출력 확인

# 출력은 10개 분류 각각에 대한 값으로 나타난다. 어떤 분류에 대해서 더 높은 값이 나타난다는 것은,
# 신경망이 그 이미지가 더 해당 분류에 가깝다고 예측했다는 것이다.
# 따라서, 가장 높은 값을 갖는 인덱스(index)를 결과 예측값으로 뽑는다.
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: '+''.join('%5s ' %classes[predicted[j]] for j in range(4)))
# (5) 화면 출력 확인

# 전체 테스트집합에 적용하여 일반화 성능을 측정한다.
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %(100 * correct / total))
# (5) 화면 출력 확인 및 일반화 성능 서술

# 각 분류에 대한 일반화 성능 평가
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %(classes[i], 100 * class_correct[i]/class_total[i]))
# (7) 화면 출력 확인 및 부류별 분류기의 성능 서술

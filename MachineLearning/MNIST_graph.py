import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets

import matplotlib
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)

        return x5

train_data = datasets.MNIST(root="MNIST_data/", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root="MNIST_data/", train=False, transform=torchvision.transforms.ToTensor(), download=True)

batch_size = 800
test_batch_size = 10000
epochs = 10
lr = 0.01
momentum = 0.9

data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_function = nn.CrossEntropyLoss()

# loss 값과 정확도에 대한 진행 상황을 저장할 변수 설정
history = {'val_loss': [],
           'val_acc': []}

def learning():
    for e in range(epochs):

        for data, target in data_loader:
            data = data.view(-1, 784)
            pred = model.forward(data)
            target = target

            optimizer.zero_grad()
            loss = loss_function(pred, target)
            loss.backward()
            optimizer.step()

        print("TRAING EPOCHS: " + str(e) + "\nLOSS: " + str(loss.data.numpy()))

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(test_batch_size, -1)
                target = target
                output = model.forward(data)
                test_loss += loss_function(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('Accuracy: {}/{} ({:.0f}%)\n'.format
              (correct, len(test_loader.dataset),
               100. * correct / len(test_loader.dataset)))

        # loss값과 정확도 진행 상황 저장
        history['val_loss'].append(loss.data.numpy())
        history['val_acc'].append(100. * correct / len(test_loader.dataset))

def loss_epoch():
    ## 폰트 설정
    font_name = matplotlib.font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    matplotlib.rc('font', family=font_name)
    ## 도화지 생성
    fig = plt.figure()
    ## Loss vs 학습량 그래프 그리기
    plt.plot(range(epochs), history['val_loss'], label='Loss', color='darkred')
    ## 축 이름
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs 학습량 그래프')
    plt.grid(linestyle='--', color='lavender')

def acc_epoch():
    ## 폰트 설정
    font_name = matplotlib.font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    matplotlib.rc('font', family=font_name)
    ## 도화지 생성
    fig = plt.figure()
    ## 정확도 vs 학습량 그래프 그리기
    plt.plot(range(epochs), history['val_acc'], label='Acc', color='darkred')
    ## 축 이름
    plt.xlabel('epochs')
    plt.ylabel('Acc')
    plt.title('정확도 vs 학습량 그래프')
    plt.grid(linestyle='--', color='lavender')

#Subject 2 : 지난 주에 만들었던 인공신경망의 Loss vs 학습량(Epoch를 기준으로) 그리기
print("Subject(2)")
learning()
loss_epoch()

## 그래프 표시
plt.savefig('./subject2_loss+epoch.png')
plt.show()

#Subject 2 : 지난 주에 만들었던 인공신경망의 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()

## 그래프 표시
plt.savefig('./subject2_acc+epoch.png')
plt.show()



# Subject 3 : 위 주제 중 1개를 반영하여 인공신경망의 Loss vs 학습량(Epoch를 기준으로), 정확도 vs 학습량(Epoch를 기준으로)을 그립니다.
# 3) Learning Rate 의 값은 새로운 값 3개이상으로 학습하기
lr = [0.001, 0.005, 0.0001]

# loss = 0.001 일 때
print("Subject(3) - 1")

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr[0], momentum=momentum)
loss_function = nn.CrossEntropyLoss()


history = {'val_loss': [],
           'val_acc': []}

learning()

#Subject 3-1 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject3-1_loss+epoch.png')
plt.show()

    #Subject 3-1 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject3-1_acc+epoch.png')
plt.show()



# loss = 0.005 일 때
print("Subject(3) - 2")

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr[1], momentum=momentum)
loss_function = nn.CrossEntropyLoss()


history = {'val_loss': [],
           'val_acc': []}

learning()

#Subject 3-2 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject3-2_loss+epoch.png')
plt.show()

#Subject 3-2 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject3-2_acc+epoch.png')
plt.show()



# loss = 0.0001 일 때
print("Subject(3) - 3")

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr[2], momentum=momentum)
loss_function = nn.CrossEntropyLoss()


history = {'val_loss': [],
           'val_acc': []}

learning()
#Subject 3-3 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject3-3_loss+epoch.png')
plt.show()

#Subject 3-3 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject3-3_acc+epoch.png')
plt.show()


# Subject 4 : 위 주제 중 다른 1개를 반영하여 인공신경망의 Loss vs 학습량, 정확도 vs 학습량을 그립니다.
# 6) 초기화의 경우 서로 다른 초기화 값 설정 사용하기 : SGD 대신 ADAM 사용
batch_size = 800
test_batch_size = 10000
epochs = 10
lr = 0.01
momentum = 0.9

model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

history = {'val_loss': [],
           'val_acc': []}

print("Subject 4")

learning()

#Subject 4 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject4_loss+epoch.png')
plt.show()

#Subject 4 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject4_acc+epoch.png')
plt.show()



# Subject 5 : 위 주제 중 2개를 반영하여 인공신경망의 Loss vs 학습량(Epoch를 기준으로), 정확도 vs 학습량(Epoch를 기준으로)을 그립니다.
# 3) Learning Rate 의 값은 새로운 값 3개이상으로 학습하기
# 6) 초기화의 경우 서로 다른 초기화 값 설정 사용하기 = SGD 대신 ADAM 사용
lr = [0.001, 0.005, 0.0001]

## Subject 5-1 : lr = 0.001일 때
model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr[0])
loss_function = nn.CrossEntropyLoss()

history = {'val_loss': [],
           'val_acc': []}

print("Subject(5)-1")

learning()

##Subject 5-1 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject5-1_loss+epoch.png')
plt.show()

##Subject 5-1 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject5-1_acc+epoch.png')
plt.show()


## Subject 5-2 : lr = 0.005일 때
model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr[1])
loss_function = nn.CrossEntropyLoss()

history = {'val_loss': [],
           'val_acc': []}

print("Subject(5)-2")

learning()

##Subject 5-2 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject5-2_loss+epoch.png')
plt.show()

##Subject 5-2 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject5-2_acc+epoch.png')
plt.show()


## Subject 5-3 : lr = 0.0001일 때
model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr[2])
loss_function = nn.CrossEntropyLoss()

history = {'val_loss': [],
           'val_acc': []}

print("Subject(5)-3")

learning()

##Subject 5-3 :  Loss vs 학습량(Epoch를 기준으로) 그리기
loss_epoch()
## 그래프 표시
plt.savefig('./subject5-3_loss+epoch.png')
plt.show()

##Subject 5-3 : 정확도 vs 학습량(Epoch를 기준으로)을 그리기
acc_epoch()
## 그래프 표시
plt.savefig('./subject5-3_acc+epoch.png')
plt.show()
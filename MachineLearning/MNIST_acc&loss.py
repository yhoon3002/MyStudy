import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets

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

learning()
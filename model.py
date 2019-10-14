import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch_sparse as ts
from torchvision import transforms
import time

cifar10_root = "/scratch/bwasti/cifar10"
cifar10_train = torchvision.datasets.CIFAR10(cifar10_root, download=True, transform=transforms.ToTensor(), train=True)
cifar10_test  = torchvision.datasets.CIFAR10(cifar10_root, download=True, transform=transforms.ToTensor(), train=False)

mnist_root = "/scratch/bwasti/mnist"
mnist_train = torchvision.datasets.MNIST(mnist_root, download=True, transform=transforms.ToTensor(), train=True)
mnist_test  = torchvision.datasets.MNIST(mnist_root, download=True, transform=transforms.ToTensor(), train=False)

#train_loader = torch.utils.data.DataLoader(cifar10_train)
#test_loader = torch.utils.data.DataLoader(cifar10_test)
train_loader = torch.utils.data.DataLoader(mnist_train)
test_loader = torch.utils.data.DataLoader(mnist_test)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class SparseNet(nn.Module):
    def __init__(self):
        super(SparseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = ts.SparseLinear(4*4*50, 512, 0.50, 16)
        #self.fc1 = nn.Linear(4*4*50, 512)
        self.fc2 = ts.SparseLinear(512, 16, 0.50, 16)
        #self.fc2 = nn.Linear(512, 16)
        #self.ident = ts.SparseLinear(16, 16, 0.0, 8)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.ident(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, device):
  model.train()
  mu = 0.25
  avg_loss=1
  avg_time=0
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    t = time.time()
    output = model(data)
    loss = F.nll_loss(output, target)
    avg_loss = mu*loss.item() + (1-mu)*avg_loss
    loss.backward()
    optimizer.step()
    avg_time = mu*(time.time() - t)*1e3 + (1-mu)*avg_time
  print(f"loss: {avg_loss} time: {avg_time}ms per iter")

device = torch.device('cuda')

model = SparseNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
epochs = 1
for _ in range(epochs):
  train(model, optimizer, device)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
epochs = 1
for _ in range(epochs):
  train(model, optimizer, device)




# %%
from PyHessian.pyhessian.hessian import hessian
from PyHessian.density_plot import get_esd_plot
from hessian_utils import *
from PyHessian.pyhessian.utils import *
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
# from keras.datasets import mnist
import matplotlib.pyplot as plt 
import time
import copy


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# %matplotlib inline

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
cuda = torch.cuda.is_available()
torch.autograd.set_detect_anomaly(True)

# %%
mnist_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.ToTensor())
mnist_fashion_trainset = datasets.FashionMNIST(root='./data/fashion_mnist', train=True, download=True, transform=transforms.ToTensor())
cifar_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transforms.ToTensor())
# Getting mnist test data
mnist_testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transforms.ToTensor())
mnist_fashion_testset = datasets.FashionMNIST(root='./data/fashion_mnist', train=False, download=True, transform=transforms.ToTensor())
cifar_testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transforms.ToTensor())

torch.manual_seed(42)

# %%
train_X = mnist_trainset.data
train_y = mnist_trainset.targets
test_X = mnist_testset.data
test_y = mnist_testset.targets

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Sequential(
            nn.Linear(16*7*7, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16*7*7)
        x = self.linear(x)
        return x
    

# def nodewise_trace(model, hessian : hessian):
#     loader = hessian
    

# %%
train_X = (train_X / 255.0).to(device)
test_X = (test_X / 255.0).to(device)
train_y = torch.tensor(train_y, dtype=torch.int64).to(device)
test_y = torch.tensor(test_y, dtype=torch.int64).to(device)

# %%
# For convnet, change dimensions of input
# train_X = train_X.unsqueeze(1)
# test_X = test_X.unsqueeze(1)

# %%
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_X[:1000], train_y[:1000]),
    batch_size=32, shuffle=True)
hessian_loader = []
# hessian_loader contains all 1000 images

for i in range(50):
    hessian_loader.append((train_X[i].unsqueeze(0), train_y[i].unsqueeze(0)))

# %%
model_SGD = Net().to(device)
model_Adam = copy.deepcopy(model_SGD)
# model_randnewt = copy.deepcopy(model_SGD)
criterion = nn.CrossEntropyLoss()
optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=1e-2)
optimizer_Adam = torch.optim.Adam(model_Adam.parameters(), lr=1e-3)

# %%
train_losses_SGD = []
train_losses_Adam = []
traces_SGD = []
traces_Adam = []
epochs = 100
k = 100

# hessian_sgd = hessian(model_SGD, criterion, dataloader=hessian_loader, cuda=cuda)
# hessian_adam = hessian(model_Adam, criterion, dataloader=hessian_loader, cuda=cuda)
# traces_SGD.append(np.mean(hessian_sgd.trace()))
# traces_Adam.append(np.mean(hessian_adam.trace()))

# %%
# hessian_sgd = hessian(model_SGD, criterion, dataloader=hessian_loader, cuda=cuda)
# hessian_adam = hessian(model_Adam, criterion, dataloader=hessian_loader, cuda=cuda)
# get_esd_plot(*hessian_sgd.density(), title='SGD')
# get_esd_plot(*hessian_adam.density(), title='Adam')
# plt.show()

# %%
layerwise_trace_SGD = []
layerwise_trace_Adam = []
parameter_trace_SGD = []
parameter_trace_Adam = []
full_trace_SGD = []
full_trace_Adam = []
lenarr = [params.numel() for params in model_SGD.parameters()]
layer_names = [str(key) for key in model_SGD.state_dict().keys()]
print(layer_names)

# %%
for epoch in range(epochs):
    model_SGD.train()
    model_Adam.train()
    # model_randnewt.train()
    hessian_sgd = hessian(model_SGD, criterion, dataloader=hessian_loader, cuda=cuda)
    hessian_adam = hessian(model_Adam, criterion, dataloader=hessian_loader, cuda=cuda)
    
    layer_trace_SGD = hessian_sgd.layer_wise_trace(maxIter=5)
    mean_layer_trace = [np.mean(layer_trace_SGD[i]) for i in range(len(layer_trace_SGD))]
    parameter_trace_SGD.append([mean_layer_trace[i]/lenarr[i] for i in range(len(lenarr))])
    layerwise_trace_SGD.append(mean_layer_trace)
    full_trace_SGD.append(sum(mean_layer_trace))

    layer_trace_Adam = hessian_adam.layer_wise_trace(maxIter=5)
    mean_layer_trace = [np.mean(layer_trace_Adam[i]) for i in range(len(layer_trace_Adam))]
    parameter_trace_Adam.append([mean_layer_trace[i]/lenarr[i] for i in range(len(lenarr))])
    layerwise_trace_Adam.append(mean_layer_trace)
    full_trace_Adam.append(sum(mean_layer_trace))

    for i, (x, y) in enumerate(dataloader):
        optimizer_SGD.zero_grad()
        optimizer_Adam.zero_grad()
        #optimizer_randnewt.zero_grad()
        y_pred_SGD = model_SGD(x)
        y_pred_Adam = model_Adam(x)
        loss_SGD = criterion(y_pred_SGD, y)
        loss_Adam = criterion(y_pred_Adam, y)
        loss_SGD.backward()
        loss_Adam.backward()
        # optimizer_SGD.step()
        # optimizer_Adam.step()
        # Divide the gradient by the mean trace of the layer the parameter is present in.
        for name, param in model_SGD.named_parameters():
            if name in layer_names:
                layer_index = layer_names.index(name)
                mean_trace = np.mean(layerwise_trace_SGD[-1][layer_index])
                param.grad /= (mean_trace + 1e-8)
                
        optimizer_SGD.step()
        optimizer_Adam.step()
        train_losses_SGD.append(loss_SGD.item())
        train_losses_Adam.append(loss_Adam.item())
        print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, Loss_SGD: {loss_SGD.item()}, Loss_Adam: {loss_Adam.item()}")

    # plt.plot(layerwise_trace_SGD[-1], label=f'SGD epoch {epoch+1} trace {round(full_trace_SGD[-1], 2)} loss {round(train_losses_SGD[-1], 3)}')
    # plt.plot(layerwise_trace_Adam[-1], label=f'Adam epoch {epoch+1} trace {round(full_trace_Adam[-1], 2)} loss {round(train_losses_Adam[-1], 3)}')
    # plt.title(f'Layerwise Trace for SGD and Adam epoch {epoch+1}')
    # plt.xticks(range(len(layer_names)), layer_names)
    # plt.legend()
    # plt.xlabel('Layer names')
    # plt.ylabel('Trace')
    # plt.show()

    # plt.plot(parameter_trace_SGD[-1], label=f'SGD epoch {epoch+1} paratrace {round(full_trace_SGD[-1], 2)} loss {round(train_losses_SGD[-1], 3)}')
    # plt.plot(parameter_trace_Adam[-1], label=f'Adam epoch {epoch+1} paratrace {round(full_trace_Adam[-1], 2)} loss {round(train_losses_Adam[-1], 3)}')
    # plt.title(f'Parameterwise Trace for SGD and Adam epoch {epoch+1}')
    # plt.xticks(range(len(layer_names)), layer_names)
    # plt.legend()
    # plt.xlabel('Layer names')
    # plt.ylabel('paratrace')
    # plt.show()

    # # Plotting evolution of layerwise trace for SGD and Adam
    # for i in range(len(layer_names)):
    #     plt.plot([layerwise_trace_SGD[j][i] for j in range(len(layerwise_trace_SGD))], label=f'SGD layer {i} {layer_names[i]}')
    #     plt.plot([layerwise_trace_Adam[j][i] for j in range(len(layerwise_trace_Adam))], label=f'Adam layer {i} {layer_names[i]}')
    #     plt.title(f'Evolution of Layerwise Trace for SGD and Adam layer {i} {layer_names[i]}')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Trace')
    #     plt.legend()
    #     plt.show()

    



    
    # hessian_randnewt = hessian(model_randnewt, criterion, dataloader=hessian_loader, cuda=cuda)
    # traces_SGD.append(np.mean(hessian_sgd.trace()))
    # traces_Adam.append(np.mean(hessian_adam.trace()))
    #print(f"Epoch {epoch+1}/{epochs}, Trace_SGD: {traces_SGD[-1]}, Trace_Adam: {traces_Adam[-1]}")

# %%
# for i in range(len(layer_names)):
#         plt.plot([layerwise_trace_SGD[j][i] for j in range(len(layerwise_trace_SGD))], label=f'SGD layer {i} {layer_names[i]}')
#         plt.plot([layerwise_trace_Adam[j][i] for j in range(len(layerwise_trace_Adam))], label=f'Adam layer {i} {layer_names[i]}')
#         plt.title(f'Evolution of Layerwise Trace for SGD and Adam layer {i} {layer_names[i]}')
#         plt.xlabel('Epochs')
#         plt.ylabel('Trace')
#         plt.legend()
#         plt.show()

# %%
# np.save('layerwise_trace_SGD.npy', np.array(layerwise_trace_SGD))
# np.save('layerwise_trace_Adam.npy', np.array(layerwise_trace_Adam))
# np.save('parameter_trace_SGD.npy', np.array(parameter_trace_SGD))
# np.save('parameter_trace_Adam.npy', np.array(parameter_trace_Adam))

# %%
# layerwise_trace_SGD = np.load('layerwise_trace_SGD.npy')
# layerwise_trace_Adam = np.load('layerwise_trace_Adam.npy')
# parameter_trace_SGD = np.load('parameter_trace_SGD.npy')
# parameter_trace_Adam = np.load('parameter_trace_Adam.npy')



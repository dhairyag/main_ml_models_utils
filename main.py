'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from torchvision import datasets

import os

from models import *

from utils import progress_bar
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    RandomCutout(n_holes=1, length=16),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=512, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=False, num_workers=2)
#,border_mode=cv2.BORDER_CONSTANT, value=dataset_mean
'''
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Define mean of your dataset
dataset_mean = (255*0.49139968, 255*0.48215827 ,255*0.44653124) 

# Define Albumentations transformations
train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40),  # Pad to (40, 40) if necessary
    A.RandomCrop(height=32, width=32, p=1),  # RandomCrop of size (32, 32)
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                    fill_value=dataset_mean, p=0.5),  
    A.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768)),
    ToTensorV2(),  # Convert to PyTorch tensor
])


test_transforms = A.Compose([
    #A.ToFloat(max_value=255.0),  # Convert images to float format
    A.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768), always_apply=True),
    ToTensorV2(),
])
# load data sets for training and testing

# albu
class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

train_dataset = Cifar10SearchDataset(root='../data', train=True, download=True, transform=train_transforms)
test_dataset = Cifar10SearchDataset(root='../data', train=False, download=True, transform=test_transforms)
# batch size is the number of samples processed before the model is updated
batch_size = 1024

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

testloader = torch.utils.data.DataLoader(test_dataset, **kwargs)
trainloader = torch.utils.data.DataLoader(train_dataset, **kwargs)
##############################

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Model
def Model(t_name='resnet'):
    print('==> Building model..')
    global optimizer, scheduler
    # net = VGG('VGG19')
    if t_name == 'resnet':
        net = ResNet18()
    
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    return net


criterion = nn.CrossEntropyLoss()


train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def Train(model):
    for epoch in range(1, 20+1):
        print(f'Epoch {epoch}')
        train(model, device, trainloader, optimizer, criterion)
        test(model, device, testloader, criterion)
        scheduler.step()

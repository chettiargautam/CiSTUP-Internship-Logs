---
description: Benchmarking of SimCLR and NNCLR on MNIST, FashionMNIST, and CIFAR-10 Datasets
---

# Benchmarking

## Implementation of SimCLR

```python
import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import EarlyStopping, MetricMonitor
from util import TwoCropTransform, SupCon, SupConLoss, save_model
```

```python
class Encoder(torch.nn.Module):
    "Encoder network"
    def __init__(self):
        super(Encoder, self).__init__()
        # L1 (?, 28, 28, 1) -> (?, 28, 28, 32) -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
            )
        # L2 (?, 14, 14, 32) -> (?, 14, 14, 64) -> (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
            )
        # L3 (?, 7, 7, 64) -> (?, 7, 7, 128) -> (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.2)
            )
        self._to_linear = 4 * 4 * 128

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten them for FC
        return x
```

```python
class LinearClassifier(torch.nn.Module):
    """Linear classifier"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 10),
            )

    def forward(self, x):
        x = self.fc(x)
        probs = torch.nn.functional.softmax(x, dim=0)
        return probs
```

```python
def calculate_accuracy(output, target):
    "Calculates accuracy"
    output = output.data.max(dim=1,keepdim=True)[1]
    output = output == 1.0
    output = torch.flatten(output)
    target = target == 1.0
    target = torch.flatten(target)
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item() 
```

```python
def pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR'):
    "Contrastive pre-training over an epoch"
    metric_monitor = MetricMonitor()
    model.train()
    for batch_idx, (data,labels) in enumerate(contrastive_loader):
        data = torch.cat([data[0], data[1]], dim=0)
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        bsz = labels.shape[0]
        features = model(data)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if method == 'SupCon':
            loss = criterion(features, labels)
        elif method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.format(method))
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg']
```

```python
def training(epoch, model, classifier, train_loader, optimizer, criterion):
    "Training over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    classifier.train()
    for batch_idx, (data,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        with torch.no_grad():
            features = model.encoder(data)
        output = classifier(features.float())
        loss = criterion(output, labels) 
        accuracy = calculate_accuracy(output, labels)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        data.detach()
        labels.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Train      | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']
```

```python
def validation(epoch, model, classifier, valid_loader, criterion):
    "Validation over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for batch_idx, (data,labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data,labels = data.cuda(), labels.cuda()
            data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
            features = model.encoder(data)
            output = classifier(features.float())
            loss = criterion(output,labels) 
            accuracy = calculate_accuracy(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            data.detach()
            labels.detach()
    print("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']
```

```python
from tqdm.auto import tqdm
```

```python
def main():
    
    num_epochs = 50
    use_early_stopping = True
    use_scheduler = True
    head_type = 'mlp' # choose among 'mlp' and 'linear"
    method = 'SimCLR' # choose among 'SimCLR' and 'SupCon'
    save_file = os.path.join('./results/', 'model.pth')
    if not os.path.isdir('./results/'):
         os.makedirs('./results/')
    
    contrastive_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    train_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    valid_transform = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])

    print("Downloading dataset...")
    
    contrastive_set = datasets.MNIST('./data', download=True, train=True, transform=TwoCropTransform(contrastive_transform))
    train_set = datasets.MNIST('./data', download=True, train=True, transform=train_transform)
    valid_set = datasets.MNIST('./data', download=True, train=False, transform=valid_transform)

    batch_size = 64
    
    contrastive_loader = torch.utils.data.DataLoader(contrastive_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    
    print("Download finished.")

    # Part 1
    encoder = Encoder()
    model = SupCon(encoder, head=head_type, feat_dim=128)
    criterion = SupConLoss(temperature=0.07)
    if torch.cuda.is_available():
        print("CUDA Enabled")
        model = model.cuda()
        criterion = criterion.cuda()   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    contrastive_loss, contrastive_lr = [], []
    
    for epoch in tqdm(range(1, num_epochs+1)):
        loss, lr = pretraining(epoch, model, contrastive_loader, optimizer, criterion, method=method)
        if use_scheduler:
            scheduler.step()
        contrastive_loss.append(loss)
        contrastive_lr.append(lr)
    
    save_model(model, optimizer, num_epochs, save_file)
    
    plt.plot(range(1,len(contrastive_lr)+1),contrastive_lr, color='b', label = 'learning rate')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Learning Rate'), plt.show()
    
    plt.plot(range(1,len(contrastive_loss)+1),contrastive_loss, color='b', label = 'loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()
    
    # Part 2
    model = SupCon(encoder, head=head_type, feat_dim=128)
    classifier = LinearClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    
    ckpt = torch.load(save_file, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    train_losses , train_accuracies = [],[]
    valid_losses , valid_accuracies = [],[]
    
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=30, verbose=False, delta=1e-4)
 
    for epoch in range(1, num_epochs+1):
        
        train_loss, train_accuracy = training(epoch, model, classifier, train_loader, optimizer, criterion)
        valid_loss, valid_accuracy = validation(epoch, model, classifier, valid_loader, criterion)
        
        if use_scheduler:
            scheduler.step()
            
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
             
        if use_early_stopping: 
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print('Early stopping at epoch', epoch)
                #model.load_state_dict(torch.load('checkpoint.pt'))
                break
     
    plt.plot(range(1,len(train_losses)+1), train_losses, color='b', label = 'training loss')
    plt.plot(range(1,len(valid_losses)+1), valid_losses, color='r', linestyle='dashed', label = 'validation loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()
     
    plt.plot(range(1,len(train_accuracies)+1),train_accuracies, color='b', label = 'training accuracy')
    plt.plot(range(1,len(valid_accuracies)+1),valid_accuracies, color='r', linestyle='dashed', label = 'validation accuracy')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Accuracy'), plt.show()
```

## MNIST

\[Epoch: 001] Train | Loss: 2.3025 | Accuracy: 0.8699 \[Epoch: 001] Validation | Loss: 2.3025 | Accuracy: 0.8832 \[Epoch: 002] Train | Loss: 2.3026 | Accuracy: 0.8858 \[Epoch: 002] Validation | Loss: 2.3025 | Accuracy: 0.8995

> early stopping counter: 1 out of 30 \[Epoch: 003] Train | Loss: 2.3026 | Accuracy: 0.8982 \[Epoch: 003] Validation | Loss: 2.3026 | Accuracy: 0.8931 early stopping counter: 2 out of 30 \[Epoch: 004] Train | Loss: 2.3026 | Accuracy: 0.8752 \[Epoch: 004] Validation | Loss: 2.3026 | Accuracy: 0.8452 early stopping counter: 3 out of 30 \[Epoch: 005] Train | Loss: 2.3026 | Accuracy: 0.8531 \[Epoch: 005] Validation | Loss: 2.3026 | Accuracy: 0.8017 early stopping counter: 4 out of 30 \[Epoch: 006] Train | Loss: 2.3026 | Accuracy: 0.8278 \[Epoch: 006] Validation | Loss: 2.3026 | Accuracy: 0.7916 early stopping counter: 5 out of 30 \[Epoch: 007] Train | Loss: 2.3026 | Accuracy: 0.8260 \[Epoch: 007] Validation | Loss: 2.3026 | Accuracy: 0.8016 early stopping counter: 6 out of 30 \[Epoch: 008] Train | Loss: 2.3026 | Accuracy: 0.8227 \[Epoch: 008] Validation | Loss: 2.3026 | Accuracy: 0.7667 early stopping counter: 7 out of 30 \[Epoch: 009] Train | Loss: 2.3026 | Accuracy: 0.8324 \[Epoch: 009] Validation | Loss: 2.3026 | Accuracy: 0.8867 ... \[Epoch: 031] Train | Loss: 2.3026 | Accuracy: 0.8876 \[Epoch: 031] Validation | Loss: 2.3026 | Accuracy: 0.8867 early stopping counter: 30 out of 30
>
> ## FashionMNIST
>
> \[Epoch: 001] Train | Loss: 2.3026 | Accuracy: 0.8836 \[Epoch: 001] Validation | Loss: 2.3026 | Accuracy: 0.8827 \[Epoch: 002] Train | Loss: 2.3026 | Accuracy: 0.8835 \[Epoch: 002] Validation | Loss: 2.3026 | Accuracy: 0.8652
>
> > early stopping counter: 1 out of 30 \[Epoch: 003] Train | Loss: 2.3026 | Accuracy: 0.8510 \[Epoch: 003] Validation | Loss: 2.3026 | Accuracy: 0.8532 early stopping counter: 2 out of 30 \[Epoch: 004] Train | Loss: 2.3026 | Accuracy: 0.8588 \[Epoch: 004] Validation | Loss: 2.3026 | Accuracy: 0.8626 early stopping counter: 3 out of 30 \[Epoch: 005] Train | Loss: 2.3026 | Accuracy: 0.8664 \[Epoch: 005] Validation | Loss: 2.3026 | Accuracy: 0.8520 early stopping counter: 4 out of 30 \[Epoch: 006] Train | Loss: 2.3026 | Accuracy: 0.8499 \[Epoch: 006] Validation | Loss: 2.3026 | Accuracy: 0.8468 early stopping counter: 5 out of 30 \[Epoch: 007] Train | Loss: 2.3026 | Accuracy: 0.8480 \[Epoch: 007] Validation | Loss: 2.3026 | Accuracy: 0.8541 early stopping counter: 6 out of 30 \[Epoch: 008] Train | Loss: 2.3026 | Accuracy: 0.8194 \[Epoch: 008] Validation | Loss: 2.3026 | Accuracy: 0.7485 early stopping counter: 7 out of 30 \[Epoch: 009] Train | Loss: 2.3026 | Accuracy: 0.8880 \[Epoch: 009] Validation | Loss: 2.3026 | Accuracy: 0.8996 ... \[Epoch: 031] Train | Loss: 2.3026 | Accuracy: 0.9000 \[Epoch: 031] Validation | Loss: 2.3026 | Accuracy: 0.9005 early stopping counter: 30 out of 30
> >
> > ## CIFAR-10
> >
> > \[Epoch: 001] Contrastive Pre-train | Loss: 2.6929 | Learning Rate: 0.0010 \[Epoch: 002] Contrastive Pre-train | Loss: 1.8169 | Learning Rate: 0.0010 \[Epoch: 003] Contrastive Pre-train | Loss: 1.5628 | Learning Rate: 0.0010 \[Epoch: 004] Contrastive Pre-train | Loss: 1.4074 | Learning Rate: 0.0010 \[Epoch: 005] Contrastive Pre-train | Loss: 1.3119 | Learning Rate: 0.0010 \[Epoch: 006] Contrastive Pre-train | Loss: 1.2198 | Learning Rate: 0.0010 \[Epoch: 007] Contrastive Pre-train | Loss: 1.1831 | Learning Rate: 0.0010 \[Epoch: 008] Contrastive Pre-train | Loss: 1.1147 | Learning Rate: 0.0010 \[Epoch: 009] Contrastive Pre-train | Loss: 1.0801 | Learning Rate: 0.0010 \[Epoch: 010] Contrastive Pre-train | Loss: 1.0354 | Learning Rate: 0.0010 \[Epoch: 011] Contrastive Pre-train | Loss: 1.0157 | Learning Rate: 0.0010 \[Epoch: 012] Contrastive Pre-train | Loss: 1.0028 | Learning Rate: 0.0010 \[Epoch: 013] Contrastive Pre-train | Loss: 0.9703 | Learning Rate: 0.0010 \[Epoch: 014] Contrastive Pre-train | Loss: 0.9478 | Learning Rate: 0.0010 \[Epoch: 015] Contrastive Pre-train | Loss: 0.9229 | Learning Rate: 0.0010 \[Epoch: 016] Contrastive Pre-train | Loss: 0.9182 | Learning Rate: 0.0010 \[Epoch: 017] Contrastive Pre-train | Loss: 0.9136 | Learning Rate: 0.0010 \[Epoch: 018] Contrastive Pre-train | Loss: 0.9221 | Learning Rate: 0.0010 \[Epoch: 019] Contrastive Pre-train | Loss: 0.9022 | Learning Rate: 0.0010 \[Epoch: 020] Contrastive Pre-train | Loss: 0.8911 | Learning Rate: 0.0010 \[Epoch: 021] Contrastive Pre-train | Loss: 0.8665 | Learning Rate: 0.0009 \[Epoch: 022] Contrastive Pre-train | Loss: 0.8526 | Learning Rate: 0.0009 \[Epoch: 023] Contrastive Pre-train | Loss: 0.8503 | Learning Rate: 0.0009 \[Epoch: 024] Contrastive Pre-train | Loss: 0.8513 | Learning Rate: 0.0009 \[Epoch: 025] Contrastive Pre-train | Loss: 0.8504 | Learning Rate: 0.0009 ... \[Epoch: 049] Contrastive Pre-train | Loss: 0.7545 | Learning Rate: 0.0008 \[Epoch: 050] Contrastive Pre-train | Loss: 0.7507 | Learning Rate: 0.0008

## Implementaion of NNCLR on CIFAR-10

```python
# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead
from lightly.models.modules import NNMemoryBankModule
```

```python
class NNCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
```

```python
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = NNCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

memory_bank = NNMemoryBankModule(size=4096)
memory_bank.to(device)

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SimCLRCollateFunction(input_size=32)
```

```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
```

```python
print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x0, x1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0, p0 = model(x0)
        z1, p1 = model(x1)
        z0 = memory_bank(z0, update=False)
        z1 = memory_bank(z1, update=True)
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
```

Starting Training epoch: 00, loss: 5.91997 epoch: 01, loss: 5.64455 epoch: 02, loss: 5.56317 epoch: 03, loss: 5.51402 epoch: 04, loss: 5.47789 epoch: 05, loss: 5.45315 epoch: 06, loss: 5.43040 epoch: 07, loss: 5.41552 epoch: 08, loss: 5.40150 epoch: 09, loss: 5.39256

## Cheers!

import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils import data
import torch.nn as nn
import time
import random
from torch.nn import functional as F
import torchvision.transforms as transforms
import sys


# ============= step 1: 数据获取及预处理 ================
def load_data(file_path):
    source = pd.read_csv(file_path)
    processed_data = []
    for i in range(0, len(source)):
        item = source.iloc[i, 1]
        pixels = [float(j) for j in item.split(" ")]
        if not any(pixels):
            continue
        origin = []
        for j in range(0, 48):
            origin.append(pixels[48 * j: 48 * j + 48])
        origin = torch.tensor([origin])
        processed_data.append([origin, source.iloc[i, 0]])

        flip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        flipped = flip(origin)
        processed_data.append([flipped, source.iloc[i, 0]])
        """""    
        #随机旋转
        rot = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        #水平翻转
        flip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        flipped = flip(origin)

        processed1 = rot(origin)
        processed2 = rot(flipped)
        temp1 = [processed1, source.iloc[i, 0]]
        temp2 = [processed2, source.iloc[i,0]]
        processed_data.append(temp1)
        processed_data.append(temp2)
        """

    batch_size = 512
    random.shuffle(processed_data)
    train_iter = data.DataLoader(processed_data[:int(len(source) * 0.8)], batch_size, shuffle=True)
    valid_iter = data.DataLoader(processed_data[int(len(source) * 0.8):], batch_size, shuffle=False)

    return train_iter, valid_iter

# ==================== step 2: 网络构建 ======================

# VGG
class VGG(nn.Module):
    def __init__(self, arch: object, num_classes=7) -> object:
        super(VGG, self).__init__()
        self.in_channels = 1
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return F.softmax(self.fc3(out))


def VGG_16():
    return VGG([2, 2, 3, 3, 3])

# AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_calsses=7, init_waight=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classfier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全连接
            nn.Linear(in_features=128*5*5 , out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_calsses),
        )
        if init_waight:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classfier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) #正态分布赋值
                nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                               nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                               nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                               nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                               nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                               nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                               nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                               nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stages = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 7))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.fc(x)

        return x

    def stage_calculate(self, stage, x):
        stage = stage - 1
        for i in range(0, stage):
            x = self.stages[i](x)

        return x[0]

# =================== step 3: 训练及输出 =====================

# train
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self):
        return sum(self.times)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(reduce_sum(cmp.type(y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def train_plot(net, train_iter, valid_iter, num_epochs, opt, device=try_gpu()):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = opt
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                        legend=['loss', 'train acc', 'valid acc'])
    timer = Timer()
    best_acc = 0

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % 10 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        if valid_acc > best_acc:
            print("New best accuracy: {}".format(valid_acc))
            best_acc = valid_acc
            torch.save(net.state_dict(), "Net.param")
        animator.add(epoch + 1, (None, None, valid_acc))
        print("Epoch: {}/{}, train accuray: {}, valid accuracy: {}".format(epoch, num_epochs, train_acc, valid_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

if __name__ == "__main__":
    print("Start loading...")
    train_iter, valid_iter = load_data(sys.argv[1])
    print("Start training...")
    net = Net()
    train_lr, train_epochs = 0.001, 50
    train_plot(net, train_iter, valid_iter, train_epochs, torch.optim.Adam(net.parameters(), lr=train_lr))
    plt.savefig("Result.jpg")
    plt.show()

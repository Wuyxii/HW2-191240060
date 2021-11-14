import sys

import torch
import pandas as pd
import torch.nn as nn


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

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def predict(file_path, param_path):
    print("Start loading...")
    net = Net().to(try_gpu())
    net.load_state_dict(torch.load(param_path))
    net.eval()

    test = pd.read_csv(file_path)

    print("Start predicting...")
    result = []
    for i in range(0, len(test)):
        map = torch.tensor([float(j) for j in test.iloc[i, 0].split(" ")]).reshape(48, 48).unsqueeze(0).unsqueeze(0).to(
            try_gpu())
        arr = net(map)
        result.append(int((torch.max(arr, dim=-1)).indices.cpu()))

    index = [i for i in range(1, len(test) + 1)]
    output = pd.DataFrame({'ID': index, 'emotion': result})
    output.to_csv('Answer.csv', index=None, encoding='utf8')

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])

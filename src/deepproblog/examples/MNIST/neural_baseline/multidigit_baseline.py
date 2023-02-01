from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from deepproblog.examples.MNIST.data import addition
from deepproblog.utils.logger import Logger
from deepproblog.utils.stop_condition import StopOnPlateau


import torch
from torch import nn as nn


class Separate_Baseline(nn.Module):
    def __init__(self, batched=False, probabilities=True):
        super(Separate_Baseline, self).__init__()
        self.batched = batched
        self.probabilities = probabilities
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
        )
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x, y):
        if not self.batched:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x = self.encoder(x)
        y = self.encoder(y)
        x = x + y
        x = x.view(-1, 16 * 8 * 2)
        x = self.classifier(x)
        if self.probabilities:
            x = self.activation(x)
        if not self.batched:
            x = x.squeeze(0)

        return x


class Separate_Baseline_Multi(nn.Module):
    def __init__(self, n=4):
        super(Separate_Baseline_Multi, self).__init__()
        self.n = n
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16 * 4 * 4 * self.n // 2, 100),
        )
        self.classifier2 = nn.Sequential(
            nn.ReLU(), nn.Linear(100 * 2, 128), nn.ReLU(), nn.Linear(128, 199)
        )

    def forward(self, imgs1, imgs2):
        imgs1 = [self.encoder(x) for x in imgs1]
        imgs2 = [self.encoder(x) for x in imgs2]
        x1, x2 = torch.cat(imgs1, 2), torch.cat(imgs2, 2)
        x1, x2 = (
            x1.view(-1, 16 * 4 * 4 * self.n // 2),
            x2.view(-1, 16 * 4 * 4 * self.n // 2),
        )
        x1, x2 = self.classifier(x1), self.classifier(x2)
        x = torch.cat([x1, x2], 1)
        x = self.classifier2(x)
        return x

def test_addition(dset):
    confusion = np.zeros(
        (199, 199), dtype=np.uint32
    )  # First index actual, second index predicted
    correct = 0
    n = 0
    for d1, d2, l in dset:
        d1 = [x.unsqueeze(0) for x in d1]
        d2 = [x.unsqueeze(0) for x in d2]
        outputs = net.forward(d1, d2)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print("Accuracy: ", acc)
    return acc


test_dataset = addition(2, "test")

Train = namedtuple("Train", ["logger"])
modelname = "Separate_multi"

for N in [14900]:
    # for N in [150, 1500, 14000]:
    batch_size = 2
    test_period = 5000
    log_period = 100
    train_dataset = addition(2, "train").subset(N)
    val_dataset = addition(2, "train").subset(N, N + 100)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    running_loss = 0.0
    log = Logger()
    i = 1
    net = Separate_Baseline_Multi()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    stop_condition = StopOnPlateau("Accuracy", patience=5)
    train_obj = Train(log)
    j = 1
    while not stop_condition.is_stop(train_obj):
        print("Epoch {}".format(j))
        for d1, d2, label in trainloader:
            optimizer.zero_grad()

            outputs = net(d1, d2)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
            if i % log_period == 0:
                print("Iteration: ", i, "\tAverage Loss: ", running_loss / log_period)
                log.log("loss", i, running_loss / log_period)
                running_loss = 0
            if i % test_period == 0:
                log.log("Accuracy", i, test_addition(val_dataset))
            i += 1
        j += 1
    log.comment("Accuracy\t{}".format(test_addition(test_dataset)))
    log.write_to_file("log/{}_{}".format(modelname, N))

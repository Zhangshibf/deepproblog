from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

method = "exact"
num = [1, 2]
for N in num:
    N = int(N)
    name = "addition_{}_{}".format(method, N)

    train_set = addition(N, "train")
    test_set = addition(N, "test")
    network = MNIST_Net()

    pretrain = 0
    if pretrain is not None and pretrain > 0:
        network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    loader = DataLoader(train_set, 2, False)
    for i in loader:
        print(i)
        break

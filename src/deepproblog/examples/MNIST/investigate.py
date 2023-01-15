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
N = 1

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train")
test_set = addition(N, "test")
path = "/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/mnist_net"
network = MNIST_Net()
network.load_state_dict(torch.load(path)['model_state_dict'])
print(network)
print("done")

#model.set_engine(ExactEngine(model), cache=True)
#model.add_tensor_source("train", MNIST_train)
#model.add_tensor_source("test", MNIST_test)

#loader = DataLoader(train_set, 2, False)
#train = train_model(model, loader, 1, log_iter=100, profile=0)
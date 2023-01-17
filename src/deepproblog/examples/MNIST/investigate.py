import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from deepproblog.examples.MNIST.network import MNIST_Net
_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}

class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]

MNIST_test = MNIST_Images("test")

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
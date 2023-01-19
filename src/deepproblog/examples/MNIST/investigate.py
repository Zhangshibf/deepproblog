import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition

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

path = ["/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/addition1/mnist_net","/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/addition2/mnist_net"]
for i in path:
    network = MNIST_Net()
    network.load_state_dict(torch.load(path)['model_state_dict'])
    print(network)
    print("done")

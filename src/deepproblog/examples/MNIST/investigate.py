import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition

_DATA_ROOT = Path(__file__).parent
def test_mnistnet(cnn,dataset):
    pass


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    datasets = {"test": torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=False, download=True, transform=transform)}
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size=100, shuffle=True)
    for i in test_loader:
        print(i)
    path = ["/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/addition1/mnist_net","/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/addition2/mnist_net"]
    for i in path:
        network = MNIST_Net()
        network.load_state_dict(torch.load(i)['model_state_dict'])
        accuracy,loss,cm = test_mnistnet(network,datasets["test"])

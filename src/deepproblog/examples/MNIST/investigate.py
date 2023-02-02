import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from deepproblog.examples.MNIST.network import MNIST_Net
from torchmetrics import ConfusionMatrix, Recall
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition


def test_mnistnet(cnn,test_loader):
    cnn.eval()
    correct = 0
    total = 0
    cm = list()
    accuracy = list()
    recall = list()
    with torch.no_grad():
        for data in test_loader:
            #actually there is only one epoch...
            images, labels = data
            outputs = cnn(images)

            #confusion matrix
            confmat = ConfusionMatrix(task="multiclass", num_classes=10)
            cm.append(confmat(outputs, labels))

            #accuracy and recall
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy.append((100 * correct / total))

            rec = Recall(task="multiclass", average='micro', num_classes=10)
            recall.append(rec(outputs,labels))

    return accuracy,recall,cm


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    datasets = {"test": torchvision.datasets.MNIST(root="/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/dataset", train=False, download=True, transform=transform)}
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size=100000, shuffle=True)
#single digit addition CNN, multi digir addition CNN, single digit baseline,
    path = ["/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/addition1/mnist_net","/home/CE/zhangshi/mlfornlp/mlnlp/src/deepproblog/examples/MNIST/snapshot/addition2/mnist_net"]
    name = ["Single digit CNN","Multi digit CNN"]
    for i in path:
        network = MNIST_Net()
        network.load_state_dict(torch.load(i)['model_state_dict'])
        accuracy,recall,cm = test_mnistnet(network,test_loader)
        print("Model name: {}".format(name[i]))
        print("Accuracy: {}".format(accuracy))
        print("Recall: {}".format(recall))
        print("Confusion Matrix: {}".format(cm))

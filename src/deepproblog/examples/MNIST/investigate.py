import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from deepproblog.examples.MNIST.network import MNIST_Net
from torchmetrics import ConfusionMatrix, Recall
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition


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

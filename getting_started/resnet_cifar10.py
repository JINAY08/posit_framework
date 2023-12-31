
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from .bfp_ops import PositLinear, PositConv2d, unpack_bfp_args
import torch.optim as optim
from tqdm import tqdm, trange

PATH = './cifar_net.pth'

# 1. Load and normalizing the CIFAR10 training and test datasets using ``torchvision``
def prepare_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=12)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=12)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes

# 2. Define a ResNet in HBFP
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bfp_args={}):
        super(BasicBlock, self).__init__()
        self.conv1 = PositConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **bfp_args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PositConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, **bfp_args)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PositConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, **bfp_args),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bfp_args={}):
        super(Bottleneck, self).__init__()
        self.conv1 = PositConv2d(in_planes, planes, kernel_size=1, bias=False, **bfp_args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PositConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **bfp_args)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PositConv2d(planes, self.expansion*planes, kernel_size=1, bias=False, **bfp_args)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PositConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, **bfp_args),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(ResNet, self).__init__()
        self.bfp_args = unpack_bfp_args(dict(vars(args)))
        self.in_planes = 64

        self.conv1 = PositConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, **self.bfp_args)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                               bfp_args=self.bfp_args))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args)

def train(net, trainset, trainloader, testset, testloader, classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    # print("Device: ", device)
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 1:
            print("Training on", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # powersgd = initpowersgd(net)

    
    for epoch in trange(20, desc='epoch'):
        correct = 0
        total = 0
        pass
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc='iteration'), 0):
            pass
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.step()
            # optimizer_step(optimizer, powersgd)
            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                tqdm.write('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('The training accuracy: %d %%' % (
            100 * correct / total))
    print('Finished Training')


    torch.save(net.state_dict(), PATH)

def test_model(net, trainset, trainloader, testset, testloader, classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    dataiter = iter(testloader)
    # images, labels = dataiter.next()
    images, labels = next(dataiter) #added this instead of line 219
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    net = nn.DataParallel(net) # added this to resolve issue of missing key(s)
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # How the network performs on the whole dataset.
    correct = 0
    total = 0
    print('The accuracy on the test dataset is being calculated...')
    with torch.no_grad():
        for data in tqdm(testloader, desc='test iteration'):
            pass
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('The accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # What are the classes that performed well
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    print('The accuracy of each class is being calculated...')
    with torch.no_grad():
        for data in tqdm(testloader, desc='test iteration'):
            pass
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('The accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def resnet18_cifar10(args):
    print(args)
    trainset, trainloader, testset, testloader, classes = prepare_data()
    net = ResNet18(args)
    train(net, trainset, trainloader, testset, testloader, classes, args)
    test_model(net, trainset, trainloader, testset, testloader, classes, args)

from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from timer import Timer

LOG_DIR = "./logs/tb"

MODEL_TYPES    = ["HYBRID"]#["CNN", "TRANSFORMER", "DeepViTNet", "HYBRID"]
OPTIMIZER_TYPE = "ADAM"     # ADA_DELTA, ADAM
DATASETS       = ["MNIST", "CIFAR10", "CIFAR100"] 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class CNNNet(nn.Module):
    def __init__(self, num_classes, channels, fc_size):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
    
class TransformerNet(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channels):
        super(TransformerNet, self).__init__()
        
        from vit_pytorch import ViT
        self.model = ViT(
                        image_size = image_size,
                        patch_size = patch_size,
                        num_classes = num_classes,
                        dim = 128,
                        depth = 6,
                        heads = 4,
                        mlp_dim = 256,
                        channels=channels,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )

    def forward(self, x):
        
        x = self.model(x)

        output = F.log_softmax(x, dim=1)
        return output

class HybridNet(nn.Module):
    def __init__(self, image_size, channels, num_classes):
        super(HybridNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        image_width = int((math.sqrt(image_size) - 4) / 2)

        from vit_pytorch import ViT
        self.vit = ViT(
                        image_size = image_width ** 2,
                        patch_size = int(image_width / 2),
                        num_classes = num_classes,
                        dim = 128,
                        depth = 6,
                        heads = 4,
                        mlp_dim = 256,
                        channels = 64,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.vit(x)

        output = F.log_softmax(x, dim=1)
        return output

class DeepViTNet(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channels):
        super(DeepViTNet, self).__init__()
        
        from vit_pytorch.deepvit import DeepViT
        self.model = DeepViT(
                        image_size = image_size,
                        patch_size = patch_size,
                        num_classes = num_classes,
                        dim = 128,
                        depth = 6,
                        heads = 4,
                        mlp_dim = 256,
                        channels=channels,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )

    def forward(self, x):
        
        x = self.model(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, dataset, modeltype, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            with Timer("{} {} Inference".format(modeltype, dataset)):
                output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    writer.add_scalar('Accuracy/{} {}'.format(modeltype, dataset), accuracy, epoch)
    return accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Transformer Experiments')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    # Reduce precision to speed up training
    torch.set_float32_matmul_precision('medium')

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    accuracies = {}
    parameters = {}
    for MODEL_TYPE in MODEL_TYPES:
        for DATASET in DATASETS:

            if DATASET == "MNIST":
                train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
                test_set = datasets.MNIST('./data', train=False, transform=transform)
                image_size = 28*28
                patch_size = 14
                num_classes = 10
                channels = 1
                fc_size = 9216
            elif DATASET == "CIFAR10":
                train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
                test_set = datasets.CIFAR10('./data', train=False, transform=transform)
                image_size = 32*32
                patch_size = 16
                num_classes = 10
                channels = 3
                fc_size = 12544
            elif DATASET == "CIFAR100":
                train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
                test_set = datasets.CIFAR100('./data', train=False, transform=transform)
                image_size = 32*32
                patch_size = 16
                num_classes = 100
                channels = 3
                fc_size = 12544

            train_loader = torch.utils.data.DataLoader(train_set,**train_kwargs)
            test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

            if MODEL_TYPE == "CNN":
                model = CNNNet(num_classes, channels, fc_size).to(device)
            elif MODEL_TYPE == "HYBRID":
                model = HybridNet(image_size, channels, num_classes).to(device)
            elif MODEL_TYPE == "TRANSFORMER":
                model = TransformerNet(image_size, patch_size, num_classes, channels).to(device)
            elif MODEL_TYPE == "DeepViTNet":
                model = DeepViTNet(image_size, patch_size, num_classes, channels).to(device)

            if OPTIMIZER_TYPE == "ADA_DELTA":    
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            elif OPTIMIZER_TYPE == "ADAM":
                optimizer = optim.Adam(model.parameters(), lr=args.lr/10000)
            
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            #scheduler = CosineAnnealingLR(optimizer, args.epochs)

            max_accuracy = 0
            writer = SummaryWriter(log_dir=LOG_DIR)
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                accuracy = test(model, device, test_loader, DATASET, MODEL_TYPE, writer, epoch)
                max_accuracy = max(accuracy, max_accuracy)
                scheduler.step()

            accuracies["{} {}".format(MODEL_TYPE, DATASET)] = max_accuracy
            parameters["{} {}".format(MODEL_TYPE, DATASET)] = count_parameters(model)

            if args.save_model:
                torch.save(model.state_dict(), "{}_cnn.pt".format(DATASET))

    Timer().report_phases()
    print(accuracies)
    print(parameters)


if __name__ == '__main__':
    main()
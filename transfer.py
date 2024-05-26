import os

import torch
import torchvision

from model import model_loader
from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def init_data(dataset_name, batch_size=64, device=torch.device('cpu')):
    class GrayScaleToRGB(object):
        def __call__(self, img):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img

    if dataset_name == 'mnist':
        input_size = 224
        in_channels = 1
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

    elif dataset_name == 'fashion_mnist':
        input_size = 28
        in_channels = 1
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True, transform=transform)

    elif dataset_name == 'caltech_101':
        input_size = 224
        in_channels = 3
        num_classes = 101
        transform = transforms.Compose([
            GrayScaleToRGB(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.Caltech101(root='./dataset', download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    elif dataset_name == 'caltech_256':
        input_size = 224
        in_channels = 3
        num_classes = 257
        transform = transforms.Compose([
            GrayScaleToRGB(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.Caltech256(root='./dataset', download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if device == torch.device('cuda'):
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    else:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return (input_size, in_channels, num_classes, train_data_loader, val_data_loader)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', choices=['lenet', 'ann'])
    parser.add_argument('--load-weight')
    parser.add_argument('--dataset', '-d', choices=['mnist', 'fashion_mnist', 'caltech_101', 'caltech_256'])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--criterion', default='cross_entropy')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=30)
    args = parser.parse_args()

    # Device
    use_gpu = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Dataset and dataloader
    input_size, in_channels, num_classes, train_data_loader, val_data_loader = init_data(args.dataset, batch_size=args.batch_size, device=device)

    # Model
    model, optimizer = model_loader.get_pretrained_model_and_convert(args.model, input_size=input_size, num_classes=num_classes, device=device, weight_path=args.load_weight)
    start_epoch = 1

    # Criterion
    if args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Log
    if not os.path.isdir('./runs'):
        os.mkdir('./runs')
    writer = SummaryWriter()

    # Save model weights
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    min_val_loss = 1e9
    max_val_acc = 0.0

    # Train
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_data_loader, desc=f'Epoch {epoch}'):
            optimizer.zero_grad()
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        scheduler.step()
        train_loss /= len(train_data_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_data_loader:
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_loss /= len(val_data_loader.dataset)
        accuracy = correct / total

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)

        if (epoch) % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), f'./checkpoints/{args.model}_{epoch}.pt')
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'./checkpoints/{args.model}_{epoch}_loss.pt')
        if max_val_acc <= accuracy:
            max_val_acc = accuracy
            torch.save(model.state_dict(), f'./checkpoints/{args.model}_{epoch}_acc.pt')


if __name__ == '__main__':
    main()

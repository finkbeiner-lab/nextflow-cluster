"""Basic CNN for classification."""

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import random
import pandas as pd
import os, stat
import argparse
from time import time
import wandb

# os.environ["WANDB_SILENT"] = "true"

transform = transforms.Compose(
    # [transforms.ToTensor(),
    # [transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
    [transforms.Lambda(lambda image: torch.from_numpy(image.astype(np.float32))),
     transforms.Resize(size=(224, 224), antialias=True),
     #  transforms.Normalize((0.5,), (0.5,))])
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print(f'CUDA Available: {torch.cuda.is_available()}')


def collate_fn(batch):
    return tuple(zip(*batch))


class Net(nn.Module):
    def __init__(self, num_classes, h, w, num_channels):
        super().__init__()
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1.
        # self.conv1 = nn.Conv2d(num_channels, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 72 * 72, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_classes)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 26 * 26, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # # print(x.size())
        # x = self.pool(F.relu(self.conv2(x)))
        # # print(x.size())

        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # # print(x.size())

        # x = F.relu(self.fc1(x))
        # # print(x.size())

        # x = F.relu(self.fc2(x))
        # # print(x.size())

        # x = self.fc3(x)

        return x


class DirectoryDataset(Dataset):
    def __init__(self, directory='/gladstone/finkbeiner/linsley/josh/dogs_vs_cats/train/S_folder',
                 n_samples=0, transform=None, target_transform=None):
        dirs = glob(os.path.join(directory, '*'))
        self.label_names = []
        for d in dirs:
            if os.path.isdir(d):
                self.label_names.append(os.path.basename(d))
        # reset dirs to match with labels
        dirs = [os.path.join(directory, lbl) for lbl in self.label_names]
        self.label2num_dct = {lbl: i for i, lbl in enumerate(self.label_names)}
        self.num2label_dct = {i: lbl for lbl, i in self.label2num_dct.items()}
        self.files = []
        self.labels = []
        for lbl, d in zip(self.label_names, dirs):
            _files = glob(os.path.join(d, '*'))
            _labels = [self.label2num_dct[lbl]] * len(_files)
            self.files += _files
            self.labels += _labels

        print(f'num files: {len(self.files)}')
        if n_samples > 0:
            random.seed(11)
            self.files = random.sample(self.files, n_samples)
            random.seed(11)
            self.labels = random.sample(self.labels, n_samples)
        print(f'shortened num files: {len(self.files)}')

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        label = self.labels[idx]
        img = np.array(Image.open(f))
        img = np.moveaxis(img, -1, 0)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class Train:
    def __init__(self, opt):
        self.opt = opt
        self.hyperparams = vars(self.opt)
        self.opt.batch_size = int(self.opt.batch_size)
        self.opt.epochs = int(self.opt.epochs)
        self.opt.learning_rate = float(self.opt.learning_rate)
        self.opt.momentum = float(self.opt.momentum)
        self.opt.use_wandb = int(self.opt.use_wandb)
        self.opt.num_channels = int(self.opt.num_channels)
        self.opt.n_samples = int(self.opt.n_samples)
        self.classes = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.label_column = None
        self.wandbdir = os.path.join(self.opt.savedir, 'wandb')
        self.modeldir = os.path.join(self.opt.savedir, 'Models')
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        if self.opt.use_wandb:
            if not os.path.exists(self.wandbdir):
                os.mkdir(self.wandbdir)
            os.chmod(self.wandbdir, 0o0777)
            wandb.init("CNN-Dir", mode='offline', dir=self.wandbdir)

    def run(self):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        print(f'Device {self.device}')
        assert self.device == 'cuda:0', 'gpu not used'

        trainset = DirectoryDataset(self.opt.traindir,
                                    self.opt.n_samples,
                                    transform=transform)
        valset = DirectoryDataset(self.opt.traindir,
                                  self.opt.n_samples,
                                  transform=transform)
        num_classes = len(trainset.label_names)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.opt.batch_size,
                                                  shuffle=True, num_workers=0,
                                                  pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.opt.batch_size,
                                                shuffle=False, num_workers=4,
                                                pin_memory=True)
        # TODO: detect height, width, channels
        self.model = Net(num_classes=num_classes, h=300, w=300, num_channels=self.opt.num_channels)
        self.model.to(self.device)
        # self.model = torch.compile(net, mode='reduce-overhead')
        modelpath = os.path.join(self.modeldir, 'net.pth')
        self.hyperparams['modelpath'] = modelpath
        if self.opt.use_wandb:
            wandb.config.update(self.opt)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        # optimizer = optim.SGD(net.parameters(), lr=self.opt.learning_rate, momentum=self.opt.momentum)
        # train_opt = torch.compile(self.train_one_epoch, mode='reduce-overhead')
        for epoch in range(self.opt.epochs):  # loop over the dataset multiple times
            epoch_loss,train_acc = self.train_one_epoch(trainloader, )
            print(f'Epoch {epoch + 1}: {epoch_loss / len(trainloader)}')
            print(f'Epoch {epoch + 1}: {train_acc / len(trainloader)}')
            if self.opt.use_wandb: wandb.log({"train_loss_epoch": epoch_loss / len(trainset)})
            with torch.no_grad():
                val_acc = 0
                val_epoch_loss = 0
                for j, (X, y) in enumerate(valloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    # calculate outputs by running images through the network
                    y_pred = self.model(X)
                    valloss = self.criterion(y_pred, y)
                    val_epoch_loss += valloss.item()
                    # the class with the highest energy is what we choose as prediction
                    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                    val_acc += (y_pred_class == y).sum().item() / len(y_pred)

                    if self.opt.use_wandb: wandb.log({"val_loss": valloss.item()})
                val_acc = val_acc / len(valloader)
                val_epoch_loss = val_epoch_loss / len(valloader)
                print(f'Accuracy of the network on validation images: {val_acc * 100} %')
                print(f'Loss of the network on validation images: {val_epoch_loss:.3f}')

        print('Finished Training')

        torch.save(self.model.state_dict(), modelpath)
        print('Saved model')
        if self.opt.use_wandb:
            wandb.finish()
        print('Done')

    def train_one_epoch(self, trainloader):
        self.model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        train_acc = 0.
        for batch, (X, y) in enumerate(trainloader):
            print(f'Batch {batch}')
            X = X.to(self.device)
            y = y.to(self.device)
            # get the inputs; data is a list of [inputs, labels]
            strt = time()
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            # with torch.autocast(device_type='cuda'):
            y_pred = self.model(X)
            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            # print statistics
            if self.opt.use_wandb: wandb.log({"loss": loss.item()})
            running_loss += loss.item()
            epoch_loss += loss.item()
            print(f'loss: {loss.item()}')
            # print(f'batch time: {time() - strt}')
            if batch % 100 == 0:
                print(f'[{batch + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            # print epoch loss
        epoch_loss = epoch_loss / len(trainloader)
        train_acc = train_acc / len(trainloader)
        print(f'Epoch Loss: {epoch_loss}')
        print(f'Epoch Acc: {train_acc}')
        return epoch_loss, train_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Text status',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
    )
    parser.add_argument('--traindir', default='/gladstone/finkbeiner/linsley/josh/dogs_vs_cats/train/S_folder',
                        type=str)
    parser.add_argument('--savedir', default='/gladstone/finkbeiner/linsley/josh/dogs_vs_cats/train', type=str)
    parser.add_argument('--num_channels', default=3)
    parser.add_argument('--n_samples', default=100)
    parser.add_argument('--epochs', default=25)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--learning_rate', default=1e-6)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--use_wandb', default=0, type=int, help="Log training with wandb.")
    args = parser.parse_args()
    print(args)
    # trainset = DirectoryDataset(args.traindir,
    #                             args.n_samples,
    #                             transform=transform)
    #
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
    #                                           shuffle=True, num_workers=0,
    #                                           pin_memory=True)
    # for X, y in trainloader:
    #     img = X[0].detach().numpy()
    #     img = np.moveaxis(img, 0, -1)
    #     img -= np.min(img)
    #     img /= np.max(img)
    #     img *= 255
    #     img = np.uint8(img)
    #     plt.imshow(img)
    #     plt.show()
    Tr = Train(args)
    Tr.run()

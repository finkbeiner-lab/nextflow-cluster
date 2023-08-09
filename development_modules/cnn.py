"""Basic CNN for classification."""

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import argparse
from db_util import Ops
from sql import Database
import wandb

transform = transforms.Compose(
    # [transforms.ToTensor(),
    [transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
     transforms.Normalize((0.5,), (0.5,))])
#  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def collate_fn(batch):
    return tuple(zip(*batch))


class Net(nn.Module):
    def __init__(self, num_classes, h, w, num_channels):
        super().__init__()
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1.
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 72 * 72, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.size())

        x = F.relu(self.fc1(x))
        # print(x.size())

        x = F.relu(self.fc2(x))
        # print(x.size())

        x = self.fc3(x)

        return x


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_type, label_name=None, transform=None, target_transform=None):
        """
        celldata: Dataframe from database with image names and classes
        class_labels: str, column name from df
        crops_per_tile: int, max number of crops to sample per tile

        For efficiency, sample multiple crops from tile.
        """
        # input single crop

        # get tiles
        print('Data length', len(df))
        self.df = df.sample(frac=1)
        unique_lbls = np.unique(self.df[label_type])

        self.label2num_dct = {lbl: i for i, lbl in enumerate(unique_lbls)}
        self.num2label_dct = {i: lbl for lbl, i in self.label2num_dct.items()}
        labels = self.df[label_type].values
        numeric_labels = [self.label2num_dct[lbl] for lbl in labels]
        self.img_labels = torch.tensor(numeric_labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.croppath
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Train:
    def __init__(self, opt):
        self.opt = opt
        self.hyperparams = vars(self.opt)
        self.opt.batch_size = int(self.opt.batch_size)
        self.opt.epochs = int(self.opt.epochs)
        self.opt.learning_rate = float(self.opt.learning_rate)
        self.opt.momentum = float(self.opt.momentum)
        self.opt.use_wandb = int(self.opt.use_wandb)
        self.Dbops = Ops(opt)
        self.Db = Database()
        self.classes = None
        self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        self.modeldir = os.path.join(self.analysisdir, 'Models')
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        if self.opt.use_wandb:
            wandb.init("CNN", mode='offline')
        # possible classes, as specific as possible
        # Gedi ratio, divide one intensity by another.
        # Live dead based on gedi ratio, either manual or automatically calculated
        # celltype
        # treatment
        # wells
        # Tiles

    def get_classes(self):
        experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
        filter_dicts = []
        classes = []
        df = None
        # treatment
        if self.opt.label_type == 'name':
            if self.opt.classes is None:
                # use all classes
                # TODO: make case insensitive
                classes = self.Db.get_table_value('dosagedata', self.opt.label_type, dict(experimentdata_id=experimentdata_id,
                                                                             kind=self.opt.label_name))  # inhibitor, treatment, antibody
                classes = np.unique(classes)
            else:
                classes = self.opt.classes.split(',')
                # assert len(classes) > 1, 'must have multiple classes if training'
            print('classes', classes)
            self.classes = classes
            df = self.Dbops.get_df_for_training(['celldata', 'cropdata', 'dosagedata'])
            df = df[df['name'].isin(self.classes)]
        elif self.opt.label_type == 'celltype':
            if self.opt.classes is None:
                # use all classes
                classes = self.Db.get_table_value('welldata', 'celltype', dict(experimentdata_id=experimentdata_id,
                                                                               ))
                classes = np.unique(classes)
            else:
                classes = self.opt.classes.split(',')
            print('classes', classes)
            self.classes = classes
            df = self.Dbops.get_df_for_training(['celldata', 'cropdata'])
            print('columns', df.columns)
            df = df[df['celltype'].isin(self.classes)]
        else:
            assert 0, f'label type {self.opt.label_type} not in selection'
        return df

    def run(self):
        df = self.get_classes()
        trainset = ImageDataset(df, label_type=self.opt.label_type,
                                label_name=self.opt.label_name,
                                transform=transform)
        num_classes = len(np.unique(self.classes))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.opt.batch_size,
                                                  shuffle=True, num_workers=2)

        net = Net(num_classes=num_classes, h=300, w=300, num_channels=1)
        modelpath = os.path.join(self.modeldir, 'net.pth')
        self.hyperparams['modelpath'] = modelpath
        if self.opt.use_wandb:
            wandb.config.update(self.opt)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.opt.learning_rate, momentum=self.opt.momentum)

        for epoch in range(self.opt.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            epoch_loss = 0.0

            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                if self.opt.use_wandb: wandb.log({"loss": loss.item()})

                running_loss += loss.item()
                epoch_loss += outputs.shape[0] * loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            # print epoch loss
            print(epoch + 1, epoch_loss / len(trainset))
            if self.opt.use_wandb: wandb.log({"train_loss_epoch": epoch_loss / len(trainset)})
        print('Finished Training')

        torch.save(net.state_dict(), modelpath)
        print('Saved model')
        if self.opt.use_wandb:
            wandb.finish()
        print('Done')


class Deploy:
    def __init__(self, opt):
        self.opt = opt
        self.opt.batch_size = int(self.opt.batch_size)
        self.opt.learning_rate = float(self.opt.learning_rate)
        self.opt.momentum = float(self.opt.momentum)

    def check_image(self, testloader):
        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{self.opt.classes[labels[j]]:5s}' for j in range(4)))

    def check_single_prediction(self, model, images):
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{self.opt.classes[predicted[j]]:5s}'
                                      for j in range(4)))

    def run(self):

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.opt.batch_size,
                                                 shuffle=False, num_workers=2)
        modelpath = './cifar_net.pth'
        net = Net()
        net.load_state_dict(torch.load(modelpath))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.opt.classes}
        total_pred = {classname: 0 for classname in self.opt.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.opt.classes[label]] += 1
                    total_pred[self.opt.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


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
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--label_type', type=str, choices=['celltype', 'name'], help="Column name in database")
    parser.add_argument('--label_name', type=str, help="Match the kind of dosage added. Treatment, Antibody, Inhibitor, etc.")
    parser.add_argument('--classes', type=str, help="Comma separated list of classes. If all classes in experiment, leave blank.")
    parser.add_argument('--img_norm_name', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument("--wells_toggle",
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle",
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels",
                        help="Morphology Channel")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    parser.add_argument('--use_wandb', default=1, type=int, help="Log training with wandb.")
    args = parser.parse_args()
    print(args)
    Tr = Train(args)
    Tr.run()

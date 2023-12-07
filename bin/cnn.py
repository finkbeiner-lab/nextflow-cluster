#!/opt/conda/bin/python

"""Basic CNN for classification."""

import numpy as np
from glob import glob
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image
import imageio
from torch.utils.data import Dataset
import pandas as pd
import os
import argparse
from db_util import Ops
from time import time
from sql import Database
import uuid
import wandb
import datetime
from tqdm import tqdm

__author__ = 'Josh Lamstein'
__copyright__ = 'Gladstone Institutes 2023'


now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)

# os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONFIG_DIR"] = '/gladstone/finkbeiner/lab/GALAXY_INFO/.config'
os.environ["WANDB_CACHE_DIR"] = '/gladstone/finkbeiner/lab/GALAXY_INFO/.cache'

# transform = transforms.Compose(
#     # [transforms.ToTensor(),
#     # [transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
#     [transforms.Lambda(lambda image: torch.from_numpy(image.astype(np.float32))),
#      transforms.Resize((224, 224), antialias=True),
#      transforms.RandomHorizontalFlip(),
#      transforms.Normalize((0.5,), (0.5,))])
#  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print(f'CUDA Available: {torch.cuda.is_available()}')


def collate_fn(batch):
    return tuple(zip(*batch))

class BuildTransform:
    def __init__(self, target_image_size:tuple, num_channels:int, imagenet:bool):
        
        if imagenet:
            norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
            if num_channels < 3:
                norm = [norm[0][:num_channels], norm[1][:num_channels]]
        else:
            norm = [[0.5]*num_channels, [0.5]*num_channels]
        
        self.train_transform = transforms.Compose(
            [transforms.Lambda(lambda image: torch.from_numpy(image.astype(np.float32))),
            transforms.Resize(target_image_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(norm[0], norm[1])])
        
        self.eval_transform = transforms.Compose(
            [transforms.Lambda(lambda image: torch.from_numpy(image.astype(np.float32))),
            transforms.Resize(target_image_size, antialias=True),
            transforms.Normalize(norm[0], norm[1])])
        
    
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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class NetBN(nn.Module):
    def __init__(self, num_classes, h, w, num_channels):
        super().__init__()
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1.
        # self.conv1 = nn.Conv2d(num_channels, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.pool(self.relu(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class NetDropout(nn.Module):
    def __init__(self, num_classes, h, w, num_channels):
        super().__init__()
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1.
        # self.dropout1 = nn.Dropout2d(p=0.2)
        # self.dropout2 = nn.Dropout2d(p=0.3)
        # self.dropout3 = nn.Dropout2d(p=0.4)
        self.dropout = nn.Dropout(p=0.4)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class ResNet:
    def __init__(self, num_classes, num_channels):
        model_ft = models.resnet18(weights='IMAGENET1K_V1')
        for param in model_ft.parameters():  # freeze weights
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        if num_channels != 3:
            layer = model_ft.conv1
            
            # Creating new Conv2d layer
            new_layer = nn.Conv2d(in_channels=num_channels, 
                            out_channels=layer.out_channels, 
                            kernel_size=layer.kernel_size, 
                            stride=layer.stride, 
                            padding=layer.padding,
                            bias=layer.bias)

            copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights

            #Copying the weight of the `copy_weights` channel of the old layer to the extra channels of the new layer
            if num_channels > 3:
                # Copying the weights from the old to the new layer
                new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

                for i in range(num_channels - layer.in_channels):
                    channel = layer.in_channels + i
                    new_layer.weight.data[:, channel:channel+1, :, :] = layer.weight.data[:, copy_weights:copy_weights+1, : :].clone()
                
            else:
                new_layer.weight.data[:, :num_channels, :, :] = layer.weight.data[:, :num_channels, :, :].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)
            model_ft.conv1 = new_layer
        self.model= model_ft


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_type, transform=None, target_transform=None):
        """
        celldata: Dataframe from database with image names and classes
        class_labels: str, column name from df
        crops_per_tile: int, max number of crops to sample per tile

        For efficiency, sample multiple crops from tile.
        """
        # input single crop

        # get tiles
        print('Data length', len(df))
        # Celldata id is based off morphology channel. Cropdata_id connects celldata_id with crop channel.
        self.grouped = df.groupby('celldata_id')
        self.groupkeys = list(self.grouped.groups.keys())
        self.label_type = label_type

        # label mapping
        unique_lbls = np.unique(df[self.label_type])
        self.label2num_dct = {lbl: i for i, lbl in enumerate(unique_lbls)}
        self.num2label_dct = {i: str(lbl) for lbl, i in self.label2num_dct.items()}

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        celldata_id = self.groupkeys[idx]
        df = self.grouped.get_group(celldata_id)
        celldata_id = str(celldata_id)
        experimentdata_id = str(df.experimentdata_id.iloc[0])
        welldata_id = str(df.welldata_id.iloc[0])
        df = df.sort_values('channel')
        df_label = df[self.label_type].iloc[0]
        num_label = self.label2num_dct[df_label]
        label = torch.tensor(num_label)
        images = []
        for i, row in df.iterrows():
            image = Image.open(row.croppath)
            images.append(np.array(image))
        
        img = np.stack(images, axis=0)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label, experimentdata_id, welldata_id, celldata_id


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
        self.opt.target_image_size = (int(self.opt.target_image_size), int(self.opt.target_image_size))
        self.opt.use_imagenet = bool(self.opt.use_imagenet)
        self.opt.n_samples = int(self.opt.n_samples)
        self.Dbops = Ops(opt)
        self.Db = Database()
        self.classes = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.label_column = None
        self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
        self.wandbdir = os.path.join(self.analysisdir, 'wandb')
        self.modeldir = os.path.join(self.analysisdir, 'Models')
        self.checkimagedir = os.path.join(self.analysisdir, 'CheckModelImages')
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        if not os.path.exists(self.checkimagedir):
            os.makedirs(self.checkimagedir)
        if self.opt.use_wandb:
            if not os.path.exists(self.wandbdir):
                os.mkdir(self.wandbdir)
            # os.chmod(self.wandbdir, 0o0777)
            wandb.init("CNN", mode='offline', dir=self.wandbdir)
        self.model_id = uuid.uuid4()
        self.Db.add_row('modeldata', dict(id=self.model_id,
                                          experimentdata_id=experimentdata_id,
                                          ))
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
                classes = self.Db.get_table_value('dosagedata', self.opt.label_type,
                                                  dict(experimentdata_id=experimentdata_id,
                                                       kind=self.opt.label_name))  # inhibitor, treatment, antibody
                classes = np.unique(classes)
            else:
                classes = self.opt.classes.split(',')
                # assert len(classes) > 1, 'must have multiple classes if training'
            print('classes', classes)
            self.classes = classes
            df = self.Dbops.get_df_for_training(
                ['celldata', 'cropdata', 'dosagedata', 'channeldata'])  # 'dosagedata', 'channeldata'
            _g = df.groupby(['celldata_id'])
            g_keys = list(_g.groups.keys())
            print('key 0', g_keys[0])
            example = _g.get_group(g_keys[0])
            print('EX 0', example)
            df = df[df['name'].isin(self.classes)]
            self.label_column = 'name'

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
            df = self.Dbops.get_df_for_training(['celldata', 'cropdata', 'channeldata', 'dosagedata'])
            print('columns', df.columns)
            df = df[df['celltype'].isin(self.classes)]
            self.label_column = 'celltype'
        elif self.opt.label_type == 'stimulate':
            if self.opt.classes is None:
                # use all classes
                classes = np.unique([True, False])
            else:
                classes = self.opt.classes.split(',')
            print('classes', classes)
            self.classes = classes
            df = self.Dbops.get_df_for_training(['celldata', 'cropdata', 'channeldata', 'dosagedata'])
            print('columns', df.columns)
            df = df[df['stimulate'].isin(self.classes)]
            self.label_column = 'stimulate'

        else:
            assert 0, f'label type {self.opt.label_type} not in selection'
        print('label column', self.label_column)
        return df

    def apply_filters(self, df):
        """Apply filters"""
        print(self.opt.filters)
        for col, val in self.opt.filters:
            print(col, val)
            df = df.loc[df[col] == val]
        return df

    def train_val_test_split(self, df, balance_method='cutoff'):
        """Split data into train, validation, testing dataframes"""
        # Number of samples is by celldata_id (training per cell) and the classification

        # If the morphology channel images, but other channels didn't, filter out
        df = df.groupby('celldata_id').filter(lambda x: len(x)==self.opt.num_channels)

        sizes = df.groupby(self.label_column).size()
        cutoff = min(sizes) // self.opt.num_channels
        max_size = max(sizes)
        # cutoff other data
        if balance_method == 'cutoff':  # sample the same number of celldata_ids from each group
            label_group = df.groupby(self.label_column)
            dfs = []
            for name, _df in label_group:
                g = _df.groupby('celldata_id')
                shuf = np.arange(g.ngroups)
                np.random.shuffle(shuf)
                dfs.append(_df[g.ngroup().isin(shuf[:cutoff])])
            balanced_df = pd.concat(dfs)
            _sizes = balanced_df.groupby(self.label_column).size()
            for _s in _sizes:
                print(_s // self.opt.num_channels, cutoff)
                assert _s // self.opt.num_channels == cutoff, 'class size not what was intended'
        else:
            balanced_df = df
        if self.opt.n_samples > 0:  # randomly sample from groupby
            g = balanced_df.groupby(['celldata_id', self.label_column])
            print(f'Grouped data length: {len(g)}')
            shuf = np.arange(g.ngroups)
            np.random.seed(121)
            np.random.shuffle(shuf)
            balanced_df = balanced_df[g.ngroup().isin(shuf[:self.opt.n_samples])]  # change 2 to what you need :-)
        # split
        print(balanced_df.head())
        balanced_g = balanced_df.groupby(['celldata_id', self.label_column])
        g_keys = list(balanced_g.groups.keys())
        print('key 1', g_keys[0])
        example = balanced_g.get_group(g_keys[0])
        print('EX 1', example)
        shuf = np.arange(balanced_g.ngroups)
        np.random.seed(121)
        np.random.shuffle(shuf)
        print('length of shuffle', len(shuf))
        print(f'Balanced group length: {len(balanced_g)}')

        train = balanced_df[balanced_g.ngroup().isin(shuf[:int(len(shuf) * .7)])]  # change 2 to what you need :-)
        val = balanced_df[
            balanced_g.ngroup().isin(shuf[int(len(shuf) * .7):int(len(shuf) * .85)])]  # change 2 to what you need :-)
        test = balanced_df[balanced_g.ngroup().isin(shuf[int(len(shuf) * .85):])]  # change 2 to what you need :-)

        # train = balanced_df.sample(frac=0.7)
        # valtest = balanced_df.drop(train.index)
        # val = valtest.sample(frac=0.5)
        # test = valtest.drop(val.index)
        train_sizes = train.groupby(['celldata_id']).size()
        val_sizes = val.groupby(['celldata_id']).size()
        test_sizes = test.groupby(['celldata_id']).size()
        print('Dataset sizes:')
        print(f'Train {len(train_sizes)}')
        print(f'Val {len(val_sizes)}')
        print(f'Test {len(test_sizes)}')
        return train, val, test

    def run(self):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        print(f'Device {self.device}')
        assert self.device == 'cuda:0', 'gpu not used'
        df = self.get_classes()
        # df = self.apply_filters(df)
        print('df channels', df.channel.unique())
        df = df.drop_duplicates(subset=['croppath'])  # TODO: duplicate croppaths
        train_df, val_df, test_df = self.train_val_test_split(df, balance_method='cutoff')
        Early = EarlyStopper(patience=3, min_delta=0)
        Tran = BuildTransform(self.opt.target_image_size, self.opt.num_channels, imagenet=self.opt.use_imagenet)
        trainset = ImageDataset(train_df, label_type=self.opt.label_type,
                                transform=Tran.train_transform)
        valset = ImageDataset(val_df, label_type=self.opt.label_type,
                              transform=Tran.eval_transform)
        assert trainset.num2label_dct == valset.num2label_dct, 'num2label dicts must be identical'
        num_classes = len(np.unique(self.classes))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.opt.batch_size,
                                                  shuffle=True, num_workers=4,
                                                  pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.opt.batch_size,
                                                shuffle=False, num_workers=4,
                                                pin_memory=True)
        # TODO: detect height, width, channels
        if self.opt.model_type == 'resnet':
            Res = ResNet(num_classes=num_classes, num_channels=self.opt.num_channels)
            self.model = Res.model
        elif self.opt.model_type == 'cnn':
            self.model = Net(num_classes=num_classes, h=300, w=300,num_channels=self.opt.num_channels)
        elif self.opt.model_type == 'cnn_with_bn':
            self.model = NetBN(num_classes=num_classes, h=300, w=300, num_channels=self.opt.num_channels)
        elif self.opt.model_type=='cnn_with_dropout':
            self.model = NetDropout(num_classes=num_classes, h=300, w=300, num_channels=self.opt.num_channels)
        else:
            self.model=None
            assert 0, f'Model type {self.opt.model_type} not found'
                        
        self.model.to(self.device)
        modelpath = os.path.join(self.modeldir, f'model_{TIMESTAMP}.pth')
        self.hyperparams['modelpath'] = modelpath
        if self.opt.use_wandb:
            wandb.config.update(self.opt)
        self.Db.update('modeldata',
                       update_dct=dict(n_samples=self.opt.n_samples if self.opt.n_samples > 0 else len(trainset),
                                       num_channels=self.opt.num_channels,
                                       momentum=self.opt.momentum,
                                       learning_rate=self.opt.learning_rate,
                                       batch_size=self.opt.batch_size,
                                       modeltype=self.opt.model_type,
                                       epochs=self.opt.epochs,
                                       optimizer=self.opt.optimizer),
                       kwargs=dict(id=self.model_id))
        self.criterion = nn.CrossEntropyLoss()
        if self.opt.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        elif self.opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.opt.learning_rate, momentum=self.opt.momentum)
        else:
            assert 0, f'Optimizer not found {self.opt.optimizer}'

        # train_opt = torch.compile(self.train_one_epoch, mode='reduce-overhead')
        if self.opt.check_images:
            # save images
            i = 0
            
            for x in trainloader:
                print('labels', x[1])
                for im, lbl in zip(x[0], x[1]):
                    lbl = lbl.detach().cpu().numpy()
                    i += 1
                    im = im.detach().cpu().numpy()
                    im = np.moveaxis(im, 0, -1)
                    
                    im -= np.min(im)
                    im = im * (255 / np.max(im))
                    im = np.uint8(im)
                    blank = np.zeros_like(im)[..., 0]
                    
                    im = np.dstack((im, blank))
                    savepath = os.path.join(self.checkimagedir, f'checkimage-{i}-lbl-{lbl}.png')
                    imageio.v3.imwrite(savepath, im)
                print(f'Saved to {self.checkimagedir}')
                if i > 100:
                    exit(0)
                        
                        

        should_stop = False
        for epoch in range(1, self.opt.epochs + 1):  # loop over the dataset multiple times
            strt = time()
            print(f'Epoch {epoch}')
            train_loss, train_acc = self.train_one_epoch(trainloader, epoch, trainset.num2label_dct, should_stop)
            print(f'Epoch time: {time() - strt:.2f}')
            if self.opt.use_wandb:
                wandb.log({"train_loss_epoch": train_loss})
            if self.opt.use_wandb:
                wandb.log({"train_acc_epoch": train_acc})
            self.model.eval()
            with torch.no_grad():
                val_acc = 0
                val_loss = 0
                preddct = []
                with tqdm(total=len(valloader), desc=f'Validation: Epoch {epoch+1}') as pbar:
                
                    for j, (X, y, experimentdata_ids, welldata_ids, celldata_ids) in enumerate(valloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        # calculate outputs by running images through the network
                        y_pred = self.model(X)
                        val_loss_batch = self.criterion(y_pred, y)
                        val_loss += val_loss_batch.item()
                        # the class with the highest energy is what we choose as prediction
                        _, y_pred_class = torch.max(y_pred, 1)
                        batch_acc = (y_pred_class == y).sum().item() / len(y_pred)
                        val_acc += batch_acc

                        if self.opt.use_wandb:
                            wandb.log({"val_loss": val_loss_batch.item()})

                        # Save to db
                        if epoch == self.opt.epochs or should_stop:
                            preddct = []
                            np_y_pred_class = y_pred_class.detach().cpu().numpy()
                            np_y = y.detach().cpu().numpy()
                            np_y_pred = y_pred.detach().cpu().numpy()
                            for _y, _y_pred, _y_pred_class, experimentdata_id, welldata_id, celldata_id in zip(np_y,
                                                                                                            np_y_pred,
                                                                                                            np_y_pred_class,
                                                                                                            experimentdata_ids,
                                                                                                            welldata_ids,
                                                                                                            celldata_ids):
                                preddct.append(dict(id=uuid.uuid4(),
                                                    model_id=self.model_id,
                                                    experimentdata_id=uuid.UUID(experimentdata_id),
                                                    welldata_id=uuid.UUID(welldata_id),
                                                    celldata_id=uuid.UUID(celldata_id),
                                                    stage='val',
                                                    prediction=float(_y_pred_class.item()),
                                                    groundtruth=float(_y.item()),
                                                    prediction_label=valset.num2label_dct[_y_pred_class],
                                                    groundtruth_label=valset.num2label_dct[_y]))
                                preddct_check = dict(model_id=self.model_id, celldata_id=uuid.UUID(celldata_id))
                                self.Db.delete_based_on_duplicate_name('modelcropdata', preddct_check)
                            self.Db.add_row('modelcropdata', preddct)
                        pbar.set_postfix(loss=val_loss_batch.item(), accuracy=100. * batch_acc)
                        pbar.update(1)

                val_acc = val_acc / len(valloader)
                val_loss = val_loss / len(valloader)
                print(f'Accuracy of the network on validation images: {val_acc:.2f}')
                print(f'Loss of the network on validation images: {val_loss:.2f}')
                if should_stop:
                    print(f'Stopping at epoch {epoch + 1}')
                    break
                should_stop = Early.early_stop(val_loss)
                should_stop = False
                print(f'Early Stopping: {should_stop}')

        print('Finished Training')

        torch.save(self.model.state_dict(), modelpath)
        self.Db.update('modeldata', update_dct=dict(modelpath=modelpath,
                                                    train_acc=train_acc,
                                                    train_loss=train_loss,
                                                    val_loss=val_loss,
                                                    val_acc=val_acc),
                       kwargs=dict(id=self.model_id))
        print('Saved model')
        if self.opt.use_wandb:
            path_to_wandb = wandb.run.dir
            wandbparent, _ = os.path.split(path_to_wandb)
            wandbpath = os.path.join(wandbparent, f'run-{wandb.run.id}.wandb')
            self.Db.update('modeldata', update_dct=dict(wandbpath=wandbpath),
                           kwargs=dict(id=self.model_id))
            print(f'Run: \n wandb sync -p galaxy {wandbpath}')
            wandb.finish()
        print('Done')

    def train_one_epoch(self, trainloader, epoch=1, num2label_dct=None, should_stop=False):
        self.model.train()
        train_loss = 0.0
        running_loss = 0.0
        train_acc = 0.
        with tqdm(total=len(trainloader), desc=f'Training: Epoch {epoch+1}') as pbar:
        
            for batch, (X, y, experimentdata_ids, welldata_ids, celldata_ids) in enumerate(trainloader):
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
                _, y_pred_class = torch.max(y_pred, 1)
                batch_acc = (y_pred_class == y).sum().item() / len(y_pred)
                train_acc += batch_acc
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                # print statistics
                if self.opt.use_wandb:
                    wandb.log({"loss": loss.item()})
                running_loss += loss.item()
                train_loss += loss.item()
                # print(f'loss: {loss.item()}')
                # print(f'batch time: {time() - strt}')
                if batch % 100 == 0:
                    # print(f'Running loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                if (epoch == self.opt.epochs or should_stop) and num2label_dct is not None:
                    # Save to db
                    preddct = []
                    np_y_pred_class = y_pred_class.detach().cpu().numpy()
                    np_y = y.detach().cpu().numpy()
                    np_y_pred = y_pred.detach().cpu().numpy()
                    for _y, _y_pred, _y_pred_class, experimentdata_id, welldata_id, celldata_id in zip(np_y, np_y_pred,
                                                                                                    np_y_pred_class,
                                                                                                    experimentdata_ids,
                                                                                                    welldata_ids,
                                                                                                    celldata_ids):
                        preddct.append(dict(id=uuid.uuid4(),
                                            model_id=self.model_id,
                                            experimentdata_id=uuid.UUID(experimentdata_id),
                                            welldata_id=uuid.UUID(welldata_id),
                                            celldata_id=uuid.UUID(celldata_id),
                                            stage='train',
                                            prediction=float(_y_pred_class.item()),
                                            groundtruth=float(_y.item()),
                                            prediction_label=num2label_dct[_y_pred_class],
                                            groundtruth_label=num2label_dct[_y]))
                        preddct_check = dict(model_id=self.model_id, celldata_id=uuid.UUID(celldata_id))
                        self.Db.delete_based_on_duplicate_name('modelcropdata', preddct_check)
                    self.Db.add_row('modelcropdata', preddct)
                pbar.set_postfix(loss=loss.item(), accuracy=100. * batch_acc)
                pbar.update(1)
        # print epoch loss
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc / len(trainloader)
        print(f'Epoch Acc: {train_acc:.2f}')
        print(f'Epoch Loss: {train_loss:.2f}')
        return train_loss, train_acc


def filter_parser(s):
    try:
        x, y = map(str, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Must be of format columnname,value")


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
    parser.add_argument('--experiment', default = '20230901-3-msneuron-cry2-KS4', type=str)
    parser.add_argument('--label_type', type=str, default='stimulate', choices=['celltype', 'name', 'stimulate'], help="Column name in database")
    parser.add_argument('--label_name', type=str, default=None,
                        help="Match the kind of dosage added. Treatment, Antibody, Inhibitor, etc.")
    parser.add_argument('--classes', type=str, default=None,
                        help="Comma separated list of classes. If all classes in experiment, leave blank.")
    parser.add_argument('--img_norm_name', choices=['division', 'subtraction', 'identity'], default='identity', type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--filters', default=[['name', 'cry2mscarlet']], help="Filter based on columnname, filtername, i.e. name,cry2mscarlet",
                        dest="filters", type=list, nargs='+')

    parser.add_argument('--model_type', default='cnn_with_dropout', choices=['cnn', 'cnn_with_bn', 'cnn_with_dropout', 'resnet'])
    parser.add_argument('--target_image_size', default=224)
    parser.add_argument('--use_imagenet', default=1)
    parser.add_argument('--num_channels', default=2)
    parser.add_argument('--n_samples', default=100)
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--check_images', default=0)

    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='all',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='T0',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc", default='RFP1,RFP2',
                        dest="chosen_channels",
                        help="Morphology Channel")
    parser.add_argument('--tile', default=0, type=int,
                        help="Select single tile to segment. Default is to segment all tiles.")
    parser.add_argument('--use_wandb', default=0, type=int, help="Log training with wandb.")
    args = parser.parse_args()
    print(args)
    Tr = Train(args)
    Tr.run()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
import time
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import customData
import argparse

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--sign', type=str, metavar='N',
                    help='sign')
parser.add_argument('--round', type=str, metavar='N',
                    help='round')
args = parser.parse_args()
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = torch.cuda.is_available()
logs = []
batch_size = 64
num_class = 2
n_round = '1'
train_image_datasets = customData(img_path='./datasets/archive/train/',
                                np_path = './datasets/train_samples_%s.npy' %args.sign,
                                dict_path=('./datasets/dic_%s.npy' %args.sign),
                                data_transforms=data_transforms,
                                dataset='train')
test_image_datasets = customData(img_path='./datasets/archive/train/',
                                np_path = './datasets/test_samples_%s.npy' %args.sign,
                                dict_path=('./datasets/dic_%s.npy' %args.sign),
                                data_transforms=data_transforms,
                                dataset='test')
image_datasets = {'train': train_image_datasets, 'test': test_image_datasets}

# wrap your data and label into Tensor
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=batch_size,
                                             shuffle=True) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# get model and replace the original fc layer with your fc layer
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_class)
if use_gpu:
    model_ft = model_ft.cuda()

# define cost function
criterion = nn.CrossEntropyLoss()


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)


exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10, eta_min=1e-7)
from sklearn.metrics import roc_auc_score
import tqdm
train_losses = []
train_aucs = []
val_losses = []
val_aucs = []

for i in range(100):
    f = open('./results_'+args.sign+'_'+args.round+'.txt', 'a+')
    gt_labels = []
    predict_p = []
    running_loss = 0
    for data in dataloders['train']:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs.data, 1)
        prediction = torch.argmax(outputs.data, 1)
        gt_labels.extend(labels.cpu().numpy().tolist())
        predict_p.extend(nn.functional.softmax(outputs).cpu().detach().numpy()[:,1])
        loss = criterion(outputs, labels)
        running_loss+=loss.data.item()
        loss.backward()
        optimizer_ft.step()
    s = 'epoch '+str(i)+' train AUC: '+str(roc_auc_score(gt_labels,predict_p))
    f.write(s+'\n')
    print(s)
    train_epoch_loss = running_loss/dataset_sizes['train']
    train_losses.append(train_epoch_loss)
    train_aucs.append(roc_auc_score(gt_labels,predict_p))
    
    with torch.no_grad():
        gt_labels = []
        predict_p = []
        running_loss = 0
        for data in dataloders['test']:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients

            # forward
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs.data, 1)
            prediction = torch.argmax(outputs.data, 1)
            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(nn.functional.softmax(outputs).cpu().detach().numpy()[:,1])
            loss = criterion(outputs, labels)
            running_loss+=loss.data.item()
        val_epoch_loss = running_loss / dataset_sizes['test']
        exp_lr_scheduler.step(val_epoch_loss)
        val_losses.append(val_epoch_loss)
        val_aucs.append(roc_auc_score(gt_labels,predict_p))
        s = 'epoch '+str(i)+' test AUC: '+str(roc_auc_score(gt_labels,predict_p))
        print(s)
        f.write(s+'\n')
    f.close()

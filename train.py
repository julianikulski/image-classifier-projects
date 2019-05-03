import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
import helper
import utility

# command line arguments
parser = argparse.ArgumentParser(description = 'Image classifier')
# get image directory from user
parser.add_argument('data_directory', type=str, help='enter a data directory containing pictures as a training set')
# enable gpu
parser.add_argument('--gpu', action='store_true', help='enter --gpu to enable GPU mode, leave it out to stay with CPU mode')
# create checkpoint file
parser.add_argument('--save_dir', '--save_directory', type=str, help='enter a file name to save the checkpoints')
# choose network architecture
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16', 'densenet161'], help='enter vgg13, vgg16 or densenet161 as network architecture')
# set hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.001, help='set learning rate')
parser.add_argument('--epochs', type=int, default=5, help='set number of epochs')
parser.add_argument('--hidden_units', type=int, default=512, help='set hidden_units')
args = parser.parse_args()

# reading in the categories to names file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = args.data_directory
cuda = args.gpu
hidden_units = args.hidden_units
learning_rate = args.learning_rate
epochs = args.epochs

# loading and transforming the data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
trainloader, validloader, testloader = utility.transform(train_dir, valid_dir, test_dir)

device = torch.device('cuda' if cuda == True else 'cpu')

# adjusting the pretrained model
model = getattr(models, args.arch)(pretrained=True)
model = helper.set_classifier(model, hidden_units)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Move the model to GPU mode, if specified
model.to(device)

# Training the model
helper.train(epochs, 60, trainloader, validloader, model, optimizer, criterion)

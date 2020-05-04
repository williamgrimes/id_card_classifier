"""Train and predict on a ResNet50 model with 3 output classes for id card
image validation as FULL_VISIBILITY, NO_VISIBILITY, or PARTIAL_VISIBILITY"""
import argparse
import cv2
import logging
import os
import random
import time
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd

from operator import itemgetter
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

from torch.autograd import Variable

now_= time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
logger = logging.getLogger(__name__)

data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')
classes_dir = os.path.join(data_dir, 'classes')
csv_labels = os.path.join(data_dir, 'gicsd_labels.csv')
model_file = 'artifacts/model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rectify_image = lambda img: img[:,:,0]

def preprocess_images(csv_labels, images_dir, classes_dir, **kwargs):
    """Rectify image to remove noisy channels, and create images directory
    for each class label in data/classes"""
    labels_df = pd.read_csv(csv_labels)
    labels_df.columns = labels_df.columns.str.replace(' ', '')
    labels_df = labels_df.apply(lambda  # strip whitespace
            x: x.str.strip() if x.dtype == "object" else x)
    os.mkdir(classes_dir)
    for class_ in labels_df['LABEL'].unique():
        class_folder = os.path.join(classes_dir, class_)
        os.mkdir(class_folder)
        logging.info(f'Creating class folder: {class_folder}')
    for index, row in labels_df.iterrows():
        image = cv2.imread(os.path.join(images_dir, row['IMAGE_FILENAME']))
        image = rectify_image(image)
        cv2.imwrite(os.path.join(classes_dir, row['LABEL'],
                    row['IMAGE_FILENAME']), image)

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """ Balance classes in batch since many more FULL_VISIBILITY labels
    https://github.com/galatolofederico/pytorch-balanced-batch/blob/master/sampler.py
    """
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) if len(
                self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][
                self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx] #FIXME drop item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception(
                    'You should pass the tensor of labels to the constructor '
                    'as second argument')

    def __len__(self):
        return self.balanced_max * len(self.keys)

def load_split_train_test(datadir, valid_size, img_size, batch_size, **kwargs):
    """Create train test split and sample with equal samples from each class
    using BalancedBatchSampler"""
    _transformer = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.CenterCrop(200),
        transforms.ToTensor()])
    data = datasets.ImageFolder(datadir, transform=_transformer)
    n = len(data)
    n_test = int(valid_size * n)
    idx = list(range(n))
    random.shuffle(idx)
    def balanced_sample():
        test_data = torch.utils.data.Subset(data, idx[:n_test])
        test_labels = [l[-1] for l in test_data]
        train_data = torch.utils.data.Subset(data, idx[n_test:])
        train_labels = [l[-1] for l in train_data]
        test_sampler = BalancedBatchSampler(test_data, labels=test_labels)
        train_sampler = BalancedBatchSampler(train_data, labels=train_labels)
        return train_data, test_data, train_sampler, test_sampler
    def random_sample():
        train_data = datasets.ImageFolder(datadir, transform=_transformer)
        test_data = datasets.ImageFolder(datadir, transform=_transformer)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train_sampler, test_sampler = list(
            map(SubsetRandomSampler, [train_idx, test_idx]))
        return train_data, test_data, train_sampler, test_sampler
    #train_data, test_data, train_sampler, test_sampler = random_sample()
    train_data, test_data, train_sampler, test_sampler = balanced_sample()
    test_loader = torch.utils.data.DataLoader(test_data,
        sampler=test_sampler, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_data,
        sampler=train_sampler, batch_size=batch_size)
    logger.info(f'Create DataLoader train/test split with {valid_size} ratio, '
                f'{len(train_data)} train data, and {len(test_data)} test, '
                f'resize images to {img_size}, with batches of {batch_size}')
    return train_loader, test_loader

def train_model(train_loader, test_loader, epochs, print_every, learning_rate,
                **kwargs):
    """Train resnet last layer"""
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 3),
        nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    model.to(device)
    train_losses, test_losses = [], []
    logger.info(f'Training model {model}')
    logger.info(f'Training {epochs} epochs, and learning rate {learning_rate}')
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(
                            equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(test_loader))
                logger.info(f'Epoch {epoch + 1}/{epochs}.. '
                            f'Train loss: {running_loss / print_every:.3f}.. '
                            f'Test loss: {test_loss / len(test_loader):.3f}.. '
                            f'Test accuracy: {accuracy / len(test_loader):.3f}')
                running_loss = 0
                model.train()
    return model

def predict_image(model, predict, img_size, **kwargs):
    """Predict the class label for a raw image in data/images directory"""
    image = cv2.imread(predict)
    np_im = rectify_image(np.array(image))
    np_im = np.dstack((np_im, np_im, np_im))
    image = Image.fromarray(np_im)
    _transforms = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor()])
    image_tensor = _transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    _data = datasets.ImageFolder(classes_dir, transform=_transforms)
    classes = _data.classes
    logger.info(f'{predict} predicted class {classes[index]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true',
                       help='Train neural network and create model.')
    group.add_argument('--predict',
                       help='Image to run prediction on.')
    # PARAMS
    parser.add_argument('--img_size', default=224,
                        help='Size of image to be passed to ResNet')
    parser.add_argument('--valid_size', default=0.2,
                        help='Ratio of train/test split to validate model.')
    parser.add_argument('--batch_size', default=120,
                        help='Number of images in each batch.')
    # MODEL PARAMS
    parser.add_argument('--learning_rate', default=0.003,
                        help='The step size at each iteration.')
    parser.add_argument('--epochs', default=30,
                        help='Number of passes through the training dataset.')
    parser.add_argument('--print_every', default=1,
                        help='Log train test loss and accuracy every x steps.')
    args = parser.parse_args()
    kwargs_ = dict(vars(args))
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO,
        filename=f'{now_}_{"train" if kwargs_["train"] else "predict"}.log',
        filemode='w', format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    if kwargs_['train']:
        if not os.path.exists(classes_dir):
            logger.info(f'Classes dir does not exist creating: {classes_dir}')
            preprocess_images(csv_labels, images_dir, classes_dir, **kwargs_)
        logger.info(f'Classes dir exists at {classes_dir}, start training')
        train_loader, test_loader = load_split_train_test(classes_dir,
            **kwargs_)
        model = train_model(train_loader, test_loader, **kwargs_)
        logger.info(f'Training finished saving model to {model_file}')
        torch.save(model, model_file)
    elif kwargs_['predict']:
        logger.info(f'Loading model from {model_file}')
        model = torch.load(model_file)
        predict_image(model, **kwargs_)

import os
import torch
import torchvision
import torchaudio
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

# print(torch.__version__)
# print(torchvision.__version__)
from toolkit.toolkitset import Toolkits
import toolkit.toolkitset as tks

toolkit = Toolkits()

toolkit.download_dataset(dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz', 
                         file_dir = './data')

toolkit.extract_file(file_path = './data/cifar10.tgz', 
                     type = 'r:gz', extract_dir = './data')

toolkit.show_info(file_path = './data/cifar10.tgz')

dir_list = os.listdir(path = './data/cifar10')
print(dir_list)

classes = os.listdir(path= './data/cifar10/test')
print(classes)

# example of cifar10 dataset
data_dir = './data/cifar10'
airplane_files = os.listdir('./data/cifar10/test/airplane')
print(f'No of test examples of airplanes: {len(airplane_files)}')
print(airplane_files[5:10])


dataset = toolkit.load_image_dataset('./data/cifar10/train', ToTensor())
"""
each element is a tuple
contains a image tensor and a label
each image tensor in the shape [3, 32, 32] (32 * 32 pixel with 3 channels - RGB)
"""
#image, label = dataset[0]
#print(image.shape)
#print(image)

#list of classes is stored in the .classes property
print(dataset.classes)
img, label = dataset[0]

plt.rcParams['font.size'] = 12

def show_example(img, label):
    print(f'Label: {dataset.classes[label]} ({label})')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

"""
show_example(*dataset[0])
show_example(*dataset[30])
"""

""" 
training and dataset

Training set
Validation set
Test set
"""
random_seed = 77
torch.manual_seed(random_seed)
# print(len(dataset))

# a small portion(5000) from traning data as valiation data
val_size = 5000
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# print(len(train_dataset)), print(len(val_dataset))


# create data loaders for training and validation
# load data in batches
batch_size = 64

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(train_dataset, batch_size*2, shuffle=True, num_workers=2, pin_memory=True)

"""
show batch images from the dataset
with matplot library
more details to look up into
"""
# look at the batches of images from the dataset using the make_grid method from torchvision
def show_batch(dataload):
    dataload_iter = iter(dataload)
    images, labels = next(dataload_iter)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([]); ax.set_yticks([])
    ax. imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.show()

#show_batch(train_dl)
#show_batch(val_dl)

"""
define a simple convolutional neural network
"""

# a foo kernel function
def apply_kernel(image, kernel):
    ri, ci = image.shape
    rk, ck = kernel.shape
    ro, co = ri - rk + 1, ci - ck + 1
    output = torch.zeros([ro, co])
    for i in range(ro):
        for j in range(co):
            output[i, j] = torch.sum(image[i:i+rk, j:j+ck] * kernel)
    return output

sample_image = torch.tensor([
    [3, 3, 2, 1, 0], 
    [0, 0, 1, 3, 1], 
    [3, 1, 2, 2, 3], 
    [2, 0, 0, 2, 2], 
    [2, 0, 0, 0, 1]
], dtype=torch.float32)

sample_kernel = torch.tensor([
    [0, 1, 2], 
    [2, 2, 0], 
    [0, 1, 2]
], dtype=torch.float32)

#print(apply_kernel(sample_image, sample_kernel))

# kernel, max-pooling

import torch.nn as nn
import torch.nn.functional as func

"""
nn.Sequential in PyTorch is a container module 
that allows to stack multiple layers or modules sequentially. 
"""

simple_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2, 2)
)

for images, labels in train_dl:
    print(images.shape)
    out = simple_model(images)
    print(out.shape)
    break

# calculate accuracy for a classification taks in PyTorch
def accuracy(outputs, labels):
    # Compute predicted labels by finding the index of maximum log-probability
    _, predictions = torch.max(outputs, 1)
    # calculate accuracy
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

# define a model which contains methods
class ImageClassification(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = func.cross_entropy(out, labels) # calculate loss
        return loss

    def valiation_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = func.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class Cifar10CnnModel(ImageClassification):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)


model = Cifar10CnnModel()
print(model)

for images, labels in train_dl:
    print(images.shape)
    out = model(images)
    print(out.shape)
    print(out[0])
    break

# load data into gpu
tks.get_defualt_device()

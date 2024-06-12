import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as tf
import torch.optim as optim
import time

# training on GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# the output of torchvision datasets are PILImage images of range [0,1]
from PIL import Image

transform = tf.Compose([
    #tf.Resize((128, 128)),
    tf.ToTensor(), # Convert the image to a PyTorch Tensor
    tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

batch_size = 6
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, 
                                          download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=True, num_workers=5)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                         shuffle=True, num_workers=5)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

################################################
# define a convolutional neural network
import torch.nn as nn
import torch.nn.functional as func

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

convNet = ConvNet()

convNet.to(device)

trained_model_path = './cifar_covnet.pth'

# yolo test
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# define loss function amd optimizer

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(convNet.parameters(), lr=0.001,momentum=0.9)


training_start_time = time.time()
# train the convolutional network
for epoch in range(10):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data is in format of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)    

        # initialize the parameter gradients to zeros
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = convNet(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            print('training time : %.3f seconds' % (time.time() - training_start_time))

print('training finished...')

torch.save(convNet.state_dict(), trained_model_path)

testiter = iter(testloader)
images, labels = next(testiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# load the trained model
convNet = ConvNet()
convNet.load_state_dict(torch.load('./cifar_covnet.pth'))

outputs = convNet(images)

notimportant, predicted = torch.max(outputs, 1)
#print(predicted[0])
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))


# preformance on the whole dataset
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = convNet(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

class ConvNetFoo(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
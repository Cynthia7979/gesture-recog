import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy
import gesture from model as gmodel

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 500
N_CLASSES = 3

# get the mean and std
trans_getstat = transforms.Compose([
    transforms.RandomResizedCrop(137),
    transforms.GrayScale()
    ])
stat_data = ImageFolder(root=r'~/data', transform=trans_getstat)
stat_mean, stat_std = getStat(stat_data)
del stat_data

# load the data
transform = transforms.Compose([
    transforms.RandomResizedCrop(137),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(360),
    transforms.GrayScale(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [stat_mean],
                         std  = [stat_std]),
    ])

trainData = datasets.ImageFolder("~/data/train", transform)
testData = 

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for X, _ in train_loader:
        mean[0] += X.mean()
        std[0] += X.std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


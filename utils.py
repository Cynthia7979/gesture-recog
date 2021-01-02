import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy
import os

'''
def getStat(datafolder:str, trans=transforms.ToTensor()):
    traindata = datasets.ImageFolder(datafolder, trans)
    trainLoader = torch.utils.data.DataLoader(dataset=traindata) 
    image_means = torch.stack([t.mean(1).mean(1) for t, c in trainLoader])
    image_means.mean(0)
    return image_means
'''

'''    
def getStat(train_data):
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
'''

def save_loss_values(directory:str, epoch:int, all_loss:list):
    with open(os.path.join(directory, str(epoch) + "_all"), "w") as fp:
        for i in all_loss:
            fp.write(str(i) + '\n')
        fp.flush()
    return


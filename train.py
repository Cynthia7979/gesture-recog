import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy
import os
import model.gesture as gmodel

# define
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

# data
trainData = datasets.ImageFolder("~/data/train", transform)
testData = datasets.ImageFolder("~/data/train", transform)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

# starts training 
model = gmodel()    # n_class = 3
model.cuda()

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(EPOCH):
    avg_loss = 0
    count = 0

    loss_list = []

    for images, labels in trainLoader:
        images = images.cuda()
        labels = labels.cuda()

        # Forward
        optimizer.zero_grad()
        _, outputs = model(images)
        # Calculate loss
        loss = cost(outputs, labels)
        loss_list += loss       # store all loss
        avg_loss += loss.data   # get avg loss
        cnt += 1
        # Backward + Optimize
        loss.backward()
        optimizer.step()
    # Schedule
    scheduler.step(avg_loss)
    
    # save loss value
    os.mkdir("visual-loss")
    save_loss_values("visual-loss", epoch, loss_list)
    with open("visual-loss/" + str(epoch) + "_avg", "w") as fp:
        fp.write(str(avg_loss / cnt))

    # auto save the model
    if 0 == epoch % 30:
        torch.save(model.state_dict(), 'cnn.pkl')


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

def save_loss_values(directory:str, epoch:int, all_loss:list):
    with open(directory + str(epoch) + "_all", "w") as fp:
        for i in all_loss:
            fp.write(str(i) + '\n')
        fp.flush()
    return


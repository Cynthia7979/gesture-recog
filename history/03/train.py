import torch
import torch.nn as tnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import gesture as gmodel
#from utils import getStat
from utils import save_loss_values

# define
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 700
N_CLASSES = 3
# from https://discuss.pytorch.org/t/solved-how-do-i-display-a-grayscale-image/35653
stat_mean = 0.1307
stat_std = 0.3081

# some trainng
if "__main__" == __name__:

# train
    # use gpu1
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # load the data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(137),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(360),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((stat_mean,),
                             (stat_std,)),
        ])

    # data
    trainData = datasets.ImageFolder("~/data/train", transform)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)

    # starts training 
    print('start...')
    print('constructing the model...')
    model = gmodel()    # n_class = 3
    print('copy to the video card...')
    model.cuda()
    print('done')

    # Loss, Optimizer & Scheduler
    print('initializing...')
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    print('done')

    print('start training.')
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
            loss_list += [float(loss.data)] # store all loss
            avg_loss += loss.data           # get avg loss
            count += 1
            print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/count))
            # Backward + Optimize
            loss.backward()
            optimizer.step()
        # Schedule
        scheduler.step(avg_loss)
        
        # save loss value
        if not os.path.exists('visual-loss'):
            os.mkdir("visual-loss")
        save_loss_values("visual-loss", epoch, loss_list)
        with open("visual-loss/" + str(epoch) + "_avg", "w") as fp:
            fp.write(str(float(avg_loss / count)))

        # auto save the model
        if 0 == (epoch + 1) % 30:
            torch.save(model, 'cnn.pkl')

    torch.save(model, 'cnn.pkl')

'''
# get the mean and std
    trans_getstat = transforms.Compose([
        transforms.RandomResizedCrop(137),
        transforms.Grayscale()
        ])
#stat_data = datasets.ImageFolder('~/data/train')#, transform=trans_getstat)
    stat_mean, stat_std = getStat('~/data/train', trans_getstat)
    del stat_data
'''


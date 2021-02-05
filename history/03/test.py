import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from train import BATCH_SIZE
from train import stat_mean
from train import stat_std
from model import gesture as gmodel

# some test
if "__main__" == __name__:

# test
    # use gpu1
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # load the data
    print('load the data...')
    transform = transforms.Compose([
        transforms.RandomResizedCrop(137),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(360),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((stat_mean,),
                             (stat_std,)),
        ])

    testData = datasets.ImageFolder("~/data/test", transform)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)
    print('done')

    # load the model
    #model = gmodel()
    print('constructing the model')
    model = gmodel()
    print('load the model')
    model.load_state_dict(torch.load('cnn.pkl'))
    model.cuda()
    print('done')

    # test the model
    print('lock the model')
    model.eval()
    correct = 0
    total = 0
    print('start testing')
    for images, labels in testLoader:
        images = images.cuda()
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100* correct/total))


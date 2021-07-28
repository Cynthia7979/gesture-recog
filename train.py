import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 250
N_CLASSES = 3
PATIENCE = 20

def simple_vgg_conv_block(in_list, out_list, k_list, p_list):
    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    return tnn.Sequential(*layers)
    
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        #self.layer0 = simple_vgg_conv_block([1,3], [3,3], [3,3], [0,0])
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

# some train
if "__main__" == __name__:
    # ref: https://blog.csdn.net/batmanchen/article/details/109897788
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(360),
        transforms.RandomAffine(360),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.8),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ]),
        ])

    testTransform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ]),
        ])

    trainData = dsets.ImageFolder('~/data/train', transform)
    testData = dsets.ImageFolder('~/data/test', testTransform)

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

          
    # use gpu1
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # loss value and acc
    train_losses = []
    test_losses = []
    acc_list = []
    # early stop
    early_stop = EarlyStopping(patience=PATIENCE, verbose=True)

    vgg16 = VGG16(n_classes=N_CLASSES)
    vgg16.cuda()

    # Loss, Optimizer & Scheduler
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train the model
    for epoch in range(EPOCH):
        
        # train set
        vgg16.train()
        avg_loss = 0
        cnt = 0
        for images, labels in trainLoader:
            images = images.cuda()
            labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            _, outputs = vgg16(images)
            loss = cost(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            print("[E: %d] \tloss: %f  \tavg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        train_losses += [avg_loss / cnt]

        # test set
        vgg16.eval()
        valid_losses = []
        # acc
        correct = 0
        total = 0
        # loss
        avg_loss = 0
        cnt = 0
        for images, labels in testLoader:
            images = images.cuda()
            _, outputs = vgg16(images)
            # acc
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
            # loss
            labels = labels.cuda()
            loss = cost(outputs, labels)
            valid_losses += [loss.item()]
            avg_loss += loss.data
            cnt += 1
            print("Test losses: %f" % (loss.data))
        # acc
        acc_list += [correct / total * 100]
        # loss
        test_losses += [avg_loss / cnt]
        valid_loss = avg_loss / cnt
        
        # early stop
        early_stop(valid_loss, vgg16)

        # save the model
        if 0 == epoch % 10:
            torch.save(vgg16.state_dict(), 'cnn.pkl')
            torch.save(vgg16, 'module.pkl')

        if early_stop.early_stop:
            print("Early stopping at epoch: " + str(epoch))
            break

    # save output data... (loss values...)
    with open('train-loss', 'w') as fp:
        for i in train_losses:
            fp.write(str(i) + '\n')
    with open('test-loss', 'w') as fp:
        for i in test_losses:
            fp.write(str(i) + '\n')

    # Test the model
    vgg16.eval()
    correct = 0
    total = 0

    for images, labels in testLoader:
        images = images.cuda()
        _, outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100* correct/total))

    # Save the Trained Model
    torch.save(vgg16.state_dict(), 'cnn.pkl')
    torch.save(vgg16, 'module.pkl')


import torch 
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
import datetime
import numpy as np
from PIL import Image

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 500
N_CLASSES = 3

class AddSaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h,w,c = img.shape
        Nd = self.density
        Sd = 1-Nd
        mask = np.random.choice((0,1,2),size=(h,w,1),p=[Nd/2.0,Nd/2.0,Sd])
        mask = np.repeat(mask,c,axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

#亮度随机变化
transform1 =  transforms.Compose([
   # AddSaltPepperNoise(0.15),
    #transforms.ColorJitter(0.5,0.5,0.5,0.25),
    transforms.RandomResizedCrop(226),
    #transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.449],std=[0.226])
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ])
    ])

transform2 = transforms.Compose([
    #AddSaltPepperNoise(0.15),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449],std=[0.226])
    #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    #                     std  = [ 0.229, 0.224, 0.225 ])
    ])

transform3 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(0.6,(0.02,0.33),(0.3,3.3),0),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ])
    ])

print(type(transform2))
my_transform = transforms.RandomApply([transform2,transform3],0.5)
print(type(my_transform))

trainData = dsets.ImageFolder('~/mnt/e/ai/haiwai/data//train', transform1)
testData = dsets.ImageFolder('~/mnt/e/ai/haiwai/data//test', transform1)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

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

if __name__ == "__main__":
   # pretrained_net = torch.load("cnn_grayscale.pkl")
    vgg16 = VGG16(n_classes=N_CLASSES)
    #vgg16.load_state_dict(pretrained_net)
    vgg16.cuda()

    # Loss, Optimizer & Scheduler
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    #train begin
    start = datetime.datetime.now()

    fp2 = open("loss_all.txt",'w')
    # Train the model
    for epoch in range(EPOCH):
        #if epoch<20:
        #    LEARNING_RATE = 0.01
        #    optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
        #else:
        #    LEARNING_RATE = 0.005
        #    optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)

        avg_loss = 0
        cnt = 0
        fp = open(str(epoch)+"_all.txt",'w')
        for images, labels in trainLoader:
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            _, outputs = vgg16(images)
            loss = cost(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            print("[E: %d] loss:%f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
            #save this time lost into "epoch_all.txt"
            fp.write(str(loss.data))
            fp.write('\r\n')#add line break

            loss.backward()
            optimizer.step()

        fp.close()
        scheduler.step(avg_loss)
        #save the model
        if 0==epoch%10:
            torch.save(vgg16.state_dict(),'cnn_best.pkl')
    
        #save avg_loss
        fp2.write(str(avg_loss/cnt))
        fp2.write("\r\n")

    fp2.close()

    # Test the model
    vgg16.eval()
    correct = 0
    total = 0

    fp3 = open("acc_all.txt",'w')
    for images, labels in testLoader:
        images = images.cuda()
        _, outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100* correct/total))
        fp3.write(str(100*correct/total))
        fp3.write('\r\n')

    fp3.close()
    #train end
    end = datetime.datetime.now()
    print((end-start).seconds/60)

    # Save the Trained Model
    torch.save(vgg16.state_dict(), 'cnn_best.pkl')


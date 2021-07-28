from vgg import VGG16
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

BATCH_SIZE = 10
LEARNING_RATE = 0.05
EPOCH = 300
N_CLASSES = 3

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = dsets.ImageFolder('~/data/train', transform)
testData = dsets.ImageFolder('~/data/test', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

vgg16 = VGG16(n_classes=N_CLASSES)
vgg16.cuda()

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


# Train the model
for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    if 10 == epoch:
        LEARNING_RATE = 0.01
        optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    if 65 == epoch:
        LEARNING_RATE = 0.005
        optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    if 100 == epoch:
        LEARNING_RATE = 0.001
        optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    if 170 == epoch:
        LEARNING_RATE = 0.0005
        optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    for images, labels in trainLoader:
        images = images.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, outputs = vgg16(images)
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)
    # save the model
    if 0 == epoch % 10:
        torch.save(vgg16.state_dict(), 'cnn.pkl')

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
torch.save(vgg16, 'model.pkl')


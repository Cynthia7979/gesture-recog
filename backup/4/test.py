import torch
from train import VGG16
from train import testLoader

if "__main__" == __name__:
    vgg16 = VGG16(n_classes=3)
    vgg16.load_state_dict(torch.load("cnn.pkl"))
    vgg16.cuda()
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
        

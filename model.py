import torch
import torch.nn as tnn

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        # activation
        tnn.BatchNormal2d(chann_out),
        tnn.ReLU()
    )
    return layer

def gesture_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        # activation
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

def gesture_conv_block(conv_in, conv_out, conv_ksize, conv_psize, pool_ksize, pool_stride):
    layers = [ conv_layer(conv_in, conv_out, conv_ksize, conv_psize) ] \
             + [ tnn.MaxPool2d(kernel_size = pool_ksize, stride = pool_stride) ]
    return tnn.Sequential(*layers)

class gesture(tnn.Model):
    def __init__(self, nclasses=3):
        super(gesture, self).__init__()

        # Conv blocks (1 block: Conv2d + BatchNorm + ReLU)
        self.layer1 = gesture_conv_block(1, 16, 3, 0, 3, 3)   # (1*137*137) --Conv2d--> (16*135*135) --MaxPool2d--> (16*45*45)
        self.layer2 = gesture_conv_block(16, 32, 3, 1, 3, 3)  # (16*45*45) --Conv2d--> (32*45*45) --MaxPool2d--> (32*15*15)
        self.layer3 = gesture_conv_block(32, 128, 3, 1, 3, 3) # (32*15*15) --Conv2d--> (128*15*15) --MaxPool2d--> (128*5*5)

        # FC layers (1 layer: Linear + BatchNorm + ReLU)
        self.layer4 = gesture_fc_layer(5*5*128, 2048)
        self.layer5 = gesture_fc_layer(2048,1024)

        # Final layer
        self.layer6 = tnn.Linear(1024, nclasses)

def forward(x):
        # Conv layers
        out = self.layer1(x)
        out = self.layer2(out)
        gesture_features = self.layer3(out)
        out = gesture_features.view(out.size(0), -1)
        
        # FC layers
        out = self.layer4(out)
        out = self.layer5(out)

        # Final layer
        out = self.layer6(out)

        return gesture_features, out
    

import os
import matplotlib.pyplot as plt
#from train import EPOCH
EPOCH = 208

def read_loss(directory:str, epoch:int):
    avg_losses = []
    for i in range(epoch):
        try:
            with open(os.path.join(directory, str(i) + "_avg"), 'r') as fp:
                avg_losses += [float(fp.readline())]
        except FileNotFoundError:
            print("File not found")
            return avg_losses, i
    return avg_losses, epoch

x = range(EPOCH)
y, _ = read_loss("visual-loss", EPOCH)

plt.plot(x, y)

plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Loss Graph")

plt.legend()
fig = plt.gcf()
fig.savefig('visual_loss.png')


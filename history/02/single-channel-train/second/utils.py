import torch
import numpy
import os

def save_loss_values(directory:str, epoch:int, all_loss:list):
    with open(directory + str(epoch) + "_all", "w") as fp:
        for i in all_loss:
            fp.write(str(i) + '\n')
        fp.flush()
    return


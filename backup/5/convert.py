import torch
from train import VGG16

pth = r"module.pkl"
model = torch.load(pth, map_location=torch.device('cpu'))

tracem = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
tracem.save(r"module.pt")


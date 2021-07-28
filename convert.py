'''
import torch
from train import VGG16

pth = r"module.pkl"
model = torch.load(pth, map_location=torch.device('cpu'))

tracem = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
tracem.save(r"module.pt")
'''
import torch
from train import VGG16

module = VGG16(n_classes=3)
module.load_state_dict(torch.load('module.pkl',  map_location=torch.device('cpu')).state_dict())
module.eval()
trace_module = torch.jit.trace(module, torch.rand(1, 3, 224, 224))


'''
print(trace_module.code)
output = trace_module(torch.ones(1, 3, 224, 224))
print(output)
trace_module.save('model.pt')
'''


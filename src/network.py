import torch.nn as nn
from torch.nn import Flatten, Sequential

class neural_network(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.flatten: Flatten = nn.Flatten() 
        self.network_stack: Sequential = nn.Sequential( nn.Linear(in_features = 28*28, out_features = 512), nn.ReLU(), nn.Linear(in_features = 512, out_features = 10))

    def forward(self,x): 
        x = self.flatten(x) 
        output = self.network_stack(x) 
        
        return output
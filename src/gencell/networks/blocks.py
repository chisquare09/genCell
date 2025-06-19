import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            DenseBatchNorm(dim, dim*2,  activation='relu'),
            DenseBatchNorm(dim*2, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class DenseBatchNorm(nn.Module):
    def __init__(self,in_units,out_units,activation=None):
        super().__init__()
        layers = [
            nn.Linear(in_units,out_units),
            nn.BatchNorm1d(out_units)
        ]
        if activation == 'relu':
            layers.append(nn.ReLU())
        self.dense_bn_act = nn.Sequential(*layers)

    def forward(self,x):
        return self.dense_bn_act(x)
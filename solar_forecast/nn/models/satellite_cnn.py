import torch.nn as nn

# import attention
import torch.nn.functional as F
from solar_forecast.nn.layers.attention import Attention

class SatelliteCNN(nn.Module):
    def __init__(self, input_channels, conv1_channels, conv2_channels, kernel_size, pool_size, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, conv1_channels, kernel_size=kernel_size, padding=1) 
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=1) 
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.attn = Attention(conv2_channels)  
        self.fc = nn.Linear(conv2_channels, feature_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))  

        B, C, H, W = x.size()
        x = x.view(B, H * W, C) # reshape               
        x = self.attn(x)                     
        out = self.fc(x)                  
        return out


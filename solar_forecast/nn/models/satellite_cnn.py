import torch.nn as nn

# import attention
import torch.nn.functional as F
from solar_forecast.nn.layers.attention import Attention

class SatelliteCNN(nn.Module):
    def __init__(self, input_channels=8*13, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.attn = Attention(input_dim=128)  
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))  

        B, C, H, W = x.size()
        x = x.view(B, H * W, C)               
        x = self.attn(x)                     
        out = self.fc(x)                  
        return out


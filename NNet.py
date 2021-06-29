import torch
import torch.nn as nn
import torch.nn.functional as F

class GooseNet(nn.Module):
    def __init__(self, n_channels):
        super(GooseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 12, out_channels = n_channels, 
                               kernel_size = 3, padding = (1,1), padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels = n_channels, out_channels = 2 * n_channels, 
                               kernel_size = 3, padding = (1,1), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels = 2 * n_channels, out_channels = 4 * n_channels, 
                               kernel_size = 3, padding = (1,1), padding_mode='circular')
        self.final_conv = nn.Conv2d(in_channels = 4 * n_channels, out_channels = 1, 
                               kernel_size = 5, padding = 0)
        
    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(0)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        north = x[:,:, 0:5, 3:8]
        east = torch.rot90(x[:,:, 1:6, 4:9], 1, [2,3])
        south = torch.rot90(x[:,:, 2:7, 3:8], 2, [2,3])
        west = torch.rot90(x[:,:, 1:6, 2:7], 3, [2,3])
        
        # т.к. выходной вектор сети имеет форму [batch_size,1,1,4] - число батчей, число каналов, высота, ширина 
        # (последний слой - это сверточная сеть, тут не очень очевидно с выходом, может добавить линейный слой?)
        # убираем лишние оси
        north = self.final_conv(north).squeeze(1).squeeze(2)
        east = self.final_conv(east).squeeze(1).squeeze(2)
        south = self.final_conv(south).squeeze(1).squeeze(2)
        west = self.final_conv(west).squeeze(1).squeeze(2)
        # Склеиваем все результаты в один батч
        return torch.cat([north, east, south, west], 1)


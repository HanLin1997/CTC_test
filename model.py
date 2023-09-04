import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 120, kernel_size=3)
        self.conv2 = nn.Conv2d(120, 100, kernel_size=3)
        self.conv3 = nn.Conv2d(100, 85, kernel_size=3)
        self.conv4 = nn.Conv2d(85, 66, kernel_size=3)


    def forward(self, x):
        #x = x.view(8, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  #(24, 259)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  #(11, 128)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  #(4, 63)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  #(1, 30)
        return F.log_softmax(x, dim=1)
    
if __name__ == '__main__':
    x = torch.randn(2, 1, 50, 520)
    net = Net()
    y = net(x)
    y = y.view(30, 2, 66)
    y.shape
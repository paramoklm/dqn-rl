import torch
import torch.nn as nn

class DQN(nn.Module):
    '''
    Our DQN model
    '''
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(3136, 512)  # 3136 = 64 * 7 * 7
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDenoiser(nn.Module):
    INPUT_SIZE = (64, 64)

    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # encoder layers #
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers #
        # transpose layer, a kernel of 2 and a stride of 2
        # will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 64, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # encode #
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x = self.dropout(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        # x = self.dropout(x)

        # decode #
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # x = self.dropout(x)
        x = F.relu(self.t_conv2(x))
        # x = self.dropout(x)
        x = F.relu(self.t_conv3(x))
        # x = self.dropout(x)
        # transpose again, output should have a sigmoid applied
        x = torch.sigmoid(self.conv_out(x))

        return x
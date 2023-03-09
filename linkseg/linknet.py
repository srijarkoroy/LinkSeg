import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision.models import resnet


class BasicBlock(nn.Module):
  
    """
    Class for each Basic Block in ResNet18
    
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
      
        """
        Parameters:
        
        - in_planes: input channels of image passed to block
        
        - out_planes: output channels required
        
        - stride(default: 1): stride to be assigned
        
        - kernel_size: kernel size for conv in each block
        
        - stride(default: 1): stride to be assigned to each block
        
        - padding(default: 0): amount of padding to be assigned to each block
        
        - groups(default: 1): groups to be assigned to each block
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None

        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
      
        """
        Parameters:
        
        - in_planes: no. of input channels
        
        - out_planes: no. of output channels
        
        - kernel_size: kernel size for conv in each block
        
        - stride(default: 1): stride to be assigned to each block
        
        - padding(default: 0): amount of padding to be assigned to each block
        
        - groups(default: 1): groups to be assigned to each block
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """
      
        super(Encoder, self).__init__()

        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        
        """
        Parameters:
        
        - in_planes: no. of input channels
        
        - out_planes: no. of output channels
        
        - kernel_size: kernel size for conv transpose
        
        - stride(default: 1): stride to be assigned to conv transpose
        
        - padding(default: 0): amount of padding to be assigned to conv transpose
        
        - output_padding(default: 0): output padding to be assigned to conv transpose
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """
        
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):

        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x  


class Linknet(nn.Module):

    """
    Generate Model Architecture
    """

    def __init__(self, n_classes = 1):

        """
        Parameters:
        
        n_classes(default: 1): number of output neurons
        
        """

        super(LinkNet, self).__init__()

        # base = resnet.resnet18(pretrained=True)

        # self.in_block = nn.Sequential(
        #     base.conv1,
        #     base.bn1,
        #     base.relu,
        #     base.maxpool
        # )

        # self.encoder1 = base.layer1
        # self.encoder2 = base.layer2
        # self.encoder3 = base.layer3
        # self.encoder4 = base.layer4

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):

        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):

        """
        This class is used for building a convolutional block. Each convolutional block contains:

        - Conv2d layers (2)
        - BatchNorm2d layers following each Conv2d  
        - ReLU activation after each BatchNorm2d

        Parameters:

        - in_channel: no. of input channels to the convolutional block

        - out_channel: no. of output channels from each convolutional block
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel):

        """
        This class is used for building a encoder block. Each encoder block contains:

        - Convolutional block (1)
        - MaxPool2d following the convolutional block (1)  

        Parameters:

        - in_channel: no. of input channels to the encoder block

        - out_channel: no. of output channels from each encoder block
        """

        super().__init__()

        self.conv = ConvBlock(in_channel, out_channel)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel):

        """
        This class is used for building a decoder block. Each decoder block contains:

        - ConvTranspose2d layers (2)
        - Concatenation of upsampled and skip 
        - Convolutional block (2)

        Parameters:

        - in_channel: no. of input channels to the decoder block

        - out_channel: no. of output channels from each decoder block
        """

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channel+out_channel, out_channel)

    def forward(self, inputs, skip):

        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class LinkNet(nn.Module):

    def __init__(self):

        """
        Main LinkNet model
        """

        super().__init__()

        """ Encoder """
        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.b = ConvBlock(512, 1024)

        """ Decoder """
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        # Encoder 
        s1, p1 = self.encoder1(inputs)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        bottleneck = self.b(p4)

        # Decoder 
        d1 = self.decoder1(bottleneck, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        out = self.outputs(d4)

        return out


## Usage ##

# if __name__ == "__main__":

#     noise = torch.randn(2, 3, 512, 512)

#     model = LinkNet()
#     print("Model:", model)
#     print("\nOutput:", model(noise).shape)
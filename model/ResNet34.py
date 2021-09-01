import torch
import torchvision.models.resnet as resnet
from model.common.res_block import ResidualBlock
class ResNet34(torch.nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.maxArg=8
        self.output_size = [176,304]
        #------------------------------- ENCODER ------------------------------------------
        self.conv7by7 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding = 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder1=ResidualBlock(64, 64, 3, False)
        self.encoder2=ResidualBlock(64, 128,4, True)
        self.encoder3=ResidualBlock(128, 256,3, True)
        self.encoder4=ResidualBlock(256, 512,2, True)

        self.maxPooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #------------------------------- DECODER ------------------------------------------

        self.decoder1 = torch.nn.Sequential(
#             torch.nn.Conv2d(512, 256, kernel_size=1, stride = 1, padding = 0),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.ReLU(),
        )
        self.decoder2 = torch.nn.Sequential(
#             torch.nn.Conv2d(256, 128, kernel_size=1, stride = 1, padding = 0),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(64, 64 ,kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0),
#             torch.nn.Conv2d(128, 128, kernel_size=7, stride = 2, padding = 3),
            
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv7by7(x)
        
        out = self.encoder1(out)
        out = self.maxPooling(out)
        out = self.encoder2(out)
        out = self.maxPooling(out)
        out = self.encoder3(out)
        out = self.maxPooling(out)
        out = self.encoder4(out)

        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)

        return out
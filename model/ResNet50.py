import torch
import torchvision.models.resnet as resnet
from model.common.res_block import ResidualBlock
class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.maxArg=8
        self.output_size = [368,640]
        #------------------------------- ENCODER ------------------------------------------
        self.conv7by7 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding = 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2=self.make_layer(64, 64, 256, 3, True, False)
        self.conv3=self.make_layer(256, 128, 512, 4, False, False)
        self.conv4=self.make_layer(512, 256, 1024, 6, False, False)
        self.conv5=self.make_layer(1024, 512, 2048, 3, False, True)
        self.avp=torch.nn.AvgPool2d(kernel_size=7, stride=1)


        self.maxPooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #------------------------------- DECODER ------------------------------------------
        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 512,  kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512,  kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256,  kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 64,  kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32,  kernel_size=3, stride=1,  padding=1),
            torch.nn.Conv2d(32, 32,  kernel_size=3, stride=1,  padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 7,  kernel_size=3, stride=2, output_padding=1, padding=1)
        )
        
    def make_layer(self, in_dim, mid_dim, out_dim, repeats, dim_down = True, scale_down=False):
        layers = []
        layers.append(ResidualBlock(in_dim, mid_dim, out_dim, dim_down=True,  scale_down=scale_down, bottleNeck = True))
        for i in range(1, repeats):
            layers.append(ResidualBlock(out_dim, mid_dim, out_dim, dim_down=False,  scale_down=False, bottleNeck = True))
        return torch.nn.Sequential(*layers)

    def forward(self, x):

  

        out = self.conv7by7(x)
#         print("------------In 2 ----------------")
        out = self.conv2(out)
#         print("------------In 3 ----------------")
        out = self.conv3(out)
#         print("------------In 4 ----------------")
        out = self.conv4(out)
#         print("------------In 5 ----------------")
        out = self.conv5(out)
        out = self.decoder1(out)
        # out = self.decoder2(out)
        out = self.decoder3(out)

        return out
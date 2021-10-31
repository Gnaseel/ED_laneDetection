import torch
from model.common.res_block import ResidualBlock
class ResNet34_delta(torch.nn.Module):
    def __init__(self):
        super(ResNet34_delta, self).__init__()
        self.maxArg=8
        self.output_size = [368,640]
        #------------------------------- ENCODER ------------------------------------------
        self.conv7by7 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding = 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2=self.make_layer(64, 64, 64, 3, True, False)
        self.conv3=self.make_layer(64, 64, 128, 4, False, False)
        self.conv4=self.make_layer(128, 128, 256, 6, False, False)
        self.conv5=self.make_layer(256, 256, 512, 3, False, False)


        self.maxPooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.decoder1 = torch.nn.Sequential(
            # torch.nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, output_padding=1, padding=1),
            # torch.nn.ReLU(),
            torch.nn.Conv2d(512, 256,  kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512,  kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128,  kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64,  kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32,  kernel_size=3, stride=1,  padding=1),
            torch.nn.Conv2d(32, 32,  kernel_size=3, stride=1,  padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1,  kernel_size=3, stride=2, output_padding=1, padding=1)
        )
        
    def make_layer(self, in_dim, mid_dim, out_dim, repeats, dim_down = True, scale_down=False):
        layers = []
        layers.append(ResidualBlock(in_dim, mid_dim, out_dim, dim_down=True,  scale_down=scale_down, bottleNeck = False))
        for i in range(1, repeats):
            layers.append(ResidualBlock(out_dim, mid_dim, out_dim, dim_down=False,  scale_down=False, bottleNeck = False))
        return torch.nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv7by7(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.decoder1(out)
        out = self.decoder3(out)
        # # out = self.decoder2(out)

        return out
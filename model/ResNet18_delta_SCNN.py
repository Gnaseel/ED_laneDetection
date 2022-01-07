import torch
from model.common.res_block import ResidualBlock
import torch.nn.functional as F
class ResNet18_delta_SCNN(torch.nn.Module):
    def __init__(self):
        super(ResNet18_delta_SCNN, self).__init__()
        self.maxArg=8
        self.output_size = [368,640]
        #------------------------------- ENCODER ------------------------------------------
        self.conv7by7 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding = 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2=self.make_layer(64, 64, 64, 2, True, False)
        self.conv3=self.make_layer(64, 64, 128, 2, False, False)
        self.conv4=self.make_layer(128, 128, 256, 2, False, False)
        self.conv5=self.make_layer(256, 256, 512, 2, False, False)


        self.maxPooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256,  kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128,  kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64,  kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32,  kernel_size=3, stride=1,  padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32,  kernel_size=3, stride=1,  padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 2,  kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ReLU(),
        )

        self.dim_set = torch.nn.Sequential( 
            torch.nn.Conv2d(512, 128, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
        )

        self.dim_set_re = torch.nn.Sequential(
            torch.nn.Conv2d(128, 512, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
        )

        self.conv_d = torch.nn.Conv2d(128, 128, (1, 9), padding=(0, 4), bias=False)
        self.conv_u = torch.nn.Conv2d(128, 128, (1, 9), padding=(0, 4), bias=False)
        self.conv_r = torch.nn.Conv2d(128, 128, (9, 1), padding=(4, 0), bias=False)
        self.conv_l = torch.nn.Conv2d(128, 128, (9, 1), padding=(4, 0), bias=False)

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

        out = self.dim_set(out)
        for i in range(1, out.shape[2]):
            out[..., i:i+1, :].add_(F.relu(self.conv_d(out[..., i-1:i, :])))

        for i in range(out.shape[2] - 2, 0, -1):
            out[..., i:i+1, :].add_(F.relu(self.conv_u(out[..., i+1:i+2, :])))

        for i in range(1, out.shape[3]):
            out[..., i:i+1].add_(F.relu(self.conv_r(out[..., i-1:i])))

        for i in range(out.shape[3] - 2, 0, -1):
            out[..., i:i+1].add_(F.relu(self.conv_l(out[..., i+1:i+2])))
        out = self.dim_set_re(out)

        out = self.decoder1(out)
        out = self.decoder3(out)

        return out

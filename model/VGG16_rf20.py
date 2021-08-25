import torch
import torch.nn as nn

class VGG16_rf20(torch.nn.Module):
    def __init__(self):
        super(VGG16_rf20, self).__init__()

        #------------------------------- ENCODER ------------------------------------------
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self.encoder3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5, stride = 5)
        )
        #------------------------------- DECODER ------------------------------------------

        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=7, stride=5, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)
        return out
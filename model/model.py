import torch
import torch.nn as nn

class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        #------------------------------- INCODER ------------------------------------------
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
        #------------------------------- DECODER ------------------------------------------

        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            torch.nn.ReLU(),
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0),
            # torch.nn.ReLU(),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        # print("DATA = {}".format(x))
        # print("SAHPE {} ".format(x.shape))
        # print("TYPE {} ".format(type(x)))
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.decoder1(out)
        out = self.decoder2(out)
        
        return out
    # def train(self, epoch):
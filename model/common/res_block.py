import torch
# from torch.autograd.grad_mode import F
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, dim_down, scale_down=False, bottleNeck = False):

        super(ResidualBlock, self).__init__()

        self.dim_down = dim_down
        self.scale_down = scale_down


        # if scale_down:
        #     dim_down=False
        self.chanel_same = torch.nn.Sequential(
            torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.chanel_dim_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        )

        # self.end = 
        self.relu = torch.nn.ReLU()
        self.bottleNeck = bottleNeck
        # print("{} {} {}".format(in_dim, out_dim, dim_down))
        
        self.insert_block =  self.getIdenlayer(in_dim, out_dim, dim_down)

        if self.bottleNeck:
            self.block = self.bottleneck_block(in_dim, mid_dim, out_dim, dim_down)
        else:
            self.block = self.std_block2(in_dim, out_dim)

    def getIdenlayer(self, in_dim, out_dim, dim_down):
        if dim_down:
            if not self.scale_down:
                return torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_dim)
                )
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(out_dim)
            )
        else:
            return torch.nn.Sequential()

    def bottleneck_block(self, in_dim, mid_dim, out_dim, dim_down=False):
        layers=[]
        # print("----------------222222---")

        if dim_down and self.scale_down:
            layers.append(torch.nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride = 2, padding=0))
        else:
            layers.append(torch.nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride = 1, padding=0))
        layers.extend([
            torch.nn.BatchNorm2d(mid_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(mid_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(out_dim),
        ])
        return torch.nn.Sequential(*layers)

    def std_block(self, in_dim, mid_dim, out_dim, dim_down=False):
        layers=[]
        if dim_down and self.scale_down:
            layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(torch.nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride = 1, padding=1))
        layers.extend([
            torch.nn.BatchNorm2d(mid_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_dim),
        ])
        return torch.nn.Sequential(*layers)

    def std_block2(self, in_dim, out_dim):
            a= torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim , kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim ),
            )
            return a
    def forward(self, x):
        iden = self.insert_block(x)
        x = self.block(x)
        x += iden
        x = self.relu(x)
        return x
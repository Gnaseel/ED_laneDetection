import torch
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, repeat, dim_changed):

        super(ResidualBlock, self).__init__()

        self.dim_changed = dim_changed

        self.chanel_same = torch.nn.Sequential(
            torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.chanel_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        )

        # self.end = 
        self.relu = torch.nn.ReLU()
        self.repeat= repeat
        
    def forward(self, x):
        out = iden = x 
        for i in range(self.repeat):
            iden = out
            if i == 0 and self.dim_changed:
                out = self.chanel_down(out)
                out = self.relu(out)
                out = self.chanel_same(out)
            else:

                out = self.chanel_same(out)
                out = self.relu(out)
                out = self.chanel_same(out)
                out += iden
            
            out = self.relu(out)
        
        return out
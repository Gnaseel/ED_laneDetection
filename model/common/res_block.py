import torch
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, repeat):

        super(ResidualBlock, self).__init__()
        self.block_start = torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        )
        self.relu = torch.nn.ReLU()
        self.repeat= repeat
        
    def forward(self, x):
        out = self.block_start(x)
        for i in range(self.repeat-1):
            iden = out
            out = self.block(out)
            out += iden
        out = self.relu(out)
        return out

        # if out.shape[1]==x.shape[1]:
        #     out +=x
        # print("X")
        # print(x.shape[1])
        # print("OUT")
        # print(out.shape[1])
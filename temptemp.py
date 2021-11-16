import torch

# input = torch.tensor([2,1,0,1,2])

# input2 = torch.tensor([0.3,0.5,0.9,0.6,0.4])
# print(input)

# input2_pad = torch.nn.functional.pad(input2, (1,0), value=0)[:-1]

# print(input2 > input2_pad)
# output = torch.where(input2 > input2_pad, input, input*-1)

# print("--------INPUT----")
# print(input2)
# print(input2_pad)
# print("--------INPUT----")
# print(output)

temp = torch.tensor([[180, 125],
        [180, 250],
        [180, 390]])
print(temp.shape)
utemp = torch.unsqueeze(temp, dim=1)
print(utemp.shape)

tt = torch.zeros([5,6,2])
tt2 = torch.zeros([2,2])
tt2[0] = torch.tensor([0,2])
tt2[1] = torch.tensor([1,3])
print(tt2)
tt[0:2,0,:] = tt2
print(tt)
import torch
import cv2
import numpy as np
from torch._C import dtype
from torch.utils.data import TensorDataset, DataLoader
from model.model import myModel

model = myModel()
# model.load_state_dict(torch.load('D:\\Lane\\weight_file\\2021_08_24\\pre.pth'))
model.load_state_dict(torch.load('D:\\Lane\\weight_file\\2021_08_24\\epoch_15_index_0.pth'))
model.eval()

img = cv2.imread('D:\\lane_dataset\\train_set\\clips\\0313-1\\660\\20.jpg')
# img2 = cv2.imread('D:\\lane_dataset\\train_set\\clips\\0313-1\\60\\1.jpg')
img = cv2.resize(img, (300, 180))
ten = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()
print(type(ten))
print(ten)

out = model(ten)

print(type(out[0]))
print(out[0].shape)
print(out[0])

# temp1 = out[0].permute(1,2,0)
temp = out[0].permute(1,2,0).detach().numpy()

cv2.imshow("ori",img)

# print(type(temp))
print(out[0].shape)
print(temp.shape)
cv2.imshow("output",temp)
for idx in range(10):
    cv2.imshow("THRESHOLD "+str(idx),(temp-idx*0.05)*100)
cv2.imwrite("abc.png",temp*100)
np.savetxt("ttt.txt",np.squeeze(temp,axis=2), fmt='%.2f')
    # cv2.imwrite('hihihi.png', predicted_image[2])

cv2.waitKey(0)
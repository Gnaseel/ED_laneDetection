from torch._C import device
from model.model import myModel
import torch
import torch.nn as nn
import numpy as np
from loader import MyDataset
import time
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import os
str_time = time.strftime('%Y_%m_%d', time.localtime(time.time()))
PATH = './weight_file/'+str_time
os.makedirs(PATH, exist_ok=True)
x_train, x_test, y_train, y_test =  np_load = np.load("D:\\lane_dataset\\image_data_0816.npy", allow_pickle=True)

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()


# x_train_t = torch.tensor(x_train)
# x_train_t = x_train_t.permute(0,3,1,2)


train_dataset = TensorDataset(x_train.permute(0,3,1,2), y_train)
# train_dataset.data = x_train.permute(0,3,1,2)
# train_dataset.target = y_train
# train_dataset.len=2719

# test_dataset = Dataset()
# test_dataset.data = x_test.permute(0,3,1,2)
# test_dataset.target = y_test
# temp_loader = MyDataset("D:\\lane_dataset\\imagedata_0816.npy")



data_loader = DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)

model = myModel()

summary(model, (3, 180, 300),device='cpu')
# model.forward(train_dataset.data[0:2])
criterion = torch.nn.BCELoss(weight=torch.tensor([60]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(30):
    for index, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()  # 기울기 초기화
        # print("{} DATA = {}".format(index, data))
        # print("SAHPE {} ".format(data.shape))
        output = model(data)

        loss = criterion(np.squeeze(output, axis = 1).float(), target)
        loss.backward()  # 역전파
        optimizer.step()
        print("IDX {}".format(index))
        print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))
        if index % 10 == 0:
            file_path = PATH +'/epoch_'+str(epoch) + '_index_'+str(index)+'.pth'
            # file_path = PATH
            torch.save(model.state_dict(),file_path)

# model.eval()  # test case 학습 방지를 위함
# test_loss = 0
# correct = 0
# with torch.no_grad():
#     for data, target in x_test,y_test:
#         output = model(data)
#         test_loss += criterion(output, target).item() # sum up batch loss
#         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#                 test_loss, correct, len(x_test),
#                 100. * correct / len(x_test)))

print("FINISHED!")


# print("DATA = {}".format(x_train[0]))
# print("SAHPE {} ".format(x_train.shape))
# print("TYPE {} ".format(type(x_train[0])))
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
import torch
import numpy as np
import cv2
import time
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import os

class Trainer():
    

    def __init__(self, args):
        print("self")
        self.dataset_path = "D:\\lane_dataset\\image_data_0816.npy"
        # self.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/image_data_0816.npy"
        self.model_path = ""
        self.cfg = args
        

    def train(self):

        # --------------------- Path Setting -------------------------------------------
        str_time = time.strftime('%Y_%m_%d', time.localtime(time.time()))
        PATH = '../weight_file/'+str_time
        os.makedirs(PATH, exist_ok=True)

        # --------------------- Load Dataset -------------------------------------------

        x_train, x_test, y_train, y_test =  np_load = np.load( self.dataset_path , allow_pickle=True)
        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()

        train_dataset = TensorDataset(x_train.permute(0,3,1,2), y_train)
        test_dataset = TensorDataset(x_test.permute(0,3,1,2), y_test)



        data_loader = DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
        if self.cfg.backbone == "VGG16":
            model = myModel()
            summary(model, (3, 180, 300),device='cpu')
        elif self.cfg.backbone == "VGG16_rf20":
            model = VGG16_rf20()
            summary(model, (3, 180, 300),device='cpu')
        elif self.cfg.backbone == "ResNet34":
            model = ResNet34()
            summary(model, (3, 180, 300),device='cpu')


        # print(model)
        # --------------------- Train -------------------------------------------
        criterion = torch.nn.BCELoss(weight=torch.tensor([60]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()

        for epoch in range(30):
            for index, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()  # gradient init
                # data
                output = model(data)


                output2np = output.permute(0,3,2,1)
                arr = output2np.detach().numpy()
                # print("Output Shape = {}".format(output.shape))
                # print("Output_p Shape = {}".format(output2np.shape))
                # #----------------------------
                # print("DATA type {}".format(type(arr)))
                # print("DATA shape {}".format(arr.shape))
                output_resize = np.array([])
                for idx in range(arr.shape[0]):
                    # print("INPUT type {}".format(arr[idx].shape))
                    
                    app = cv2.resize(arr[idx], (180,300))
                    # print("INPUT2 type {}".format(app.shape))
                    # app = cv2.resize(arr[idx].permute(), (180,300))

                    # print("arr  shape {}".format(arr[idx].shape))
                    # print("DATA shape {}".format(app.shape))
                    # print("APP shape {}".format(app.shape))
                    output_resize = np.append(output_resize, app)
                    # print("DATA shape {}".format(arr.shape))

                output_resize = np.reshape(output_resize,(arr.shape[0], 300,180,1))
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!{}".format(output_resize.shape))

                output_reTensor = torch.from_numpy(output_resize).permute(0,3,2,1).float()
                output_reTensor.requires_grad=True
                # output_reTensor.grad_fn="SigmoidBackward"
                # print("RESULT Shape = {}".format(output_reTensor.shape))
                # np.savetxt('out1.txt', output[0][0].detach().numpy())
                # np.savetxt('out2.txt', output_reTensor[0][0].detach().numpy())
                # print("OUTPUT DATA = {}".format(output))
                # print("RESULT DATA = {}".format(output_reTensor))

                #----------------------------
                loss = criterion(np.squeeze(output_reTensor, axis = 1).float(), target)
                loss.backward()  # backProp
                optimizer.step()
                print("IDX {}".format(index))
                print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))
                if index % 10 == 0:
                    file_path = PATH +'/epoch_'+str(epoch) + '_index_'+str(index)+'.pth'
                    torch.save(model.state_dict(),file_path)

        print("Train Finished.")
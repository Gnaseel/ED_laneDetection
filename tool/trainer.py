from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
from model.ResNet34_lin import ResNet34_lin
from tool.logger import Logger
import tool.data_preprocess as datas

import torch
import numpy as np
import cv2
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os

class Trainer():

    def __init__(self, args):
        print("self")
        self.dataset_path = "D:\\lane_dataset\\image_data_0816.npy"
        # self.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/image_data_0816.npy"
        self.model_path = ""
        self.log_path = ""
        self.cfg = args
        self.epoch = 0
        self.index = 0
        self.loss = 0
        self.logger = Logger()
        self.model = None

    def train_lane_lin(self):
        # --------------------- Path Setting -------------------------------------------

        self.logger.makeLogDir()

        # --------------------- Load Dataset -------------------------------------------
        
        data_loader = self.getDataLoader()

        # --------------------- Train -------------------------------------------
        weights = torch.ones(7)
        weights[0] = 0.4
        criterion = torch.nn.NLLLoss(weight=weights, reduction="mean")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.model.train()

        for epoch in range(70):
            for index, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()  # gradient init
                target2 = self.getTarget(target)

                #---------------------------- Get Loss ----------------------------------------
                
                loss = criterion(F.log_softmax(self.model(data), dim=1), target2.long())
                loss.backward()  # backProp
                optimizer.step()
                self.loss = loss.item()
                #---------------------------- Logging ----------------------------------------
                self.dataUpdate(epoch, index)
                if index % 10 == 0 or True:
                    self.logger.logging(self)

        print("Train Finished.")


    def getTarget_ex(self, target):
        arr = target.detach().numpy()
        target_resize = np.array([])
        for idx in range(target.shape[0]):
            app = cv2.resize(arr[idx], (304,176))
            target_resize = np.append(target_resize, app)
        target_resize = np.reshape(target_resize,(target.shape[0], 176, 304))
        target_reTensor = torch.tensor(torch.from_numpy(target_resize).float(), requires_grad=False)        
        return target_reTensor
    
    def getTarget(self, target):
        arr = target.detach().numpy()
        list = []
        for ins in arr:
            list2 = []
            for row in ins:
                row = np.delete(row, 1, 1)
                row = np.delete(row, 1, 1)
                row = np.squeeze(row, 1)
                list2.append(row)
            list.append(list2)
        new_img=np.array(list)
        target_resize = np.array([])
        for idx in range(target.shape[0]):
            app = cv2.resize(new_img[idx], (304,176), interpolation=cv2.INTER_NEAREST)
            target_resize = np.append(target_resize, app)
        target_resize = np.reshape(target_resize,(target.shape[0], 176, 304))
        torch_img = torch.from_numpy(target_resize.astype(np.int64))
        return torch_img
    
    def dataUpdate(self, epoch, index):
        self.epoch = epoch
        self.index = index
        return

    def getDataLoader(self):

        x_train, x_test, y_train, y_test  = np.load( self.dataset_path , allow_pickle=True)
        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()

        train_dataset = TensorDataset(x_train.permute(0,3,1,2), y_train)
        test_dataset = TensorDataset(x_test.permute(0,3,1,2), y_test)
        data_loader = DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
        return data_loader

    def getModel(self):
        if self.cfg.backbone == "VGG16":
            self.model = myModel()
            summary(self.model, (3, 180, 300),device='cpu')
        elif self.cfg.backbone == "VGG16_rf20":
            self.model = VGG16_rf20()
            summary(self.model, (3, 180, 300),device='cpu')
        elif self.cfg.backbone == "ResNet34":
            self.model = ResNet34()
            summary(self.model, (3, 176, 304),device='cpu')
        elif self.cfg.backbone == "ResNet34_lin":
            self.model = ResNet34_lin()
            self.dataset_path = "D:\\lane_dataset\\img_lane.npy"
            summary(self.model, (3, 176, 304),device='cpu')
        return self.model

    def train(self):
        # --------------------- Path Setting -------------------------------------------
        self.logger.makeLogDir()

        # --------------------- Load Dataset -------------------------------------------
       
        data_loader = self.getDataLoader()
        self.getModel()

        # print(self.model)
        # --------------------- Train -------------------------------------------
        criterion = torch.nn.BCELoss(weight=torch.tensor([40]), reduction="mean")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.model.train()

        for epoch in range(9000):
            for index, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()  # gradient init
                output = self.model(data)

                target_reTensor = self.getTarget_ex(target)
                loss = criterion(np.squeeze(output, axis = 1).float(), target_reTensor)
                loss.backward()  # backProp
                optimizer.step()

                #---------------------------- Logging ----------------------------------------
                self.dataUpdate(epoch, index)
                if index % 10 == 0 or True:
                    self.logger.logging(self)

        print("Train Finished.")

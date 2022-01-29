from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
from model.ResNet50 import ResNet50
from tool.logger import Logger
from back_logic.delta_distance import delta_degree
from back_logic.delta_distance import delta_distance
import time

import torch
import numpy as np
import cv2
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
import torch.nn.functional as nnf

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
        self.device='cpu'
        self.weight=None

        # self.datasets_path=[0 for i in range(0,100)]
        self.datasets_path=[]
    
    def train_seg(self):
        # --------------------- Path Setting -------------------------------------------

        self.logger.setLogger(self.device)
        print('학습을 진행하는 기기:',self.device)
        print('Segmentation Train')


        # --------------------- Load Dataset -------------------------------------------
        
        # data_loader = self.getDataLoader_from_np(self.device)

        # --------------------- Train -------------------------------------------
        wt = [1,1]
        self.setWeight(wt)
        print("WT = {}".format(wt))
        print("WT = {}".format(self.weight))

        self.logger.wanna_log = self.weight
        self.logger.makeLogDir()
        self.logger.writeTrainingHead(self)

        criterion = torch.nn.NLLLoss(weight=self.weight, reduction="mean").to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(70000):
            for data_set in self.datasets_path:
                data_loader = self.getDataLoader_from_np(self.device, data_set)
                print("DATASET IDX = {}".format(data_set))

                for index, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()  # gradient init
                    # print(target.shape)
                    target2 = self.getTarget_single(target.detach())

                    #---------------------------- Get Loss ----------------------------------------
                    # print("333 Output Shape {}".format(F.log_softmax(self.model(data), dim=1).shape))
                    # print("444 Output Shape {}".format(target2.long().shape))

                    # Custom Loss
                    loss = self.getCustomHeatloss(self.model(data), target2.long())
                    # Official Loss
                    # loss = criterion(F.log_softmax(self.model(data), dim=1), target2.long())
                    loss.backward()  # backProp
                    optimizer.step()
                    self.loss = loss.item()
                    #---------------------------- Logging ----------------------------------------
                    self.dataUpdate(epoch, index)
                    self.logger.printTrainingLog(self)
                    self.logger.saveTrainingtxt(self)
                    
            if True:
            # if epoch % 10 == 0:
                print("LOG!!")
                self.logger.logging(self)

        print("Train Finished.")

        return
    

    def train_deg(self):
        # --------------------- Path Setting -------------------------------------------
        self.logger.setLogger(self.device)
        print('학습을 진행하는 기기:',self.device)
        print("Model = train_deg")
        # --------------------- Load Dataset -------------------------------------------
        # data_loader = self.getDataLoader_from_np(self.device)

        # --------------------- Train -------------------------------------------
        wt = [1]
        self.setWeight(wt)
        print("WT = {}".format(wt))
        print("WT = {}".format(self.weight))

        self.logger.wanna_log = self.weight
        self.logger.makeLogDir()
        self.logger.writeTrainingHead(self)

        criterion = torch.nn.L1Loss(reduction="mean").to(self.device)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2000)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.model = self.model.to(self.device)
        self.model.train()
        d=delta_distance()
        d.setDevice(self.device)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(70000):
            epoch_start = time.time()

            for data_set in self.datasets_path:
                data_loader = self.getDataLoader_from_np(self.device, data_set)
                print("DATASET IDX = {}".format(data_set))
                for index, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()  # gradient init
                    target2 = self.getTarget_onlyLane(target.detach())
                    delta_right_list, delta_right_exist_list = d.getDeltaRightMap(target2)
                    # delta_right_list, delta_right_exist_list = d.getLaneExistHeight(target2)
                    delta_up_list, delta_up_exist_list = d.getDeltaVerticalMap(target2)

                    #---------------------------- Get Loss ----------------------------------------
                    output = self.model(data)
                    # print("OUtput Shape {}".format(output.shape))
                    # loss = []
                    totalLoss=1e-9
                    for idx, lane_tensor in enumerate(delta_right_exist_list):
                        if len(lane_tensor)==0:
                            continue
                        selected_output = torch.index_select(output[idx,0],0, lane_tensor.to(self.device))
                        selected_target = torch.index_select(delta_right_list[idx],0, lane_tensor.to(self.device))
                        # loss.append(criterion(selected_output, selected_target.float()))
                        totalLoss+=criterion(selected_output, selected_target.float())
                        # print(criterion(selected_output, selected_target.float()))
                        # print(lane_tensor)

                    for idx, lane_tensor in enumerate(delta_up_exist_list):
                        if len(lane_tensor)==0:
                            continue
                        selected_output = torch.index_select(output[idx,1],1, lane_tensor.to(self.device))
                        selected_target = torch.index_select(delta_up_list[idx],1, lane_tensor.to(self.device))
                        # loss.append(criterion(selected_output, selected_target.float()))
                        totalLoss+=criterion(selected_output, selected_target.float())
                        # print(criterion(selected_output, selected_target.float()))

                    # print(totalLoss)
                    totalLoss.backward()  # backProp
                    optimizer.step()
                    self.loss = totalLoss.item()
                    #---------------------------- Logging ----------------------------------------

                    self.dataUpdate(epoch, index)
                    if index%20==0:
                        self.logger.printTrainingLog(self)
                    # end = time.time()
                    # print("TOTAL {}".format(end-start))
                    # print("ADDED {}".format(end2-start2))
            if True:
                # if epoch % 10 == 0:
                time_str = "LOG!!  Time = {}".format(time.time() - epoch_start)
                print(time_str)
                self.logger.logging(self)
                self.logger.saveTxt(time_str)

        print("Train Finished.")

        return

    def train_total(self):
        # --------------------- Path Setting -------------------------------------------
        self.logger.setLogger(self.device)
        print('학습을 진행하는 기기:',self.device)
        print("Model = train_total")
        # --------------------- Load Dataset -------------------------------------------
        # data_loader = self.getDataLoader_from_np(self.device)

        # --------------------- Train -------------------------------------------
        wt = [1, 1]
        wt_40 = [1, 1]
        self.setWeight(wt)
        print("WT = {}".format(wt))
        print("WT = {}".format(self.weight))

        self.logger.wanna_log = self.weight
        self.logger.makeLogDir()
        self.logger.writeTrainingHead(self)

        criterion = torch.nn.L1Loss(reduction="mean").to(self.device)
        criterion_heat = torch.nn.NLLLoss(weight=torch.tensor([1,1], dtype=torch.float), reduction="mean").to(self.device)
        criterion_heat_40 = torch.nn.NLLLoss(weight=torch.tensor([1,40], dtype=torch.float), reduction="mean").to(self.device)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2000)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.model = self.model.to(self.device)
        self.model.train()
        d=delta_distance()
        d.setDevice(self.device)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(70000):
            epoch_start = time.time()

            for data_set in self.datasets_path:
                data_loader = self.getDataLoader_from_np(self.device, data_set)
                print("DATASET IDX = {}".format(data_set))
                for index, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()  # gradient init
                    target2 = self.getTarget_onlyLane(target.detach())
                    delta_right_list, delta_right_exist_list = d.getDeltaRightMap(target2)
                    delta_up_list, delta_up_exist_list = d.getDeltaVerticalMap(target2)

                    #---------------------------- Get Loss ----------------------------------------
                    output = self.model(data)
                    # print("OUtput Shape {}".format(output.shape))
                    # loss = []
                    totalLoss=1e-9
                    loss_dx=0
                    loss_dy=0
                    loss_seg=0
                    loss__weighted_seg=0

                    for idx, lane_tensor in enumerate(delta_right_exist_list):
                        if len(lane_tensor)==0:
                            continue
                        selected_output = torch.index_select(output[idx,0],0, lane_tensor.to(self.device))
                        selected_target = torch.index_select(delta_right_list[idx],0, lane_tensor.to(self.device))
                        # loss.append(criterion(selected_output, selected_target.float()))
                        loss_dx += criterion(selected_output, selected_target.float())
                        # print(criterion(selected_output, selected_target.float()))
                        # print(lane_tensor)

                    for idx, lane_tensor in enumerate(delta_up_exist_list):
                        if len(lane_tensor)==0:
                            continue
                        selected_output = torch.index_select(output[idx,1],1, lane_tensor.to(self.device))
                        selected_target = torch.index_select(delta_up_list[idx],1, lane_tensor.to(self.device))
                        # loss.append(criterion(selected_output, selected_target.float()))
                        loss_dy += criterion(selected_output, selected_target.float())
                        # print(criterion(selected_output, selected_target.float()))

                    loss_seg = criterion_heat(F.log_softmax(output[:,[2,3]], dim=1), target2.long())
                    loss_weighted_seg = criterion_heat_40(F.log_softmax(output[:,[2,4]], dim=1), target2.long())
                    custom_heat_loss = self.getCustomHeatloss(output[:,[2,4]], target2.long())
                    totalLoss = (loss_dx + loss_dy)*0.01 + loss_seg + loss_weighted_seg
                    # print(totalLoss)
                    totalLoss.backward()  # backProp
                    optimizer.step()
                    self.loss = totalLoss.item()
                    #---------------------------- Logging ----------------------------------------

                    self.dataUpdate(epoch, index)
                    if index%10==0:
                        self.logger.printTrainingLog(self)
                        print("         loss_dx     = {}".format(loss_dx))
                        print("         loss_dy     = {}".format(loss_dy))
                        print("         loss_seg    = {}".format(loss_seg))
                        print("         loss_seg_40 = {}".format(loss_weighted_seg))
                    # end = time.time()
                    # print("TOTAL {}".format(end-start))
                    # print("ADDED {}".format(end2-start2))
            # if True:
            if epoch % 10 == 0:
                time_str = "LOG!!  Time = {}".format(time.time() - epoch_start)
                print(time_str)
                self.logger.logging(self)
                self.logger.saveTxt(time_str)

        print("Train Finished.")

        return


    def getTarget_ex(self, target):
        arr = target.detach().numpy()
        target_resize = np.array([])
        for idx in range(target.shape[0]):
            app = cv2.resize(arr[idx], (304,176))
            target_resize = np.append(target_resize, app)
        target_resize = np.reshape(target_resize,(target.shape[0], 176, 304))
        target_reTensor = torch.tensor(torch.from_numpy(target_resize).float(), requires_grad=False)        
        return target_reTensor
    
    def getTarget_single(self, target):
        target2 = torch.unsqueeze(target, 1)
        target3 = torch.squeeze(nnf.interpolate(target2, size=(368, 640), mode='nearest'))
        return target3
    def getTarget_onlyLane(self, target):
        target2 = torch.unsqueeze(target, 1)
        target3 = torch.squeeze(nnf.interpolate(target2, size=(368, 640), mode='nearest'))
        return target3

    def getTarget(self, target):
        target2 = (target[:,:,:,0:1]).permute(0,3,1,2)
        target3 = torch.squeeze(nnf.interpolate(target2[:,:,:,:], size=(368, 640), mode='nearest'))
        return target3
    
    def dataUpdate(self, epoch, index):
        self.epoch = epoch
        self.index = index
        return
 
    def getDataLoader_from_np(self, device, path):
        print("DEVICE {}    PATH {}".format(device, path))
        x_train, x_test, y_train, y_test  = np.load( path , allow_pickle=True)
        x_train = torch.from_numpy(x_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        train_dataset = TensorDataset(x_train.permute(0,3,1,2), y_train)
        if self.cfg.backbone=="ResNet50":
            batch_size=4
        else:
            batch_size=8
        data_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        
#         print(x_train.requires_grad)
#         print(y_train.requires_grad)
        return data_loader


    def setWeight(self, list):
        weights = torch.ones(len(list))
        for idx, item in enumerate(list):
            weights[idx] = item
        self.weight = weights
        return
    def train(self):
        # --------------------- Path Setting -------------------------------------------
        self.logger.makeLogDir()

        # --------------------- Load Dataset -------------------------------------------
       
        data_loader = self.getDataLoader_from_np()
#         self.getModel()

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

    def getCustomHeatloss(self, output, target):
        output_log = F.log_softmax(output, dim=1)
        one_hot_target = F.one_hot(target).permute(0,3,1,2)
        one_hot_target[:,1:,:]  *=60
        val = one_hot_target*output_log*torch.pow(1-F.softmax(output, dim=1), 2)
        custom_nll_loss = -val.sum()/(val.shape[0]*val.shape[1]*val.shape[2]*val.shape[3])
        # print(output_log)
        # nll_loss = torch.nn.NLLLoss()
        # official_nll_loss = nll_loss(output_log, target.long())
        # print("CUSTOM")
        # print(custom_nll_loss*2)
        # print("Official")
        # print(official_nll_loss)

        return custom_nll_loss

# def myloss(outputs, targets):
#     onehot = torch.nn.functional.one_hot(targets).float()
#     reshape = np.transpose(onehot, (0,3,1,2))
#     logsoft_out = nn.LogSoftmax(dim=1)
#     logsoft_out_value = logsoft_out(outputs)
#     hadamrd = logsoft_out_value*reshape
#     sum = torch.sum(hadamrd, dim=1)
#     return -torch.sum(sum)
# myloss(outputs, targets)
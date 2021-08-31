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
            summary(model, (3, 176, 304),device='cpu')


        # print(model)
        # --------------------- Train -------------------------------------------
        criterion = torch.nn.BCELoss(weight=torch.tensor([40]), reduction="sum")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        model.train()

        for epoch in range(70):
            for index, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()  # gradient init
                # data
                output = model(data)


                # print("target Shape = {}".format(target.shape))
                # print("target Shape = {}".format(target.shape))
                # print("target Shape = {}".format(target.shape))
                # print("target Shape = {}".format(target.shape))
                # target2np = target.permute(2,1,0)
                # print("target_p Shape = {}".format(target2np.shape))
                arr = target.detach().numpy()
                # #----------------------------
                target_resize = np.array([])
                # print("ARRSIZE ================== {}".format(arr.shape))
                # print("ARRSIZE1 ================== {}".format(arr[0].shape))
                # print("ARRSIZE2 ================== {}".format(arr[1].shape))
                # cv2.imshow("SDFSDF", arr[0])
                for idx in range(target.shape[0]):
                    app = cv2.resize(arr[idx], (304,176))
                    # print("app Shape = {}".format(app.shape))
                    # cv2.imshow("APP", app)
                    # cv2.waitKey(0)

                    # app Shape = (304, 176)
                    target_resize = np.append(target_resize, app)


                # print("TARGET {}".format(target.shape[0]))
                target_resize = np.reshape(target_resize,(target.shape[0], 176, 304))
                # cv2.imshow("tar 1{}",target_resize[0])
                # cv2.imshow("tar 2{}",target_resize[1])
                # cv2.imshow("tar 3{}",target_resize[99])
                # cv2.waitKey(0)
                # print("RETAR SHPAE {}".format(target_resize[0].shape))

                target_reTensor = torch.tensor(torch.from_numpy(target_resize).float(), requires_grad=False)
            
                # print("RESULT Shape = {}".format(target_reTensor.shape))
                # output_reTensor = torch.from_numpy(output_resize).permute(0,3,2,1).float()
                # output_reTensor.grad_fn="SigmoidBackward"
                # print("OUTPUT DATA = {}".format(output))
                # print("RESULT DATA = {}".format(output_reTensor))

                # print("TARGET = {}".format(target.shape))
                # print("TARGET2 = {}".format(target_reTensor.shape))
                loss = criterion(np.squeeze(output, axis = 1).float(), target_reTensor)

                # print(target)
                # print(target_reTensor)
                # np.savetxt('out1.txt', torch.Tensor(target[0]).numpy(),"%d")
                # np.savetxt('out2.txt', torch.Tensor(target_reTensor[0]).numpy(),fmt="%d")
                
                # dis = np.squeeze(output, axis = 1)
                # print("SHAPE = {} ".format(dis.shape))
                # cv2.imshow("tar ",output[0])
            
                #----------------------------
                # loss = criterion(np.squeeze(output, axis = 1).float(), target)

                # np.savetxt('out1.txt', np.squeeze(output_reTensor.detach(), axis = 1).float()[3])
                # np.savetxt('out2.txt', np.squeeze(output.detach(), axis = 1).float()[3])


                loss.backward()  # backProp
                optimizer.step()
                print("IDX {}".format(index))
                print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))
                if index % 10 == 0:
                    file_path = PATH +'/epoch_'+str(epoch) + '_index_'+str(index)+'.pth'
                    torch.save(model.state_dict(),file_path)

        print("Train Finished.")
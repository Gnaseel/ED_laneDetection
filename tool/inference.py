from numpy.core.arrayprint import format_float_positional
import torch
import cv2
import numpy as np
from tool.scoring import Scoring
import data.sampleColor as myColor
from model.VGG16 import myModel
import matplotlib.pyplot as plt
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
from model.ResNet50 import ResNet50
from model.ResNet34_lin import ResNet34_lin
import torch.nn.functional as nnf
import glob
import os
import torch.nn.functional as F
import torch
import time
from back_logic.segmentation import EDseg
from back_logic.evaluate import EDeval

class Inference():
    def __init__(self, args):
        self.cfg = args
#         self.device = torch.device('cuda:2')
        self.device = torch.device('cpu')
        self.model_path = self.cfg.model_path
        self.image_path = self.cfg.image_path
        self.image_save_path = self.cfg.save_path
        self.gt_path = self.cfg.save_path
        self.max_arg = 0
        self.model = None
        self.max_arg = 0
        # self.model.maxArg


    def inference(self):
        #----------------------- Get Model ---------------------------------------------

        self.max_arg = self.model.maxArg
        #----------------------- Get Image ---------------------------------------------

        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))
        input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()

        #----------------------- Inference ---------------------------------------------

        output_tensor = self.model(input_tensor)
        output_image = output_tensor[0].permute(1,2,0).detach().numpy()
        #----------------------- Show Image ---------------------------------------------
        if self.cfg.show:
            self.show_image(img)
        #----------------------- Save Image ---------------------------------------------
        else:
            self.save_image(output_image)


    def getSoftMaxImgfromTensor(self, input_tensor):
        output_tensor = self.model(input_tensor)
        m = torch.nn.Softmax(dim=0)
        cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
        return cls_soft
    def save_image(self, image, fileName):
        # --------------------------Save segmented map
        fir_dir = os.path.join(self.image_save_path,str(fileName)+".jpg")
        print(fir_dir)

        # --------------------------Save SegImg ____ 
        image = cv2.resize(image, (300,180))
        cv2.imwrite(fir_dir, image)
        return
    def save_image_dir(self, filePaths):
        
        print("SAVE IMAGE")
        # --------------------------Save segmented map
        for file in filePaths:
            print("PATH : {}".format(file))
            img = cv2.imread(file)
            img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))

            # print(img.shape)
            input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float().to(self.device)
            cls_soft = self.getSoftMaxImgfromTensor(input_tensor)
            img_idx = cls_soft.indices.to('cpu').numpy()
            img_val = cls_soft.values.to('cpu').numpy()
            # output_tensor = self.model(input_tensor)
            # m = torch.nn.Softmax(dim=0)
            # cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
            score = Scoring()
            score.prob2lane(img_idx, img_val, 40, 350, 5 )
            score.getLanebyH_sample(160, 710, 10)
            path_list = file.split(os.sep)
#             print("path list = {}".format(path_list))
            cls_img = cls_soft.indices.detach().to('cpu').numpy().astype(np.uint8)
            cls_img = cv2.cvtColor(cls_img, cv2.COLOR_GRAY2BGR)
            cls_img = cv2.resize(cls_img, dsize = (1280, 720), interpolation=cv2.INTER_NEAREST)*30
            raw_img = cv2.resize(img, dsize = (1280, 720), interpolation=cv2.INTER_NEAREST)
            gt_path = os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/seg_label", *path_list[8:-1],"20.png")
#             print("GT path = {}".format(gt_path))
            gt_img = cv2.imread(gt_path)*30

            # score = self.getScoreInstance2(img_tensor, file)



#             gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            
#             print("LEN = {}".format(len(score.lane_list)))
#             for idx, lane in enumerate(score.lanes):
#                 print("IDX = {}".format(idx))
#                 if len(lane) <=2:
#                     continue
#                 for idx2, node in enumerate(lane):
#                     print(node)
        
            for idx, lane in enumerate(score.lane_list):
#                 print("IDX = {}".format(idx))
                # print(myColor.color_list[idx])
                if len(lane) <=2:
                    continue
                for idx2, height in enumerate(range(160, 710+1, 10)):
                    # print(lane[idx2])
                    if lane[idx2] > 0:
                        cls_img = cv2.circle(cls_img, (lane[idx2],height), 15, myColor.color_list[idx])
                        gt_img = cv2.circle(gt_img, (lane[idx2],height), 15, myColor.color_list[idx])
                        raw_img = cv2.circle(raw_img, (lane[idx2],height), 15, myColor.color_list[idx])
                    idx2+=1
                idx+=1
            
            
            fileName = "20_raw"
            fir_dir = os.path.join(self.image_save_path,path_list[8] + "_" + path_list[9] + "_" + str(fileName)+".jpg")
            print("Raw Path = {}".format(fir_dir))
            cv2.imwrite(fir_dir, raw_img)
            fileName = "20_segmented"
            fir_dir = os.path.join(self.image_save_path,path_list[8] + "_" + path_list[9] + "_" + str(fileName)+".jpg")
            print("Seg Path = {}".format(fir_dir))
            cv2.imwrite(fir_dir, cls_img)
            gtfileName = "20_ground_truth"
            fir_dir = os.path.join(self.image_save_path,path_list[8] + "_" + path_list[9] + "_" + str(gtfileName)+".jpg")
            print("GT Path = {}".format(fir_dir))
            cv2.imwrite(fir_dir, gt_img)
            
            
    def show_image(self, img):

        input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()
        output_tensor = self.model(input_tensor)
        m = torch.nn.Softmax(dim=0)


        cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
        score = Scoring()
        score.prob2lane(cls_soft, 40, 150, 5 )

        cls_img = cls_soft.indices.detach().numpy().astype(np.uint8)
        cls_img = cv2.cvtColor(cls_img, cv2.COLOR_GRAY2BGR)
        for idx, lanes in enumerate(score.lanes):
            if len(lanes) <=2:
                continue
            for node in lanes:
                cls_img = cv2.circle(cls_img, (node[1],node[0]), 3, myColor.color_list[idx])
        cv2.imshow("ori",img)
        img =cv2.resize(img, (1280, 720))
        cls_img = cv2.resize(cls_img, (1280, 720), interpolation=cv2.INTER_NEAREST)


        #----------- Show Image ---------------
        output_image = output_tensor[0].detach().numpy()
        for idx, out_img in enumerate(output_image):
            cv2.imshow("output {}".format(idx),(255-out_img*80))
            idx +=1
        cv2.imshow("SOFT MAX",cls_img*30)
        cv2.imshow("ori_re",cv2.resize(img, (1280, 720)))


        if self.cfg.showAll:
            cv2.waitKey(30)
        else:
            cv2.waitKey()
        return

    def inference_all(self):
        

        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))

        for folder_path in folder_list:       
            sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            for folder in sub_folder_list:
                #----------------------- Get Image ---------------------------------------------
                filepath = os.path.join(folder,"20.jpg")
                img = cv2.imread(filepath)
                img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))
                self.show_image(img)
                
            print("Inference Finished!")
        return 

    def inference_dir(self):
        start_time = time.time()
        total_time = time.time()
#         print("Total Inference time = {0:0.3f}".format(end_time-start_time))
        print("SEDFSDFSDF")
        lanelist = []
        pathlist = []
        self.model = self.model.to(self.device)
        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))
        file_num=0
        file_list=[]
        start_time = time.time()
        
        for folder_path in folder_list:       
            sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            for folder in sub_folder_list:
                filepath = os.path.join(folder,"20.jpg")
                path_list = filepath.split('/')
                re_path = os.path.join(*path_list[:])
                pathlist.append(os.path.join(*path_list[-4:]))
                #----------------------- Get Image ---------------------------------------------
                
#                 img = cv2.imread(filepath)
#                 img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))
#                 img_tensor = torch.from_numpy(img).to(self.device)
                

#                 img_tensor = torch.unsqueeze(img_tensor, dim=0).permute(0,3,1,2).float()
#                 score = self.getScoreInstance2(img_tensor, re_path)
                
                
                img = cv2.imread(filepath)
                img_tensor = torch.from_numpy(img).to(self.device)
                
                img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))

                img_tensor = nnf.interpolate( torch.unsqueeze(img_tensor, dim=0).permute(0,3,1,2), size=(self.model.output_size[0], self.model.output_size[1])).float()
                score = self.getScoreInstance2(img_tensor, re_path)

                lanelist.append(score.lane_list)
#                 print("HJIOHIHIHIH")
                if len(lanelist)%100==0:
                    print("Idx {}".format(len(lanelist)))
                    end_time = time.time()
                    print(end_time-start_time)
                    start_time = end_time
        print("Inference Finished!")
        print("Time = {}".format(time.time()-total_time))
        return lanelist, pathlist
    
    
    def getScoreInstance2(self, input_tensor, path):
        
        path_list = path.split('/')
        start = time.time()
#         print(input_tensor.shape)
        output_tensor = self.model(input_tensor).to(self.device)
        inference_time = time.time()

#         print("Inference time={}".format(inference_time - start))
        
        m = torch.nn.Softmax(dim=0)
        cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)   
    
        softmax_time = time.time()
#         print("softmax_time time={}".format(softmax_time - inference_time))
        
        score = Scoring()
        img_idx = cls_soft.indices.to('cpu').numpy()
        img_val = cls_soft.values.to('cpu').numpy()
        score.prob2lane(img_idx, img_val, 40, 365, 5)
        
        seg2lane_time = time.time()
#         print("seg2lane_time time={}".format(seg2lane_time - softmax_time))
        
        score.getLanebyH_sample(160, 710, 10)
#         print("path = {}".format(path))
        getH_time = time.time()
#         print("getH_time time={}".format(getH_time - seg2lane_time)) 
#         os.system('clear')
        return score
    def inference_instance(self, input):
        output_tensor = self.model(input)
        output = output_tensor[0].permute(1,2,0).detach().numpy()
        return output
        
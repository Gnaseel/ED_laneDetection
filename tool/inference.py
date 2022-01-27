from builtins import zip
from genericpath import exists
from numpy.core.arrayprint import format_float_positional
import torch
import cv2
import numpy as np
from tool.scoring import Scoring
import matplotlib.pyplot as plt
import torch.nn.functional as nnf
import glob
import os
import time
import math
from back_logic.segmentation import EDseg
from back_logic.evaluate import EDeval
from back_logic.image_saver import ImgSaver
from back_logic.network_logic import Network_Logic
from back_logic.postprocess_logic import PostProcess_Logic
from back_logic.laneBuilder import LaneBuilder
from back_logic.laneBuilder import Lane

class Inference():
    def __init__(self, args):
        self.cfg = args
        self.device = torch.device('cpu')
        self.model_path = self.cfg.model_path
        self.image_path = self.cfg.image_path
        self.image_save_path = os.path.dirname(self.cfg.model_path)+"/Image"
        self.gt_path = self.cfg.save_path
        self.model = None
        self.model2 = None
        # tuSimple
        self.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/test_set"
        #cuLane
        # self.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/CULane"
        self.time_network = 0
        self.time_post_process= 0

    def inference_instance(self, img, filepath=""):
        start_time = time.time()
        img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))
        pl = PostProcess_Logic()
        pl.device = self.device
        #Loop
        input_tensor = torch.unsqueeze(torch.from_numpy(img).to(self.device), dim=0).permute(0,3,1,2).float()
        output_tensor = self.model(input_tensor)
        out_delta = output_tensor[0].permute(1,2,0).cpu().detach().numpy()
        output_tensor = torch.squeeze(self.model2(input_tensor))

        model_out_time = time.time()

        out_heat = torch.where(output_tensor[1] > output_tensor[0], 1, 0).cpu().detach().numpy()
        # print(out_heat.shape)
        # print("NON ZERO {}".format(torch.count_nonzero(out_heat)))
        
        builder = LaneBuilder()
        # my_lane_list = builder.getKeyfromDelta(out_delta)
        my_lane_list = builder.getLanefromHeat(out_heat, out_delta)
        
        # NEW

        # Trad
        # nl = Network_Logic()
        # nl.device=self.device
        # my_lane_list = nl.getScoreInstance_deg(filepath, out_heat, out_delta)

        # my_lane_list = pl.post_process(my_lane_list)
        if len(my_lane_list)>5:
            my_lane_list = my_lane_list[0:5]
        lane_output_time = time.time()
        
        self.time_network +=model_out_time-start_time
        self.time_post_process +=lane_output_time - model_out_time

        return my_lane_list
    def inference(self):

        #----------------------- Get Image ---------------------------------------------
        # /clips/0530/1492626047222176976_0/20.jpg
        path = os.path.join(self.dataset_path, self.image_path)
        img = cv2.imread(path)
        img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))

        #----------------------- Inference ---------------------------------------------

        output_image = None

        self.print_inference_option()
        nl = Network_Logic()
        nl.device=self.device
        pl = PostProcess_Logic()
        pl.device=self.device
        imgSaver = ImgSaver(self.cfg)
        imgSaver.device = self.device
        output_image = nl.inference_np2tensor_instance(img, self.model)
        #----------------------- Show Image ---------------------------------------------
        if self.cfg.show:
            self.show_image(img)
        #----------------------- Save Image ---------------------------------------------
        else:
            if self.cfg.backbone=="ResNet34":
                self.save_image_softmax(img, output_image,"_arrowed")

            elif self.cfg.backbone=="ResNet34_delta":
                imgSaver.save_image_delta(img, output_image, "del")
            
            elif self.cfg.backbone=="ResNet34_deg" or self.cfg.backbone=="ResNet18_delta_SCNN" or self.cfg.backbone=="ResNet34_delta_SCNN":
  
                input_tensor = torch.unsqueeze(torch.from_numpy(img).to(self.device), dim=0).permute(0,3,1,2).float()
                output_tensor = torch.squeeze(self.model2(input_tensor))
                heat_img = torch.where(output_tensor[1] > output_tensor[0], 1, 0).cpu().detach().numpy()

                # score = Scoring()
                # score.device = self.device     
                # Trad
                # score = nl.getScoreInstance_deg("temp_path", heat_img, output_image.cpu().detach().numpy())
                # score.lane_list = pl.post_process(score.lane_list)

                # New
                builder = LaneBuilder()
                # lane_list = builder.getKeyfromDelta(output_image.cpu().detach().numpy())
                my_lane_list = builder.getLanefromHeat(heat_img, output_image)
                # if len(my_lane_list)>5:
                #     my_lane_list = my_lane_list[0:5]

                # score.lane_list = self.getKeyfrom(output_image.cpu().detach().numpy())
                # score.lane_list = pl.post_process(score.lane_list)

                # print(score.lane_list)
                # return
                imgSaver.save_image_deg_basic(img, output_image, "del")                                    # circle, arrow, raw delta_map, delta_key
                imgSaver.save_image_deg_total(img, output_image, heat_img, "del")                     # total arrow
                # imgSaver.save_image_deg(img, heat_img, my_lane_list, self.image_path, "del")       # heat, lane, GT
                imgSaver.save_image_deg(img, output_tensor, my_lane_list, self.image_path, "del")       # heat, lane, GT
                # ev = EV.LaneEval.bench_one_instance(score.lane_list, img_path, gt_path)

            
            elif self.cfg.backbone=="ResNet34_seg":
                imgSaver.save_image_seg(self.model, img, output_image, "seg")
            else:
                print("N")

    def show_image(self, img):

        input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()
        output_tensor = self.model(input_tensor)
        m = torch.nn.Softmax(dim=0)


        cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
        # score = Scoring()
        # score.prob2lane(cls_soft, 40, 150, 5 )

        cls_img = cls_soft.indices.detach().numpy().astype(np.uint8)
        # cls_img = cv2.cvtColor(cls_img, cv2.COLOR_GRAY2BGR)
        # for idx, lanes in enumerate(score.lanes):
        #     if len(lanes) <=2:
        #         continue
        #     for node in lanes:
        #         cls_img = cv2.circle(cls_img, (node[1],node[0]), 3, myColor.color_list[idx])
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
#hen 465
    def inference_dir_deg(self):
        start_idx=0
        end_idx=3000
        # start_idx=200
        # end_idx=250
        print_time_mode = False
        print_time_mode = True
        self.print_inference_option()
        total_time = time.time()
        print("Inference_deg")
        lanelist = []
        pathlist = []
        self.model = self.model.to(self.device)
        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))
        file_num=0
        file_list=[]
        # start_time = time.time()
        idx=0

        for folder_path in folder_list:       
            sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            for folder in sub_folder_list:
                idx+=1
                if idx<start_idx:
                    continue
                if idx > end_idx:
                    return lanelist, pathlist
                start_time = time.time()

                filepath = os.path.join(folder,"20.jpg")
                path_list = filepath.split('/')
                re_path = os.path.join(*path_list[:])
                pathlist.append(os.path.join(*path_list[-4:]))
                #----------------------- Get Image ---------------------------------------------

                

                img = cv2.imread(filepath)
                input_time = time.time()


                ## HERE 
                my_lane_list = self.inference_instance(img, filepath)
                pl = PostProcess_Logic()
                # my_lane_list = pl.post_process(my_lane_list)

                # score = nl.getScoreInstance_deg(filepath, out_heat, out_delta)
                postprocess_output_time = time.time()
                # print(my_lane_list)
                lanelist.append(my_lane_list)




                # if print_time_mode:
                    # print("Get Network Output time {}".format(getNetwokr_output_time-input_time))
                    # print("Get Lane Model time {}".format(lane_output_time-getNetwokr_output_time))
                    # print("Post Process time {}".format(postprocess_output_time-lane_output_time))
                    # print("Total time {}".format(postprocess_output_time-input_time))
                # print("")

                # score = self.getScoreInstance_deg(img, filepath)
                # if True:
                if len(lanelist)%100==0:
                    print("Idx {}".format(len(lanelist)))
                    end_time = time.time()
                    print(end_time-start_time)
                    start_time = end_time

        print("Inference Finished!")
        print("Time = {}".format(time.time()-total_time))
        print("Network time = {}".format(self.time_network / len(lanelist)))
        print("PostProcess time = {}".format(self.time_post_process / len(lanelist)))
        return lanelist, pathlist
    
    def inference_dir(self):
        start_time = time.time()
        total_time = time.time()
        #         print("Total Inference time = {0:0.3f}".format(end_time-start_time))
        print("SEDFSDFSDF")
        lanelist = []
        pathlist = []
        self.model = self.model.to(self.device)
        self.model2 = self.model2.to(self.device)
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

                
                img = cv2.imread(filepath)
                img_tensor = torch.from_numpy(img).to(self.device)
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
        output_tensor = self.model(input_tensor).to(self.device)
        inference_time = time.time()

        
        m = torch.nn.Softmax(dim=0)
        cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)   
    
        softmax_time = time.time()
        
        score = Scoring()
        img_idx = cls_soft.indices.to('cpu').numpy()
        img_val = cls_soft.values.to('cpu').numpy()
        score.prob2lane(img_idx, img_val, 40, 365, 5)
        
        seg2lane_time = time.time()
        
        score.getLanebyH_sample(160, 710, 10)
        getH_time = time.time()
        return score

    def print_inference_option(self):
        print("------------ Inference Parameter -------------------")
        print("View Mode       : {}".format(self.cfg.show))
        print("Model Name      : {}".format(self.cfg.model_path))
        print("Model Path      : {}".format(self.cfg.backbone))
        print("Device Name     : {}".format(self.device))
        print("Image Dir path  : {}".format(self.image_path))
        print("Image save path : {}".format(self.image_save_path))
        print("Image gt_path   : {}".format(self.gt_path))
        
    
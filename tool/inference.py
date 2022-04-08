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
# import config.config_window as CFG
from model.ResNet18_seg_SCNN import ResNet18_seg_SCNN
import json
class Inference():
    def __init__(self, args):
        self.cfg = args
        self.device = self.cfg.device
        self.model_path = self.cfg.model_path
        self.image_path = self.cfg.image_path
        self.dataset_path = self.cfg.dataset_path
        self.image_save_path = os.path.dirname(self.cfg.model_path)+"/Image"
        self.gt_path = self.cfg.save_path
        self.model = self.cfg.model.to(self.device)
        self.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        

        self.model2 = ResNet18_seg_SCNN()
        self.model2.to(self.device)
        self.model2.eval()
        self.model2.load_state_dict(torch.load(self.cfg.heat_model_path, map_location='cpu'))

        self.output_size=(0,0)
        # self.model = None
        # tuSimple
        #cuLane
        # self.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/CULane"
        self.time_network = 0
        self.time_post_process= 0

    def inference_instance(self, img, filepath=""):

        start_time = time.time()
        if self.cfg.dataset=="tuSimple":
            output_size = (640, 368)
        elif self.cfg.dataset=="cuLane":
            output_size = (800, 300)
        img = cv2.resize(img, output_size)
        # pl = PostProcess_Logic()
        # pl.device = self.device
        #Loop
        input_tensor = torch.unsqueeze(torch.from_numpy(img).to(self.device), dim=0).permute(0,3,1,2).float()
        output_delta_tensor = self.model(input_tensor)
        out_delta = output_delta_tensor[0].permute(1,2,0).cpu().detach().numpy()
        output_tensor = torch.squeeze(self.model2(input_tensor))

        model_out_time = time.time()
        #prev
        # out_heat = torch.where(output_tensor[1] > output_tensor[0], torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)).cpu().detach().numpy()

        #Changed
        th=-1
        out_heat = torch.where(output_tensor[0] < th, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)).cpu().detach().numpy()
        
        
        builder = LaneBuilder(self.cfg)
        builder.device = self.cfg.device
        # my_lane_list = builder.getKeyfromDelta(out_delta)


        my_lane_list = builder.getLanefromHeat(output_tensor, out_delta, img)
        if len(my_lane_list)>5:
            my_lane_list = my_lane_list[0:5]
        lane_output_time = time.time()
        
        self.time_network +=model_out_time-start_time
        self.time_post_process +=lane_output_time - model_out_time

        return my_lane_list, None, None
        # return my_lane_list, output_tensor, output_delta_tensor
    
    def inference(self):

        #----------------------- Get Image ---------------------------------------------
        # /clips/0530/1492626047222176976_0/20.jpg
        path = os.path.join(self.dataset_path, self.image_path)
        img = cv2.imread(path)
        img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))

        #----------------------- Inference ---------------------------------------------
        my_lane_list, output_heat_tensor, output_delta_tensor = self.inference_instance(img, path)


        # imgSaver = ImgSaver(self.cfg)
        # imgSaver.device = self.device
        # imgSaver.save_image_deg_basic(img, output_delta_tensor, "del")                                    # circle, arrow, raw delta_map, delta_key
        # imgSaver.save_image_deg(img, output_heat_tensor, my_lane_list, self.image_path, "del")       # heat, lane, GT
        return my_lane_list


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
        idx=0
        start_idx=0
        end_idx=3000
        # start_idx=300
        # end_idx=310
        self.print_inference_option()
        total_time = time.time()
        print("Inference_deg")
        lanelist = []
        pathlist = []
        self.model = self.model.to(self.device)
        imgSaver = ImgSaver(self.cfg)

        # path=self.cfg.image_path
        # folder_list= glob.glob(os.path.join(path,"*"))
        # file_num=0
        start_time = time.time()
        if self.cfg.dataset_path.split(os.sep)[-1]=="tuSimple":
            test_list = self.get_test_list_tuSimple()
        else:
            test_list = self.get_test_list_cuLane()

        evaluator = EDeval()

        for file_path in test_list:

            idx+=1
            if idx<start_idx:
                continue
            if idx > end_idx:
                return lanelist, pathlist
            full_file_path = os.path.join(self.dataset_path, file_path)
            # re_path = os.path.join(*file_path[:])
            #----------------------- Get Image ---------------------------------------------

            img = cv2.imread(full_file_path)
            input_time = time.time()

            # keypoint_list = evaluator.getKeypoint(file_path)
            # imgSaver.img_keypoint_save(full_file_path, keypoint_list, file_path) 
            # return
            ## HERE 
            my_lane_list, temp, temp2 = self.inference_instance(img, full_file_path)
            pathlist.append(file_path)
            lanelist.append(my_lane_list)
            end_time = time.time()

            # print("TIME = {}".format(end_time-input_time))
            # print("")

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
        # print("Total Inference time = {0:0.3f}".format(end_time-start_time))
        lanelist = []
        pathlist = []
        self.model = self.model.to(self.device)
        self.model2 = self.model2.to(self.device)
        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))
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
        
        score = Scoring(self.cfg)
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
        
    def get_test_list_tuSimple(self):
        # test_list_path = os.path.join(self.cfg.dataset_path, "test_label.json")
        test_list_path = os.path.join(self.cfg.dataset_path, "test_label.json")
        test_list = []
        # print(test_list_path)

        f = open(test_list_path,'r')
        lines = f.readlines()
        for line in lines:
            json_object = json.loads(line)
            # print(json_object["raw_file"])
            test_list.append(json_object["raw_file"])
        return test_list

    def get_test_list_cuLane(self):
        # test_list_path = os.path.join(self.cfg.dataset_path, "test_label.json")
        test_list_path = os.path.join(self.cfg.dataset_path, "list", "test.txt")
        test_list_path = os.path.join(self.cfg.dataset_path, "list", "test_split", "test0_normal.txt")
        test_list = []
        # print(test_list_path)

        f = open(test_list_path,'r')
        lines = f.readlines()
        for line in lines:
            if line[0]==os.sep:
                line = line[1:]
            test_list.append(line.strip()) # 1-> abs to rel
        return test_list
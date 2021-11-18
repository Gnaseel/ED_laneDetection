from builtins import zip
from genericpath import exists
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
import time
from back_logic.segmentation import EDseg
from back_logic.evaluate import EDeval
from back_logic.image_saver import ImgSaver
import math

class Inference():
    def __init__(self, args):
        self.cfg = args
#         self.device = torch.device('cuda:2')
        self.device = torch.device('cpu')
        self.model_path = self.cfg.model_path
        self.image_path = self.cfg.image_path
        self.image_save_path = os.path.dirname(self.cfg.model_path)+"/Image"
        self.gt_path = self.cfg.save_path
        self.max_arg = 0
        self.model = None
        self.model2 = None
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

        # output_tensor = self.model(input_tensor)
        # output_image = output_tensor[0].permute(1,2,0).detach().numpy()
        output_image = None

        self.print_inference_option()
        imgSaver = ImgSaver(self.cfg)
        imgSaver.device = self.device
        #----------------------- Show Image ---------------------------------------------
        if self.cfg.show:
            self.show_image(img)
        #----------------------- Save Image ---------------------------------------------
        else:
            if self.cfg.backbone=="ResNet34":
                print("TEMP")
                # self.save_image(img, output_image,"temped")
                self.save_image_softmax(img, output_image,"_arrowed")

            elif self.cfg.backbone=="ResNet34_delta":
                self.save_image_delta(img, output_image, "del")
            
            elif self.cfg.backbone=="ResNet34_deg":
                # self.save_image_deg(img, output_image, "del")
                imgSaver.save_image_deg(self.model, self.model2, img, output_image, "del")
            
            elif self.cfg.backbone=="ResNet34_seg":
                self.save_image_seg(img, output_image, "seg")
            else:
                print(self.image_save_path)
                # return
                # self.save_image(img, output_image,"_seged")
                self.save_image_delta(img, output_image,"_arrowed")
                self.save_image_softmax(img, output_image,"_arrowed")
                # self.save_image_dir(self.image_save_path)


    def getSoftMaxImgfromTensor(self, input_tensor):
        output_tensor = self.model(input_tensor)
        m = torch.nn.Softmax(dim=0)
        cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
        return cls_soft

    def save_image(self, img, output_image, fileName):
        # --------------------------Save segmented map
        os.makedirs(self.image_save_path, exist_ok=True)

        back_fir_dir = os.path.join(self.image_save_path,str(fileName)+"_back.jpg")
        lane_fir_dir = os.path.join(self.image_save_path,str(fileName)+"_lane.jpg")
        print(output_image.shape)
        # back = np.squeeze(output_image[0], axis=0)
        # lane = np.squeeze(output_image[1], axis=0)
        back = output_image[:,:,0]
        lane = output_image[:,:,1]
        print(back.shape)
        print(lane.shape)

        # print(fir_dir)
        # return
        # --------------------------Save SegImg ____ 
        back = cv2.resize(back, (640,368))
        lane = cv2.resize(lane, (640,368))
        cv2.imwrite(back_fir_dir, back*50)
        cv2.imwrite(lane_fir_dir, lane*50)
        return
    def save_image_seg(self, img, output_image, fileName):
        seg_folder_name = "seged"
        seg_dir_name= os.path.join(self.image_save_path, seg_folder_name)
        # --------------------------Save segmented map
        os.makedirs(seg_dir_name, exist_ok=True)

        back_fir_dir = os.path.join(seg_dir_name,str(fileName)+"_back.jpg")
        lane_fir_dir = os.path.join(seg_dir_name,str(fileName)+"_lane.jpg")

        output_image = self.inference_np2np_instance(img, self.model)

        print(output_image.shape)
        # back = np.squeeze(output_image[0], axis=0)
        # lane = np.squeeze(output_image[1], axis=0)
        back = output_image[:,:,0]
        lane = output_image[:,:,1]
        np.savetxt(os.path.join(seg_dir_name,str(fileName)+"_back.txt"), back, fmt='%3f')
        np.savetxt(os.path.join(seg_dir_name,str(fileName)+"_lane.txt"), lane, fmt='%3f')

        lane_cfd_th = np.where(lane > 0.5)
        print("SDFSDF")
        print(lane_cfd_th)
        print(lane_cfd_th[0].shape)
        print(lane_cfd_th[1].shape)
        # lane = (lane-3)*50
        lane = (lane+0.5)*100
        back = (back)*20
        # for h,w in zip(lane_cfd_th[0], lane_cfd_th[1]):
        #     cv2.circle(lane, (w,h), 3, (255,0,0), -1)

        # print(back.shape)
        # print(lane.shape)

        # print(fir_dir)
        # return
        # --------------------------Save SegImg ____ 
        # back = cv2.resize(back, (640,368))
        # lane = cv2.resize(lane, (640,368))
        cv2.imwrite(back_fir_dir, back)
        cv2.imwrite(lane_fir_dir, lane)

        return
    def save_image_softmax(self, img, output_image, fileName):
        # --------------------------Save segmented map
        softmax_folder_name = "softmaxed"
        softmax_dir_name= os.path.join(self.image_save_path, softmax_folder_name)
        os.makedirs(softmax_dir_name, exist_ok=True)
        print(os.path.join(self.image_save_path, softmax_folder_name))
        input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float().to(self.device)

        # soft_image = self.getSoftMaxImgfromTensor(input_tensor)
        output_tensor = self.model(input_tensor)

        for idx, seg_tensor in enumerate(output_tensor[0]):
            # print(seg_tensor.shape)
            seg_image = seg_tensor.cpu().detach().numpy()
            # print(seg_image.shape)
            cv2.imwrite(os.path.join(softmax_dir_name, "_{}.jpg".format(idx)), seg_image*30)
        return
    def save_image_delta(self, image, output_image, fileName, delta_height=10, delta_threshold = 20):
        # return
        # --------------------------Save segmented map
        delta_folder_name = "delta"
        delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
        os.makedirs(delta_dir_name, exist_ok=True)

        right_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_right_lane2.jpg")
        up_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_up_lane2.jpg")
        up_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_up_circle_lane2.jpg")
        right_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_right_circle_lane2.jpg")
        raw_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw2.jpg")
        raw_fir_txt_dir = os.path.join(delta_dir_name,str(fileName)+"_raw2.txt")
        output_image = self.inference_np2np_instance(image, self.model)


        print(output_image.shape)
        print(output_image[:,:,0].shape)
        output_right_image = np.copy(image)
        output_right_circle_image = np.copy(image)
        output_up_image = np.copy(image)
        output_up_circle_image = np.copy(image)
        delta_right_image = output_image[:,:,0]
        delta_up_image = output_image[:,:,1]
        # i=height, j=width
        for i in range(130, delta_right_image.shape[0], 20):
            idx = 0
            for j in range(10, delta_right_image.shape[1], 30):
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if delta_right_image[i,j] > delta_threshold:
                    continue
                idx +=1
                # print("COORD {} {}".format(j, i))
                # print("PRE  delta_right_image[i,j] {}".format(delta_right_image[i,j]))
                # print("POST delta_right_image[i,j] {}".format(delta_right_image[i,j]))
                startPoint = (j, i)
                direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                endPoint = (int(delta_right_image[i,j])*direction + j, i)
                
                cv2.circle(output_right_image, startPoint, 1, (255,0,0), -1)
                output_right_image = cv2.arrowedLine(output_right_image, startPoint, endPoint, (0,0,255), 1)
                endPoint = (int(delta_up_image[i,j])*direction + j, i-delta_height)

                output_right_image = cv2.arrowedLine(output_right_image, startPoint, endPoint, (0,255,0), 1)

                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                # print("End point = {} //// {} {}".format(endPoint, endPoint[0], endPoint[1]))
                direction= -1 if delta_right_image[endPoint[1],endPoint[0]+3] > delta_right_image[endPoint[1],endPoint[0]-3] else 1
                endPoint2 = (int(delta_right_image[endPoint[1],endPoint[0]])*direction + endPoint[0], i-delta_height)

                output_right_image = cv2.arrowedLine(output_right_image, endPoint, endPoint2, (255,0,0), 1)

        for i in range(130, delta_up_image.shape[0], 10):
            idx = 0
            for j in range(10, delta_up_image.shape[1], 20):
                if j+11 > delta_up_image.shape[1] or j-11 < 0:
                    continue
                if delta_up_image[i,j] > delta_threshold:
                    continue
                idx +=1
                # print("COORD {} {}".format(j, i))
                # print("PRE  delta_up_image[i,j] {}".format(delta_up_image[i,j]))
                # print("POST delta_up_image[i,j] {}".format(delta_up_image[i,j]))
                startPoint = (j, i)
                direction= -1 if delta_up_image[i,j+3] > delta_up_image[i,j-3] else 1
                endPoint = (int(delta_up_image[i,j])*direction + j, i-delta_height)
                cv2.circle(output_up_image, startPoint, 1, (255,0,0), -1)
                output_up_image = cv2.arrowedLine(output_up_image, startPoint, endPoint, (0,0,255), 1)
        #--------------------------------------------------------------------------------------------------------------------------------------


        for i in range(130, output_right_circle_image.shape[0], 3):
            idx = 0
            for j in range(10, output_right_circle_image.shape[1], 5):
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if delta_right_image[i,j] > delta_threshold:
                    continue
                direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                endPoint = (int(delta_right_image[i,j])*direction + j, i)
                output_right_circle_image = cv2.circle(output_right_circle_image, endPoint, 1, (0,0,255), -1)
        for i in range(130, output_up_circle_image.shape[0], 3):
            idx = 0
            for j in range(10, output_up_circle_image.shape[1], 5):
                if j+11 > delta_up_image.shape[1] or j-11 < 0:
                    continue
                if delta_up_image[i,j] > delta_threshold:
                    continue
                direction= -1 if delta_up_image[i,j+3] > delta_up_image[i,j-3] else 1
                endPoint = (int(delta_up_image[i,j])*direction + j, i-delta_height)
                output_up_circle_image = cv2.circle(output_up_circle_image, endPoint, 1, (0,0,255), -1)
        cv2.imwrite(right_fir_dir, output_right_image)
        cv2.imwrite(up_fir_dir, output_up_image)
        cv2.imwrite(up_circle_fir_dir, output_up_circle_image)
        cv2.imwrite(right_circle_fir_dir, output_right_circle_image)

        # delta_up_image = np.squeeze(delta_up_image)
        cv2.imwrite(raw_fir_dir, delta_right_image)
        # print(delta_up_image)
        # print(delta_up_image.shape)
        np.savetxt(raw_fir_txt_dir, delta_up_image, fmt='%3d')   
    def save_image_deg(self, image, output_image, fileName, delta_height=10, delta_threshold = 50):
        # return
       
        # --------------------------Save segmented map
        delta_folder_name = "delta"
        delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
        os.makedirs(delta_dir_name, exist_ok=True)

        right_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_right_arrow.jpg")
        up_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_up_arrow.jpg")
        up_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_up_circle.jpg")
        right_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_right_circle.jpg")
        raw_right_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw_right.jpg")
        raw_up_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw_up.jpg")
        raw_fir_txt_dir = os.path.join(delta_dir_name,str(fileName)+"_raw2.txt")
        total_arrow_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_total_arrow.jpg")
        total_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_total_circle.jpg")


        heat_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_heat.jpg")
        key_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_heat_key.jpg")


        
        output_image = self.inference_np2np_instance(image, self.model)


        # print(output_image.shape)
        # print(output_image[:,:,0].shape)
        deg_padding = 3
        deg_image = np.zeros((image.shape[0]+deg_padding*2,image.shape[1]+deg_padding*2))
        # print("ZERO SHPAPE {}".format(deg_image.shape))
        output_right_image = np.copy(image)
        output_right_circle_image = np.copy(image)
        output_up_image = np.copy(image)
        output_up_circle_image = np.copy(image)
        output_total_circle_image = np.copy(image)
        output_total_arrow_image = np.copy(image)
        output_key_image = np.copy(image)
        delta_right_image = output_image[:,:,0]
        delta_up_image = output_image[:,:,1]


        output_image2  = self.inference_np2np_instance(image, self.model2)[:,:,1]
        cv2.imwrite(heat_fir_dir, (output_image2+1)*50)

        # i=height, j=width
        for i in range(130, delta_right_image.shape[0], 20):
            idx = 0
            for j in range(10, delta_right_image.shape[1], 30):
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if delta_right_image[i,j] > delta_threshold:
                    continue
                idx +=1
                # print("COORD {} {}".format(j, i))
                # print("PRE  delta_right_image[i,j] {}".format(delta_right_image[i,j]))
                # print("POST delta_right_image[i,j] {}".format(delta_right_image[i,j]))
                startPoint = (j, i)
                direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                endPoint = (int(delta_right_image[i,j])*direction + j, i)
                
                cv2.circle(output_right_image, startPoint, 1, (255,0,0), -1)
                output_right_image = cv2.arrowedLine(output_right_image, startPoint, endPoint, (0,0,255), 1)
                endPoint = (int(delta_up_image[i,j])*direction + j, i-delta_height)

                output_right_image = cv2.arrowedLine(output_right_image, startPoint, endPoint, (0,255,0), 1)

                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                # print("End point = {} //// {} {}".format(endPoint, endPoint[0], endPoint[1]))
                direction= -1 if delta_right_image[endPoint[1],endPoint[0]+3] > delta_right_image[endPoint[1],endPoint[0]-3] else 1
                endPoint2 = (int(delta_right_image[endPoint[1],endPoint[0]])*direction + endPoint[0], i-delta_height)

                output_right_image = cv2.arrowedLine(output_right_image, endPoint, endPoint2, (255,0,0), 1)

        delta = 10
        arrow_size=1
        min_threshold = 5
        threshold = 35
        
        for i in range(90, delta_up_image.shape[0]-10, delta):
            idx = 0
            for j in range(10, delta_up_image.shape[1]-10, delta):

                
                horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                vertical_direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1



                startPoint = (j, i)


                x1 = int(delta_right_image[i,j])*horizone_direction + j
                y1 = i

                x2 = j
                y2 = int(delta_up_image[i,j])*vertical_direction + i
                m = (y2-y1)/(x2-x1+0.00001)
                # y-y1 = m*(x-x1)
                # y-y1 = mx  - mx1
                # mx - y +y1 - mx1 = 0
                a = m
                b = -1
                c = y1 - m*x1
                newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
                newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)

                endpoint_up_arrow = (j, int(delta_right_image[i,j])*vertical_direction +i)
                endpoint_total_arrow = (int(delta_right_image[i,j])*horizone_direction + j, int(delta_up_image[i,j])*vertical_direction + i)
                endpoint_total_arrow = (int(newx), int(newy))
                
                endpoint_temp_arrow = (int(newx-5/(m+0.00001)), int(newy-5))
                # if int(newy) < image.shape[0] and int(newx) < image.shape[1] and int(newy) > 0 and int(newx) > 0:
                #     deg_image[int(newy)+deg_padding,int(newx)+deg_padding] = m

                dist = abs(int(delta_right_image[i,j])) + abs(int(delta_up_image[i,j]))

                deg = str( (delta_up_image[i,j]*vertical_direction) / (delta_right_image[i,j]*horizone_direction))[0:4]
                deg = str( (delta_up_image[i,j]*vertical_direction))[0:2] + " / "+str(delta_right_image[i,j]*horizone_direction)[0:2]
                deg=str(m)[0:5]



                deg= (math.atan2(y2-y1, x2-x1)*180.0/math.pi)
                while deg<0:
                    deg+=180
                while deg>180:
                    deg-=180
                if int(newy) < image.shape[0] and int(newx) < image.shape[1] and int(newy) > 0 and int(newx) > 0:
                    deg_image[int(newy)+deg_padding,int(newx)+deg_padding] = deg
                # deg=str(deg)[0:5]
                cv2.circle(output_up_image, startPoint, 1, (255,0,0), -1)
                output_total_arrow_image = cv2.circle(output_total_arrow_image, startPoint, 1, (255,0,0), -1)
                if dist > threshold or dist < min_threshold:
                    continue
                
                
                
                if vertical_direction<0:
                    output_up_image = cv2.arrowedLine(output_up_image, startPoint, endpoint_up_arrow, (0,0,255), 1)
                    output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,0,255), arrow_size)
                    output_right_circle_image= cv2.circle(output_right_circle_image, endpoint_total_arrow, 1, (0,0,255), -1)
                    # output_total_circle_image = cv2.putText(output_total_circle_image,deg,endpoint_total_arrow,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

                else:
                    output_up_image = cv2.arrowedLine(output_up_image, startPoint, endpoint_up_arrow, (0,255, 0), 1)
                    output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,255, 0), arrow_size)
                    output_right_circle_image= cv2.circle(output_right_circle_image, endpoint_total_arrow, 1, (0,255,0), -1)
                    # output_total_circle_image = cv2.putText(output_total_circle_image,deg,endpoint_total_arrow,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                if i>130 and deg<170 and deg > 10:
                    output_total_circle_image= cv2.arrowedLine(output_total_circle_image,endpoint_total_arrow,  endpoint_temp_arrow, (0,255,0), 1)
                # output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,255, 0), arrow_size)
        
        score = Scoring()
        score.device = self.device     
        lane_tensor = torch.zeros([10,30,2]).to(self.device)
        # print(path)
        # return
        key_height =  170
        first_key_tensor = score.getLocalMaxima_heatmap_re(output_image2, key_height).to(self.device)
        while first_key_tensor.shape[0]==0 or key_height>300:
            key_height += 10
            first_key_tensor = score.getLocalMaxima_heatmap_re(output_image2, key_height).to(self.device)
        key_height = 170
        while first_key_tensor.shape[0]==0 or key_height<100:
            key_height -= 10
            first_key_tensor = score.getLocalMaxima_heatmap_re(output_image2, key_height).to(self.device)

        if first_key_tensor.shape[0]==0:
            score.tensor2lane(lane_tensor)
            score.getLanebyH_sample(160, 710, 10)
            return score

        terminal_tensor = torch.zeros([10,2], dtype = torch.long).to(self.device)
        # terminal = 
        lt_idx = 7
        lane_tensor[0:first_key_tensor.shape[0], lt_idx] = first_key_tensor
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        lane_num=first_key_tensor.shape[0]

        # print("KeyPoint  {}".format(first_key_tensor))
        # print("Terminal = {}".format(terminal_tensor))
        #---------------- HEAT to INS ---------------------------------------------------------------
        for height in range(key_height+10, 330, 10):
            print("Height {}".format(height))
            new_key =  score.getLocalMaxima_heatmap_re(output_image2, height)
            # print("-----------------------------------------------------------------------------------")
            # print("New Key = {}".format(new_key))
            lane_num = score.chainKey2(new_key, terminal_tensor, deg_image, lane_num)
            lt_idx +=1
            lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:4,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        # print("--------------IUDUDUDUDUUDUDUD")
        lt_idx=7
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        # print("New Terminal = {}".format(terminal_tensor))
        for height in range(key_height-10, 90, -10):
            print("Height {}".format(height))
            new_key =  score.getLocalMaxima_heatmap_re(output_image2, height)
            # print("New Key = {}".format(new_key))
            lane_num = score.chainKey2(new_key, terminal_tensor, deg_image, lane_num)
            lt_idx -=1
            lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:3,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        get_pre_lane = time.time()
        # print("Get Pre lane Time = {}".format(get_pre_lane - get_degMap_output))
        # lane_tensor[:,0] *= (720.0/368.0)
        # lane_tensor[:,1] *= (720.0/368.0)
        score.tensor2lane(lane_tensor)
        # return
        score.getLanebyH_sample_deg(160, 710, 10)
        for idx, lane in enumerate(lane_tensor):
            # if idx != 2:
            #     continue
            for point in lane:
                output_key_image = cv2.circle(output_key_image, (int(point[1]), int(point[0])), 3, myColor.color_list[idx], -1)

        cv2.imwrite(key_fir_dir, output_key_image)
        
        # return
        # for height in range(330, 60, -10):

        #     seed_ternsor = score.getLocalMaxima_heatmap_re(output_image2, height)
        #     print("Height = {}, key = {}".format(height, len(seed_ternsor)))
        #     for point in seed_ternsor:
        #         output_key_image = cv2.circle(output_key_image, (int(point[1]), int(point[0])), 1, (255,0,0), -1)
        #         # print("--------------------------------------------")
        #         for i in range(0,25):
        #             y = int(point[0])+deg_padding + i%5-2
        #             x = int(point[1])+deg_padding + i//5-2
        #             if deg_image[y,x]==0:
        #                 continue
        #             # print("DEG = {}".format(deg_image[y,x]))

        #     key_tensor = torch.cat([key_tensor, seed_ternsor]).to(self.device)

        #     print(seed_ternsor)
        # cv2.imwrite(key_fir_dir, output_key_image)
        # print("KEY TENSOR")
        # print(key_tensor)
        #--------------------------------------------------------------------------------------------------------------------------------------

        threshold = 40
        for i in range(130, output_right_circle_image.shape[0]-20, delta):
            idx = 0
            for j in range(10, output_right_circle_image.shape[1]-10, delta):
                # if j+11 > delta_right_image.shape[0] or j-11 < 0:
                #     continue
                if delta_right_image[i,j] > threshold:
                    continue
                # if i+11 > delta_up_image.shape[1] or i-11 < 0:
                #     continue
                # if delta_up_image[i,j] > delta_threshold:
                #     continue
                # endPoint = (int(delta_up_image[i,j])*direction + j, i-delta_height)
                endPoint = (j, int(delta_up_image[i,j])*direction +i)
                # output_up_circle_image = cv2.circle(output_up_circle_image, endPoint, 1, (0,0,255), -1)
                horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                endPoint = (int(delta_right_image[i,j])*horizone_direction + j, i)
                output_right_circle_image = cv2.circle(output_right_circle_image, endPoint, 1, (0,0,255), -1)

                
        #         vertical_direction= -1 if delta_up_image[i,j+3] > delta_up_image[i,j-3] else 1

        #         endpoint_circle = (int(delta_right_image[i,j])*horizone_direction + j, int(delta_up_image[i,j])*vertical_direction + i)
        #         output_total_circle_image= cv2.circle(output_right_circle_image, endpoint_circle, 1, (0,0,255), -1)


        for i in range(130, output_up_circle_image.shape[0], 3):
            idx = 0
            for j in range(10, output_up_circle_image.shape[1], 5):
                if i+11 > delta_up_image.shape[1] or i-11 < 0:
                    continue
                if delta_up_image[i,j] > 10:
                    continue
                direction= -1 if delta_up_image[i,j+3] > delta_up_image[i,j-3] else 1
                # endPoint = (int(delta_up_image[i,j])*direction + j, i-delta_height)
                endPoint = (j, int(delta_up_image[i,j])*direction +i)
                output_up_circle_image = cv2.circle(output_up_circle_image, endPoint, 1, (0,0,255), -1)

        cv2.imwrite(right_fir_dir, output_right_image)
        cv2.imwrite(up_fir_dir, output_up_image)
        cv2.imwrite(right_circle_fir_dir, output_right_circle_image)
        cv2.imwrite(up_circle_fir_dir, output_up_circle_image)
        cv2.imwrite(total_circle_fir_dir, output_total_circle_image)
        cv2.imwrite(total_arrow_fir_dir, output_total_arrow_image)

        # delta_up_image = np.squeeze(delta_up_image)
        cv2.imwrite(raw_right_fir_dir, delta_right_image)
        cv2.imwrite(raw_up_fir_dir, delta_up_image)
        # print(delta_up_image)
        # print(delta_up_image.shape)
        np.savetxt(raw_fir_txt_dir, delta_up_image, fmt='%3d')
    def save_image_dir_deg(self, filePaths):
        for file in filePaths:
            print("PATH : {}".format(file))
            img = cv2.imread(file)
            img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))

            # print(img.shape)
            #             input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float().to(self.device)
            #             cls_soft = self.getSoftMaxImgfromTensor(input_tensor)
            #             img_idx = cls_soft.indices.to('cpu').numpy()
            #             img_val = cls_soft.values.to('cpu').numpy()
            #             # output_tensor = self.model(input_tensor)
            #             # m = torch.nn.Softmax(dim=0)
            #             # cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
            score = self.getScoreInstance_deg(img, file)
            # score.prob2lane(img_idx, img_val, 40, 350, 5 )
            # score.getLanebyH_sample(160, 710, 10)
            path_list = file.split(os.sep)
            raw_img = cv2.resize(img, dsize = (1280, 720))
            cls_img = self.inference_np2np_instance(img, self.model2)[:,:,1]
            cls_img = cv2.resize(cls_img, dsize = (1280, 720))
            cls_img = (cls_img-0.4)*30
            gt_path = os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/seg_label", *path_list[8:-1],"20.png")
            #             print("GT path = {}".format(gt_path))
            gt_img = cv2.imread(gt_path)*30

        
            for idx, lane in enumerate(score.lane_list):
                if len(lane) <=2:
                    continue
                for idx2, height in enumerate(range(160, 710+1, 10)):
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

        
            for idx, lane in enumerate(score.lane_list):
                if len(lane) <=2:
                    continue
                for idx2, height in enumerate(range(160, 710+1, 10)):
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

    def inference_dir_deg(self):
        self.print_inference_option()
        start_time = time.time()
        total_time = time.time()
        print("Inference_deg")
        lanelist = []
        pathlist = []
        self.model = self.model.to(self.device)
        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))
        file_num=0
        file_list=[]
        start_time = time.time()
        idx=0
        for folder_path in folder_list:       
            sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            for folder in sub_folder_list:
                idx+=1
                filepath = os.path.join(folder,"20.jpg")
                path_list = filepath.split('/')
                re_path = os.path.join(*path_list[:])
                pathlist.append(os.path.join(*path_list[-4:]))
                #----------------------- Get Image ---------------------------------------------
                
                img = cv2.imread(filepath)
                img = cv2.resize(img, (self.model.output_size[1], self.model.output_size[0]))
                score = self.getScoreInstance_deg(img, filepath)
                # print("------------------TEMP")
                # if idx > 50:
                #     return lanelist, pathlist
                # continue
                # return None, None
                lanelist.append(score.lane_list)
# #                 print("HJIOHIHIHIH")
                if len(lanelist)%100==0:
                    print("Idx {}".format(len(lanelist)))
                    end_time = time.time()
                    print(end_time-start_time)
                    start_time = end_time
        print("Inference Finished!")
        print("Time = {}".format(time.time()-total_time))
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
    def getScoreInstance_deg(self, input_image, path):
        start_time = time.time()

        deg_padding = 3
        deg_image = np.zeros((input_image.shape[0]+deg_padding*2,input_image.shape[1]+deg_padding*2))

        input_tensor = torch.from_numpy(np.expand_dims(input_image, axis=0)).permute(0,3,1,2).float().to(self.device)
        output_tensor = self.model(input_tensor)
        out_delta =output_tensor[0].permute(1,2,0).cpu().detach().numpy()
        output_tensor = self.model2(input_tensor)
        out_heat = output_tensor[0].permute(1,2,0)[:,:,1].cpu().detach().numpy()

        delta_right_image = out_delta[:,:,0]
        delta_up_image = out_delta[:,:,1]

        get_model_output = time.time()
        # print("Get Model Output Time = {}".format(get_model_output-start_time))

        delta = 5
        arrow_size=4
        min_threshold = 5
        threshold = 35

        for i in range(90, delta_up_image.shape[0]-10, delta):
            for j in range(10, delta_up_image.shape[1]-10, delta):
                
                horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                vertical_direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
                startPoint = (j, i)

                x1 = int(delta_right_image[i,j])*horizone_direction + j
                y1 = i
                x2 = j
                y2 = int(delta_up_image[i,j])*vertical_direction + i
                m = (y2-y1)/(x2-x1+0.00001)
                a = m
                b = -1
                c = y1 - m*x1
                newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
                newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)

                dist = abs(int(delta_right_image[i,j])) + abs(int(delta_up_image[i,j]))

                deg = str( (delta_up_image[i,j]*vertical_direction) / (delta_right_image[i,j]*horizone_direction))[0:4]
                deg = str( (delta_up_image[i,j]*vertical_direction))[0:2] + " / "+str(delta_right_image[i,j]*horizone_direction)[0:2]
                deg=str(m)[0:5]

                deg= (math.atan2(y2-y1, x2-x1)*180.0/math.pi)
                while deg<0:
                    deg+=180
                while deg>180:
                    deg-=180
                if int(newy) < input_image.shape[0] and int(newx) < input_image.shape[1] and int(newy) > 0 and int(newx) > 0:
                    deg_image[int(newy)+deg_padding,int(newx)+deg_padding] = deg
        get_degMap_output = time.time()
        # print("Get DegMap Time = {}".format(get_degMap_output-get_model_output))
        # print("MODEL 2 NAME = {}".format)
        score = Scoring()
        score.device = self.device     
        lane_tensor = torch.zeros([10,30,2]).to(self.device)
        # print(path)
        # return
        key_height =  170
        first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)
        while first_key_tensor.shape[0]==0 or key_height>300:
            key_height += 10
            first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)
        key_height = 170
        while first_key_tensor.shape[0]==0 or key_height<100:
            key_height -= 10
            first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)

        if first_key_tensor.shape[0]==0:
            score.tensor2lane(lane_tensor)
            score.getLanebyH_sample(160, 710, 10)
            return score

        terminal_tensor = torch.zeros([10,2], dtype = torch.long).to(self.device)
        # terminal = 
        lt_idx = 7
        lane_tensor[0:first_key_tensor.shape[0], lt_idx] = first_key_tensor
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        lane_num=first_key_tensor.shape[0]

        # print("KeyPoint  {}".format(first_key_tensor))
        # print("Terminal = {}".format(terminal_tensor))
        #---------------- HEAT to INS ---------------------------------------------------------------
        for height in range(key_height+10, 330, 10):
            new_key =  score.getLocalMaxima_heatmap_re(out_heat, height)
            # print("-----------------------------------------------------------------------------------")
            # print("New Key = {}".format(new_key))
            lane_num = score.chainKey2(new_key, terminal_tensor, deg_image, lane_num)
            lt_idx +=1
            lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:4,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        # print("--------------IUDUDUDUDUUDUDUD")
        lt_idx=7
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        # print("New Terminal = {}".format(terminal_tensor))
        for height in range(key_height-10, 90, -10):
            new_key =  score.getLocalMaxima_heatmap_re(out_heat, height)
            # print("New Key = {}".format(new_key))
            lane_num = score.chainKey2(new_key, terminal_tensor, deg_image, lane_num)
            lt_idx -=1
            lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:3,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        get_pre_lane = time.time()
        # print("Get Pre lane Time = {}".format(get_pre_lane - get_degMap_output))
        # lane_tensor[:,0] *= (720.0/368.0)
        # lane_tensor[:,1] *= (720.0/368.0)
        score.tensor2lane(lane_tensor)
        # return
        score.getLanebyH_sample_deg(160, 710, 10)
        # score.getLanebyH_sample_deg(160, 710, 10)
        get_gt_lane = time.time()
        # print("Get GT lane Time = {}".format(get_gt_lane - get_pre_lane))

        # print(" -------------- N of Lane {}".format(len(score.lane_list)))
        # for lane in score.lane_list:
        #     print("        ---- {}".format(len(lane)))
        # print("SCORE LANE")
        # for lane in score.lane_list:
        #     print(lane)
        #     print("\n")

        #------------------------------------------------------------------- Save Image ---------------
        # idx=0
        # raw_image = np.copy(input_image)
        # resize_image = np.copy(input_image)
        # resize_image = cv2.resize(resize_image, (1280, 720))

        # # print("LANELIST = {}".format(score.lane_list))
        # for lane in lane_tensor:
        #     idx+=1
        #     for point in lane:
        #         raw_image = cv2.circle(raw_image, (int(point[1]), int(point[0])), 4, myColor.color_list[idx], -1)
        # idx=0
        # for lane in score.lane_list:
        #     # if idx != 2:
        #     #     continue
        #     # print("LANE = {}".format(lane))
        #     lane_idx = 0
        #     for height in range(160, 711, 10):
        #         # print("POINT {}".format(point))
        #         resize_image = cv2.circle(resize_image, (int(lane[lane_idx]), height), 4, myColor.color_list[idx], -1)
        #         lane_idx+=1
        #     idx+=1
        # os.makedirs(os.path.join(self.image_save_path, "Check_image"), exist_ok=True)

        # score_dir_raw = os.path.join(self.image_save_path, "Check_image",path[-12:-7]+"_data_raw.jpg")
        # score_dir = os.path.join(self.image_save_path, "Check_image", path[-12:-7]+"_data.jpg")
        # print(score_dir)
        # print(score_dir)

        # cv2.imwrite(score_dir_raw, raw_image)
        # cv2.imwrite(score_dir, resize_image)
        return score
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
    def inference_np2np_instance(self, image, model):
        # input_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).to(self.device).permute(0,3,1,2).float()
        input_tensor = torch.unsqueeze(torch.from_numpy(image).to(self.device), dim=0).permute(0,3,1,2).float()
        output_tensor = model(input_tensor)
        # print("Tensor {}".format(output_tensor.shape))
        output = output_tensor[0].permute(1,2,0).cpu().detach().numpy()
        return output
    def inference_np2tensor(self, image, model):
        input_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).permute(0,3,1,2).float().to(self.device)
        output_tensor = model(input_tensor)
        # print("Tensor {}".format(output_tensor.shape))
        output = output_tensor[0].permute(1,2,0)
        return output
    def print_inference_option(self):
        print("------------ Inference Parameter -------------------")
        print("View Mode       : {}".format(self.cfg.show))
        print("Model Name      : {}".format(self.cfg.model_path))
        print("Model Path      : {}".format(self.cfg.backbone))
        print("Device Name     : {}".format(self.device))
        print("Image Dir path  : {}".format(self.image_path))
        print("Image save path : {}".format(self.image_save_path))
        print("Image gt_path   : {}".format(self.gt_path))
        
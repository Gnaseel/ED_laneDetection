import os
import numpy as np
import torch
import data.sampleColor as myColor
from tool.scoring import Scoring
import evaluator.lane as EV
from back_logic.network_logic import Network_Logic
import cv2
import time
import math

##
# @file image_saver.py
# @brief package for save image


class ImgSaver:
        def __init__(self, cfg):
            self.cfg=cfg
            self.image_save_path = os.path.dirname(self.cfg.model_path)+"/Image"
            self.device = torch.device('cpu')

        def save_image_seg(self, model, img, output_image, fileName):
            seg_folder_name = "seged"
            seg_dir_name= os.path.join(self.image_save_path, seg_folder_name)
            # --------------------------Save segmented map
            os.makedirs(seg_dir_name, exist_ok=True)
    
            back_fir_dir = os.path.join(seg_dir_name,str(fileName)+"_back.jpg")
            lane_fir_dir = os.path.join(seg_dir_name,str(fileName)+"_lane.jpg")
    
            output_image = self.inference_np2np_instance(img, model)
    
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
            cv2.imwrite(back_fir_dir, back)
            cv2.imwrite(lane_fir_dir, lane)
    
            return

        def save_image_softmax(self, model, img, output_image, fileName):
            # --------------------------Save segmented map
            softmax_folder_name = "softmaxed"
            softmax_dir_name= os.path.join(self.image_save_path, softmax_folder_name)
            os.makedirs(softmax_dir_name, exist_ok=True)
            print(os.path.join(self.image_save_path, softmax_folder_name))
            input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float().to(self.device)

            # soft_image = self.getSoftMaxImgfromTensor(input_tensor)
            output_tensor = model(input_tensor)

            for idx, seg_tensor in enumerate(output_tensor[0]):
                # print(seg_tensor.shape)
                seg_image = seg_tensor.cpu().detach().numpy()
                # print(seg_image.shape)
                cv2.imwrite(os.path.join(softmax_dir_name, "_{}.jpg".format(idx)), seg_image*30)
            return

        def save_image_deg(self, image, out_heat, score, img_path, fileName, delta_height=10, delta_threshold = 50):

            # --------------------------Save segmented map
            delta_folder_name = "delta"
            delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
            os.makedirs(delta_dir_name, exist_ok=True)

            raw_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw.jpg")
            gt_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_GT.jpg")
            seged_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_Seged.jpg")
            heat_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_heat.jpg")
            heat_key_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_heat_key.jpg")
            lane_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_lane.jpg")

            gt_path = os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/seg_label", *img_path.split(os.sep)[1:-1],"20.png")
            # gt_path = os.path.join(img_path)
            print("GT PAHT {}".format(gt_path))
            gt_img = cv2.imread(gt_path)*30
            seged_image = cv2.resize(out_heat.cpu().detach().numpy()[:,:,1]*50, (1280,720))
            output_key_image = cv2.resize(np.copy(image), (1280,720))
            output_lane_image = cv2.resize(np.copy(image), (1280,720))


            nl = Network_Logic()
            nl.device = self.device

            # score = Scoring()
            # score.device = self.device     
            # score = nl.getScoreInstance_deg("temp_path", out_heat, out_delta)
            key_list = nl.getKeypoint(out_heat[:,:,1])
            for idx, lane in enumerate(key_list):
                output_key_image = cv2.circle(output_key_image, (int(lane[1]*1280.0/640.0),int(lane[0]*720.0/368.0)), 5, myColor.color_list[0], -1)
            for idx, lane in enumerate(score.lane_list):
                if len(lane) <=2:
                    continue
                for idx2, height in enumerate(range(160, 710+1, 10)):
                    if lane[idx2] > 0:
                        seged_image = cv2.circle(seged_image, (int(lane[idx2]),height), 15, myColor.color_list[idx])
                        # gt_img = cv2.circle(gt_img, (int(lane[idx2]),height), 15, myColor.color_list[idx])
                        # raw_img = cv2.circle(raw_img, (lane[idx2],height), 15, myColor.color_list[idx])
                        if idx > 10:
                            idx = 10
                        output_lane_image = cv2.circle(output_lane_image, (int(lane[idx2]),height), 5, myColor.color_list[idx], -1)
                    idx2+=1
                idx+=1
            gt_path = "./evaluator/gt.json"
            # print("LANE {}".format(type(score.lane_list)))
            print("LANE {}".format(score.lane_list))
            # print("Image path {}".format(img_path))
            ev = EV.LaneEval.bench_one_instance(score.lane_list, img_path, gt_path)
            cv2.imwrite(raw_fir_dir, image)
            cv2.imwrite(heat_key_fir_dir, output_key_image)
            cv2.imwrite(lane_fir_dir, output_lane_image)
            out_heat = out_heat.cpu().detach().numpy()
            cv2.imwrite(heat_fir_dir, (out_heat[:,:,1]+3)*50)
            gt_img = np.where(gt_img>0, 255, 0)
            cv2.imwrite(gt_fir_dir, gt_img)
            cv2.imwrite(seged_fir_dir, seged_image)

        def save_image_deg_basic(self, image, output_image, fileName, delta_height=10, delta_threshold = 50):
            # return
            output_image = output_image.cpu().detach().numpy()
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

            delta_key_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_key.jpg")

        
            # output_image = self.inference_np2np_instance(image, model)

            output_right_image = np.copy(image)
            output_right_circle_image = np.copy(image)
            output_up_image = np.copy(image)
            output_up_circle_image = np.copy(image)
            output_delta_key_image = np.copy(image)

            delta_right_image = output_image[:,:,0]
            delta_up_image = output_image[:,:,1]

            nl = Network_Logic()
            nl.device = self.device
            delta_key_list = nl.getKeypoint(output_image[:,:,0],  threshold = 2.5,reverse = True)
            for idx, lane in enumerate(delta_key_list):
                output_delta_key_image = cv2.circle(output_delta_key_image, (int(lane[1]),int(lane[0])), 2, myColor.color_list[0], -1)
                # output_delta_key_image = cv2.circle(output_delta_key_image, (int(lane[1]*1280.0/640.0),int(lane[0]*720.0/368.0)), 5, myColor.color_list[0], -1)

            # Arrow, Circle Image
            for i in range(130, delta_right_image.shape[0], 20):
                for j in range(10, delta_right_image.shape[1], 30):
                    startPoint = (j, i)

                    if j+11 > delta_right_image.shape[1] or j-11 < 0:
                        continue
                    if delta_right_image[i,j] < delta_threshold:
                        direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                        endPoint = (int(delta_right_image[i,j])*direction + j, i)
                        # cv2.circle(output_right_image, startPoint, 1, (255,0,0), -1)
                        output_right_circle_image = cv2.circle(output_right_circle_image, endPoint, 1, (0,0,255), -1)
                        output_right_image = cv2.arrowedLine(output_right_image, startPoint, endPoint, (0,0,255), 1)
                    if delta_up_image[i,j] < delta_threshold:
                        direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
                        endPoint = (j, int(delta_up_image[i,j])*direction +i)
                        # cv2.circle(output_up_image, startPoint, 1, (255,0,0), -1)
                        output_up_image = cv2.arrowedLine(output_up_image, startPoint, endPoint, (0,0,255), 1)
                        output_up_circle_image = cv2.circle(output_up_circle_image, endPoint, 1, (0,0,255), -1)


            cv2.imwrite(right_fir_dir, output_right_image)
            cv2.imwrite(up_fir_dir, output_up_image)
            cv2.imwrite(right_circle_fir_dir, output_right_circle_image)
            cv2.imwrite(up_circle_fir_dir, output_up_circle_image)
            cv2.imwrite(raw_right_fir_dir, delta_right_image)
            cv2.imwrite(raw_up_fir_dir, delta_up_image)
            cv2.imwrite(delta_key_fir_dir, output_delta_key_image)

        def save_image_deg_total(self, image, output_image, fileName):
            # --------------------------Save segmented map

            delta_folder_name = "delta"
            delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
            os.makedirs(delta_dir_name, exist_ok=True)

            output_total_arrow_image = np.copy(image)

            delta_right_image = output_image[:,:,0]
            delta_up_image = output_image[:,:,1]

            total_arrow_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_total_arrow.jpg")

            delta = 3
            arrow_size= 2
            min_threshold = 5
            threshold = 60

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

                    endpoint_total_arrow = (int(delta_right_image[i,j])*horizone_direction + j, int(delta_up_image[i,j])*vertical_direction + i)
                    endpoint_total_arrow = (int(newx), int(newy))

                    dist = abs(int(delta_right_image[i,j])) + abs(int(delta_up_image[i,j]))

                    output_total_arrow_image = cv2.circle(output_total_arrow_image, startPoint, 2, (0,255,255), -1)
                    if dist > threshold or dist < min_threshold:
                        continue
                    
                    if vertical_direction<0:
                        output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,0,255), arrow_size)

                    else:
                        output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,255, 0), arrow_size)
            cv2.imwrite(total_arrow_fir_dir, output_total_arrow_image)
            return
        
        def save_image_dir_deg(self, delta_model, heat_model, filePaths, num_of_good = 5):
            for file_idx, file in enumerate(filePaths):
                # print("PATH : {}".format(file))
                img = cv2.imread(file)
                img = cv2.resize(img, (delta_model.output_size[1], delta_model.output_size[0]))

                nl = Network_Logic()
                nl.device=self.device
                input_tensor = torch.unsqueeze(torch.from_numpy(img).to(self.device), dim=0).permute(0,3,1,2).float()
                output_tensor = delta_model(input_tensor)
                out_delta = output_tensor[0].permute(1,2,0).cpu().detach().numpy()
                output_tensor = heat_model(input_tensor)
                out_heat = output_tensor[0].permute(1,2,0)
                score = nl.getScoreInstance_deg(file, out_heat, out_delta)

                # score.prob2lane(img_idx, img_val, 40, 350, 5 )
                # score.getLanebyH_sample(160, 710, 10)
                path_list = file.split(os.sep)
                raw_img = cv2.resize(img, dsize = (1280, 720))
                cls_img = self.inference_np2np_instance(img,heat_model)[:,:,1]
                cls_img = cv2.resize(cls_img, dsize = (1280, 720))
                cls_img = (cls_img-0.4)*30
                # print("PATH LIST = {}".format(path_list))
                # print("IDX = {}".format(path_list.index('clips')))
                clip_idx = path_list.index('clips')
                gt_path = os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/seg_label", *path_list[clip_idx+1:-1],"20.png")

                # print("GT PATH = {}".format(gt_path))
                gt_img = cv2.imread(gt_path)*30


                for idx, lane in enumerate(score.lane_list):
                    if len(lane) <=2:
                        continue
                    for idx2, height in enumerate(range(160, 710+1, 10)):
                        if lane[idx2] > 0:
                            cls_img = cv2.circle(cls_img, (lane[idx2],height), 5, myColor.color_list[idx if idx <=10 else 10], -1)
                            gt_img = cv2.circle(gt_img, (lane[idx2],height), 5, myColor.color_list[idx if idx <=10 else 10], -1)
                            raw_img = cv2.circle(raw_img, (lane[idx2],height), 5, myColor.color_list[idx if idx <=10 else 10], -1)
                        idx2+=1
                    idx+=1

                if file_idx > num_of_good:
                    dir_name = os.path.join(self.image_save_path, "check", "good_del")
                else:
                    dir_name = os.path.join(self.image_save_path, "check", "bad_del")
                os.makedirs(dir_name, exist_ok = True)
                fileName = "raw"
                fir_dir = os.path.join(dir_name,path_list[clip_idx+1] + "_" + path_list[clip_idx+2] + "_" + str(fileName)+".jpg")
                # print("Raw Path = {}".format(fir_dir))
                cv2.imwrite(fir_dir, raw_img)
                fileName = "segmented"
                fir_dir = os.path.join(dir_name,path_list[clip_idx+1] + "_" + path_list[clip_idx+2] + "_" + str(fileName)+".jpg")
                # print("Seg Path = {}".format(fir_dir))
                cv2.imwrite(fir_dir, cls_img)
                gtfileName = "ground_truth"
                fir_dir = os.path.join(dir_name,path_list[clip_idx+1] + "_" + path_list[clip_idx+2] + "_" + str(gtfileName)+".jpg")
                # print("GT Path = {}".format(fir_dir))
                cv2.imwrite(fir_dir, gt_img)
            return

        def save_image_delta(self, image, output_image, fileName, delta_height=10, delta_threshold = 50):
            # return
            output_image = output_image.cpu().detach().numpy()
            # --------------------------Save segmented map
            delta_folder_name = "delta"
            delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
            os.makedirs(delta_dir_name, exist_ok=True)

            raw_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw.jpg")
            right_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_right.jpg")
            right_10_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_up_10.jpg")

            # up_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_up_circle.jpg")
            # right_circle_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_right_circle.jpg")

            # raw_right_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw_right.jpg")
            # raw_up_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw_up.jpg")

            # delta_key_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_key.jpg")

        
            # output_image = self.inference_np2np_instance(image, model)

            # output_right_image = np.copy(image)
            # output_right_circle_image = np.copy(image)
            # output_up_image = np.copy(image)
            # output_up_circle_image = np.copy(image)
            # output_delta_key_image = np.copy(image)

            # delta_right_image = output_image[:,:,0]
            # delta_up_image = output_image[:,:,1]

            # nl = Network_Logic()
            # nl.device = self.device
            # delta_key_list = nl.getKeypoint(output_image[:,:,0],  threshold = 5.5,reverse = True)
            # for idx, lane in enumerate(delta_key_list):
            #     output_delta_key_image = cv2.circle(output_delta_key_image, (int(lane[1]),int(lane[0])), 5, myColor.color_list[0], -1)
            #     # output_delta_key_image = cv2.circle(output_delta_key_image, (int(lane[1]*1280.0/640.0),int(lane[0]*720.0/368.0)), 5, myColor.color_list[0], -1)

            # # Arrow, Circle Image
            # for i in range(130, delta_right_image.shape[0], 20):
            #     for j in range(10, delta_right_image.shape[1], 30):
            #         startPoint = (j, i)

            #         if j+11 > delta_right_image.shape[1] or j-11 < 0:
            #             continue
            #         if delta_right_image[i,j] < delta_threshold:
            #             direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
            #             endPoint = (int(delta_right_image[i,j])*direction + j, i)
            #             # cv2.circle(output_right_image, startPoint, 1, (255,0,0), -1)
            #             output_right_circle_image = cv2.circle(output_right_circle_image, endPoint, 1, (0,0,255), -1)
            #             output_right_image = cv2.arrowedLine(output_right_image, startPoint, endPoint, (0,0,255), 1)
            #         if delta_up_image[i,j] < delta_threshold:
            #             direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
            #             endPoint = (j, int(delta_up_image[i,j])*direction +i)
            #             # cv2.circle(output_up_image, startPoint, 1, (255,0,0), -1)
            #             output_up_image = cv2.arrowedLine(output_up_image, startPoint, endPoint, (0,0,255), 1)
            #             output_up_circle_image = cv2.circle(output_up_circle_image, endPoint, 1, (0,0,255), -1)

            print("SHAPE {}".format(output_image.shape))
            cv2.imwrite(raw_fir_dir, image)
            cv2.imwrite(right_fir_dir, output_image[:,:,0])
            cv2.imwrite(right_10_fir_dir, output_image[:,:,1])
            # cv2.imwrite(right_circle_fir_dir, output_right_circle_image)
            # cv2.imwrite(up_circle_fir_dir, output_up_circle_image)
            # cv2.imwrite(raw_right_fir_dir, delta_right_image)
            # cv2.imwrite(raw_up_fir_dir, delta_up_image)
            # cv2.imwrite(delta_key_fir_dir, output_delta_key_image)
         
        def save_image_dir(self, model, filePaths):

            print("SAVE IMAGE")
            # --------------------------Save segmented map
            for file in filePaths:
                print("PATH : {}".format(file))
                img = cv2.imread(file)
                img = cv2.resize(img, (model.output_size[1], model.output_size[0]))

                # print(img.shape)
                input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float().to(self.device)
                cls_soft = self.getSoftMaxImgfromTensor(input_tensor)
                img_idx = cls_soft.indices.to('cpu').numpy()
                img_val = cls_soft.values.to('cpu').numpy()
                # output_tensor = model(input_tensor)
                # m = torch.nn.Softmax(dim=0)
                # cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
                score = Scoring()
                score.prob2lane(img_idx, img_val, 40, 350, 5 )
                score.getLanebyH_sample(160, 710, 10)
                path_list = file.split(os.sep)
                cls_img = cls_soft.indices.detach().to('cpu').numpy().astype(np.uint8)
                cls_img = cv2.cvtColor(cls_img, cv2.COLOR_GRAY2BGR)
                cls_img = cv2.resize(cls_img, dsize = (1280, 720), interpolation=cv2.INTER_NEAREST)*30
                raw_img = cv2.resize(img, dsize = (1280, 720), interpolation=cv2.INTER_NEAREST)
                gt_path = os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/seg_label", *path_list[8:-1],"20.png")
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
                # print("Raw Path = {}".format(fir_dir))
                cv2.imwrite(fir_dir, raw_img)
                fileName = "20_segmented"
                fir_dir = os.path.join(self.image_save_path,path_list[8] + "_" + path_list[9] + "_" + str(fileName)+".jpg")
                # print("Seg Path = {}".format(fir_dir))
                cv2.imwrite(fir_dir, cls_img)
                gtfileName = "20_ground_truth"
                fir_dir = os.path.join(self.image_save_path,path_list[8] + "_" + path_list[9] + "_" + str(gtfileName)+".jpg")
                # print("GT Path = {}".format(fir_dir))
                cv2.imwrite(fir_dir, gt_img)
        
        def inference_np2np_instance(self, image, model):
            # input_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).to(self.device).permute(0,3,1,2).float()
            input_tensor = torch.unsqueeze(torch.from_numpy(image).to(self.device), dim=0).permute(0,3,1,2).float()
            output_tensor = model(input_tensor)
            # print("Tensor {}".format(output_tensor.shape))
            output = output_tensor[0].permute(1,2,0).cpu().detach().numpy()
            return output

        def getSoftMaxImgfromTensor(self, input_tensor):
            output_tensor = self.model(input_tensor)
            m = torch.nn.Softmax(dim=0)
            cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
            return cls_soft

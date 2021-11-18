import os
import numpy as np
import torch
import data.sampleColor as myColor
from tool.scoring import Scoring
import cv2
import time
import math
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
        def save_image_delta(self, model, image, output_image, fileName, delta_height=10, delta_threshold = 20):
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
            output_image = self.inference_np2np_instance(image, model)


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
        def save_image_deg(self, model, model2, image, output_image, fileName, delta_height=10, delta_threshold = 50):
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


            # print("-------------------------------")
            # print(image.shape)
            output_image = self.inference_np2np_instance(image, model)


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


            output_image2  = self.inference_np2np_instance(image, model2)[:,:,1]
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
        def save_image_dir_deg(self, model, model2, filePaths):
            for file in filePaths:
                print("PATH : {}".format(file))
                img = cv2.imread(file)
                img = cv2.resize(img, (model.output_size[1], model.output_size[0]))

                # print(img.shape)
                #             input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float().to(self.device)
                #             cls_soft = self.getSoftMaxImgfromTensor(input_tensor)
                #             img_idx = cls_soft.indices.to('cpu').numpy()
                #             img_val = cls_soft.values.to('cpu').numpy()
                #             # output_tensor = model(input_tensor)
                #             # m = torch.nn.Softmax(dim=0)
                #             # cls_soft = torch.max(m(output_tensor[0][:]).detach(), 0)
                score = self.getScoreInstance_deg(img, file)
                # score.prob2lane(img_idx, img_val, 40, 350, 5 )
                # score.getLanebyH_sample(160, 710, 10)
                path_list = file.split(os.sep)
                raw_img = cv2.resize(img, dsize = (1280, 720))
                cls_img = self.inference_np2np_instance(img, model2)[:,:,1]
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

        def inference_np2np_instance(self, image, model):
            # input_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).to(self.device).permute(0,3,1,2).float()
            input_tensor = torch.unsqueeze(torch.from_numpy(image).to(self.device), dim=0).permute(0,3,1,2).float()
            output_tensor = model(input_tensor)
            # print("Tensor {}".format(output_tensor.shape))
            output = output_tensor[0].permute(1,2,0).cpu().detach().numpy()
            return output
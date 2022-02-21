import os
import numpy as np
import torch
import data.sampleColor as myColor
from tool.scoring import Scoring
import evaluator.lane as EV
from back_logic.network_logic import Network_Logic
from back_logic.laneBuilder import LaneBuilder
from back_logic.laneBuilder import Lane
import cv2
import time
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons

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

        def save_image_deg(self, image, out_heat, lane_list, img_path, fileName, delta_height=10, delta_threshold = 50):

            # --------------------------Save segmented map
            delta_folder_name = "delta"
            delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
            os.makedirs(delta_dir_name, exist_ok=True)

            extention = "png"
            raw_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_raw."+extention)
            gt_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_GT."+extention)
            seged_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_Seged."+extention)
            heat_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_heat."+extention)
            heat_fir_dir_raw = os.path.join(delta_dir_name,str(fileName)+"_heat_raw."+extention)
            heat_fir_dir_back = os.path.join(delta_dir_name,str(fileName)+"_heatback."+extention)
            heat_key_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_heat_key."+extention)
            lane_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_lane."+extention)

            # gt_path = os.path.join(img_path)
            # seged_image = cv2.resize(out_heat.cpu().detach().numpy()[:,:]*50, (1280,720))
            # out_heat = torch.unsqueeze(out_heat, dim=2)
            # print(image.shape)
            # print("Heat Shape {}".format(out_heat[1].cpu().detach().numpy().shape))
            ## HERE!!
            seged_image = cv2.resize(np.copy(image), (1280,720))
            output_key_image = cv2.resize(np.copy(image), (1280,720))
            output_lane_image = cv2.resize(np.copy(image), (1280,720))


            nl = Network_Logic()
            nl.device = self.device

            # score = Scoring()
            # score.device = self.device     
            # score = nl.getScoreInstance_deg("temp_path", out_heat[1], out_delta)
            key_list = nl.getKeypoint(out_heat[1])
            for idx, lane in enumerate(key_list):
                output_key_image = cv2.circle(output_key_image, (int(lane[1]*1280.0/640.0),int(lane[0]*720.0/368.0)), 5, myColor.color_list[0], -1)
            for idx, lane in enumerate(lane_list):
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
            # print("LANE {}".format(lane_list))
            if img_path.find("\\") != -1:
                img_path = img_path.replace("\\", "/")
            a, p, n = EV.LaneEval.bench_one_instance(lane_list, img_path, "./evaluator/gt.json")
            cv2.imwrite(raw_fir_dir, image)
            cv2.imwrite(heat_key_fir_dir, output_key_image)
            cv2.imwrite(lane_fir_dir, output_lane_image)
            # print("SHAPE2 {}".format(out_heat[1].shape))
            out_heat = out_heat.cpu().detach().numpy()
            heat_raw = np.where(out_heat[1] > out_heat[0], 200, 0)
            # heat_img = np.where(out_heat[1] > out_heat[0], 1, 0)
            # un, co = np.unique(out_heat[1], return_counts=True)
            # print(dict(zip(un,co)))
            cv2.imwrite(heat_fir_dir_raw, heat_raw )
            threshold=10
            # out_heat[1] = np.where(out_heat[1] > threshold, 10, -1)
            # torch.where(out_heat[1] > torch.tensor(0.5), torch.tensor(10), torch.tensor(0))
            # out_heat[0] = np.where(out_heat[0] < threshold, threshold, out_heat[0])
            out_heat[0] = cv2.normalize(out_heat[0], None, 0, 255, cv2.NORM_MINMAX)

            th=0
            cv2.imwrite(heat_fir_dir, (out_heat[1]+th)*10)
            # cv2.imwrite(heat_fir_dir_back, (out_heat[0]+th)*10)
            cv2.imwrite(heat_fir_dir_back, out_heat[0]*10)
            # gt_path = os.path.join("/workspace/data/tuSimple/seged_gt", *img_path.split(os.sep)[1:-1],"20.jpg")
            gt_path = os.path.join("/workspace/data/tuSimple/seged_gt", img_path)[:-3]+"png"
            print(gt_path)
            print(gt_path)
            print(img_path)
            print(img_path)
            if not os.path.isfile(gt_path):
                gt_path = os.path.join(os.sep,*img_path)[:-3]+"png"
                # print("IGM PATH {}".format(os.path.join(gt_path, *img_path.split(os.sep)[-3:])[:-3]+"png"))
            if os.path.isfile(gt_path):
                gt_img = cv2.imread(gt_path)
                print("GT PAHT {}".format(gt_path))
                gt_img = np.where(gt_img>0, 255, 0)
                cv2.imwrite(gt_fir_dir, gt_img)
            else:
                print("GT not found {}".format(gt_path))
            cv2.imwrite(seged_fir_dir, seged_image)

        def save_image_deg_basic(self, image, output_image, fileName, delta_height=10, delta_threshold = 30):
            delta_threshold_min=3
            # return
            output_image = output_image[0].permute(1,2,0).cpu().detach().numpy()
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

            output_right_arrow_image = np.copy(image)
            output_right_circle_image = np.copy(image)
            output_up_image = np.copy(image)
            output_up_circle_image = np.copy(image)
            output_delta_key_image = np.copy(image)

            delta_right_image = output_image[:,:,0]
            delta_up_image = output_image[:,:,1]

            # nl = Network_Logic()
            # nl.device = self.device
            # delta_key_list = nl.getKeypoint(output_image[:,:,0],  threshold = 2.5,reverse = True)
            # for idx, lane in enumerate(delta_key_list):
            #     output_delta_key_image = cv2.circle(output_delta_key_image, (int(lane[1]),int(lane[0])), 2, myColor.color_list[0], -1)
                # output_delta_key_image = cv2.circle(output_delta_key_image, (int(lane[1]*1280.0/640.0),int(lane[0]*720.0/368.0)), 5, myColor.color_list[0], -1)

            # Arrow, Circle Image
            lane_in_height=[0 for i in range(6)]
            key_list=[]
            key_up_list=[]
            for i in range(130, delta_right_image.shape[0], 10):

                width_list=[]
                for j in range(10, delta_right_image.shape[1], 10):

                    startPoint = (j, i)
                    output_right_circle_image = cv2.circle(output_right_circle_image, startPoint, 1, (255,0,0), -1)
                    output_up_circle_image = cv2.circle(output_up_circle_image, startPoint, 1, (255,0,0), -1)

                    if j+11 > delta_right_image.shape[1] or j-11 < 0:
                        continue
                    if  delta_threshold_min < delta_right_image[i,j] and delta_right_image[i,j] < delta_threshold:
                        direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                        width_list.append(int(delta_right_image[i,j])*direction + j)
                        endPoint = (int(delta_right_image[i,j])*direction + j, i)
                        # cv2.circle(output_right_arrow_image, startPoint, 1, (255,0,0), -1)
                        output_right_circle_image = cv2.circle(output_right_circle_image, endPoint, 1, (0,0,255), -1)
                        output_right_arrow_image = cv2.arrowedLine(output_right_arrow_image, startPoint, endPoint, (0,0,255), 1)
                        
                    if  delta_threshold_min < delta_up_image[i,j] and delta_up_image[i,j] < delta_threshold:
                        direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
                        endPoint = (j, int(delta_up_image[i,j])*direction +i)
                        # cv2.circle(output_up_image, startPoint, 1, (255,0,0), -1)
                        output_up_image = cv2.arrowedLine(output_up_image, startPoint, endPoint, (0,0,255), 1)
                        output_up_circle_image = cv2.circle(output_up_circle_image, endPoint, 1, (0,0,255), -1)
   

                if len(width_list)==0:
                    continue

                count=1
                
                buf_count=1
                buf=last=width_list[0]
                point_list=[]
                point_up_list=[]
                # print("width_list {}".format(width_list))
                for idx in width_list[1:]:
                    if idx > last+40:
                        count +=1
                        point_list.append([i, int(buf/buf_count)])
                        # print("ADDED !! {}, {}".format(buf, buf_count))
                        buf_count=1
                        buf=idx
                    else:
                        buf_count+=1
                        buf+=idx
                    last = idx
                if buf_count != 0:
                    # print("ADDED !! {}, {}".format(buf, buf_count))
                    point_list.append([i, int(buf/buf_count)])

                for idxs in point_list:
                    # print("IDXS {}".format(idxs))
                    idx = idxs[1]
                    output_right_circle_image = cv2.circle(output_right_circle_image, (idx, i), 3, (0,255,0), -1)
                    min_abs =  100

                    for point in range(idx-40, idx+41, 2):
                        if  0<point and point < delta_right_image.shape[1] and 7 < delta_up_image[i,point] and delta_up_image[i,point] < 13:
                            # print("!! {} {}".format(point, delta_right_image.shape[1]))
                            direction = -1 if delta_up_image[i-5,point] > delta_up_image[i+5,point] else 1
                            if direction == -1: # Go Down
                                continue
                            resi = abs(idx-point)
                            if resi < min_abs:
                                new_10_start_point = (idx, i)
                                new_10_point = (point, i-10*direction)
                                min_abs = resi
                    if min_abs < 99:
                        output_right_circle_image = cv2.arrowedLine(output_right_circle_image, new_10_start_point, new_10_point, (0,255,255), 2)
                        point_up_list.append([new_10_start_point[0], new_10_point[0]])
                            # output_right_circle_image = cv2.circle(output_right_circle_image, new_10_point, 1, (0,255,255), -1)

                # -------------- Get Lane Num --------------------
                # print("new_width_list {}".format(point_list))
                # print("Height {}, Count {}".format(i, count))
                if count>5:
                    count=5
                lane_in_height[count] +=1
                key_list.append(point_list)
                key_up_list.append(point_up_list)
            key_up_list = key_up_list[1:]
            builder = LaneBuilder()
            lane_data = Lane()
            lane_data = builder.buildLane(key_list,  delta_up_image)
            for idx, lane in enumerate(lane_data.lane_list):
                for point in lane:
                    output_right_circle_image = cv2.circle(output_right_circle_image, (point[1], point[0]), 5, myColor.color_list[idx if idx <=10 else 10], -1)
            myList = lane_data.convert_tuSimple()
            # print(myList)
            # print("Lane Num {}".format(lane_data.lanes_num))   
            # print("Lane Idx Num {}".format(lane_data.lane_idx))   
            
            # print("Key List {}".format(key_list))   
            # print("Key Up List {}".format(key_up_list))   
            cv2.imwrite(right_fir_dir, output_right_arrow_image)
            cv2.imwrite(up_fir_dir, output_up_image)
            cv2.imwrite(right_circle_fir_dir, output_right_circle_image)
            cv2.imwrite(up_circle_fir_dir, output_up_circle_image)
            cv2.imwrite(raw_right_fir_dir, delta_right_image)
            cv2.imwrite(raw_up_fir_dir, delta_up_image)
            cv2.imwrite(delta_key_fir_dir, output_delta_key_image)

        def save_image_deg_total(self, image, output_image, heat_map, fileName):
            # --------------------------Save segmented map
            # heat_map = heat_map.cpu().detach().numpy()
            delta_folder_name = "delta"
            delta_dir_name= os.path.join(self.image_save_path, delta_folder_name)
            os.makedirs(delta_dir_name, exist_ok=True)

            output_total_arrow_image = np.copy(image)

            delta_right_image = output_image[:,:,0]
            delta_up_image = output_image[:,:,1]

            total_arrow_fir_dir = os.path.join(delta_dir_name,str(fileName)+"_delta_total_arrow.jpg")
            img_height = delta_up_image.shape[0]
            img_width = delta_up_image.shape[1]
            delta = 5
            arrow_size= 2
            min_threshold = 10
            threshold = 25
            temp_tensor = torch.zeros([3,10000])
            tensor_idx=0
            for i in range(90, delta_up_image.shape[0]-10, delta):
                for j in range(10, delta_up_image.shape[1]-10, delta):


                    horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                    vertical_direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
                    startPoint = (j, i)
                    delta_right_val = int(delta_right_image[i,j])
                    delta_up_val = int(delta_up_image[i,j])

       
                    x1 = delta_right_val*horizone_direction + j
                    y1 = i
                    x2 = j
                    y2 = delta_up_val*vertical_direction + i
                    m = (y2-y1)/(x2-x1+0.00001)
                    a = m
                    b = -1
                    c = y1 - m*x1
                    newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
                    newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)
                    endpoint_total_arrow = (delta_right_val*horizone_direction + j, delta_up_val*vertical_direction + i)
                    endpoint_total_arrow = (int(newx), int(newy))
                    if newx<0 or img_width < newx or newy<0 or img_height<newy:
                        continue
                    if heat_map[int(newy), int(newx)] < -3.5:
                        continue
                    dist = abs(delta_right_val) + abs(delta_up_val)

                    output_total_arrow_image = cv2.circle(output_total_arrow_image, startPoint, 2, (0,255,255), -1)
                    if dist > threshold or dist < min_threshold:
                        continue
                    deg= (math.atan2(y2-y1, x2-x1)*180.0/math.pi)
                    if deg < 5 or deg > 175:
                        continue
                    if delta_right_image[i,j]<3 or delta_up_image[i,j]<3:
                        deg=0
                    while deg<0:
                        deg+=180
                    while deg>180:
                        deg-=180


                    temp_tensor[0,tensor_idx] = newx
                    temp_tensor[1,tensor_idx] = newy
                    temp_tensor[2,tensor_idx] = deg
                    tensor_idx+=1                    
                    if vertical_direction<0:
                        output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,0,255), arrow_size)
                    else:
                        output_total_arrow_image = cv2.arrowedLine(output_total_arrow_image, startPoint, endpoint_total_arrow, (0,255, 0), arrow_size)
            # self.save_image_cluster(temp_tensor[:,0:tensor_idx-1])
            cv2.imwrite(total_arrow_fir_dir, output_total_arrow_image)
            return
        
        # lane_tensor N*3 tensor (x,y,deg)
        def save_image_cluster(self, lane_tensor):
            # lane_tensor
            # for lane in lane_tensor[:,:]:
            #     plt.scatter(lane[0], lane[1], lane[2])
            # plt.subplot()
            db = DBSCAN(eps=20, min_samples=10).fit(torch.transpose(lane_tensor, 0, 1))
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            print(len(np.where(labels==-1)[0]))
            # print(np.where(labels==-1))
            print(len(np.where(labels==0)[0]))
            print(len(np.where(labels==1)[0]))
            print(len(np.where(labels==2)[0]))
            print(len(np.where(labels==3)[0]))

            plt.figure(figsize=(50, 25))
            ax = plt.subplot(1, 2, 1, projection='3d')
            ax.set_zlim(-10, 3600)
            nt = lane_tensor[2]*20
            ax.invert_yaxis()
            ax.scatter(lane_tensor[0], lane_tensor[1], nt,  s = 15,  marker='o')
            # ax.scatter(lane_tensor[0], lane_tensor[1], nt,c = labels,  s = 15,  marker='o')



            ax2 = plt.subplot(1, 2, 2) 
            ax2.invert_yaxis()
            ax2.scatter(lane_tensor[0], lane_tensor[1])
            # ax2.scatter(lane_tensor[0], lane_tensor[1], c = labels)
            # plt.subplot(1, 3, 3) 
            # X, Y = make_moons(noise=0.07, random_state=1)
            # print("X SHAPE {} {}".format(X.shape, X))
            # print("Y SHAPE {} {}".format(Y.shape, Y))
            # plot_clusters(data, cluster.DBSCAN, (), {'eps':0.020})
            plt.savefig('mmmmmmm.png')

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax2 = fig.add_subplot()
            # ax.scatter(lane_tensor[0], lane_tensor[1], lane_tensor[2])
            # ax2.scatter(lane_tensor[0], lane_tensor[1])
            # fig.savefig('mmmmmmm.png')
            return

        def save_image_dir_deg(self, inferencer, path, dir_name):
            
            # for file_idx, file in enumerate(paths):
            img = cv2.imread(path)
            lane_list, temp, temp = inferencer.inference_instance(img)
            path_list = path.split(os.sep)
            clip_idx = path_list.index('clips')
            raw_img = self.plot_lane_img(img, lane_list)
            gt_img = cv2.imread(os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/seg_label", *path_list[clip_idx+1:-1],"20.png"))

            gt_path = "./evaluator/gt.json"
            img_path = os.path.join(*path.split(os.sep)[-4:])    

            # PATH : /home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/clips/0601/1494453723505661848/20.jpg
            # IMG PATH = clips/0601/1494453723505661848/20.jpg
            # print("PATH : {}".format(path))
            # print("IMG PATH = {}".format(img_path))            
            # print("dir_name PATH = {}".format(dir_name))            
            ev = EV.LaneEval.bench_one_instance(lane_list, img_path, gt_path)
            # print("EV = {}".format(ev))
            raw_img = cv2.putText(raw_img, str(ev[0])[0:5], (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
 
            dir_name = os.path.join(self.image_save_path, "check_" + self.cfg.model_path.split(os.sep)[-1], dir_name)
            os.makedirs(dir_name, exist_ok = True)
            fileName = "raw"
            fir_dir = os.path.join(dir_name,path_list[clip_idx+1] + "_" + path_list[clip_idx+2] + "_" + str(fileName)+".jpg")
            cv2.imwrite(fir_dir, raw_img)
            gtfileName = "ground_truth"
            # fir_dir = os.path.join(dir_name,path_list[clip_idx+1] + "_" + path_list[clip_idx+2] + "_" + str(gtfileName)+".jpg")
            # cv2.imwrite(fir_dir, gt_img*50)
            return
            
        def plot_lane_img(self, img, lane_list):
            # print(type(lane_list))
            # print(lane_list)
            raw_img = img
            for idx, lane in enumerate(lane_list):
                if len(lane) <=2:
                    continue
                for idx2, height in enumerate(range(160, 710+1, 10)):
                    if lane[idx2] > 0:
                        raw_img = cv2.circle(raw_img, (int(lane[idx2]),int(height)), 5, myColor.color_list[idx if idx <=10 else 10], -1)
            return raw_img
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

            print("SHAPE {}".format(output_image.shape))
            cv2.imwrite(raw_fir_dir, image)
            cv2.imwrite(right_fir_dir, output_image[:,:,0])
            cv2.imwrite(right_10_fir_dir, output_image[:,:,1])
         
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

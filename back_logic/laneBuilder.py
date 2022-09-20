import torch
import numpy as np
import time
import data.sampleColor as myColor
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.signal import *
from sklearn.neighbors import KernelDensity
import cv2
import math
class Lane:
    def __init__(self):
        self.candi_num= 50
        self.height_num = 80
        self.lane_list = np.zeros((self.candi_num, self.height_num, 2), dtype=np.int)
        self.lane_idx = [0 for i in range(self.candi_num)]
        # self.lane_predicted = [0 for i in range(self.candi_num)]
        self.lanes_num=0
        return
    def delete_last(self):
        # print(self.lanes_num)
        # print(self.lane_idx)
        # print("PRE")
        for lane_idx in range(self.lanes_num):
            self.lane_idx[lane_idx] -=1
            self.lane_list[lane_idx, self.lane_idx[lane_idx]] = [0,0]
            
        
    def addCount(self, idx):
        if idx >= self.candi_num:
            print("IDX is Too BIG!!!")
            return
        self.lane_idx[idx] +=1
        return
    def addKey(self, up_lane, idx):
        if up_lane[0]!=0:
            # lane_list = [height(idx), lane_idx, 2], up_lane = [height, width]
            if idx > 25 or self.lane_idx[idx] > 70:
                print("Lane idx unormal {} {}!!!!!!!!!!".format(idx, self.lane_idx[idx]))
            self.lane_list[idx, self.lane_idx[idx]] = up_lane 
            self.addCount(idx)
        else:
            # Interpolate!!
            return
        return
    def resize_lane(self):
        for lane_idx in range(self.lanes_num):
            for height_idx in range(self.lane_idx[lane_idx]):
                self.lane_list[lane_idx, height_idx,0] = int(self.lane_list[lane_idx, height_idx,0] * 720/368) 
                self.lane_list[lane_idx, height_idx,1] = int(self.lane_list[lane_idx, height_idx,1] * 1280/640) 
        return
    # lane_tensor = numOfLane*height*(x,y)   3dim tensor
    def tensor2lane(self):
        # print(self.lane_list[:self.lanes_num])
        # print("===============")
        # print("===============")
        # print("===============")
        re_lane = []
        for lane_idx in range(self.lanes_num):
            lane=[]
            for height_idx in range(self.lane_list.shape[1]):
                if self.lane_list[lane_idx, height_idx, 0] <1 or self.lane_list[lane_idx, height_idx, 1] <1:
                    continue
                point = [self.lane_list[lane_idx, height_idx,0], self.lane_list[lane_idx, height_idx,1]]
                lane.append(point)
            re_lane.append(lane)
        # print(re_lane)
        return re_lane
    def printLane(self):
        print("----------- LANE ----------")
        for idx, lane in enumerate(self.lane_list):
            if idx>5:
                break
            print(lane[0:self.lane_idx[idx] +2])
            # for item in range():
        print(self.lane_idx)
        return
    def convert_tuSimple(self):
        self.resize_lane()
        return_lane=[]
        for lane_idx in range(self.lanes_num):
            add_lane=[]
            height_start = self.lane_list[lane_idx,0,0]
            # print("Height Start = {}".format(height_start))
            start_idx = int((360- height_start)/10)
            # print("start_idx = {}".format(start_idx))
            for i in range(start_idx):
                add_lane.append(-2)
            for height_idx in range(self.lane_idx[lane_idx]):
                add_lane.append(self.lane_list[lane_idx, height_idx,1])
            while len(add_lane) !=56:
                add_lane.append(-2)
            add_lane.reverse()
            return_lane.append(add_lane)

        return return_lane
class LaneBuilder:

    def __init__(self, args):
        self.device = torch.device('cpu')
        self.cfg = args
        self.kde_time=0
        self.nor_time=0
        return

    # Key = [height, width], delta_image = 348*640 delta_up_image
    # Return = New key of (height-10)
    def getUpLane(self, key, delta_image, height_delta=10, window_width = 40):
        height = key[0]
        width = key[1]
        min_abs=100
        new_10_point = np.array([key[0]-10,key[1]], dtype=np.int)
        point_sum = 0
        point_count = 0
        for point in range(width-window_width, width+window_width+1, 2):
            # print("     Point {}".format(point))
            if  0<point and point < delta_image.shape[1] and height_delta*0.7 < delta_image[height,point] and delta_image[height,point] < height_delta*1.3:
                # print("HERE")
                # print("!! {} {}".format(point, delta_right_image.shape[1]))
                direction = -1 if delta_image[height-2,point] > delta_image[height+2,point] else 1
                if direction == -1: # Go Down
                    continue
                point_sum +=point
                point_count +=1
                resi = abs(width-point)
                if resi < min_abs:
                    new_10_point = [height-height_delta*direction, point]
                    min_abs = resi
        # print("PRE {}".format(new_10_point[1]))
        if point_count!=0:
            new_10_point[1] = int(point_sum/point_count)
        # print("POST {}".format(new_10_point[1]))
        return new_10_point

    def getKeyfromHeat(self, heat_image,delta_right_image, min_height, threshold=-3):
        threshold = 0
        delta_threshold_min = 3
        delta_threshold = 20

        key_list=[]
        height_delta=5
        for i in range(min_height, heat_image.shape[0], height_delta):
            width_list=[]
            for j in range(10, heat_image.shape[1], 3):
                if heat_image[i,j] > threshold:
                    width_list.append(j)


                # get Key From delta
                # if j+11 > delta_right_image.shape[1] or j-11 < 0:
                #     continue
                # if  delta_threshold_min < delta_right_image[i,j] and delta_right_image[i,j] < delta_threshold:
                #     direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                #     width_list.append(int(delta_right_image[i,j])*direction + j)

            if len(width_list)==0:
                continue
            # print("Height {} , Pre Key LIST {}".format(i, width_list))
            point_list = self.widthCluster(width_list, i, 30, use_mean_width=False)
            # print("Height {} , PPPPP Key LIST {}".format(i, point_list))
            # print("Key LIST {}".format(point_list))
            # lane_in_height[count] +=1
            if point_list is not None:
                key_list.append(point_list)
        return key_list

    def getKeyfromHeat_adaptiveThreshold(self, heat_img,delta_right_image, min_height, max_height, threshold, width=40):

        key_list=[]
        height_delta=5
        for height in range(min_height, max_height, height_delta):
            newnew = (heat_img[height]-heat_img[height].min())/(heat_img[height].max()-heat_img[height].min())
            if height<190:
                threshold = 0.93
                width = 20
            elif height <250:
                threshold = 0.80
                width = 60
            else:
                threshold = 0.80
                width = 80
            abc = torch.where(newnew>threshold)
            # 
            a =np.array(abc[0]).reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
            s = np.linspace(0,heat_img.shape[1], heat_img.shape[1])
            e = kde.score_samples(s.reshape(-1,1))
            mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
            maa =  find_peaks(heat_img[height].detach().numpy())
            # maa = argrelextrema(nnn, np.greater)[0]
            # print("     MA = {}".format(ma.tolist()))
            heat_width_list = ma
            if heat_width_list is None or len(heat_width_list)==0:
                continue
            horizon_heat_key = self.widthCluster(ma, height, width, use_mean_width=False)
            if horizon_heat_key is not None:
                key_list.append(horizon_heat_key)
        return key_list

    def addDelta2Heat(self, delta_list, heat_list):
        new_list=[]

        for delta_item in delta_list:
            addFlag=True
            for heat_item in heat_list:
                dist = abs(heat_item - delta_item)
                if dist < 50:
                    addFlag=False
            if addFlag:
                new_list.append(delta_item)
        return new_list+heat_list
    # sort lanes left to right
    def sort_lane(self, lane_data):
        # sort_idx [idx, lastAbscissa] ->sorting by Abscissa
        # Lane list = [ [height, abscissa] * point ] *n_of_lane
        sort_idx_list = []
        lane_list = []
        lane_list_temp = np.zeros((lane_data.candi_num, lane_data.height_num, 2), dtype=np.int)
        lane_idx_temp = [0 for i in range(lane_data.candi_num)]

        for idx, lane in enumerate(lane_data.lane_list):
            # print(lane[lane_data.lane_idx[idx]-1])
            # 
            sort_idx_list.append([idx, lane[lane_data.lane_idx[idx]-1][1]])
            if idx>=lane_data.lanes_num-1:
                break
        # print(sort_idx_list)
        sort_idx_list.sort(key=lambda x : x[1])
        # print(sort_idx_list)
        # print(lane_data.lane_idx)
        for idx, sort_idx in enumerate(sort_idx_list):
            lane_list.append(lane_data.lane_list[sort_idx[0]])
            # print(lane_data.lane_list[sort_idx[0]])
            # lane_data.lane_list
            lane_list_temp[idx] = lane_data.lane_list[sort_idx[0]]
            lane_idx_temp[idx] = lane_data.lane_idx[sort_idx[0]]
        lane_data.lane_list = lane_list_temp
        lane_data.lane_idx = lane_idx_temp
        # print(lane_data.lane_idx)
        return lane_data
    def predict_horizon_v2(self, heat_img, lane_data, bottom_height, top_height, Trad = False):
        lane_data = self.sort_lane(lane_data)
        # print(lane_data.lane_idx)

        threshold = -1
        # heat_img = heat_tensor.cpu().detach().numpy()
        
        horizon_lane_tensor=np.zeros((lane_data.candi_num, 40, 2), dtype = np.int)

        lane_data.lane_list = np.concatenate((lane_data.lane_list, horizon_lane_tensor), axis = 1)
        # print("Lane Count = {}".format(lane_data.lanes_num))
        # top_height = 150
        doubled_lane = []
        temp_key_return = []
        height_delta=5
        height = bottom_height

        print("TOP = {}".format(top_height))
        print("     SHPE = {}".format(heat_img.shape))
        print("     SHPE = {} {}".format(bottom_height, top_height))
        for height in range(bottom_height, top_height,5):
            sum_th = 0
            count_th=0
            width_list=[]
            pradicted_lane_key=[]
            notFound_lane_idx=[]
            horizon_heat_key = None
            if not Trad:
                nor_start_time = time.time()
                sm = torch.nn.Softmax(dim=0)

                heat_img[height] = (heat_img[height]-heat_img[height].min())/(heat_img[height].max()-heat_img[height].min()) # Minmax scaling
                # newnew = sm(heat_img[height]) # Softmax normalize

                nor_end_time = time.time()
                self.nor_time += nor_end_time-nor_start_time

                # KDE Start
                kde_start_time = time.time()
                abc = torch.where(heat_img[height]>0.00615)
                # abc = torch.where(newnew>0.007)
                a =np.array(abc[0]).reshape(-1, 1)
                # print(f"---------A {a}")
                # print(f"---------A {a.shape}")

                if len(a)==0:
                    continue
                kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
                s = np.linspace(0,heat_img.shape[1], heat_img.shape[1])
                e = kde.score_samples(s.reshape(-1,1))
                mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
                # maa =  find_peaks(heat_img[height].detach().numpy())
                kde_end_time = time.time()
                self.kde_time += kde_end_time-kde_start_time
                # print(f"---------MAA {maa}")
                # print(f"---------MA {ma}")
                # maa = argrelextrema(nnn, np.greater)[0]
                # print("     MA = {}".format(ma.tolist()))
                heat_width_list = ma
                if heat_width_list is None or len(heat_width_list)==0:
                    continue
                horizon_heat_key = self.widthCluster(ma, height, 30,  use_mean_width=True)
                # horizon_heat_key = self.widthCluster(ma, height, 50,  use_mean_width=False)
                # print("RAW {}".format(horizon_heat_key))

                # ma = ma.reshape(-1,1)
                # ma = np.pad(ma, ((0,0),(1,0)), 'constant', constant_values=height)

                # print("     ORI = {}".format(horizon_heat_key))
                # print("     ORI = {}".format(type(horizon_heat_key)))
                # return
                # horizon_heat_key = ma
                sum_th+=threshold
                count_th+=1
                # print("Height = {}, th = {}".format(height, threshold))
                # print(horizon_heat_key)
                if horizon_heat_key is not None:
                    temp_key_return.append(horizon_heat_key)
                    # temp_key_return.append(horizon_heat_key)
            # Trad
            if Trad:
                ## 1. Get Key
                heat_width_list=[]
                for j in range(10, heat_img.shape[1], 3):
                    if  heat_img[height,j] > 0: #prev
                    # if  heat_img[height,j] < heat_threshold:
                        heat_width_list.append(j)
                if len(heat_width_list)==0:
                    continue
                # print(heat_width_list)
                # horizon_heat_key = self.widthCluster(heat_width_list, height,5, use_mean_width=False)
                horizon_heat_key = self.widthCluster(heat_width_list, height, 50)

                # print(horizon_heat_key)
                if horizon_heat_key is not None:
                    temp_key_return.append(horizon_heat_key)
  


            if horizon_heat_key is None or len(horizon_heat_key)==0:
                continue
            # print(horizon_heat_key)
            ## 2. Predicte Key
            key_min = [ 1000 for i in range(len(horizon_heat_key))]
            for idx, lane in enumerate(lane_data.lane_list):
                if idx in doubled_lane:
                    continue
                if lane_data.lane_idx[idx] <1:
                    continue
                last_idx_of_lane = lane_data.lane_idx[idx]-1

                if abs(lane[last_idx_of_lane,0] - height) > 30:
                    continue
                
                # GET dx
                if lane_data.lane_idx[idx] >=2:
                    dx = lane[lane_data.lane_idx[idx] - 1 ,1] - lane[lane_data.lane_idx[idx] - 2 ,1]
                    dy = lane[lane_data.lane_idx[idx] - 1 ,0] - lane[lane_data.lane_idx[idx] - 2 ,0]
                    delta_height = lane[lane_data.lane_idx[idx] - 2 ,0] - height
                    delta_width = dx*(delta_height/(dy+0.01))
                # else lane_data.lane_idx[idx] >2:
                else:
                    dx=0
                    dy=5
                    # print("ELSE")
                    # delta_width = 
                # print("DX = {}".format(dx))
                delta_width = lane[last_idx_of_lane,1] + dx
                min=1000
                predicted_key = [height, delta_width] 
                # predicted_key = None
                for heat_key_idx, heat_key in enumerate(horizon_heat_key):
                    # print("Key {}".format(heat_key))
                    dist = abs(heat_key[1]-delta_width)
                    # print("     DIST {}".format(dist))

                    deg = 180
                    if lane_data.lane_idx[idx] >2:
                        point1 = lane[last_idx_of_lane-2]
                        point2 = lane[last_idx_of_lane-1]
                        point3 = heat_key
                        # print("LANE {}".format(lane))
                        # print("LANE idx {}".format(lane_data.lane_idx[idx]))
                        deg = self.getLaneAngle(point1, point2, point3)
                    deg -=180
                    # if deg < 145 or deg > 250:
                    if abs(deg) > 70:
                        continue


                    if min>dist and dist < 50 and dist < key_min[heat_key_idx]:
                        min=dist
                        key_min[heat_key_idx]=dist
                        # predicted_key = [height, int(heat_key[1]*0.8+delta_width*0.2)]
                        predicted_key = [height-dy, int(heat_key[1])]
                        # print("     Changed !! {} {}".format(predicted_key, idx))

                
                if predicted_key is not None:
                    pradicted_lane_key.append([predicted_key, idx])


            ## 4. Add key
            for idx, lane in enumerate(pradicted_lane_key):
                lane_data.addKey(lane[0], lane[1])
            
            ## 5. Delete doubled Lane

            for idx in range(lane_data.lanes_num-1):
                if idx in doubled_lane:
                    continue
                next_idx = idx+1
                while next_idx in doubled_lane:
                    next_idx +=1
                if next_idx >= lane_data.lanes_num:
                    continue
                if abs(lane_data.lane_list[next_idx,lane_data.lane_idx[next_idx]-1, 0] - height) > 30:
                    continue
                lane_abscissa = lane_data.lane_list[idx, lane_data.lane_idx[idx]-1, 1]
                next_abscissa = lane_data.lane_list[next_idx, lane_data.lane_idx[next_idx]-1, 1]
                # print(lane_data.lane_list[idx][lane_data.lane_idx[idx]][1])
                if next_abscissa - 10 < lane_abscissa:
                    doubled_lane.append(idx)
                    # print("Doubled!!")
                    # print("IDX {}/ {}".format(idx, next_idx))
                    # print("Abscissa {} / {}".format(lane_abscissa, next_abscissa))
            if len(doubled_lane)+1 == lane_data.lanes_num:
                break
        # if not Trad:
        #     print("     mean TH = {}".format(sum_th/count_th))
        # print(f"        KDE Time = {self.kde_time}, Nor Time = {self.nor_time}")
        return lane_data, temp_key_return
    def getMaxHeight(self, heat_img):
        max_height = 80
        for idx, height in enumerate(heat_img):
            # print(height.shape)
            # print(type(height))
            if np.max(height)>0:
                # print("MAX HEIGHT = {}".format(idx))
                return idx
        return max_height
    
    # key_list = [ [height,width] * n_of_key ] * n_of_height
    # heat_img = [ background_img, lane_img ]
    def getLanefromHeat(self, heat_img, delta_img, temp_raw_image=None, idx=0):

        draw_mode = True
        # draw_mode = False
        th=-3
        heat_lane = heat_img[0].cpu()
        heat_lane2 = heat_img[1].cpu()

        compat_heat = torch.where(heat_img[0] < heat_img[1], torch.tensor(10).to(self.device), torch.tensor(0).to(self.device)).cpu().detach().numpy()
        max_height = self.getMaxHeight(compat_heat)


        key_list=self.getKeyfromHeat(compat_heat, delta_img[:,:,0], 200, 0)  #Trad
        # key_list=self.getKeyfromHeat_adaptiveThreshold(heat_lane2, delta_img[:,:,0], max_height, heat_lane2.shape[0]-20, 0.8, 40) # Adaptive_threshold
        # key_list2=self.getKeyfromHeat_adaptiveThreshold(heat_lane2, delta_img[:,:,0], 160, 200, 0.93, 20) # Adaptive_threshold

        # print(key_list)
        # key_list = key_list+key_list2
        lane_data = Lane()
        lane_data = self.buildLane(key_list,  delta_img[:,:,1])
        lane_data2 = Lane()
        lane_data2 = self.buildLane(key_list,  delta_img[:,:,1])

        # max_height = self.getMaxHeight(heat_lane2.detach().numpy())
        heat_np = heat_lane2.detach().numpy()
        heat_com_np = heat_lane2.detach().numpy()
        
        sm = torch.nn.Softmax(dim=0)
                # 
        # for i, item in enumerate(heat_np):
        # #     heat_np[i] = sm(heat_np[i]) # Softmax normalize
        # for height in range(heat_np.shape[0]):
        #     heat_np[height] = (heat_np[height]-heat_np[height].min())/(heat_np[height].max()-heat_np[height].min()) # Minmax scaling
        # time.sleep(1.0)
        def softmax(x):
            y = np.exp(x - np.max(x))
            f_x = y / np.sum(np.exp(x))
            return f_x
        
        # for height in range(heat_np.shape[0], max_height):
        print("SSSSSSSSSHAPE {}".format(heat_np.shape))
        for height in range(max_height, heat_np.shape[0]):
            heat_np[height] = softmax(heat_np[height])
        cv2.imwrite("data/heat.png",(heat_np)*100000000)
        cv2.imwrite("data/heat_com.png",compat_heat*100)
        
        
        # lane_data2, up_key_list = self.predict_horizon_v2(compat_heat, lane_data, 5, max_height, Trad = True) #STD 80
        # lane_data2, up_key_list2 = self.predict_horizon_v2(compat_heat, lane_data2, 5, max_height, Trad = True) #STD 80
        
        # lane_data2, up_key_list = self.predict_horizon_v2(heat_lane2, lane_data, max_height, 200, Trad = False) #STD 80
        lane_data2, up_key_list = self.predict_horizon_v2(compat_heat, lane_data, max_height, 200, Trad = True) #STD 80
        # print(f"Lane data = ")
        # print(up_key_list)
        # for height in up_key_list:
        #     print(height)
        new_list = lane_data2.tensor2lane()
        new_list.sort(key=len, reverse=True)
        re_list=[]
        for item in new_list:
            if len(item) > 5:
                re_list.append(item)
        print(f"Lane data = {re_list}")

        if self.cfg.dataset == "tuSimple":
            re_list = self.getLanebyH_sample_deg(re_list, 160, 710, 10)

        if self.cfg.dataset == "cuLane":
            for lane in re_list:
                for node in lane:
                    node[0] = int(node[0]/300*590)
                    node[1] = int(node[1]/800*1640)
        # elif self.cfg.dataset == "cuLane":
        #     continue

        # cv2.imwrite("SIBAL22.png",heat_img[0].cpu().detach().numpy()*10)
        # cv2.imwrite("SIBAL33.png",heat_img[1].cpu().detach().numpy()*10)
        # self.temp_Key_drawer(temp_raw_image.copy(), key_list)

        # draw_mode = False 
        draw_mode = True 
        if draw_mode:
            if self.cfg.dataset=="tuSimple":
                output_size = (640, 368)
            elif self.cfg.dataset=="cuLane":
                output_size = (800, 300)
            resized_raw_img = cv2.resize(temp_raw_image, output_size)
            for height in key_list:
                # print(key)
                for key in height:
                    resized_raw_img = cv2.circle(resized_raw_img, (key[1], key[0]), 2, (0,0,255), -1)
                                # cv2.circle(output_right_circle_image, startPoint, 1, (255,0,0), -1)
            cv2.imwrite("data/keys2.png",resized_raw_img)

            resized_raw_img = cv2.resize(temp_raw_image, output_size)
            for height in up_key_list:
                # print(key)
                for key in height:
                    resized_raw_img = cv2.circle(resized_raw_img, (key[1], key[0]), 2, (0,0,255), -1)
                                # cv2.circle(output_right_circle_image, startPoint, 1, (255,0,0), -1)
            cv2.imwrite("data/keys_up2.png",resized_raw_img)
            resized_raw_img = cv2.resize(temp_raw_image, output_size)
            all_key = up_key_list+key_list
            for height in all_key:
                # print(key)
                for key in height:
                    resized_raw_img = cv2.circle(resized_raw_img, (key[1], key[0]), 2, (0,255,255), -1)
                                # cv2.circle(output_right_circle_image, startPoint, 1, (255,0,0), -1)
            cv2.imwrite("data/keys_all.png",resized_raw_img)
            
            # all_key2 = up_key_list2+key_list
            # for height in all_key2:
            #     # print(key)
            #     for key in height:
            #         resized_raw_img = cv2.circle(resized_raw_img, (key[1], key[0]), 2, (0,0,255), -1)
            #                     # cv2.circle(output_right_circle_image, startPoint, 1, (255,0,0), -1)
            # cv2.imwrite("data/keys_all2.png",resized_raw_img)
            self.temp_lane_drawer_laneObject(temp_raw_image.copy(), lane_data, "data/temptemp2")

        return re_list


    def getLaneAngle(self, point1, point2, point3):
        at1 = math.atan2((point1[0]-point2[0]), (point1[1]-point2[1]))
        at2 = math.atan2((point3[0]-point2[0]), (point3[1]-point2[1]))
        rad = at1-at2
        deg = rad*180/math.pi
        # print(point1)
        # print(point2)
        # print(point3)
        # print(deg)
        return deg
    def buildLane(self, key_list,  delta_up_image):
        key_list_copy = key_list.copy()
        key_list_copy.reverse()
        # key_up_list.reverse()
        lane_data = Lane()
        if len(key_list_copy)==0:
            return lane_data
        # First Update
        for idx, first_key in enumerate(key_list_copy[0]):
            # print(first_key)
            lane_data.lane_list[idx, 0] =  np.array(first_key)
            # lane_data.addCount(idx)
            # up_lane = self.getUpLane(first_key, delta_up_image)
            lane_data.addKey(first_key, idx)
            lane_data.lanes_num +=1


        # keys = key of One height, key_list_copy = key of All height
        temp = 0
        for keys in key_list_copy[1:]:
            # print("KEYS {}".format(keys))
            # lane_data.printLane()

            for key in keys:
                # print("     KEY {}".format(key))
                min_dist = 100
                added_idx = np.array([0,0])
                for idx, lane in enumerate(lane_data.lane_list):
                    if lane_data.lane_idx[idx] == 0:
                        continue

                    if lane_data.lane_idx[idx] >=4:
                        dx = lane[lane_data.lane_idx[idx] - 1 ,1] - lane[lane_data.lane_idx[idx] - 2 ,1] 
                        dx += lane[lane_data.lane_idx[idx] - 2 ,1] - lane[lane_data.lane_idx[idx] - 3 ,1] 
                        dx += lane[lane_data.lane_idx[idx] - 3 ,1] - lane[lane_data.lane_idx[idx] - 4 ,1] 
                        dx /=3
                        dy = lane[lane_data.lane_idx[idx] - 2 ,0] - lane[lane_data.lane_idx[idx] - 1 ,0]
                        delta_height = abs(lane[lane_data.lane_idx[idx] - 1 ,0] - key[0])
                        delta_width = dx*(delta_height/(dy+0.01))
                    else:
                        delta_width = 0
                    # delta_width = lane[lane_data.lane_idx[idx]-1,1] + dx
                    
                    dist = abs(lane[lane_data.lane_idx[idx]-1, 1]+delta_width - key[1])
                    height_dist = abs(lane[lane_data.lane_idx[idx]-1, 0] - key[0])

                    # print("         LANE {}, Key =  {}".format(lane[lane_data.lane_idx[idx]-1, 1], key))
                    # print("         Dist = {}".format(dist))
                    deg = 180
                    if lane_data.lane_idx[idx] >2:
                        point1 = lane[lane_data.lane_idx[idx]-2]
                        point2 = lane[lane_data.lane_idx[idx]-1]
                        point3 = key
                        # print("LANE {}".format(lane))
                        # print("LANE idx {}".format(lane_data.lane_idx[idx]))
                        deg = self.getLaneAngle(point1, point2, point3)
                    deg -=180
                    # if deg < 145 or deg > 250:
                    if abs(deg) > 65 and lane_data.lane_idx[idx]>3 and abs(lane[lane_data.lane_idx[idx]-1, 1] - key[1]) > 50:
                        continue
                        
                    if dist < min_dist and height_dist < 40:
                        min_dist = dist
                        added_idx = idx
                        # print("Mindist - {}".format(min_dist))  
                
                if min_dist < 50:
                    # print(" Added Idx - {}".format(added_idx))  
                    # print(lane_data.lane_idx)      
                    lane_data.lane_list[ added_idx, lane_data.lane_idx[added_idx]] = key
                    lane_data.addKey(key, added_idx)
                else:
                    # print("KEY {}".format(key))
                    lane_data.addKey(key, lane_data.lanes_num)
                    lane_data.lanes_num+=1
        
        lane_data.delete_last()
        return lane_data

    def getLanebyH_sample_deg(self, lanes,  h_start, h_end, h_step):
        lane_list = []
        for lane in lanes:
            for node in lane:
                node[0] = int(node[0]/368*720)
                node[1] = int(node[1]/640*1280)
            lane.reverse()
        
        for lane in lanes:
            # print(lane)
            if len(lane)==0:
                continue
            new_single_lane=[]
            cur_height_idx = 0
            lane_ended=False
            while True:
                
                y = lane[cur_height_idx][0]
                x = lane[cur_height_idx][1]
                if x<0 and y < 0:
                    cur_height_idx+=1
                    continue
                else:
                    break
            # print("LANE DATA ---------------")
            # print(lane)
            
            # end+1 = for height_sample number
            for height in range(h_start, h_end+1, h_step):
                # print(height) # 

                if lane_ended or cur_height_idx==len(lane):
                    # print("1 X = {}".format(-2))
                    # print("Lane Ended {}".format(lane_ended))
                    # print("Lane Idx   {}".format(cur_height_idx))
                    new_single_lane.append(-2)
                    continue
                # print("Cur Height = {}".format(cur_height))
                cur_height = lane[cur_height_idx][0]
                # print("Cur Idx = {}".format(cur_height_idx))
                # print("Cur Height = {}/{}".format(cur_height, height))

                if height < cur_height and cur_height_idx == 0:
                    new_single_lane.append(-2)
                    continue
                if cur_height_idx == len(lane)-1 and height > cur_height:
                    new_single_lane.append(-2)
                    continue
                
                if cur_height < height:
                    while cur_height < height:
                        cur_height_idx +=1
                        cur_height = lane[cur_height_idx][0]
                        if cur_height_idx == len(lane)-1:
                            break

                    # print("INDEX = {}".format(cur_height_idx))
                dx = lane[cur_height_idx][1] -lane[cur_height_idx-1][1] 
                dy = lane[cur_height_idx][0] -lane[cur_height_idx-1][0] 
                # while False:
                while True:
                    # print("IDX {}".format(cur_height_idx))
                    dx = lane[cur_height_idx][1] -lane[cur_height_idx-1][1] 
                    dy = lane[cur_height_idx][0] -lane[cur_height_idx-1][0] 
                    # print(dy)
                    if dy!=0:
                        break
                    cur_height_idx +=1
                    if cur_height_idx==len(lane):
                        lane_ended=True
                        break
                    # print(cur_height_idx)
                if dy<0 or lane_ended:
                    new_single_lane.append(-2)
                    continue
                


                # print(new_single_lane)
                subY = height - lane[cur_height_idx-1][0] 
                subX = dx*subY/dy
    
                newX = int(subX + lane[cur_height_idx-1][1])
                # print("Added  = {}.{}".format(height, newX))
                new_single_lane.append(newX)        
                # print(new_single_lane)                
            if np.mean(new_single_lane) > 0:
                lane_list.append(new_single_lane)
        self.lane_list=lane_list
        return lane_list

    ## Point Clustering (width)
    def widthCluster(self, width_list, height, distBTlane=20,  use_mean_width = False):
        point_list=[]
        width_buffer=[]
        count=1
            
        buf_count=1
        buf=last=width_list[0]
        # key_buffer.append(width_list[0])
        # print("Height {}".format(height))
        min_value = max_value = buf
        for idx in width_list[1:]:
            # print("IDX {}".format(idx))
            if idx > last+distBTlane:

                count +=1
                point_list.append([height, int(buf/buf_count)])
                if last + distBTlane*2 > idx:
                    width_buffer.append(max_value-min_value)
                # print("     COUNT {}".format(buf_count))
                # print("         min_value {} LAST {}  MAX_value {}, DIst {}".format(min_value, last, max_value, max_value-min_value))

                max_value = min_value = idx
                buf_count=1
                buf=idx
            else:
                buf_count+=1
                buf+=idx
                if idx<min_value:
                    min_value=idx
                if idx>max_value:
                    max_value=idx
            last = idx

        if buf_count != 0:
            # print("     22COUNT {}".format(buf_count))
            # print("         min_value {} MAX_value {}, DIst {}".format(min_value, max_value, max_value-min_value))
            width_buffer.append(max_value-min_value)
            point_list.append([height, int(buf/buf_count)])

        # dist_mean = mean of each lane width
        width_mean = sum(width_buffer)/len(width_buffer)
        # print("DIST = {}".format(width_mean))
        # 30 40 97.63
        # 40 40 97.60
        if (width_mean>20 or ( max(width_buffer) > 20 )) and use_mean_width:
            return None
        return point_list

    def widthCluster2(self, width_list, height, distBTlane=50):
        point_list=[]
        key_buffer=[]
        count=1
            
        buf_count=1
        buf=last=width_list[0]
        min = max = buf
        for idx in width_list[1:]:
            if idx > last+distBTlane:
                count +=1
                point_list.append([height, int(buf/buf_count)])
                max = min = idx
                buf_count=1
                buf=idx
            else:
                buf_count+=1
                buf+=idx
                if idx<min:
                    min=idx
                if idx>max:
                    max=idx
            last = idx
        if buf_count != 0:
            # print("MIN {} MAX {}, DIst {}".format(min, max, max-min))
            point_list.append([height, int(buf/buf_count)])
        return point_list
    def temp_Key_drawer(self, img, key_list):

        # print(key_list)
        for key_height in key_list:
            # print(key)
            for key in key_height:
                # cv2.circle
                img = cv2.circle(img, (key[1], key[0]), 5, (0,0,255), -1)
        cv2.imwrite("temptemp.jpg", img)
        return
    def temp_lane_drawer_laneObject(self, img, lane_data, fileName):
        # print("=======================")
        # print(lane_data.lanes_num)
        for idx, lane in enumerate(lane_data.lane_list):
            # print(lane)
            if idx>=lane_data.lanes_num:
                break
            for point in lane:
                if idx>10:
                    idx = 10
                # print("IDX {}".format(idx))
                img = cv2.circle(img, (point[1], point[0]), 3, myColor.color_list[idx], -1)
        
        # img = cv2.circle(img, (478, 120), 10, (255,255,255), -1)
        cv2.imwrite(fileName+".jpg", img)

    def temp_lane_drawer_list(self, img, lane_list, fileName):
        # print("=======================")
        # print(lane_data.lanes_num)
        for idx, lane in enumerate(lane_list):
            # print(lane)
            for point in lane:
                if idx>10:
                    idx = 10
                # print("point {} {}".format(height, point))
                img = cv2.circle(img, (point[1], point[0]), 3, myColor.color_list[idx], -1)
        # img = cv2.circle(img, (478, 120), 10, (255,255,255), -1)
        cv2.imwrite(fileName+".jpg", img)

        # print(lane_list.shape)
        # print(type(lane_list))
if __name__=="__main__":
    pl = LaneBuilder()

import torch
import numpy as np
import time
import data.sampleColor as myColor
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

    def __init__(self):
        self.device = torch.device('cpu')
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
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if  delta_threshold_min < delta_right_image[i,j] and delta_right_image[i,j] < delta_threshold:
                    direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                    width_list.append(int(delta_right_image[i,j])*direction + j)
            if len(width_list)==0:
                continue
            # print("Height {} , Pre Key LIST {}".format(i, width_list))
            point_list = self.widthCluster(width_list, i, 50)
            # print("Height {} , PPPPP Key LIST {}".format(i, point_list))
            # print("Key LIST {}".format(point_list))
            # lane_in_height[count] +=1
            if point_list is not None:
                key_list.append(point_list)
        return key_list

    def getKeyfromHeat22(self, heat_image,delta_right_image, min_height, threshold=-3):
        threshold = -5
        delta_threshold_min = 3
        delta_threshold = 20

        key_list=[]
        height_delta=5
        for i in range(min_height, heat_image.shape[0], height_delta):
            width_list=[]
            for j in range(10, heat_image.shape[1], 3):
                # get Key From delta
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if  delta_threshold_min < delta_right_image[i,j] and delta_right_image[i,j] < delta_threshold:
                    direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                    width_list.append(int(delta_right_image[i,j])*direction + j)


            # if len(width_list)==0:
            #     continue
            point_list = None
            # print("[get Key] HEIGHT {}, TH = {}". format(i, threshold))
            threshold-=2
            while point_list is None and threshold < 10:
                heat_width_list=[]
                threshold+=2
                for j in range(10, heat_image.shape[1], 3):
                    if heat_image[i,j] > threshold:
                        heat_width_list.append(j)
                if len(heat_width_list)==0:
                    break
                point_list = self.widthCluster(heat_width_list, i, 30)
                # if point_list is None:
                #     print("TH UP !! {}".format(threshold))
            if len(width_list+heat_width_list)==0:
                continue
            # point_list = self.widthCluster2(sorted(width_list+heat_width_list), i, 30)
            point_list = self.widthCluster2(sorted(self.addDelta2Heat(width_list, heat_width_list)), i, 30)

            if point_list is not None:
                key_list.append(point_list)
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
    def predict_horizon_v2(self, heat_img, lane_data, bottom_height, top_height):
        lane_data = self.sort_lane(lane_data)
        # print(lane_data.lane_idx)

        threshold = -1
        # heat_img = heat_tensor.cpu().detach().numpy()
        
        horizon_lane_tensor=np.zeros((lane_data.candi_num, 40, 2), dtype = np.int)

        lane_data.lane_list = np.concatenate((lane_data.lane_list, horizon_lane_tensor), axis = 1)
        # print("Lane Count = {}".format(lane_data.lanes_num))
        # top_height = 150
        doubled_lane = []
        for height in range(bottom_height, top_height, -5):
            width_list=[]
            # horizon_heat_key=[]
            pradicted_lane_key=[]
            notFound_lane_idx=[]


            horizon_heat_key = None
            # print("[get Key] HEIGHT {}, TH = {}". format(i, threshold))
            threshold-=1
            while horizon_heat_key is None and threshold < 10:
                heat_width_list=[]
                threshold+=1
                ## 1. Get Key
                for j in range(10, heat_img.shape[1], 3):
                    # if  heat_img[height,j] > -2: #prev
                    if  heat_img[height,j] > threshold:
                        heat_width_list.append(j)

                if len(heat_width_list)==0:
                    break
                horizon_heat_key = self.widthCluster(heat_width_list, height, 10, True)
                # if horizon_heat_key is None:
                    # print("TH UP !! {}".format(threshold))


            # print("PRE  {} / {}".format(height, heat_width_list))
            # horizon_heat_key = self.widthCluster(heat_width_list, height,10)
            # print("POST {} / {}".format(height, horizon_heat_key))
            if horizon_heat_key is None or len(horizon_heat_key)==0:
                continue
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
                # !!!!!!!!!!!!!!!!!!
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
                    if abs(deg) > 50:
                        continue


                    if min>dist and dist < 20 and dist < key_min[heat_key_idx]:
                        min=dist
                        key_min[heat_key_idx]=dist
                        # predicted_key = [height, int(heat_key[1]*0.8+delta_width*0.2)]
                        predicted_key = [height, int(heat_key[1])]
                        # print("     Changed !! {} {}".format(predicted_key, idx))

                
                if predicted_key is not None:
                    # lane_data.addKey(predicted_key, idx)
                    # print("ADD!! {} {}".format(predicted_key, idx))
                    pradicted_lane_key.append([predicted_key, idx])
                # else:
                #     notFound_lane_idx.append(idx)
                #     pradicted_lane_key.append([predicted_key, idx])
                # print("==========")

            ## 3. Post process (lane that has no key)
            # for idx in notFound_lane_idx:
            #     lane = lane_data.lane_list[idx]
            #     if lane_data.lane_idx[idx]>1:
            #         predicted_key = [height, ]
            #     pradicted_lane_key.append([predicted_key, idx])
                
            ## 4. Add key
            for idx, lane in enumerate(pradicted_lane_key):
                lane_data.addKey(lane[0], lane[1])
            
            ## 5. Delete doubled Lane

            for idx in range(lane_data.lanes_num-1):
                if idx in doubled_lane:
                    continue
                # print("IAM     {}".format(idx))
                # print("DOUBLED {}".format(len(doubled_lane)))
                # print("lanes num {}".format(lane_data.lanes_num))
                # print("IDX")
                # print(lane_data.lane_list[idx])
                # print(lane_data.lane_idx[idx])
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
        return lane_data
    def getMaxHeight(self, heat_img):
        max_height = 80
        for idx, height in enumerate(heat_img):
            # print(height.shape)
            # print(type(height))
            if np.max(height)>-3:
                # print("MAX HEIGHT = {}".format(idx))
                return idx
        return max_height
    
    # key_list = [ [height,width] * n_of_key ] * n_of_height
    def getLanefromHeat(self, heat_img, delta_img, temp_raw_image=None):
        th=-3
        heat_lane = heat_img[0].cpu()*100
        heat_lane2 = heat_img[1].cpu()
        # cv2.imwrite("SIBAL22.png",heat_img[0].cpu().detach().numpy()*10)
        # cv2.imwrite("SIBAL33.png",heat_img[1].cpu().detach().numpy()*10)
        compat_heat = torch.where(heat_img[0] < heat_img[1], torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)).cpu().detach().numpy()

        # key_list = 3d list height * key * [height,width]
        # key_list=self.getKeyfromHeat_pre(compat_heat, delta_img[:,:,0], 170) # 84.0
        # key_list=self.getKeyfromHeat(heat_lane, delta_img[:,:,0], 170, -1) # 82.9
        key_list=self.getKeyfromHeat22(heat_lane2, delta_img[:,:,0], 170, 3) # 82.9
        # print(key_list)
        resized_raw_img = cv2.resize(temp_raw_image,(640,368) )
        for height in key_list:
            # print(key)
            for key in height:
                resized_raw_img = cv2.circle(resized_raw_img, (key[1], key[0]), 2, (0,0,255), -1)
                            # cv2.circle(output_right_circle_image, startPoint, 1, (255,0,0), -1)
        # cv2.imwrite("keys2.png",resized_raw_img)

        # key_list = key_list[3:10]
        lane_data = Lane()
        lane_data = self.buildLane(key_list,  delta_img[:,:,1])

        # self.temp_lane_drawer_laneObject(temp_raw_image.copy(), lane_data, "temptemp2")

        
        max_height = self.getMaxHeight(heat_lane2.detach().numpy())
        lane_data = self.predict_horizon_v2(heat_lane2, lane_data, 165, max_height) #STD 80

        new_list = lane_data.tensor2lane()
        new_list.sort(key=len, reverse=True)

        new_list = self.getLanebyH_sample_deg(new_list, 160, 710, 10)
        # self.temp_Key_drawer(temp_raw_image.copy(), key_list)
        return new_list

    def extendLane(self, lane_list):
        for idx, lane in enumerate(lane_list):
            # print("----------")
            # print(lane_list[idx])
            if len(lane)<5:
                continue
            height=lane[0][0]+5
            width =lane[0][1]
            delta = int((lane[0][1] - lane[4][1])/4)
            extend_list=[]
            
            while height < 370 and( 0<width and width <640):
                extend_list.append([height, width+delta])
                height += 5
                width += delta
            extend_list.reverse()
            lane_list[idx] = extend_list + lane_list[idx]
            # print(lane_list[idx])
            # if lane[0]
        return lane_list
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
        key_list.reverse()
        # key_up_list.reverse()
        lane_data = Lane()
        if len(key_list)==0:
            return lane_data
        # First Update
        for idx, first_key in enumerate(key_list[0]):
            # print(first_key)
            lane_data.lane_list[idx, 0] =  np.array(first_key)
            # lane_data.addCount(idx)
            # up_lane = self.getUpLane(first_key, delta_up_image)
            lane_data.addKey(first_key, idx)
            lane_data.lanes_num +=1
        # print("========================")
        # print(key_list[0])
        # print("========================")
        # print(key_list[1])

        # keys = key of One height, key_list = key of All height
        temp = 0
        for keys in key_list[1:]:
            # print("KEYS {}".format(keys))
            # lane_data.printLane()

            for key in keys:
                # print("     KEY {}".format(key))
                min_dist = 100
                added_idx = np.array([0,0])
                for idx, lane in enumerate(lane_data.lane_list):
                    if lane_data.lane_idx[idx] == 0:
                        continue
                    dist = abs(lane[lane_data.lane_idx[idx]-1, 1] - key[1])
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
                    if abs(deg) > 50:
                        continue
                        
                    if dist < min_dist and height_dist < 30:
                        min_dist = dist
                        added_idx = idx
                        # print("Mindist - {}".format(min_dist))  
                
                if min_dist < 40:
                    # print(" Added Idx - {}".format(added_idx))  
                    # print(lane_data.lane_idx)      
                    lane_data.lane_list[ added_idx, lane_data.lane_idx[added_idx]] = key
                    up_lane = self.getUpLane(key, delta_up_image)
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
    def widthCluster(self, width_list, height, distBTlane=20, anyoneOption = False):
        point_list=[]
        dist_buffer=[]
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
                    dist_buffer.append(max_value-min_value)
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
            dist_buffer.append(max_value-min_value)
            point_list.append([height, int(buf/buf_count)])
        dist_mean = sum(dist_buffer)/len(dist_buffer)
        # print("DIST = {}".format(dist_mean))
        # 30 40 97.63
        # 40 40 97.60
        if dist_mean>30 or (anyoneOption and max(dist_buffer) > 40 ):
            # print("         DIST Too Big!!")
            # print("         DIST Too Big!!")
            # print("         DIST Too Big!!")
            # print("         DIST Too Big!!")
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

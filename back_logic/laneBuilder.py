import torch
import numpy as np
import time
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

    def getKeyfromDelta(self, output_image, delta_threshold = 30):
        delta_threshold_min=3

        delta_right_image = output_image[:,:,0]
        delta_up_image = output_image[:,:,1]

        lane_in_height=[0 for i in range(6)]
        key_list=[]
        key_up_list=[]
        for i in range(130, delta_right_image.shape[0], 5):
            width_list=[]
            for j in range(10, delta_right_image.shape[1], 5):
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if  delta_threshold_min < delta_right_image[i,j] and delta_right_image[i,j] < delta_threshold:
                    direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                    width_list.append(int(delta_right_image[i,j])*direction + j)
            if len(width_list)==0:
                continue
            count=1
            
            buf_count=1
            buf=last=width_list[0]
            point_list=[]
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

            # -------------- Get Lane Num --------------------
            # print("new_width_list {}".format(point_list))
            # print("Height {}, Count {}".format(i, count))
            if count>5:
                count=5
            lane_in_height[count] +=1
            key_list.append(point_list)
        key_up_list = key_up_list[1:]
        # builder = LaneBuilder()
        lane_data = Lane()
        lane_data = self.buildLane(key_list,  delta_up_image)
        # for idx, lane in enumerate(lane_data.lane_list):
        #     for point in lane:
                # output_right_circle_image = cv2.circle(output_right_circle_image, (point[1], point[0]), 5, myColor.color_list[idx if idx <=10 else 10], -1)

        new_list = lane_data.tensor2lane()
        new_list = self.getLanebyH_sample_deg(new_list, 160, 710, 10)
        return new_list
    
    def getKeyfromHeat(self, heat_image,delta_right_image, min_height):
        # lane_in_height=[0 for i in range(6)]
        delta_threshold_min = 3
        delta_threshold = 20

        key_list=[]
        for i in range(min_height, heat_image.shape[0], 5):
            width_list=[]
            for j in range(10, heat_image.shape[1], 3):
                # if j+11 > heat_image.shape[1] or j-11 < 0:
                #     continue
                if  0 < heat_image[i,j]:
                    width_list.append(j)

                # get Key From delta
                if j+11 > delta_right_image.shape[1] or j-11 < 0:
                    continue
                if  delta_threshold_min < delta_right_image[i,j] and delta_right_image[i,j] < delta_threshold:
                    direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                    width_list.append(int(delta_right_image[i,j])*direction + j)
            if len(width_list)==0:
                continue
            count=1
            
            ## Point Clustering (width)
            buf_count=1
            buf=last=width_list[0]
            point_list=[]
            for idx in width_list[1:]:
                if idx > last+40:
                    count +=1
                    point_list.append([i, int(buf/buf_count)])
                    buf_count=1
                    buf=idx
                else:
                    buf_count+=1
                    buf+=idx
                last = idx
            if buf_count != 0:
                point_list.append([i, int(buf/buf_count)])
            # print("Key LIST {}".format(point_list))
            # lane_in_height[count] +=1
            key_list.append(point_list)
        return key_list

    def predict_horizon(self, heat_image, lane_data, bottom_height, top_height):
        horizon_lane_tensor=np.zeros((lane_data.candi_num, 20, 2), dtype = np.int)
        # lane_start_idx = [20 for i in range(lane_data.candi_num)]
        # print(lane_data.lane_list[:lane_data.lanes_num])
        # print("11111111========================")

        lane_data.lane_list = np.concatenate((lane_data.lane_list, horizon_lane_tensor), axis = 1)

        # print("Lane Count = {}".format(lane_data.lanes_num))
        for height in range(bottom_height, top_height, -5):
            horizon_heat_key=[]
            pradicted_lane_key=[]
            width_list=[]

            ## 1. Get Key
            for j in range(10, heat_image.shape[1], 3):

                if  0 < heat_image[height,j]:
                    width_list.append(j)
            if len(width_list)==0:
                continue
            count=1
            ## 2. Point Clustering (width)
            buf_count=1
            buf=last=width_list[0]
            for idx in width_list[1:]:
                if idx > last+15:
                    count +=1
                    horizon_heat_key.append([height, int(buf/buf_count)])
                    buf_count=1
                    buf=idx
                else:
                    buf_count+=1
                    buf+=idx
                last = idx
            if buf_count != 0:
                horizon_heat_key.append([height, int(buf/buf_count)])
            if len(horizon_heat_key)==0:
                continue
            ## 3. Predicted Key
            for idx, lane in enumerate(lane_data.lane_list):
                if lane_data.lane_idx[idx] <2:
                    continue
                # print(lane)
                last_idx_of_lane = lane_data.lane_idx[idx]-1
                # print("Last index {}".format(last_idx_of_lane))
                # print("Last point {}".format(lane[last_idx_of_lane]))
                if abs(lane[last_idx_of_lane,0] - height) > 30:
                    continue
                if lane_data.lane_idx[idx] >5:
                    dx = lane[lane_data.lane_idx[idx] - 2 ,1] - lane[lane_data.lane_idx[idx] - 1 ,1]
                    dy = lane[lane_data.lane_idx[idx] - 2 ,0] - lane[lane_data.lane_idx[idx] - 1 ,0]
                    delta_height = lane[lane_data.lane_idx[idx] - 2 ,0] - height
                    # if dy <2:
                    #     print(lane)
                    #     return
                    delta_width = dx*(delta_height/(dy+0.01))

                    # print("Lane IDX= {}".format(lane_data.lane_idx[idx]))
                    # print("delta_widthX= {}".format(delta_width))
                    predicted_key = [height, int(lane[lane_data.lane_idx[idx] - 2 ,1] - delta_width)]
                    # print(predicted_key)
                    pradicted_lane_key.append(predicted_key)
                    for heat_key in horizon_heat_key:
                        if abs(predicted_key[1] - heat_key[1]) < 10:
                            predicted_key=heat_key
                            break
                    lane_data.addKey(predicted_key, idx)

            no_pair=[]
            for idx, pred in enumerate(pradicted_lane_key):
                for heat_key in horizon_heat_key:
                    if abs(pred[1] - heat_key[1]) < 10:
                        continue
                    no_pair.append(idx)
            # for idx, pred in enumerate(pradicted_lane_key):
            #     if idx in no_pair:




            # print("HORIZON KEY LIST {}".format(horizon_heat_key))
            # print("PREDICTED KEY LIST {}".format(pradicted_lane_key))

        return lane_data
    def getLanefromHeat(self, heat_img, delta_img):

        key_list=self.getKeyfromHeat(heat_img, delta_img[:,:,0], 170)
        lane_data = Lane()
        lane_data = self.buildLane(key_list,  delta_img[:,:,1])
        lane_data = self.predict_horizon(heat_img, lane_data, 160, 80)
        

        new_list = lane_data.tensor2lane()
        new_list = self.getLanebyH_sample_deg(new_list, 160, 710, 10)
        return new_list

    
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
            lane_data.addCount(idx)
            up_lane = self.getUpLane(first_key, delta_up_image)
            lane_data.addKey(up_lane, idx)
            lane_data.lanes_num +=1
        # print("========================")
        # print(key_list)

        # keys = key of One height, key_list = key of All height
        temp = 0
        for keys in key_list[1:]:

            for key in keys:
                # print(" KEY {}".format(key))
                min_dist = 100
                added_idx = np.array([0,0])
                for idx, lane in enumerate(lane_data.lane_list):
                    # print("Idx {}".format(idx))
                    if lane_data.lane_idx[idx] == 0:
                        # print("Continue")
                        continue
                    dist = abs(lane[lane_data.lane_idx[idx]-1, 1] - key[1])
                    height_dist = abs(lane[lane_data.lane_idx[idx]-1, 0] - key[0])

                    # print("LANE {}, Key =  {}".format(lane[lane_data.lane_idx[idx]-1, 1], key))
                    # print("Dist = {}".format(dist))
                    if dist < min_dist and height_dist < 71:
                        min_dist = dist
                        added_idx = idx
                        # print("Mindist - {}".format(min_dist))  
                        
                if min_dist < 50:
                    # print(" Added Idx - {}".format(added_idx))  
                    # print(lane_data.lane_idx)      
                    lane_data.lane_list[ added_idx, lane_data.lane_idx[added_idx]-1] = key
                    up_lane = self.getUpLane(key, delta_up_image)
                    lane_data.addKey(up_lane, added_idx)
                    # lane_data.lane_predicted[added_idx] = self.lane_predicted 

                else:
                    lane_data.addKey(key, lane_data.lanes_num)
                    up_lane = self.getUpLane(key, delta_up_image)
                    lane_data.addKey(up_lane, lane_data.lanes_num)
                    lane_data.lanes_num+=1
                    # lane_data.lane_list[lane_data.lanes_num, 0] = 
                    # else:
                    #     lane_data.addKey(up_lane, idx)
                    #     lane_data.lane_list[idx, lane_data.lane_idx[idx]-1] = key

            # if temp ==7:
            #     break
            temp+=1
        # print(lane_data.lane_list)
        # print(lane_data.lane_idx)      
        lane_data.delete_last()
        return lane_data

    def getLanebyH_sample_deg(self, lanes,  h_start, h_end, h_step):
        lane_list = []
        for lane in lanes:
            for node in lane:
                node[0] = int(node[0]/368*720)
                node[1] = int(node[1]/640*1280)
            lane.reverse()
        # print(lanes)
        
        for lane in lanes:
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
#                             continue

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
if __name__=="__main__":
    pl = LaneBuilder()

import torch
import numpy as np
import time
class Lane:
    def __init__(self):
        self.lane_list = np.zeros((30, 30, 2), dtype=np.int)
        self.lane_idx = [0 for i in range(30)]
        self.lanes_num=0
        return
    def addCount(self, idx):
        if idx >= 30:
            print("IDX is Too BIG!!!")
            return
        self.lane_idx[idx] +=1
        return
    def addKey(self, up_lane, idx):
        if up_lane[0]!=0:
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
        re_lane = []
        for lane_idx in range(self.lanes_num):
            lane=[]
            for height_idx in range(self.lane_idx[lane_idx]):
                point = [self.lane_list[lane_idx, height_idx,0], self.lane_list[lane_idx, height_idx,1]]
                lane.append(point)
            re_lane.append(lane)
        return re_lane

    def convert_tuSimple(self):
        self.resize_lane()
        return_lane=[]
        for lane_idx in range(self.lanes_num):
            add_lane=[]
            height_start = self.lane_list[lane_idx,0,0]
            print("Height Start = {}".format(height_start))
            start_idx = int((360- height_start)/10)
            print("start_idx = {}".format(start_idx))
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
    def getUpLane(self, key, delta_image):
        height = key[0]
        width = key[1]
        min_abs=100
        new_10_point = np.array([0,0], dtype=np.int)
        for point in range(width-40, width+41, 2):
            # print("     Point {}".format(point))
            if  0<point and point < delta_image.shape[1] and 7 < delta_image[height,point] and delta_image[height,point] < 13:
                # print("HERE")
                # print("!! {} {}".format(point, delta_right_image.shape[1]))
                direction = -1 if delta_image[height-5,point] > delta_image[height+5,point] else 1
                if direction == -1: # Go Down
                    continue
                resi = abs(width-point)
                if resi < min_abs:
                    new_10_start_point = (width, height)
                    new_10_point = [height-10*direction, point]
                    min_abs = resi
        if min_abs < 99:
            temp_up = new_10_point
        return new_10_point
            # point_up_list.append([new_10_start_point[0], new_10_point[0]])

        return
    def getKeyfromDelta(self, output_image, delta_threshold = 30):
        delta_threshold_min=3
        # return
        # output_image = output_image.cpu().detach().numpy()
        # --------------------------Save segmented map
    
        # output_image = self.inference_np2np_instance(image, model)

        delta_right_image = output_image[:,:,0]
        delta_up_image = output_image[:,:,1]


        lane_in_height=[0 for i in range(6)]
        key_list=[]
        key_up_list=[]
        for i in range(130, delta_right_image.shape[0], 10):
            width_list=[]
            for j in range(10, delta_right_image.shape[1], 10):
                startPoint = (j, i)
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
        lane_data = self.buildLane(key_list, key_up_list, delta_up_image)
        # for idx, lane in enumerate(lane_data.lane_list):
        #     for point in lane:
                # output_right_circle_image = cv2.circle(output_right_circle_image, (point[1], point[0]), 5, myColor.color_list[idx if idx <=10 else 10], -1)

        new_list = lane_data.tensor2lane()
        new_list = self.getLanebyH_sample_deg(new_list, 160, 710, 10)
        return new_list
        myList = lane_data.convert_tuSimple()
        print(myList)
        print("Lane Num {}".format(lane_data.lanes_num))   
        print("Lane Idx Num {}".format(lane_data.lane_idx))   
        return myList
    def buildLane(self, key_list, key_up_list, delta_up_image):
        key_list.reverse()
        key_up_list.reverse()
        lane_data = Lane()
        if len(key_list)==0:
            return lane_data
        # First Update
        for idx, key_up in enumerate(key_list[0]):
            # print(key_up)
            lane_data.lane_list[idx, 0] =  np.array(key_up)
            lane_data.addCount(idx)
            up_lane = self.getUpLane(key_up, delta_up_image)
            lane_data.addKey(up_lane, idx)
            lane_data.lanes_num +=1
        # print("========================")
        # print(key_list)

        # print("Hello world {}".format(key_list))
        # print("Lane List{}".format(lane_data.lane_list))
        # for keys, keys_up in zip(key_list[1:], key_up_list[1:]):
        #     # print("KEYs {}   ,   {}".format(keys, keys_up))

        #     min = 9000
        #     next = -1
        #     temp_up = None

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

                    # print("LANE {}, Key =  {}".format(lane[lane_data.lane_idx[idx]-1, 1], key))
                    # print("Dist = {}".format(dist))
                    if dist < min_dist:
                        min_dist = dist
                        added_idx = idx
                        # print("Mindist - {}".format(min_dist))  
                        
                if min_dist < 20:      
                    # print(" Added Idx - {}".format(added_idx))  
                    # print(lane_data.lane_idx)      
                    lane_data.lane_list[ added_idx, lane_data.lane_idx[added_idx]-1] = key
                    up_lane = self.getUpLane(key, delta_up_image)
                    lane_data.addKey(up_lane, added_idx)

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

        return lane_data
    # def tensor2lane(self, lane_tensor):
    #     return_lane = []
    #     count=0
    #     for idx, lane in enumerate(lane_tensor):
    #         if torch.count_nonzero(lane) < 4:
    #             count+=1
    #             continue
    #         new_lane=lane.tolist()
    #         new_lane.sort(key=lambda x : x[0])
    #         nz_idx = 0
    #         for idx, lane in enumerate(new_lane):
    #             if lane[0]>0 and lane[1]>0:
    #                 nz_idx = idx
    #                 break
    #         return_lane.append(new_lane[nz_idx:])
    #     return return_lane
    def getLanebyH_sample_deg(self, lanes,  h_start, h_end, h_step):
        lane_list = []
        # print("=========")
        # print("=========")
        # print("=========")
        # print("=========")
        # print("=========")
        # print("=========")
        # print("=========")
        # print("=========")
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
        # print("????????????")
        # print("????????????")
        # print("????????????")
        # print("????????????")
        # print(lane_list)
#         print("SELF LIST")
#         print(self.lane_list)
        return lane_list
if __name__=="__main__":
    pl = LaneBuilder()
    # lane_list=np.array([[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 636, 621, 613, 606, 598, 584, 572, 562, 552, 542, 527, 520, 517, 488, 485, 487, 480, 465, 451, 440, 440, 427, 416, 409, 401, 388, 377, 370, 362, 350, 337, 330, 323, 312, 298, 291, 284, 273, 259, 252, 245, 235, 220, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 680, 680, 705, 723, 731, 745, 761, 776, 788, 798, 812, 827, 840, 840, 856, 875, 890, 904, 914, 924, 947, 956, 967, 981, 996, 1011, 1022, 1030, 1045, 1060, 1074, 1088, 1096, 1107, 1122, 1137, 1145, 1153, 1162, 1183, 1195, 1204, 1220, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 798, 834, 864, 907, 949, 991, 1048, 1108, 1126, 1155, 1203, 1239, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])
    # pl.mergeLane(lane_list)
    

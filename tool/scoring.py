from sklearn.cluster import MeanShift, estimate_bandwidth
from back_logic.evaluate import EDeval
import torch
import time
import os
import numpy as np
import math
class Scoring():
    def __init__(self):

        self.imagePath=""
        self.outputPath=""
        self.lanes = []
        self.lane_length = [0 for i in range(7)]
        self.lane_list=[]
        self.device=None
        return
    def getLanebyH_sample(self,  h_start, h_end, h_step):
        # os.system('clear')
#         print("START------------")
        lane_list = []
#         print(self.lanes)
#         for lane in self.lanes:
#             print("Before {}".format(lane))
#             for node in lane:
#                 node[0] = int(node[0]/368*720)
#                 node[1] = int(node[1]/640*1280)
#             print("After {}".format(lane))
                
#             print(lane)
           
        for lane in self.lanes:

            new_single_lane=[]
            
            for node in lane:
                node[0] = int(node[0]/368*720)
                node[1] = int(node[1]/640*1280)
            
            
            # print("LANE DATA ---------------")
            # print(lane)

            cur_height_idx = 0
            lane_ended=False
            
            # end+1 = for height_sample number
            for height in range(h_start, h_end+1, h_step):
                if lane_ended or cur_height_idx==30:
                    new_single_lane.append(-2)
                    continue
                # print(height)
                cur_height = lane[cur_height_idx][0]
                # print("Cur Height = {}".format(cur_height))
                
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
                    
                    dx = lane[cur_height_idx][1] -lane[cur_height_idx-1][1] 
                    dy = lane[cur_height_idx][0] -lane[cur_height_idx-1][0] 
                    cur_height_idx +=1
                    # print(dy)
                    if dy!=0:
                        break
                    if cur_height_idx==30:
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
                new_single_lane.append(newX)                        
#             if len(new_single_lane)!=55:
#                 print("SDFSDFSDFSDF!!!!!!!!!!!!!!!!!!!")
#                 print(len(new_single_lane))
#                 time.sleep(100000)
#             print(len(new_single_lane))
            # print(new_single_lane)
            lane_list.append(new_single_lane)
        self.lane_list=lane_list
#         print("SELF LIST")
#         print(self.lane_list)
        return lane_list
    def getLanebyH_sample_deg(self,  h_start, h_end, h_step):

#         os.system('clear')
        # print("----------------------------")
        # print("Get H Sample START------------")
        lane_list = []
#         print(self.lanes)
        for lane in self.lanes:
            # print("Before {}".format(lane))
            for node in lane:
                node[0] = int(node[0]/368*720)
                node[1] = int(node[1]/640*1280)
            # print("After {}".format(lane))
                
        #     print(lane)
        # print("-------------------- FINISH")
        # print(self.lanes)

        # return
        for lane in self.lanes:

            new_single_lane=[]
            
            # for node in lane:
            #     node[0] = int(node[0]/368*720)
            #     node[1] = int(node[1]/640*1280)
            
            

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
#             if len(new_single_lane)!=55:
#                 print("SDFSDFSDFSDF!!!!!!!!!!!!!!!!!!!")
#                 print(len(new_single_lane))
#                 time.sleep(100000)
#             print(len(new_single_lane))
            # print("MEAN {}".format(np.mean(new_single_lane)))
            # print(new_single_lane)
            if np.mean(new_single_lane) > 0:
                lane_list.append(new_single_lane)
        self.lane_list=lane_list
#         print("SELF LIST")
#         print(self.lane_list)
        return lane_list

    def tensor2lane(self, lane_tensor):
        # lane_tensor[:,:,0]  *= (720.0/368.0)
        # lane_tensor[:,:,1]  *= (1280.0/640.0)
        # print("TENSOR 2 LANE------------------------------------")
        # non_zero_tensor = lane_tensor.tolist()
        # print("--- FROM ------------------------------------")
        # print(lane_tensor.shape)
        # print(lane_tensor)
        # print("--- TO -----------------------------------")
        count=0
        for idx, lane in enumerate(lane_tensor):
            # print("N of 0 {}".format(torch.count_nonzero(lane)))
            if torch.count_nonzero(lane) < 4:
                # print("TARI IDDX {}".format(idx))
                # print("TARINAI {}".format(len(torch.count_nonzero(lane))))
                # print("   TARI {}".format(torch.count_nonzero(lane)))
                count+=1
                continue
            # else:
            #     print(lane)
                # print("   ORI  {}".format(lane))
            # if False:
            # new_lane = torch.reshape(lane[torch.nonzero(lane, as_tuple=True)], (:,2)).tolist()
            # new_lane = lane[torch.nonzero(lane, as_tuple=True)]
            new_lane=lane.tolist()
            # print(new_lane)
            # .tolist()
            new_lane.sort(key=lambda x : x[0])
            nz_idx = 0
            for idx, lane in enumerate(new_lane):
                if lane[0]>0 and lane[1]>0:
                    nz_idx = idx
                    break
            # print("New Lane - {}".format(new_lane[nz_idx:]))
            self.lanes.append(new_lane[nz_idx:])
            # print(lane)
            # print("\n\n")
        # print(non_zero_tensor)
        # print("TAri countr {}".format(count))
        return
    
    def prob2lane(self, img_idx, img_val, lane_start, lane_end, lane_step, lane_num=10):
        width = img_idx.shape[1]

        lane_list=[[] for i in range(7)]
        for ordinate in range(lane_start, lane_end+1, lane_step):
        # for ordinate in range(lane_start, lane_end-50, lane_step):

            # print("ORDINATE {}".format(ordinate))
            max_value = [-2 for i in range(0,7)]
            max_idx = [-1 for i in range(0,7)]


            for abscissa in range(0, width):
                id = img_idx[ordinate][abscissa]
                if id==0: continue
                val = img_val[ordinate][abscissa]
                if val > max_value[id]:
                    max_value[id] = val
                    max_idx[id] = abscissa
            for id in range(1,7):
                if max_idx[id] is not -1:
                    lane_list[id].append([ordinate, max_idx[id]])
                    self.lane_length[id] +=1
        for lane in range(1,7):
            # if self.lane_length[lane] >=2:
            if True:
                self.lanes.append(lane_list[lane])

            # self.lanes.append(lane_list[lane])
        return
    
    def refine_deltamap(self, deltamap, heatmap):

        # print("HEatShape = {}".format(heatmap.shape))
        heatmap_horizon_pad = torch.nn.functional.pad(heatmap, (4,0), value=0)[:,:-4]
        heatmap_vertical_pad = torch.nn.functional.pad(heatmap, (0,0,4,0), value=0)[:-4]
        # print("HEATMAP IDX  {}".format(heatmap[0,0]))
        # print("HEATMAP IDX2 {}".format(heatmap_horizon_pad[0,4]))
        # print("HEATMAP IDX4 {}".format(heatmap_vertical_pad[4,0]))
        # print("heatmap_horizon_pad = {}".format(heatmap_horizon_pad.shape))
        # print("heatmap_vertical_pad = {}".format(heatmap_vertical_pad.shape))
        # print("DELTA SHAPE = {}".format(deltamap.shape))

        abc = torch.where(heatmap > heatmap_horizon_pad, deltamap[0], deltamap[0]*-1)
        abc2= torch.where(heatmap > heatmap_vertical_pad, deltamap[1], deltamap[1]*-1)

        deltamap[0] = abc
        deltamap[1] = abc2
        return

    def refine_points(self, tensor, deltamap):
        # f=open("CUDA_sibal.txt",'a')

        # f.writelines("refine_tensor = {} /// {} shape\n".format(tensor, tensor.shape))
        if tensor.shape[0]==1:
            index_int_tensor = tensor[:,1].type(torch.torch.LongTensor).to(self.device)
            index_int_tensor_vertical = tensor[:,0].type(torch.torch.LongTensor).to(self.device)
        else:
            index_int_tensor = torch.squeeze(tensor[:,1]).type(torch.torch.LongTensor).to(self.device)
            index_int_tensor_vertical = torch.squeeze(tensor[:,0]).type(torch.torch.LongTensor).to(self.device)
        # print("SELECTED : {}".format(index_int_tensor))
        getdel = torch.index_select(deltamap[0],1, index_int_tensor).permute(1,0)
        # f.writelines("GETDEL =  {} shape\n".format( getdel.shape))

        
        # print(deltamap[0,170,165])
        # print(deltamap[0,170,270])
        # print(deltamap[0,170,355])
        # print("0-----------------------")
        # print(getdel[0,170])
        # print(getdel[1,170])
        # print(getdel[2,170])


        # print("GETDEL11")
        # print(getdel)
        # print(getdel.shape)
        # print("asdfasdf")
        # print(index_int_tensor)
        # print(index_int_tensor.dtype)
        # print(torch.unsqueeze(index_int_tensor[:,0], dim=1))
        # print(torch.unsqueeze(index_int_tensor[:,0], dim=1).shape)
        getdel = torch.gather(getdel, 1, torch.unsqueeze(index_int_tensor_vertical, dim=1))

        # print("GETDEL22")
        # print(getdel)
        # print(getdel.shape)
        # # refined_tensor = tensor[:,1] + 
        # print(tensor[:,1:2])
        # print(tensor.get_device())
        # print(getdel.get_device())
        # f.writelines("----------------------------- Bkey\n {}\n".format(tensor))
        tensor[:,1:2] += getdel
        # f.writelines("----------------------------- Akey\n {}\n".format(tensor))
        # f.writelines("___\n")
        # f.writelines("___\n")
        # f.close()
        # print(tensor[:,1:2])

        return
    
    def getLaneFromsegdeg(self, heatmap, deltamap, seed, height_val = 170, delta_height=5):
        # heatmap = torch.squeeze(heatmap, dim=0)[1]
        # heatmap = torch.squeeze(heatmap, dim=0)[1]
        filter_size=5
        recep_size = filter_size
        tensor_point = torch.empty(seed.shape[0], 2, filter_size*2, filter_size*2)

        seed = torch.nn.functional.pad(torch.unsqueeze(seed, dim=1), (1,0,0,0), value=170).type(torch.FloatTensor).to(self.device)
        # print("NEW SEED {}".format(seed))
        # print("NEW SEED {}".format(seed.shape))
        # print("NEW SEED TYPE {}".format(seed.dtype))
        self.refine_points(seed, deltamap)
        # print("NEW SEED {}".format(seed))


        # tensor_point_
        lane_tensor = torch.zeros(seed.shape[0],500,2)

        torch.set_printoptions(1, sci_mode=False)
        print("---------------------------- Get Lane from Image --------------")
        f=open("CUDA_sibal.txt",'a')
        f.write("Get Lane --------------------\n\n")
        f.write("SEED - {}\n".format(seed))

        print("SEED - {}".format(seed))
        for idx, key in enumerate(seed):
            # key = seed[1]
            print("KEY = {}".format(key))
            f.write("--- KEY {}\n".format(key))

            tete2 = self.getLocalDeltampa(heatmap, deltamap, key)
            # f.write("-TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT {}\n".format(tete2))
            # print("TTT {}".format(tete2))
            
            new_key = self.getPolyLane(tete2.permute(1,2,0), heatmap, 5)
            # f.write("-NNNNNNNNNNNNNNNNNNNNNNNNNNN {}\n".format(new_key))

            # f.write("NEW KEY {}---------------------------------------\n".format(new_key))

            # new_key = self.getPolyLane(tensor_point[idx].permute(1,2,0), heatmap, 5)
            # print("NEW KEY--------------------------- {}".format(new_key))

            lane_tensor[idx][0] = key
            lane_tensor[idx][1] = new_key[0]
            downtensor = new_key[1]
            # print(local_delta.shape)
            # print(local_delta)

            # Get Up-side Lane
            new_uptensor = lane_tensor[idx][1]
            up_idx = 2
            f.write("--------------------------- UP\n")
            # f.close()
            # UP = key[0], Down = Key[1]
            while True:
                # if int(new_key[0,0]) > int(new_uptensor[0]) or (int(new_key[0,0]) == int(new_uptensor[0]) and int(new_key[0,1]) == int(new_uptensor[1])):
                #     print("Brake 33333333333")
                #     break
                # print(lane_tensor.shape)
                # f.write("11UP TENSOR = {}\n".format(new_uptensor))
                # print("UP TENSOR = {}".format(new_uptensor))
                local_delta = self.getLocalDeltampa(heatmap, deltamap, new_uptensor)
                # f.write("22UP TENSOR = {}\n".format(new_uptensor))

                # print("UP TENSOR = {}".format(new_uptensor))

                new_key = self.getPolyLane(local_delta.permute(1,2,0), heatmap, 3)
                # f.write("11Refined key {}\n".format(new_key))

                # self.refine_points(new_key, deltamap)
                # print("PRE {}".format(new_uptensor))
                # print("NEW {}".format(new_key))
                # f.write("Pre Key     {}\n".format(new_uptensor))
                # f.write("22Refined key {}\n".format(new_key))

                # print("COMPARE  --------------------------- {}///{}".format(new_key[0,0], int(new_uptensor[0])))

                # if abs(int(new_key[0,1]) - int(new_uptensor[1])) > 20:
                #     break
                # or int(new_key[0,0]) == int(new_uptensor[0])
                if int(new_key[0,1]) == 0 :
                    break
                new_uptensor = new_key[0]
                # print("NEW KEY--------------------------- {}".format(new_key[0]))
                # print("InLOOP  {}".format(heatmap[int(new_key[0,0]), int(new_key[0,1])]))
                if heatmap[int(new_key[0,0]), int(new_key[0,1])] < -5.0:
                    break
                lane_tensor[idx][up_idx] = new_uptensor
                up_idx+=1

                
                if int(new_key[0,0])-20 <0 or int(new_key[0,1])-20 <0:
                    break
                if int(new_key[0,0])+20 > heatmap.shape[0] or int(new_key[0,1])+20 > heatmap.shape[1]:
                    break
                if up_idx > 200:
                    break
            print("--------------------------- UPDOWN ---------------------------------------")
            # new_uptensor = lane_tensor[idx][0]
            # lane_tensor[idx][up_idx] = lane_tensor[idx][0]
            new_uptensor = downtensor
            new_key[1] = downtensor.to(self.device)
            # lane_tensor[idx][up_idx] = downtensor
            up_idx+=1
            f.write("--------------------------- Down\n")

            while False:
            # while True:
                # print(lane_tensor.shape)
                # print("UP TENSOR = {}".format(new_uptensor))
                #  or :
                # if abs(int(new_key[0,1]) - int(new_uptensor[1])) > 20:
                #     print("Brake 11111111111")
                #     break
                if int(new_key[1,1]) == 0:
                    print("Brake 2222222222222")
                    break
                t1 = time.time()
                local_delta = self.getLocalDeltampa(heatmap, deltamap, new_uptensor)
                t2 = time.time()
                # print("INPUT = {}".format(local_delta.permute(1,2,0)))
                new_key = self.getPolyLane(local_delta.permute(1,2,0), heatmap, 3, threshold=-1.0)
                t3 = time.time()
                print("TIME {} {}".format(t3-t2, t2-t1))
                print("New Key = {}".format(new_key))
                # self.refine_points(new_key, deltamap)

                if int(new_key[1,0]) > int(new_uptensor[0]) or (int(new_key[1,0]) == int(new_uptensor[0]) and int(new_key[1,1]) == int(new_uptensor[1])):
                    print("Brake 33333333333")
                    break
                #  
                new_uptensor = new_key[1]
                # print("NEW KEY--------------------------- {}".format(new_key[1]))
                # print("InLOOP  {}".format(heatmap[int(new_key[1,0]), int(new_key[1,1])]))
                if heatmap[int(new_key[1,0]), int(new_key[1,1])] < -5.0:
                    break
                # print("New Y = {} ///  Old Y = {}".format(int(new_key[1,0]), int(new_uptensor[0])))
                lane_tensor[idx][up_idx] = new_uptensor
                up_idx+=1
                if int(new_key[1,0])-20 <0 or int(new_key[1,1])-20 <0:
                    print("Brake 4444444444")

                    break

                if int(new_key[1,0])+20 > heatmap.shape[0] or int(new_key[1,1])+20 > heatmap.shape[1]:
                    print("Brake 5555555555555555555")

                    break
                # print(lane_tensor.get_device())
                # print(new_uptensor.get_device())
                # lane_tensor[up_idx] = torch.cat((lane_tensor, new_uptensor) ,1)
                # lane_tensor = torch.stack((lane_tensor, new_uptensor) ,0)
                if up_idx > 300:
                    print("Brake 66666666666666")

                    break


        return lane_tensor[:,3:]
        # print("tensor_point!!!!!!!!!!!!")
        # print(tensor_point)
        # print(tensor_point.shape)


        return tensor_point.permute(0,2,3,1)

    def getLocalDeltampa(self, heatmap, deltamap, key, filter_size=5):
        # print("UP TENSOR = {}".format(key))

        # print("KEY = {}".format(key))
        abscissa = int(key[1])
        ordinate = int(key[0])
        tensor_point = torch.empty(2, filter_size*2, filter_size*2)
        recep_size = filter_size

        # if abscissa+filter_size > heatmap.shape[1]:
        #     recep_size=heatmap.shape[1]-abscissa
        # if abscissa-filter_size <0:
        #     recep_size=abscissa-1
        # f=open("CUDA_sibal.txt",'a')
        # # print("ORDINATE = {}".format(ordinate))
        # # print("Abscissa = {}".format(abscissa))
        # f.write("ORDINATE = {}\n".format(ordinate))
        # f.write("Abscissa = {}\n".format(abscissa))
        # f.close()
        # local_delta = torch.index_select(deltamap,      1, torch.tensor([i       for i in range(ordinate-recep_size, ordinate+recep_size, 1)]).to(self.device)) 
        # local_delta = torch.index_select(local_delta,   2, torch.tensor([i   for i in range(abscissa-recep_size, abscissa+recep_size, 1)]).to(self.device)) 
        
        
        local_delta = torch.empty(2, filter_size*2, filter_size*2).to(self.device)
        local_delta = deltamap[:, ordinate-recep_size: ordinate+recep_size, abscissa-recep_size: abscissa+recep_size]
        # print("LOCAL DELTA SIXZE {}".format(local_delta.shape))
        # print("LOCAL2 DELTA SIXZE {}".format(local_delta2.shape))
        
        # print("LOCAL DELTA {}".format(local_delta))
        # print("LOCAL2 DELTA {}".format(local_delta2))
        # local_delta 0 = width, 1= height, 2*filter*filter dimesion
        coord_tensor = torch.tensor([ordinate, abscissa])

        range_temp = torch.arange(0, filter_size*2).repeat(filter_size*2,1).to(self.device)
        range_tensor = torch.transpose(torch.stack([range_temp, torch.transpose(range_temp, 0, 1)]),1,2)



        range_tensor[0] = range_tensor[0]+coord_tensor[0]-recep_size
        range_tensor[1] = range_tensor[1]+coord_tensor[1]-recep_size

        tensor_point[0] = local_delta[1] + range_tensor[0]
        tensor_point[1] = local_delta[0] + range_tensor[1]
        # print("UP TENSOR = {}".format(key))

        return tensor_point
    
    def getPolyLane(self, lane_tensor, heatmap, delta, threshold=-5.0):
        # print("INPUT SHAPE {}".format(lane_tensor.shape))
        # return
        # lane_tensor (filter*filter*2) 3D tensor
        return_tensor = torch.zeros(2,2).to(self.device)
        # for idx, points in enumerate(lane_tensor):
        sum_x = 0
        sum_x_list = []
        sum_y = 0
        sum_y_list = []
        count= 0
        # t1 = time.time()
        for height in lane_tensor:
            for point in height:
                # print(point)
                # print("{} {}".format(idx%5, idx//5))
                # cv2.circle(key_image, (int(point[1].item()), int(point[0].item()), 2, (0,255,0), -1))
                # print(point.shape)
                # print(point[0])
                # print(point[1])
                if int(point[0]) >= heatmap.shape[0] or int(point[1]) >= heatmap.shape[1]:
                    continue
                if heatmap[int(point[0].item()), int(point[1].item())]> threshold:
                    # cv2.circle(key_image, (int(point[1].item()), int(point[0].item())), 2, (0,255,0), -1)
                    sum_x +=point[1].item()
                    sum_y +=point[0].item()
                    count +=1
                    sum_x_list.append(point[1].item())
                    sum_y_list.append(point[0].item())
        if count==0:
            return return_tensor
        # t2 = time.time()
        # linear_model=np.polyfit(sum_x_list,sum_y_list,1)
        linear_model=np.polyfit(sum_y_list,sum_x_list,1)
        linear_model_fn=np.poly1d(linear_model)
        return_tensor[0,0] = int(sum_y/count +delta)
        return_tensor[0,1] = int(linear_model_fn(sum_y/count +delta))
        return_tensor[1,0] = int(sum_y/count -delta)
        return_tensor[1,1] = int(linear_model_fn(sum_y/count -delta))
        # t3 = time.time()
        # print("~~~~~~~~~~~~~~~~~TIME {} {}".format(t3-t2, t2-t1))

        # return_tensor = (2*2) tensor - updonwpoint*coord

        return return_tensor
    


    def getLocalMaxima_heatmap_re(self, img_tensor, height_val = 170):
        # img_tensor = torch.squeeze(img_tensor, dim=0)[1]
        # print("--------- Get Local Maxima  {}".format(img_tensor.shape))
        # st = time.time()
        # for width_tensor in img_tensor:
        width_tensor = img_tensor[height_val]
        # print("WidthTensor shape {}".format(width_tensor.shape))

        local_maxima = torch.empty(0, dtype=torch.int64)
        last=0
        for abscissa in range(0, width_tensor.shape[0], 5):
            # print(width_tensor[abscissa].item())
            if  width_tensor[abscissa].item() > -0.5 and (local_maxima.shape[0] == 0 or local_maxima[-1,1] + 15 < abscissa  ):
                # print("Coord {} / {}".format(height_val, abscissa))
                # print("SHAPE {}   ".format(local_maxima.shape[0]))
                # if local_maxima.shape[0]!=0:
                #     print("Last {}".format( local_maxima[-1,1]))
                # print("Idx {} ..... Val {}".format(abscissa, width_tensor[abscissa].item()))
                # local_maxima = torch.cat([local_maxima, torch.tensor([[height_val, abscissa]]).to(self.device)])
                local_maxima = torch.cat([local_maxima, torch.tensor([[height_val, abscissa]])])
                # print("SHPAE {}".format(local_maxima.shape))
                # local_maxima = torch.stack([local_maxima, torch.tensor([height_val, abscissa]).to(self.device)])
                last = abscissa
        # f = time.time()
        # print("TIME {}".format(f-st))
        
        # return
        # print("LOCAL MAXIMA")
        # print(local_maxima)
        return local_maxima
    

    
    def chainKey2(self, new_key, terminal, terminal_deg, degmap, lane_num, print_mode = False):

        if print_mode:
            print("Ney Key {}".format(new_key))
        if new_key.shape[0]==0:
            return lane_num, terminal_deg
        score_tensor = torch.zeros(new_key.shape[0], terminal.shape[0]).to(self.device)

        score_tensor -=100
        new_terminal_tensor = torch.zeros(new_key.shape[0]).to(self.device)
        min_list=[100 for i in new_key]

        for ter_idx, t_point in enumerate(terminal):
            if t_point[0] == 0 and t_point[1] ==0:
                continue
            count=0
            d_sum = 0
            d_list = np.array([])
            for i in range(0,49):
                y = int(t_point[0]) + i%7
                x = int(t_point[1]) + i//7
                d = degmap[y,x]
                if t_point[0]<202 and t_point[0]>199 and print_mode:
                    print("D = {}".format(d))

                if d>170 or d<10:
                    continue
                else:
                    count+=1
                    d_sum+=d
                    d_list = np.append(d_list, d)
            d_list.sort()
            if count > 3:
                d_mean = sum(d_list[len(d_list)//2-1:len(d_list)//2+1]) / 2
            elif count == 3:
                d_mean = d_list[1]
            else:
                d_mean = d_sum/(count+0.00001)

            if print_mode:
                print("\n----------Im {}".format(t_point))
                print(" - D_mean = {}".format(d_mean))
                print(" - D_devi = {}".format(np.std(d_list)))
                if np.var(d_list) > 20 and len(d_list)>1:
                    print(d_list)
                    print(d_list.shape)
                    print(d_list[0]+d_list[1])
                # print(" - D_Devi = {}".format(d_mean))
            if len(d_list)==0 or np.std(d_list) > 30:
                d_mean = (terminal_deg[ter_idx]*0.8+90*0.2)
                if print_mode:
                    print("ALTER = {}".format(d_mean))
                    
                terminal_deg[ter_idx] = d_mean
            else:
                terminal_deg[ter_idx] = (d_mean+terminal_deg[ter_idx])/2

            # if d_mean < 5 or d_mean>175:
            #     d_mean=90
            find=False
            for key_idx, point in enumerate(new_key):
            # for ter_idx, t_point in enumerate(terminal):
                if point[0]==0:
                    continue
                d_y = point[0] - t_point[0]
                predicted_d_x = d_y/(math.tan(d_mean/180.0*math.pi)+0.001)
                predicted_x = predicted_d_x + t_point[1]
                if print_mode:
                    print("Target = {}".format(point))
                    print("Predectetd_Coord  = {} / {}".format(point[0], predicted_x))
                dist = abs(point[1] - predicted_x)
                
                # if abs(point[1] - predicted_x) < 20* math.sqrt(abs(d_y/10)):
                # if abs(point[1] - predicted_x) < 20* math.sqrt(abs(d_y/10)):
                if abs(point[1] - predicted_x) < 20* math.sqrt(math.sqrt(abs(d_y/10))) and abs(predicted_d_x) < 50  and abs(d_y)<80:
                    if print_mode:
                        print("FIND !!!")
                        print("T  = {}".format(point))
                        print("PRED = {}".format(t_point))
                        print("DIST = {}".format(dist))
                    if min_list[key_idx] > dist:
                        score_tensor[key_idx, :] = -100
                        min_list[key_idx] = dist
                        score_tensor[key_idx, ter_idx] = -100 + dist
                        new_terminal_tensor[key_idx]=1
                        find = True

            if not find and np.std(d_list) > 30:
                if print_mode:
                    print("Not Founded !!! {}".format(0))
                new_d_list = np.where(d_list > 90, 180 - d_list, d_list)
                d_mean = np.mean(new_d_list)
                # d_mean = 180-d_mean
            for key_idx, point in enumerate(new_key):
            # for ter_idx, t_point in enumerate(terminal):
                if point[0]==0:
                    continue
                d_y = point[0] - t_point[0]
                predicted_d_x = d_y/(math.tan(d_mean/180.0*math.pi)+0.001)
                predicted_x = predicted_d_x + t_point[1]

                if print_mode:
                    print("Target = {}".format(point))
                    print("Predectetd_Coord  = {} / {}".format(point[0], predicted_x))
                dist = abs(point[1] - predicted_x)
                if abs(point[1] - predicted_x) < 20* math.sqrt(abs(d_y/10))and abs(predicted_d_x) < 50  and abs(d_y)<80:
                    if print_mode:

                        print("FIND !!!")
                        print("T  = {}".format(point))
                        print("PRED = {}".format(t_point))
                        print("DIST = {}".format(dist))
                    if min_list[key_idx] > dist:
                        score_tensor[key_idx, :] = -100
                        min_list[key_idx] = dist
                        score_tensor[key_idx, ter_idx] = -100 + dist
                        new_terminal_tensor[key_idx]=1
                        find = True

            if not find and np.std(d_list) > 30:
                if print_mode:
                    print("Not Founded !!! {}".format(0))
                new_d_list = np.where(d_list < 90, 180 - d_list, d_list)
                d_mean = np.mean(new_d_list)
                # d_mean = 180-d_mean
            for key_idx, point in enumerate(new_key):
            # for ter_idx, t_point in enumerate(terminal):
                if point[0]==0:
                    continue
                d_y = point[0] - t_point[0]
                predicted_d_x = d_y/(math.tan(d_mean/180.0*math.pi)+0.001)
                predicted_x = predicted_d_x + t_point[1]
                if print_mode:
                    print("Target = {}".format(point))
                    print("Predectetd_Coord  = {} / {}".format(point[0], predicted_x))
                    print("PREDECTED _DX {}".format(predicted_d_x))
                dist = abs(point[1] - predicted_x)
                if abs(point[1] - predicted_x) < 20* math.sqrt(abs(d_y/10))and abs(predicted_d_x) < 50  and abs(d_y)<80:
                    if print_mode:

                        print("FIND !!!")
                        print("T  = {}".format(point))
                        print("PRED = {}".format(t_point))
                        print("DIST = {}".format(dist))
                    if min_list[key_idx] > dist:
                        score_tensor[key_idx, :] = -100
                        min_list[key_idx] = dist
                        score_tensor[key_idx, ter_idx] = -100 + dist
                        new_terminal_tensor[key_idx]=1
                        find = True
        # if not find:
        #     print("Key inx = {}".format(key_idx))
        #     print("lane_num= {}".format(lane_num))
        #     score_tensor[key_idx, lane_num] += 10
        #     lane_num+=1
        for idx, new_terminal in enumerate(new_terminal_tensor):
            if new_terminal < 1:
                score_tensor[idx, lane_num] += 10
                lane_num+=1


        max_torch, max_idx_torch = torch.max(score_tensor, dim = 0)

        max_torch = max_torch
        max_idx_torch = max_idx_torch
        # max_idx_torch = torch.argmax(score_tensor, dim = 1)
        # print("Max torch = {}".format(max_torch))
        # print("max_idx_torch torch = {}".format(max_idx_torch))
        temp = torch.where(max_torch>-100, max_idx_torch+1,0)
        # temp = torch.where(max_torch>-100, max_idx_torch+1,0).to(self.device)

        # print("TEMP Ori {}".format(temp))
        # print("TEMP NZ  {}".format(torch.nonzero(temp, as_tuple=True)))
        # print("TEMP Sli {}".format(temp))
        terminal[torch.nonzero(temp, as_tuple=True)] = new_key[temp[torch.nonzero(temp, as_tuple=True)]-1]
        # terminal_deg[torch.nonzero(temp, as_tuple=True)] = new_deg[temp[torch.nonzero(temp, as_tuple=True)]-1]
        return lane_num, terminal_deg

        #SITA 0.5 LEFT 0.7




# N. 검출
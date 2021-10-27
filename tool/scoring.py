from back_logic.evaluate import EDeval
import torch
import time
import os
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

#         os.system('clear')
        # print("----------------------------")
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
            
            
#             print("LANE DATA ---------------")
#             print(lane)

            cur_height_idx = 0 
            
            # end+1 = for height_sample number
            for height in range(h_start, h_end+1, h_step):
                
#                 print("Cur Idx = {} / {}".format(cur_height_idx, len(lane)))
                cur_height = lane[cur_height_idx][0]
                
                if height < cur_height and cur_height_idx == 0:
                    new_single_lane.append(-2)
                    continue
                if cur_height_idx == len(lane)-1 and height > cur_height:
                    new_single_lane.append(-2)
                    continue
                
                if cur_height < height:
                    while cur_height < height:
                        cur_height_idx +=1
#                         print(cur_height_idx)
#                         print("Cur Idx = {} / {}".format(cur_height_idx, len(lane)))
#                         print("Height = {} / {}".format(cur_height, height))
                        cur_height = lane[cur_height_idx][0]
                        if cur_height_idx == len(lane)-1:
                            break
#                             continue

                    # print("INDEX = {}".format(cur_height_idx))

                dx = lane[cur_height_idx][1] -lane[cur_height_idx-1][1] 
                dy = lane[cur_height_idx][0] -lane[cur_height_idx-1][0] 
    
                subY = height - lane[cur_height_idx-1][0] 
                subX = dx*subY/dy
    
                newX = int(subX + lane[cur_height_idx-1][1])
                new_single_lane.append(newX)                        
#             if len(new_single_lane)!=55:
#                 print("SDFSDFSDFSDF!!!!!!!!!!!!!!!!!!!")
#                 print(len(new_single_lane))
#                 time.sleep(100000)
#             print(len(new_single_lane))
            lane_list.append(new_single_lane)
        self.lane_list=lane_list
#         print("SELF LIST")
#         print(self.lane_list)
        return lane_list
    def prob2lane(self, img_idx, img_val, lane_start, lane_end, lane_step):
        width = img_idx.shape[1]
#         print("WIDTH!!!!!!!!!!!!!!!!!!!")
#         print(width)
#         prob_img.indices = prob_img.indices.to('cpu').numpy()
#         prob_img.values = prob_img.values.to('cpu').numpy()
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
#         print("IN PR")
        for lane in range(1,7):
            if self.lane_length[lane] >=2:
#             if True:
                self.lanes.append(lane_list[lane])
#                 print(lane_list[lane])
#         print("OUT PR")
                
#         print("ppppSELF LIST")
#         print(len(self.lanes))
            # self.lanes.append(lane_list[lane])
        return
#     def prob2lane(self, prob_img, prob_val, lane_start, lane_end, lane_step):

#         width = prob_img.shape[1]
        
#         pi = torch.squeeze(prob_img).to(torch.device('cpu')).detach().numpy()
#         pv = torch.squeeze(prob_val).to(torch.device('cpu')).detach().numpy()

#         lane_list=[[] for i in range(7)]
#         for ordinate in range(lane_start, lane_end, lane_step):
#             max_value = [-2 for i in range(0,7)]
#             max_idx = [-1 for i in range(0,7)]


#             for abscissa in range(0, width):
# #                    id = 1
#                 id = pi[ordinate][abscissa]
# #                    print(id)
# #                    print(pi.device)
# #                    print(pi[ordinate][abscissa].device)
# #                    print(pv.device)
# #                    print(pv[ordinate][abscissa].device)
#                 if id==0: continue
# #                 print(id)
                
#                 val = pv[ordinate][abscissa]
#                 if val > max_value[id]:
#                     max_value[id] = val
#                     max_idx[id] = abscissa
                
#             for id in range(1,7):
#                 if max_idx[id] is not -1:
#                     lane_list[id].append([ordinate, max_idx[id]])
#                     self.lane_length[id] +=1
# #                     print("!!!")

#         for lane in range(1,7):
#             if self.lane_length[lane] >=2:
#                 self.lanes.append(lane_list[lane])
#         print(len(self.lanes))
# #                 print(lane_list[lane])
# #         end = time.time()
# #         print(end - start)
#                 # self.lanes.append(lane_list[lane])
#         return
from back_logic.evaluate import EDeval

class Scoring():
    def __init__(self):

        self.imagePath=""
        self.outputPath=""
        self.lanes = []
        self.lane_length = [0 for i in range(7)]
        self.lane_list=[]
        return
    def getLanebyH_sample(self,  h_start, h_end, h_step):

        # print("----------------------------")
        lane_list = []
        for lane in self.lanes:
            new_single_lane=[]
            # print("LANE DATA ---------------")
            for node in lane:
                node[0] = int(node[0]/368*720)
                node[1] = int(node[1]/640*1280)
            # print(lane)

            cur_height_idx = 0 
            # print("Cur Idx = {} / {}".format(cur_height_idx, len(lane)))

            for height in range(h_start, h_end+1, h_step):

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
                        cur_height = lane[cur_height_idx][0]
                        if cur_height_idx == len(lane)-1:
                            continue
                            print("OUT----------")

                    # print("INDEX = {}".format(cur_height_idx))

                dx = lane[cur_height_idx][1] -lane[cur_height_idx-1][1] 
                dy = lane[cur_height_idx][0] -lane[cur_height_idx-1][0] 
    
                subY = height - lane[cur_height_idx-1][0] 
                subX = dx*subY/dy
    
                newX = int(subX + lane[cur_height_idx-1][1])
                new_single_lane.append(newX)         
                    

            # print(new_single_lane)
            lane_list.append(new_single_lane)
        self.lane_list=lane_list
        return lane_list
    
    
    def prob2lane(self, prob_img, lane_start, lane_end, lane_step):
        width = prob_img.indices.shape[1]

        lane_list=[[] for i in range(7)]
        for ordinate in range(lane_start, lane_end, lane_step):
            # print("ORDINATE {}".format(ordinate))
            max_value = [-2 for i in range(0,7)]
            max_idx = [-1 for i in range(0,7)]


            for abscissa in range(0, width):
                id = prob_img.indices[ordinate][abscissa].item()
                if id==0: continue
                max_value[id]
                val = prob_img.values[ordinate][abscissa].item()
                if val > max_value[id]:
                    max_value[id] = val
                    max_idx[id] = abscissa
            for id in range(1,7):
                if max_idx[id] is not -1:
                    lane_list[id].append([ordinate, max_idx[id]])
                    self.lane_length[id] +=1

        for lane in range(1,7):
            if self.lane_length[lane] >=2:
                self.lanes.append(lane_list[lane])
                print(lane_list[lane])

            # self.lanes.append(lane_list[lane])
        return
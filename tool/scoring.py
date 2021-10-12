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
                node[0] = int(node[0]/176*720)
                node[1] = int(node[1]/304*1280)
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
                    # print("Cur coord = {} {}".format(lane[cur_height_idx][0], lane[cur_height_idx][1]))
                    # print("Pre coord = {} {}".format(lane[cur_height_idx-1][0], lane[cur_height_idx-1][1]))
                    # print("Height = {}".format(height))
                    # print("Dx = {}".format(dx))
                    # print("Dy = {}".format(dy))
                    # print("H / C = {} / {} ".format(height, cur_height))
                    # new_single_lane.append(-2)
                    # cur_height_idx +=1
                    # cur_height = lane[cur_height_idx][0]
                    

            # print(new_single_lane)
            lane_list.append(new_single_lane)
        self.lane_list=lane_list
        return lane_list
    def prob2lane(self, prob_img, lane_start, lane_end, lane_step):
        width = prob_img.indices.shape[1]
        height = prob_img.indices.shape[0]
        # print(width)
        # print(height)
        # print(len(self.lanes))
        # return
        lane_list=[[] for i in range(7)]
        for ordinate in range(lane_start, lane_end, lane_step):
            # print("ORDINATE {}".format(ordinate))
            for lane in range(1,7):
                max_value = -2
                max_idx = -1
 
                for abscissa in range(0, width):
                    # print("ABS = {}".format(prob_img.values[ordinate][abscissa]))
                    # print(prob_img.indices[ordinate][abscissa].item())
                    if prob_img.indices[ordinate][abscissa].item() is not lane:
                        continue
                    # print("HERE")
                    if prob_img.values[ordinate][abscissa] > max_value:
                        max_value = prob_img.values[ordinate][abscissa]
                        max_idx = abscissa
                        # print("MAX CHANGED")
                if max_idx is not -1:
                    lane_list[lane].append([ordinate, max_idx])
                    self.lane_length[lane] +=1
                # else:
                #     lane_list[lane].append([,]])

        # for lane in range(1,7):
        #     print(self.lane_length[lane])
        # for lane in self.lanes:
        #     print(lane)

        for lane in range(1,7):
            if self.lane_length[lane] >=2:
                self.lanes.append(lane_list[lane])
        return
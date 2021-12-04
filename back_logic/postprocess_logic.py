import torch
import numpy as np
import time
class PostProcess_Logic:

    def __init__(self):
        self.device = torch.device('cpu')
        return
    def refine_lane(self, lane):
        for idx, node in enumerate(lane):
            if idx < 4 or 50 < idx:
                continue

    def interpolate_lane(self, lane):
        for idx, node in enumerate(lane):
            if idx < 4 or 50 < idx:
                continue
            if lane[idx-1] > 0 and lane[idx] < 0:
                d1 = lane[idx-1] - lane[idx-2]
                d2 = lane[idx-2] - lane[idx-3]
                new_x = d1*0.7 + d2*0.3 +lane[idx-1]
                if 50 < new_x and new_x < 1200:
                    # print("CHANGED {} to {}".format(lane[idx], new_x))
                    lane[idx] = d1*0.7 + d2*0.3 + lane[idx-1]
        return lane


    def merge(self, lane1, lane2):
        lane = np.where(lane1 < 0 , lane2, lane1)
        lane = self.interpolate_lane(lane)
        # for idx, node in enumerate(lane):

        #     if idx < 4 or 50 < idx:
        #         continue
        #     if lane[idx-1] > 0 and lane[idx] < 0:
        #         d1 = lane[idx-1] - lane[idx-2]
        #         d2 = lane[idx-2] - lane[idx-3]
        #         new_x = d1*0.7 + d2*0.3 +lane[idx-1]
        #         if 50 < new_x and new_x < 1200:
        #             print("CHANGED {} to {}".format(lane[idx], new_x))
        #             lane[idx] = d1*0.7 + d2*0.3 + lane[idx-1]

        return lane
    
    def mergeCheck(self, lane1, lane2):
        lane1_nonzero_tuple = np.where(lane1>0)[0]
        lane2_nonzero_tuple = np.where(lane2>0)[0]
        pre_big = True
        if len(lane1_nonzero_tuple)<2 and len(lane2_nonzero_tuple)<2:
            return False
        if len(lane1_nonzero_tuple) < len(lane2_nonzero_tuple):
            pre_big=False

        if pre_big:
            big_lane = np.copy(lane1)
            small_lane = np.copy(lane2)
        else:
            big_lane = np.copy(lane2)
            small_lane = np.copy(lane1)
        if len(np.where(big_lane>0)[0]) < 3:
            return False

        start_idx = np.where(big_lane>0)[0][0]
        end_idx = np.where(big_lane>0)[0][-1]

        small_nonzero = np.where(small_lane>0)[0]

        # print("BIG")
        # print(big_lane)

        while start_idx>0:
            big_lane[start_idx-1] = 2*big_lane[start_idx] - big_lane[start_idx+1]
            start_idx-=1
        while end_idx<lane1.shape[0]-1:
            big_lane[end_idx+1] = 2*big_lane[end_idx] - big_lane[end_idx-1]
            end_idx+=1
        sub = 0
        for idx, lane in enumerate(small_lane[small_nonzero]):
            sub += abs(lane - big_lane[small_nonzero[idx]])
        sub /=len(small_nonzero)

        if sub < 80:
        # if sub < 30:
            return True
        return False
    def mergeLane(self, lane_list):
        lane_np = np.array(lane_list)
        out_list = [0 for i in range(len(lane_np))]
        for idx1, master_lane in enumerate(lane_np):
            master_value_idx = np.where(master_lane>0, 1, 0)
            master_value_tuple = np.where(master_lane>0)[0]

            for idx2, branch_lane in enumerate(lane_np):
                if idx1>=idx2:
                    continue
                branch_value_idx = np.where(branch_lane>0, 1, 0)
                branch_value_tuple = np.where(branch_lane>0)[0]
                total_length = len(master_value_tuple) + len(branch_value_tuple)


                added_list = branch_value_idx+master_value_idx

                doubled_tuple = np.where(added_list>1)[0]
                solo_tuple = np.where(added_list==1)[0]
                doubleRatio1 = len(doubled_tuple)/len(master_value_tuple)
                doubleRatio2 = len(doubled_tuple)/len(branch_value_tuple)
                soloRatio1 = len(solo_tuple)/len(master_value_tuple)
                soloRatio2 = len(solo_tuple)/len(branch_value_tuple)
                # print(" DoubleRatio1 {}".format(doubleRatio1))
                # print(" DoubleRatio2 {}".format(doubleRatio2))
                # print(" SoubleRatio1 {}".format(soloRatio1))
                # print(" SoubleRatio2 {}".format(soloRatio2))
                # if soloRatio1 > 0.7 and soloRatio2 > 0.7:
                if soloRatio1 > 0.7 and soloRatio2 > 0.7 and not (doubleRatio1 > 0.7 and doubleRatio2 > 0.7):
                    # print("     Merge Check {} and {}".format(idx1, idx2))
                    # print(master_lane)
                    if self.mergeCheck(master_lane, branch_lane):

                        lane_np[idx1] = self.merge(master_lane, branch_lane)
                        out_list[idx2] = -1
        new_lane_list=[]

        for idx, master_lane in enumerate(lane_np):
            # print("IDX {} NUM {}".format(idx, len(np.where(master_lane>0)[0])))
            # print(master_lane)
            if out_list[idx]>-1 and len(np.where(master_lane>0)[0]) > 4:
            # if out_list[idx]>-1:
                new_lane_list.append(master_lane.tolist())
        # print("LEN {}".format(len(new_lane_list)))
        return new_lane_list

    def post_process(self, lane_list):
        post_process_start = time.time()
        print("LEN OF {}".format(len(lane_list)))
        print("LEN OF {}".format(len(lane_list)))
        new_lane_list = self.mergeLane(lane_list)
        re_lane=[]
        for idx, lane in enumerate(new_lane_list):
            # if len(np.where(np.array(lane)>0)[0]) > 5:
            if True:
                re_lane.append(lane)

        for idx, lane in enumerate(re_lane):
            new_lane_list[idx] = self.interpolate_lane(lane)

        post_process_end = time.time()
        # print("Post Process time {}".format(post_process_end - post_process_start))

        return new_lane_list
        return lane_list

if __name__=="__main__":
    pl = PostProcess_Logic()
    lane_list=np.array([[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 636, 621, 613, 606, 598, 584, 572, 562, 552, 542, 527, 520, 517, 488, 485, 487, 480, 465, 451, 440, 440, 427, 416, 409, 401, 388, 377, 370, 362, 350, 337, 330, 323, 312, 298, 291, 284, 273, 259, 252, 245, 235, 220, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 680, 680, 705, 723, 731, 745, 761, 776, 788, 798, 812, 827, 840, 840, 856, 875, 890, 904, 914, 924, 947, 956, 967, 981, 996, 1011, 1022, 1030, 1045, 1060, 1074, 1088, 1096, 1107, 1122, 1137, 1145, 1153, 1162, 1183, 1195, 1204, 1220, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 798, 834, 864, 907, 949, 991, 1048, 1108, 1126, 1155, 1203, 1239, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])
    pl.mergeLane(lane_list)
    

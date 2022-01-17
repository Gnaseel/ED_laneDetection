# from back_logic.anchor import *
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm, Normalize
class EDeval():
    def __init__(self):
        self.eval_list = []
        self.bad_path_list = []
        self.good_path_list = []
        return
    def save_JSON(self, lane_list, path_list):
        lane_checker = [0 for i in range(30)]
        output_list = []
        for idx in range(len(lane_list)):
            output = dict()
            output["h_samples"]=[i for i in range(160,720,10)]
            output["lanes"] = [lane for lane in lane_list[idx]]
            output["run_time"] = 1
            output["raw_file"]=path_list[idx]
            output_str = json.dumps(output)
            lane_num = len(lane_list[idx])

            if lane_num > 10:
                lane_num=10
            lane_checker[lane_num] +=1
            # if lane_num>5:
            #     continue
            output_list.append(output_str)

        with open('./back_logic/result_li.json','w') as file:
            # json_data = json.load(json_file)
            file.write('\n'.join(output_list))
        file.close()
        return
    
    def sort_list(self):
        self.eval_list.sort(key=lambda data : data.acc)
        return

    # Table of n*n real&predicted lane
    def get_lane_table(self, eval_list):
        heat_count_tensor = np.zeros((10,10), dtype=int)
        heat_acc_tensor = np.zeros((10,10))

        acc_arr = [10,10]
        for item in eval_list:
            heat_count_tensor[item.gt_lane, item.pred_lane] +=1
            heat_acc_tensor[item.gt_lane, item.pred_lane] +=item.acc
        heat_acc_tensor = heat_acc_tensor/(heat_count_tensor+0.0000000000000001)
        # print("HEAT TENSOR = {}".format(heat_count_tensor))
        # print("HEAT TENSOR = {}".format(heat_acc_tensor))
        np.save("data/count_tensor.npy", heat_count_tensor)
        np.save("data/heat_acc_tensor.npy", heat_acc_tensor)
        return
if __name__=="__main__":

    count = np.load("data/count_tensor.npy")
    acc = np.load("data/heat_acc_tensor.npy")
    # print(count)
    # print(acc)
    count = count[0:6, 0:6]
    acc = acc[0:6, 0:6]
    l = count[0,0]+count[1,1]+count[2,2]+count[3,3]+count[4,4]+count[5,5]
    print("Collect LANE  {}".format(l))
    data_df = pd.DataFrame(count)
    data_df_acc = pd.DataFrame(acc)
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,2,1)
    ax.set_title("Lane count Heatmap")

    sns.heatmap(data_df,cmap='Reds', annot=True, ax=ax, fmt = "", norm=LogNorm())
    ax.set_ylabel("GrountTruth")
    ax.set_xlabel("predicted")
    ax.xaxis.set_ticks_position("top")

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("ACC Heatmap")
    sns.heatmap(data_df_acc,cmap='Reds', annot=True, ax=ax2, fmt = ".2%")

    plt.savefig('data/lane_count_heat_map_90.png')
    # def getH_sample_all(self, anchor_tensor, h_start, h_end, h_interval):
    #     lane_tensor=[]
    #     for anchorlist in anchor_tensor:
    #         lane_list=[[] for i in range(len(anchorlist.list))]
    #         mul = 4

    #         for h in range(h_start, h_end, h_interval):
    #             mulH = h//mul
    #             for idx, anchor in enumerate(anchorlist.list):
    #                 val = -2
    #                 for node_idx, node in enumerate(anchor.nodelist):
    #                     if node_idx+1 == len(anchor.nodelist):
    #                         break
                        
    #                     # print("H = {}, nodeY = {}, nextY = {}".format(h, node.y*mul,anchor.nodelist[node_idx+1].y*mul))

    #                     if mulH < node.y  and  anchor.nodelist[node_idx+1].y < mulH:
    #                         # print("NODE = {}".format(node_idx))
    #                         val = self.getH(node, anchor.nodelist[node_idx+1], mulH)*mul
    #                         break
    #                     elif mulH ==node.y:
    #                         val = node.x*mul
    #                 # print("IDX = {}".format(idx))
    #                 lane_list[idx].append(int(val))
    #         lane_tensor.append(lane_list)
    #         # self.printLaneData(lane_list)
    #     return lane_tensor
    # def getH_sample(self, anchorlist, h_start, h_end, h_interval):
    #     lane_list=[[] for i in range(len(anchorlist.list))]

    #     mul = 4

    #     for h in range(h_start, h_end, h_interval):
    #         mulH = h//mul
    #         for idx, anchor in enumerate(anchorlist.list):
    #             val = -2
    #             for node_idx, node in enumerate(anchor.nodelist):
    #                 if node_idx+1 == len(anchor.nodelist):
    #                     break
                    
    #                 # print("H = {}, nodeY = {}, nextY = {}".format(h, node.y*mul,anchor.nodelist[node_idx+1].y*mul))

    #                 if mulH < node.y  and  anchor.nodelist[node_idx+1].y < mulH:
    #                     print("NODE = {}".format(node_idx))
    #                     val = self.getH(node, anchor.nodelist[node_idx+1], mulH)*mul
    #                     break
    #                 elif mulH ==node.y:
    #                     val = node.x*mul
    #             # print("IDX = {}".format(idx))
    #             lane_list[idx].append(int(val))
    #     self.printLaneData(lane_list)
    #     return lane_list
    # def getH(self, node1, node2, height):
    #     tilt = (node1.x - node2.x)/(node1.y - node2.y+0.000001)
    #     dely = height - node2.y
    #     val = dely*tilt + node2.x

    #     # print("{} {} /// {} {}".format(node1.x, node1.y, node2.x, node2.y))
    #     # print("TILT {}".format(tilt))
    #     # return tilt*dely*4
    #     return val
    # def printLaneData(self, lanes):
    #     for idx, lane in enumerate(lanes):
    #         print("Lane {}".format(idx))
    #         for cor in lane:
    #             print("{} ".format(cor), end=' ')
    #     return
    
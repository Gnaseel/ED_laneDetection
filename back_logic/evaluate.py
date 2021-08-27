from back_logic.anchor import *
import json

class EDeval():
    def __init__(self):
        return
    def getH_sample_all(self, anchor_tensor, h_start, h_end, h_interval):
        lane_tensor=[]
        for anchorlist in anchor_tensor:
            lane_list=[[] for i in range(len(anchorlist.list))]
            mul = 4

            for h in range(h_start, h_end, h_interval):
                mulH = h//mul
                for idx, anchor in enumerate(anchorlist.list):
                    val = -2
                    for node_idx, node in enumerate(anchor.nodelist):
                        if node_idx+1 == len(anchor.nodelist):
                            break
                        
                        # print("H = {}, nodeY = {}, nextY = {}".format(h, node.y*mul,anchor.nodelist[node_idx+1].y*mul))

                        if mulH < node.y  and  anchor.nodelist[node_idx+1].y < mulH:
                            # print("NODE = {}".format(node_idx))
                            val = self.getH(node, anchor.nodelist[node_idx+1], mulH)*mul
                            break
                        elif mulH ==node.y:
                            val = node.x*mul
                    # print("IDX = {}".format(idx))
                    lane_list[idx].append(int(val))
            lane_tensor.append(lane_list)
            # self.printLaneData(lane_list)
            # self.save_JSON(lane_list)
        return lane_tensor

    def getH_sample(self, anchorlist, h_start, h_end, h_interval):
        lane_list=[[] for i in range(len(anchorlist.list))]

        mul = 4

        for h in range(h_start, h_end, h_interval):
            mulH = h//mul
            for idx, anchor in enumerate(anchorlist.list):
                val = -2
                for node_idx, node in enumerate(anchor.nodelist):
                    if node_idx+1 == len(anchor.nodelist):
                        break
                    
                    # print("H = {}, nodeY = {}, nextY = {}".format(h, node.y*mul,anchor.nodelist[node_idx+1].y*mul))

                    if mulH < node.y  and  anchor.nodelist[node_idx+1].y < mulH:
                        print("NODE = {}".format(node_idx))
                        val = self.getH(node, anchor.nodelist[node_idx+1], mulH)*mul
                        break
                    elif mulH ==node.y:
                        val = node.x*mul
                # print("IDX = {}".format(idx))
                lane_list[idx].append(int(val))
        self.printLaneData(lane_list)
        # self.save_JSON(lane_list)
        return lane_list
    def getH(self, node1, node2, height):
        tilt = (node1.x - node2.x)/(node1.y - node2.y+0.000001)
        dely = height - node2.y
        val = dely*tilt + node2.x

        # print("{} {} /// {} {}".format(node1.x, node1.y, node2.x, node2.y))
        # print("TILT {}".format(tilt))
        # return tilt*dely*4
        return val
    
    def printLaneData(self, lanes):
        for idx, lane in enumerate(lanes):
            print("Lane {}".format(idx))
            for cor in lane:
                print("{} ".format(cor), end=' ')
        return
    def save_JSON(self, lane_list, path_list):
        file_name="abcde"
        # with open('./back_logic/test_tasks_0627.json') as json_file:
        # list=['foo', {'bar': ('baz', None, 1.0, 2)}]
        output_list = []
        for idx in range(len(lane_list)):
            output = dict()
            output["h_sample"]=[i for i in range(160,720,10)]
            output["lanes"] = [lane for lane in lane_list[idx]]
            output["run_time"] = 1
            output["raw_file"]=path_list[idx]
            output_str = json.dumps(output)
            output_list.append(output_str)
        with open('./back_logic/test.json','w') as file:
            # json_data = json.load(json_file)
            file.write('\n'.join(output_list))
        # print(output_list[0])
        # print(output_list[1])
        return
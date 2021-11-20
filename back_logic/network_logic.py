import time
# from tool.inference import Inference
from tool.scoring import Scoring
import torch.nn.functional as F
import torch
import math
import numpy as np

class Network_Logic:
    def __init__(self):
        self.device = torch.device('cpu')
        self.printTimer=True
        return
    def getDeg(self, delta_up, delta_right):
        x1 = int(delta_right_image[i,j])*horizone_direction + j
        y1 = i
        x2 = j
        y2 = int(delta_up_image[i,j])*vertical_direction + i
        m = (y2-y1)/(x2-x1+0.00001)
        a = m
        b = -1
        c = y1 - m*x1
        newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
        newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)

        return
    def getDegmap_tensor(self, delta_up_image, delta_right_image):
        st=time.time()
        img_height = delta_up_image.shape[0]
        img_width = delta_up_image.shape[1]
        delta = 5
        arrow_size=4
        min_threshold = 5
        threshold = 35
        deg_padding = 3
        deg_image = np.zeros((img_height+deg_padding*2,img_width+deg_padding*2))

        print("ORI {}".format(delta_up_image.shape))
        print("ORI {}".format(type(delta_up_image)))
        delta_up_padding_tensor = F.pad(delta_up_image, (0,0,5,0), value=100)[:-5,:]
        delta_right_padding_tensor = F.pad(delta_right_image, (5,0,0,0), value=100)[:,:-5]

        print("PAD {}".format(delta_up_padding_tensor.shape))

        # print("ORI {}".format(delta_up_image[0,115]))
        # print("PAD {}".format(delta_up_padding_tensor[3,115]))
        # delta_up_rel_tensor = torch.zeros((img_height,img_width))
        # delta_right_rel_tensor = torch.zeros((img_height,img_width))

        delta_up_rel_tensor = torch.where(delta_up_image < delta_up_padding_tensor, delta_up_image, delta_up_image*-1)
        delta_right_rel_tensor = torch.where(delta_right_image < delta_right_padding_tensor, delta_right_image, delta_right_image*-1)
        pre = time.time()
        for i in range(90, delta_up_image.shape[0]-10, delta):
            for j in range(10, delta_up_image.shape[1]-10, delta):
                
                # horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                # vertical_direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
                # startPoint = (j, i)

                x1 = int(delta_right_rel_tensor[i,j]) + j
                y1 = i
                x2 = j
                y2 = int(delta_up_rel_tensor[i,j]) + i
                m = (y2-y1)/(x2-x1+0.00001)
                a = m
                b = -1
                c = y1 - m*x1
                newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
                newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)

                # dist = abs(int(delta_right_image[i,j])) + abs(int(delta_up_image[i,j]))

                # deg = str( (delta_up_image[i,j]*vertical_direction) / (delta_right_image[i,j]*horizone_direction))[0:4]
                # deg = str( (delta_up_image[i,j]*vertical_direction))[0:2] + " / "+str(delta_right_image[i,j]*horizone_direction)[0:2]
                # deg=str(m)[0:5]

                deg= (math.atan2(y2-y1, x2-x1)*180.0/math.pi)
                while deg<0:
                    deg+=180
                while deg>180:
                    deg-=180
                if int(newy) < img_height and int(newx) < img_width and int(newy) > 0 and int(newx) > 0:
                    deg_image[int(newy)+deg_padding,int(newx)+deg_padding] = deg
        return deg_image

    def getDegmap(self, delta_up_image, delta_right_image):
        img_height = delta_up_image.shape[0]
        img_width = delta_up_image.shape[1]
        delta = 5
        arrow_size=4
        min_threshold = 5
        threshold = 35
        deg_padding = 3
        deg_image = np.zeros((img_height+deg_padding*2,img_width+deg_padding*2))

        for i in range(90, delta_up_image.shape[0]-10, delta):
            for j in range(10, delta_up_image.shape[1]-10, delta):
                
                horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                vertical_direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1
                startPoint = (j, i)

                x1 = int(delta_right_image[i,j])*horizone_direction + j
                y1 = i
                x2 = j
                y2 = int(delta_up_image[i,j])*vertical_direction + i
                m = (y2-y1)/(x2-x1+0.00001)
                a = m
                b = -1
                c = y1 - m*x1
                newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
                newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)

                dist = abs(int(delta_right_image[i,j])) + abs(int(delta_up_image[i,j]))

                # deg = str( (delta_up_image[i,j]*vertical_direction) / (delta_right_image[i,j]*horizone_direction))[0:4]
                # deg = str( (delta_up_image[i,j]*vertical_direction))[0:2] + " / "+str(delta_right_image[i,j]*horizone_direction)[0:2]
                # deg=str(m)[0:5]

                deg= (math.atan2(y2-y1, x2-x1)*180.0/math.pi)
                while deg<0:
                    deg+=180
                while deg>180:
                    deg-=180
                if int(newy) < img_height and int(newx) < img_width and int(newy) > 0 and int(newx) > 0:
                    deg_image[int(newy)+deg_padding,int(newx)+deg_padding] = deg
        return deg_image

    def inference_np2np_instance(self, image, model):
            # input_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).to(self.device).permute(0,3,1,2).float()
            input_tensor = torch.unsqueeze(torch.from_numpy(image).to(self.device), dim=0).permute(0,3,1,2).float()
            output_tensor = model(input_tensor)
            # print("Tensor {}".format(output_tensor.shape))
            output = output_tensor[0].permute(1,2,0).cpu().detach().numpy()
            return output

    def inference_np2tensor_instance(self, image, model):
            # input_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).to(self.device).permute(0,3,1,2).float()
            input_tensor = torch.unsqueeze(torch.from_numpy(image).to(self.device), dim=0).permute(0,3,1,2).float()
            output_tensor = model(input_tensor)
            # print("Tensor {}".format(output_tensor.shape))
            output = output_tensor[0].permute(1,2,0)
            return output

    def getScoreInstance_deg(self, path, out_heat, out_delta):
        height_delta=15
        terminal_size = 30
        start_time = time.time()
        out_heat = out_heat[:,:,1]
        deg_padding = 3
        deg_image = np.zeros((out_heat.shape[0]+deg_padding*2,out_heat.shape[1]+deg_padding*2))


        delta_right_image = out_delta[:,:,0]
        delta_up_image = out_delta[:,:,1]
        # deg_image = self.getDegmap_tensor(delta_up_image, delta_right_image)

        deg_image = self.getDegmap(delta_up_image, delta_right_image)

        # get_model_output = time.time()
        get_degMap_output = time.time()
        # print("Get DegMap Time = {}".format(get_degMap_output-start_time))

        score = Scoring()
        score.device = self.device     
        lane_tensor = torch.zeros([terminal_size,60,2]).to(self.device)
        # print(path)
        # return
        key_height =  170
        first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)
        while first_key_tensor.shape[0]==0 or key_height>300:
            key_height += 10
            first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)
        key_height = 170
        while first_key_tensor.shape[0]==0 or key_height<100:
            key_height -= 10
            first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)

        if first_key_tensor.shape[0]==0:
            score.tensor2lane(lane_tensor)
            score.getLanebyH_sample(160, 710, 10)
            return score

        terminal_tensor = torch.zeros([terminal_size,2], dtype = torch.long).to(self.device)
        terminal_deg_tensor = torch.zeros([terminal_size], dtype = torch.long).to(self.device)

        # getKey_time = time.time()
        # print("Get Key Time = {}".format(getKey_time-get_degMap_output))


        # terminal = 
        lt_idx = 7
        lane_tensor[0:first_key_tensor.shape[0], lt_idx] = first_key_tensor
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        lane_num=first_key_tensor.shape[0]

        # print("KeyPoint  {}".format(first_key_tensor))
        # print("Terminal = {}".format(terminal_tensor))
        #---------------- HEAT to INS ---------------------------------------------------------------
        for height in range(key_height+10, 330, height_delta):
            new_key =  score.getLocalMaxima_heatmap_re(out_heat, height)
            # print("-----------------------------------------------------------------------------------")
            # print("New Key = {}".format(new_key))
            lane_num, terminal_deg_tensor = score.chainKey2(new_key, terminal_tensor, terminal_deg_tensor, deg_image, lane_num)
            lt_idx +=1
            lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:4,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        # print("--------------IUDUDUDUDUUDUDUD")
        lt_idx=7
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        # print("New Terminal = {}".format(terminal_tensor))
        terminal_deg_tensor = torch.zeros([terminal_size], dtype = torch.long).to(self.device)
        
        for height in range(key_height-10, 90, -height_delta):
            new_key =  score.getLocalMaxima_heatmap_re(out_heat, height)
            # print("New Key = {}".format(new_key))
            lane_num, terminal_deg_tensor = score.chainKey2(new_key, terminal_tensor, terminal_deg_tensor, deg_image, lane_num)
            lt_idx -=1
            lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:3,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        get_pre_lane = time.time()
        # print("Get Pre lane Time = {}".format(get_pre_lane - get_degMap_output))
        # lane_tensor[:,0] *= (720.0/368.0)
        # lane_tensor[:,1] *= (720.0/368.0)
        score.tensor2lane(lane_tensor)
        # return
        score.getLanebyH_sample_deg(160, 710, 10)
        # score.getLanebyH_sample_deg(160, 710, 10)
        get_gt_lane = time.time()

        return score

    def getKeypoint(self, out_heat, height_start=90, height_end=330):
        score = Scoring()
        score.device = self.device
        key_list = []
        for height in range(height_start, height_end, 10):
            out_tensor = score.getLocalMaxima_heatmap_re(out_heat,height)
            for point in out_tensor:
                key_list.append([point[0], point[1]])
        return key_list
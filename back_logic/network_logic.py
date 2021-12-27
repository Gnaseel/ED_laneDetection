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

    def getDegmap(self, delta_up_image, delta_right_image, heat_map):
        heat_map = heat_map.cpu()
        img_height = delta_up_image.shape[0]
        img_width = delta_up_image.shape[1]
        # delta = 1
        delta = 5
        arrow_size=4
        min_threshold = 5
        threshold = 35
        deg_padding = 3
        deg_image = np.zeros((img_height+deg_padding*2,img_width+deg_padding*2))

        for i in range(90, delta_up_image.shape[0]-10, delta):
            for j in range(10, delta_up_image.shape[1]-10, delta):
                
                # horizone_direction= -1 if delta_right_image[i,j+3] + delta_right_image[i,j+2] + delta_right_image[i,j+1] > delta_right_image[i,j-3] + delta_right_image[i,j-2] + delta_right_image[i,j-1] else 1
                # vertical_direction= -1 if delta_up_image[i+3,j] + delta_up_image[i+2,j] + delta_up_image[i+1,j] > delta_up_image[i-3,j] + delta_up_image[i-2,j] + delta_up_image[i-1,j] else 1

                delta_right_val = int(delta_right_image[i,j])
                delta_up_val = int(delta_up_image[i,j])


                # if img_width <= j + delta_right_val:
                #     horizone_direction = -1
                # elif j - delta_right_val < 0:
                #     horizone_direction = 1
                # elif heat_map[i,j + delta_right_val] > heat_map[i,j -  delta_right_val]:
                #     horizone_direction = 1
                # else:
                #     horizone_direction = -1

                # if img_height <= i + delta_up_val:
                #     vertical_direction = -1
                # elif i - delta_up_val < 0:
                #     vertical_direction = 1
                # elif heat_map[i + delta_up_val, j] > heat_map[i - delta_up_val, j]:
                #     vertical_direction = 1
                # else:
                #     vertical_direction = -1
                # horizone_direction= -1 if delta_right_image[i,j+3] + delta_right_image[i,j+2] + delta_right_image[i,j+1] > delta_right_image[i,j-3] + delta_right_image[i,j-2] + delta_right_image[i,j-1] else 1
                # vertical_direction= -1 if delta_up_image[i+3,j] + delta_up_image[i+2,j] + delta_up_image[i+1,j] > delta_up_image[i-3,j] + delta_up_image[i-2,j] + delta_up_image[i-1,j] else 1
                
                horizone_direction= -1 if delta_right_image[i,j+3] > delta_right_image[i,j-3] else 1
                vertical_direction= -1 if delta_up_image[i+3,j] > delta_up_image[i-3,j] else 1

                # startPoint = (j, i)

                x1 = int(delta_right_val)*horizone_direction + j
                y1 = i
                x2 = j
                y2 = int(delta_up_val)*vertical_direction + i
                m = (y2-y1)/(x2-x1+0.00001)
                a = m
                b = -1
                c = y1 - m*x1
                newx = (b*(b*j-a*i)-a*c)/(a**2+b**2)
                newy = (a*(-b*j+a*i)-b*c)/(a**2+b**2)

                dist = abs(int(delta_right_image[i,j])) + abs(int(delta_up_val))


                deg= (math.atan2(y2-y1, x2-x1)*180.0/math.pi)
                if delta_right_image[i,j]<3 or delta_up_val<3:
                    deg=0
                while deg<0:
                    deg+=180
                while deg>180:
                    deg-=180
                if int(newy) < img_height and int(newx) < img_width and int(newy) > 0 and int(newx) > 0:
                    # deg_image[i+deg_padding,j+deg_padding] = deg
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
            # print("Tensor {}".format(output_tensor.get_device()))
            # print("Tensor {}".format(input_tensor.get_device()))
            output = output_tensor[0].permute(1,2,0)
            return output

    def getScoreInstance_deg(self, path, out_heat, out_delta):
        print_mode = False
        # print("PATH {}".format(path))
        height_delta=7
        terminal_size = 50
        start_time = time.time()
        out_heat = out_heat[:,:,1]
        deg_padding = 3
        deg_image = np.zeros((out_heat.shape[0]+deg_padding*2,out_heat.shape[1]+deg_padding*2))


        delta_right_image = out_delta[:,:,0]
        delta_up_image = out_delta[:,:,1]
        # deg_image = self.getDegmap_tensor(delta_up_image, delta_right_image)

        deg_image = self.getDegmap(delta_up_image, delta_right_image, out_heat)

        # get_model_output = time.time()
        get_degMap_output = time.time()

        score = Scoring()
        score.device = self.device     
        lane_tensor = torch.zeros([terminal_size,60,2]).to(self.device)
        # print(path)
        # return
        key_height =  170
        # first_key_tensor = self.getLocalMaxima_heatmap(delta_right_image, key_height, threshold = 1.5,reverse=True).to(self.device)
        first_key_tensor = self.getLocalMaxima_heatmap(out_heat, key_height, reverse=False).to(self.device)
        # first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)
        while first_key_tensor.shape[0]==0 and key_height<300:
            key_height += 10
            # first_key_tensor = self.getLocalMaxima_heatmap(delta_right_image, key_height, threshold = 1.5,reverse=True).to(self.device)
            # first_key_tensor = self.getLocalMaxima_heatmap(out_heat, key_height, reverse=False).to(self.device)
            first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)
        key_height = 170
        while first_key_tensor.shape[0]==0 and key_height>100:
            key_height -= 10
            # first_key_tensor = self.getLocalMaxima_heatmap(delta_right_image, key_height, threshold = 1.5,reverse=True).to(self.device)
            # first_key_tensor = self.getLocalMaxima_heatmap(out_heat, key_height, reverse=False).to(self.device)
            first_key_tensor = score.getLocalMaxima_heatmap_re(out_heat, key_height).to(self.device)

        # print("Key height {}".format(key_height))

        if first_key_tensor.shape[0]==0:
            score.tensor2lane(lane_tensor)
            score.getLanebyH_sample(160, 710, 10)
            return score

        terminal_tensor = torch.zeros([terminal_size,2], dtype = torch.long)
        # terminal_tensor = torch.zeros([terminal_size,2], dtype = torch.long).to(self.device)
        terminal_deg_tensor = torch.zeros([terminal_size], dtype = torch.long)
        # terminal_deg_tensor = torch.zeros([terminal_size], dtype = torch.long).to(self.device)

        # getKey_time = time.time()
        # print("Get Key Time = {}".format(getKey_time-get_degMap_output))


        # terminal = 
        lt_idx = 25
        lane_tensor[0:first_key_tensor.shape[0], lt_idx] = first_key_tensor
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        lane_num=first_key_tensor.shape[0]

        # print("KeyPoint  {}".format(first_key_tensor))
        # print("Terminal = {}".format(terminal_tensor))

        #---------------- HEAT to INS ---------------------------------------------------------------
        for height in range(key_height+10, 350, height_delta):
            # t1 = time.time()
            
            new_key = self.getLocalMaxima_deltamap(delta_right_image, height)
            # new_key = self.getLocalMaxima_heatmap(out_heat, height, reverse=False)
            # new_key =  score.getLocalMaxima_heatmap_re(out_heat, height)
            # t2 = time.time()
            # print("-----------------------------------------------------------------------------------")
            # print("New Key = {}".format(new_key))
            lane_num, terminal_deg_tensor = score.chainKey2(new_key, terminal_tensor, terminal_deg_tensor, deg_image, lane_num)
            # t3 = time.time()
            lt_idx +=1
            lane_tensor[:,lt_idx] = terminal_tensor
            # t4 = time.time()
            # print("         TIME {} {} {}".format(t4-t3, t3-t2, t2-t1))
            # lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:4,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        # print("--------------IUDUDUDUDUUDUDUD")
        lt_idx=25
        terminal_tensor[0:first_key_tensor.shape[0]] = first_key_tensor
        # print("New Terminal = {}".format(terminal_tensor))
        terminal_deg_tensor = torch.zeros([terminal_size], dtype = torch.long).to(self.device)
        
        for height in range(key_height-10, 80, -height_delta):

            new_key = self.getLocalMaxima_deltamap(delta_right_image, height)
            # new_key = self.getLocalMaxima_heatmap(out_heat, height, reverse=False)
            # new_key =  score.getLocalMaxima_heatmap_re(out_heat, height)

            # print("New Key = {}".format(new_key))
            lane_num, terminal_deg_tensor = score.chainKey2(new_key, terminal_tensor, terminal_deg_tensor, deg_image, lane_num)
            lt_idx -=1
            lane_tensor[:,lt_idx] = terminal_tensor
            # lane_tensor[:,lt_idx] = terminal_tensor.clone().detach().to(self.device)
            # print("New Terminal = {}".format(terminal_tensor))
            # print("Lane tensor = {}".format(lane_tensor[0:3,5:12]))
            # return
        # print("Lane tensor = {}".format(lane_tensor[0:5]))
        get_pre_lane = time.time()
        # lane_tensor[:,0] *= (720.0/368.0)
        # lane_tensor[:,1] *= (720.0/368.0)
        score.tensor2lane(lane_tensor)
        # return
        score.getLanebyH_sample_deg(160, 710, 10)
        # score.getLanebyH_sample_deg(160, 710, 10)
        get_gt_lane = time.time()
        if print_mode:
            print("     Get DegMap Time = {}".format(get_degMap_output-start_time))
            print("     Get Pre lane Time = {}".format(get_pre_lane - get_degMap_output))
            print("     Tensor to lane Time = {}".format(get_gt_lane - get_pre_lane))


        return score

    def getKeypoint(self, out_heat, height_start=90, height_end=330, threshold = -0.5, reverse = False ):
        # score = Scoring()
        # score.device = self.device
        key_list = []
        for height in range(height_start, height_end, 10):
            # out_tensor = score.getLocalMaxima_heatmap_re(out_heat,height)
            out_tensor = self.getLocalMaxima_heatmap(out_heat,height, threshold=threshold, reverse=reverse)
            for point in out_tensor:
                key_list.append([point[0], point[1]])
        return key_list

    def getLocalMaxima_deltamap(self, img_tensor, height_val = 170, threshold = 10 ):
        # img_tensor = torch.squeeze(img_tensor, dim=0)[1]
        # print("--------- Get Local Maxima  {}".format(img_tensor.shape))
        # st = time.time()
        # for width_tensor in img_tensor:
        width_tensor = img_tensor[height_val]
        # print("WidthTensor shape {}".format(width_tensor.shape))

        local_maxima = torch.empty(0, dtype=torch.int64)
        last=10
        last_idx=-20
        # if reverse:
        #     last = 100

        for abscissa in range(0, width_tensor.shape[0], 5):

            # print(width_tensor[abscissa].item())
            abscissa_item = width_tensor[abscissa].item()
            if  abscissa_item < threshold and ( 0 < abscissa + abscissa_item and abscissa + abscissa_item < width_tensor.shape[0]):
                if abscissa > last_idx + 15 or local_maxima.shape[0]==0:
                    local_maxima = torch.cat([local_maxima, torch.tensor([[height_val, int(abscissa + abscissa_item)]])])
                elif abscissa_item < last:
                    local_maxima[-1,1] = abscissa + abscissa_item
                else:
                    continue
                last = abscissa_item
                last_idx = abscissa
        # print("SHAPE {}".format(local_maxima.shape))
        return local_maxima

    def getLocalMaxima_heatmap(self, img_tensor, height_val = 170, threshold = -0.5, reverse = False ):
        # img_tensor = torch.squeeze(img_tensor, dim=0)[1]
        # print("--------- Get Local Maxima  {}".format(img_tensor.shape))
        # st = time.time()
        # for width_tensor in img_tensor:
        width_tensor = img_tensor[height_val]
        # print("WidthTensor shape {}".format(width_tensor.shape))

        local_maxima = torch.empty(0, dtype=torch.int64)
        last=0
        last_idx=0
        for abscissa in range(0, width_tensor.shape[0], 5):

            # print(width_tensor[abscissa].item())
            abscissa_item = width_tensor[abscissa].item()
            if  abscissa_item > threshold:
                if abscissa > last_idx + 15 or local_maxima.shape[0]==0:
                    local_maxima = torch.cat([local_maxima, torch.tensor([[height_val, abscissa]])])
                elif abscissa_item > last:
                    local_maxima[-1,1] = abscissa
                else:
                    continue
                last = abscissa_item
                last_idx = abscissa
        #             print("LOCAL {}".format(local_maxima))

        # print("-----------------------LOCAL {}".format(local_maxima))
        return local_maxima
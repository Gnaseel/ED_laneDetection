import math

import numpy as np
import torch

class delta_distance():
    def __init__(self):
        self.target_tensor=None
        self.device=None
        return
    def setDevice(self, device):
        self.device = device
    def getDeltaDist(self,y, x):
        dist = -9999
        for i in range(self.target_tensor.shape[2]):
            data = self.target_tensor[y][i]
            if data == 0:
                continue
            dist = data-x
        return
    def getDeltaMap(self, batch_image):
        # reTensor = torch.FloatTensor([i for i in range(7 *6* 5)]).reshape(7, 6, 5)
        # print("----------In Delta Map--------")
        dim1 = batch_image.shape[0]
        dim2 = batch_image.shape[1]
        dim3 = batch_image.shape[2]

        batched_width_list=[]
        batched_exist_list=[]
        for image in batch_image:
            # print(image.shape)
            img_lane=(image==1).nonzero().to(self.device)
            # img_lane=(image==1)
            # print("image*--------------------")
            # print(image)
            # print("Lane*--------------------")
            # print(img_lane)
            
            width_list=[]
            exist_list=[]
            for idx, height in enumerate(image):
                the=(img_lane[:,0]==idx).nonzero().to(self.device)

                if the.shape[0]==0:
                    indi = torch.zeros(height.shape[0]).to(self.device)
                    # continue
                else:
                    the = torch.squeeze(the)
                    # print(the)
                    # print("Idx    Shape : {}".format(the.shape))
                    # print("Matrix Shape : {}".format(img_lane.shape))
                    ga = torch.unsqueeze(torch.index_select(img_lane, 0, the)[:,1],1)
                    # print(ga)
                    # for i in 
                    dist_tensor = torch.arange(height.shape[0]* ga.shape[0]).reshape(ga.shape[0], height.shape[0]).to(self.device)%height.shape[0]
                    # print(dist_tensor.shape)
                    dist_tensor = ga - dist_tensor

                    # print(torch.min(torch.abs(dist_tensor),0))
                    exist_list.append(idx)
                    indi = torch.min(torch.abs(dist_tensor),0).values
                width_list.append(indi)

            new_lane_img = torch.stack(width_list)
            new_exist_img = torch.Tensor(exist_list).type(torch.int64)
            # print(exist_list)
            # print(new_exist_img)
            # return
            batched_width_list.append(new_lane_img)
            batched_exist_list.append(new_exist_img)
                
        batched_dist_tensor = torch.stack(batched_width_list)
        # batched_exist_tensor = torch.stack(batched_exist_list)
        # print(batched_width_list)
        # print(batched_exist_list)
        return batched_dist_tensor, batched_exist_list

def test1():
     # t= torch.zeros((3, 5, 10))
    dum4 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum5 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum1 = torch.Tensor([1,0,0,0,0,0,0,0,0,1])
    dum2 = torch.Tensor([0,0,1,0,0,0,0,1,0,0])
    dum3 = torch.Tensor([0,0,0,0,1,1,0,0,0,0])
    t1 = torch.stack([dum4, dum5, dum1,dum2,dum3])
    dum4 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum5 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum1 = torch.Tensor([1,0,0,0,0,0,0,0,0,1])
    dum2 = torch.Tensor([0,0,1,0,0,0,0,1,0,0])
    dum3 = torch.Tensor([0,0,0,0,1,1,0,0,0,0])
    t2 = torch.stack([dum4, dum5, dum1,dum2,dum3])
    dum4 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum5 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum1 = torch.Tensor([1,0,0,0,0,0,0,0,0,1])
    dum2 = torch.Tensor([0,0,1,0,0,0,0,1,0,0])
    dum3 = torch.Tensor([0,0,0,0,1,0,1,0,0,0])
    t3 = torch.stack([dum5, dum1,dum2,dum3, dum4])
    
    # t = torch.stack(dum2)
    t = torch.stack([t1, t2, t3])
    # print(t.shape)
    # for bat in t:
    #     print(bat.shape)
    d=delta_distance()
    d.getDeltaMap(t)
# def test():
#     for i in range(3):
#         g = 5
#     print(g)
def test2():
    mytor = torch.arange(30).reshape(2,3,5)
    print("Input Torch {}".format(mytor))
    indices = torch.Tensor([0]).type(torch.int64)

    newtor = torch.index_select(mytor,2, indices)
    print("Output Torch {}".format(newtor))

    return
if __name__ =="__main__":
    # test()

    
    print("MAIN")
    test1()
    # test2()
   




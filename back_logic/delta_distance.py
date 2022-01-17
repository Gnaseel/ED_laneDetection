import math

import numpy as np
import torch
import time
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
    def getDeltaUpMap(self, batch_image, delta_height):
        batched_width_list=[]
        batched_exist_list=[]
        for image in batch_image:
            # print(image.shape)
            segmented_point=(image==1).nonzero().to(self.device)
            # segmented_point -> (N*coord) 2D tensor ... dim1 = number of segmented points, dim2 = point coord

            # print("image*--------------------")
            # print(image)
            # print(image.shape[0])
            # return

            # print("Lane*--------------------")
            # print(segmented_point)
            # segmented_point=(image==1)
            # print("Lane*--------------------")
            # print(segmented_point)
            
            width_list=[]
            exist_list=[]
            for idx, height in enumerate(image):
                
                
                if idx - delta_height  < 0: #image.shape[0]:
                    indi = torch.zeros(height.shape[0]).to(self.device)
                    width_list.append(indi)
                    continue
                laneCandi=(segmented_point[:,0]==idx - delta_height).nonzero().to(self.device)
                # laneCandi -> 2D tensor (N*coord), keypoint candidate of this height

                # No candidate in this height
                if laneCandi.shape[0]==0:
                    indi = torch.zeros(height.shape[0]).to(self.device)
                    # continue
                else:
                    laneCandi = torch.squeeze(laneCandi)
                    # print(laneCandi)
                    # print("Idx    Shape : {}".format(laneCandi.shape))
                    # print("Matrix Shape : {}".format(segmented_point.shape))
                    abscissa = torch.unsqueeze(torch.index_select(segmented_point, 0, laneCandi)[:,1],1)
                    # abscissa -> 2D tensor (N * abscissa), only abscissa of candidate
                    # print(abscissa)

                    dist_tensor = abscissa - torch.arange(height.shape[0]* abscissa.shape[0]).reshape(abscissa.shape[0], height.shape[0]).to(self.device)%height.shape[0]
                    # print(dist_tensor.shape)
                    # print(dist_tensor)
                    # dist_tensor -> 2D tensor (N * width), distance from each candidate
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

    # batch_image = batch * height * width  3D tensor
    def getDeltaRightMap(self, batch_image):
        start_time = time.time()
        # print("SHAPE {}".format(batch_image.shape))
        # reTensor = torch.FloatTensor([i for i in range(7 *6* 5)]).reshape(7, 6, 5)
        # print("----------In Delta Map--------")
        dim1 = batch_image.shape[0]
        dim2 = batch_image.shape[1]
        dim3 = batch_image.shape[2]

        batched_width_list=[]
        batched_exist_list=[]
        for image in batch_image:
            segmented_point=(image==1).nonzero().to(self.device)
            # segmented_point -> (N*coordXY) 2D tensor ... dim1 = number of segmented points, dim2 = point coord

            width_list=[]
            exist_list=[]
            for idx, height in enumerate(image):
                laneCandi=(segmented_point[:,0]==idx).nonzero().to(self.device)
                # laneCandi -> 2D tensor (N*coord), keypoint candidate of this height

                # No candidate in this height
                if laneCandi.shape[0]==0:
                    indi = torch.zeros(height.shape[0]).to(self.device)
                else:
                    laneCandi = torch.squeeze(laneCandi)
                    abscissa = torch.unsqueeze(torch.index_select(segmented_point, 0, laneCandi)[:,1],1)
                    # abscissa -> 2D tensor (N * abscissa), only abscissa of candidate
                    dist_tensor = abscissa - torch.arange(height.shape[0]* abscissa.shape[0]).reshape(abscissa.shape[0], height.shape[0]).to(self.device)%height.shape[0]
                    # dist_tensor -> 2D tensor (N * width), distance from each candidate
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
        # print("Batch Line")
        # for i in batched_exist_list:
        #     print(i.shape)
        end_time = time.time()
        # print("Time = {}".format(end_time-start_time))

        return batched_dist_tensor, batched_exist_list

    def getLaneExistHeight(self, batch_image):
        start_time = time.time()
        print("SHAPE {}".format(batch_image.shape))


        segmented_point3=torch.max(batch_image, dim = 2)
        # segmented_point3 = torch.where()
        print(segmented_point3.values[0])
        print(segmented_point3.values.shape)
        return segmented_point3.values

    def getDeltaVerticalMap(self, batch_image):
        # reTensor = torch.FloatTensor([i for i in range(7 *6* 5)]).reshape(7, 6, 5)
        # print("----------In Delta Map--------")
        batched_width_list=[]
        batched_exist_list=[]
        for image in batch_image:
            segmented_point=(image==1).nonzero().to(self.device)
            width_list=[]
            exist_list=[]
            for idx, width in enumerate(torch.transpose(image,0,1)):
                laneCandi=(segmented_point[:,1]==idx).nonzero().to(self.device)
                # laneCandi -> 2D tensor (N*idx), keypoint idx of this width in segmented point
                # print("----------------IDX  -----------------------------{}".format(idx))
                # print("width {}",format(width))
                # print(" candi {}".format(laneCandi))
                # No candidate in this height
                if laneCandi.shape[0]==0:
                    indi = torch.zeros(width.shape[0]).to(self.device)
                else:
                    laneCandi = torch.squeeze(laneCandi)
                    # print("l;aneCandi")
                    # print(laneCandi)
                    # print("Idx    Shape : {}".format(laneCandi.shape))
                    # print("Matrix Shape : {}".format(segmented_point.shape))
                    abscissa = torch.unsqueeze(torch.index_select(segmented_point, 0, laneCandi)[:,0],1)
                    # abscissa -> 2D tensor (N * abscissa), only abscissa of candidate
                    # print("ABSCISSSA")
                    # print(abscissa)

                    dist_tensor = abscissa - torch.arange(width.shape[0]* abscissa.shape[0]).reshape(abscissa.shape[0], width.shape[0]).to(self.device)%width.shape[0]
                    # print(dist_tensor.shape)
                    # print(dist_tensor)
                    # dist_tensor -> 2D tensor (N * width), distance from each candidate
                    exist_list.append(idx)
                    indi = torch.min(torch.abs(dist_tensor),0).values
                width_list.append(indi)

            new_lane_img = torch.stack(width_list)
            new_exist_img = torch.Tensor(exist_list).type(torch.int64)
            batched_width_list.append(new_lane_img)
            batched_exist_list.append(new_exist_img)
                
        batched_dist_tensor = torch.stack(batched_width_list)
        # batched_exist_tensor = torch.stack(batched_exist_list)
        # print(batched_width_list)
        # print(batched_exist_list)
        return torch.transpose(batched_dist_tensor,1,2), batched_exist_list
        return batched_dist_tensor, batched_exist_list

    def transform2deltaLoss(self, height_list, tensor)    :
        return
class delta_degree():

    def __init__(self):
        self.device = None

        return

    def setDevice(self, device):
        self.device = device

    def getDeltadegree(self,batch_image):
        batched_width_list=[]
        batched_exist_list=[]
        for image in batch_image:
            segmented_point=(image==1).nonzero().to(self.device)
            # segmented_point -> (N*coord) 2D tensor ... dim1 = number of segmented points, dim2 = point coord

            print("Lane*--------------------")
            print(segmented_point)
            
            width_list=[]
            exist_list=[]
            deg_tensor = torch.zeros(image.shape[0], image.shape[1], 7)
            print(deg_tensor.shape)
            range_list = [3,5,7,9,11,13,15]
            for idx in range_list:
                print("IDX {}".format(idx))

            return
            for idx, height in enumerate(image):
                laneCandi=(segmented_point[:,0]==idx).nonzero().to(self.device)
                # laneCandi -> 2D tensor (N*coord), keypoint candidate of this height

                # No candidate in this height
                if laneCandi.shape[0]==0:
                    indi = torch.zeros(height.shape[0]).to(self.device)
                else:
                    laneCandi = torch.squeeze(laneCandi)
                    # print(laneCandi)
                    # print("Idx    Shape : {}".format(laneCandi.shape))
                    # print("Matrix Shape : {}".format(segmented_point.shape))
                    abscissa = torch.unsqueeze(torch.index_select(segmented_point, 0, laneCandi)[:,1],1)
                    # abscissa -> 2D tensor (N * abscissa), only abscissa of candidate
                    # print(abscissa)

                    dist_tensor = abscissa - torch.arange(height.shape[0]* abscissa.shape[0]).reshape(abscissa.shape[0], height.shape[0]).to(self.device)%height.shape[0]
                    # print(dist_tensor.shape)
                    # print(dist_tensor)
                    # dist_tensor -> 2D tensor (N * width), distance from each candidate
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
        return batched_dist_tensor, batched_exist_list
    
def test1():
     # t= torch.zeros((3, 5, 10))
    dum1 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum2 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum3 = torch.Tensor([1,0,0,0,0,0,0,0,0,1])
    dum4 = torch.Tensor([0,0,1,0,0,0,0,1,0,0])
    dum5 = torch.Tensor([0,0,0,0,1,1,0,0,0,0])
    t1 = torch.stack([dum1, dum2, dum3, dum4, dum5])

    pad = torch.nn.functional.pad(t1, (2,0), value=3)
    print(pad)
    return
    dum1 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum2 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum3 = torch.Tensor([1,0,0,0,0,0,0,0,0,1])
    dum4 = torch.Tensor([0,0,1,0,0,0,0,1,0,0])
    dum5 = torch.Tensor([0,0,0,0,1,1,0,0,0,0])
    t2 = torch.stack([dum1, dum2, dum3, dum4, dum5])
    dum1 = torch.Tensor([0,0,0,0,0,0,0,0,0,0])
    dum2 = torch.Tensor([0,1,0,0,0,0,0,0,0,0])
    dum3 = torch.Tensor([1,0,0,0,0,0,0,0,0,1])
    dum4 = torch.Tensor([0,0,1,0,0,0,0,1,0,0])
    dum5 = torch.Tensor([0,0,0,0,1,0,1,0,0,0])
    t3 = torch.stack([dum1, dum2, dum3, dum4,  dum5])
    
    # t = torch.stack(dum2)
    # t = torch.stack([t1])
    t = torch.stack([t1, t2, t3])
    # print(t.shape)
    # for bat in t:
    #     print(bat.shape)
    d=delta_distance()
    batched_dist_tensor, li = d.getDeltaVerticalMap(t)
    # d.getDeltaRightMap(t)
    # batched_dist_tensor, li = d.getDeltaUpMap(t, 1)
    print(batched_dist_tensor)
    print(li)


    # d2 = delta_degree()
    # d2.getDeltadegree(t)

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
   
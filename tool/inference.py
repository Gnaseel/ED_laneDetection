import torch
import cv2
import numpy as np
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
import glob
import os

import math
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

color_list=[
    [0,0,255],
    [0,255,0],
    [255,0,0],
    [120,120,0],
    [0,120,120],
    [220,220,220],
    [100,0,100],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0]
]
class Inference():
    def __init__(self, args):
        self.cfg = args
        self.model_path = self.cfg.model_path
        self.image_path = self.cfg.image_path
        self.image_save_path = self.cfg.image_path

    def inference(self):

        #----------------------- Get Model ---------------------------------------------

        model = self.get_model()

        #----------------------- Get Image ---------------------------------------------

        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (300, 180))
        input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()


        
        #----------------------- Inference ---------------------------------------------

        output_tensor = model(input_tensor)
        output_image = output_tensor[0].permute(1,2,0).detach().numpy()

        #----------------------- Show Image ---------------------------------------------
        if self.cfg.show:
            self.show_image(img, output_image)
        #----------------------- Save Image ---------------------------------------------
        
        else:
            self.save_image(output_image)

    def get_model(self):
        if self.cfg.backbone == "VGG16":
            model = myModel()
        elif self.cfg.backbone == "VGG16_rf20":
            model = VGG16_rf20()

        temp = glob.glob('*')
        print(temp)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def save_image(self, image):
        dir = self.image_save_path.split(os.path.sep)
        fir_dir = os.path.join(*dir[:-1])+"_inference.jpg"
        cv2.imwrite(fir_dir, image)
        cv2.imwrite(os.listdir(self.image_save_path)[:-1])
        return

    def show_image(self, img, output_image):
        cv2.imshow("ori",img)
        cv2.imshow("output",output_image)


        for idx in range(10):
            cv2.imshow("THRESHOLD "+str(idx),(output_image-idx*0.05-0.1)*100)




        
        temp_image = (output_image-6*0.05-0.1)*10
        temp_image = np.squeeze(temp_image, axis=2)
        imimage = img.copy()

        max_arg=10
        arr = []
        anchorlist = anchorList()
        for y in range(temp_image.shape[0], 0,  -max_arg): # 180
            count=0
            print("Y = {}".format(y))
            for x in range(0,temp_image.shape[1], max_arg): # 300
                new_arr = temp_image[y-max_arg:y, x:x+max_arg]
                max_idx = np.argmax(new_arr)
                max = np.max(new_arr)

                max_x_idx = x + max_idx%max_arg
                max_y_idx = y - max_arg + max_idx//max_arg

                temp_image[y-max_arg:y, x:x+max_arg] = -100
                temp_image[max_y_idx, max_x_idx ]=max

                if max > -1.5:
                    # print("X = {} Y = {}".format(max_x_idx,max_y_idx))
                    imimage = cv2.circle(imimage, ( max_x_idx,max_y_idx), 1, color_list[0])
                    anchorlist.addNode(max_x_idx,max_y_idx,max)
                    count +=1
            print("{} COUNT = {}".format(y,count))
        print("DATA = {}".format(arr))

        # return
        np_arr = np.array(arr)




        anchor_image = img.copy()
        anchorlist.printList()

        laneidx=0
        for idx, anchor in enumerate(anchorlist.list):
            if len(anchor.nodelist) < 0:
                continue
            for node in anchor.nodelist:
                if True:
                # if idx ==3:
                    anchor_image = cv2.circle(anchor_image, (node.x,node.y), 1, color_list[laneidx])
            laneidx+=1
                # anchor_image = cv2.circle(anchor_image, (node.x*10+15,node.y*9+10), 10, (255-20*idx,20*idx,20*idx))
        
        vidx=1
        for idx, node in enumerate(anchorlist.list[vidx].nodelist):
            if idx==len(anchorlist.list[vidx].nodelist)-1:
                break
            print("--------------{}------------".format(idx))
            print("{} {}".format(node.x, node.y))
            print("Tilt = {}".format(anchorlist.list[vidx].tilt[idx]))
            print("Tilt avg = {}".format(anchorlist.list[vidx].tilt_avg[idx]))
            print("Tilt sub = {}".format(anchorlist.list[vidx].tilt_avg[idx] - anchorlist.list[vidx].tilt[idx]))
            print("Dist = {}".format(anchorlist.list[vidx].dist[idx]))



        cv2.imshow("111",temp_image)
        cv2.imshow("222",cv2.resize(imimage,(1280,720)))
        cv2.imshow("333",cv2.resize(anchor_image,(1280,720)))
        cv2.waitKey()
class anchorList():
    def __init__(self):
        self.list=[]
        return
    
    def getDist(self, pre_anc, new_anc):
        subx = (pre_anc.x - new_anc.x)*2
        suby = (pre_anc.y - new_anc.y)

        return math.sqrt(subx*subx + suby*suby)

    def getTilt(self, node1, node2):
        return math.atan2(node2.y-node1.y, node2.x - node1.x)*180/math.pi

    def nor_tilt(self, deg):
        while deg>180:
            deg -=360
        while deg<-180:
            deg +=180
        return deg
        
    def addNode(self,posx, posy, val):
        new_node = node()
        new_node.x=posx
        new_node.y=posy
        new_node.val=val

        
        min_dist = 300
        min_idx=-1
        # Get Min Dist
        for idx, anc in enumerate( self.list):
            dist = self.getDist(anc.nodelist[-1], new_node)
            if min_dist > dist:
                min_dist=dist
                min_idx=idx
            
        
        if min_dist > 100:
            new_anchor = anchor()
            new_anchor.nodelist.append(new_node)
            self.list.append(new_anchor)
            return



        if len(self.list[min_idx].nodelist) != 0:
            dist = self.getDist(self.list[min_idx].nodelist[-1], new_node)
            tilt = self.getTilt(self.list[min_idx].nodelist[-1], new_node)
            print("DIST = {}".format(dist))
            print("TILT = {}".format(tilt))
            if tilt >= 0:
                return
            if len(self.list[min_idx].nodelist)>4 and abs( tilt- self.list[min_idx].tilt_avg[-1])  >=60:
                return
            elif len(self.list[min_idx].nodelist)>1 and abs( tilt- self.list[min_idx].tilt_avg[-1])  >=70:
                return
        

        self.list[min_idx].nodelist.append(new_node)
        self.list[min_idx].dist.append(dist)
        self.list[min_idx].tilt.append(tilt)
        if len(self.list[min_idx].nodelist) <3:
            self.list[min_idx].tilt_avg.append(tilt)
        else:
            print("TILT {} AVG {} ".format(tilt, self.list[min_idx].tilt[-2]*0.3 + tilt*0.7))
            self.list[min_idx].tilt_avg.append(self.list[min_idx].tilt[-2]*0.3 + tilt*0.7)

        

        

       
        


        

                
    def printList(self):
        for idx, anchor in enumerate(self.list):
            print("{} Anchor Count = {}".format(idx, len(anchor.nodelist)))
            # if len(anchor.nodelist) >5:
                # for i in range(4):
                #     print("     TILT = {}".format(anchor.tilt[-4+i]))

        
class anchor():
    def __init__(self):
        self.nodelist=[]
        self.firstRun = True
        self.tilt = []
        self.dist = []
        self.tilt_avg=[]
        return
class node():
    def __init__(self):
        self.x=0
        self.y=0
        self.val = 0
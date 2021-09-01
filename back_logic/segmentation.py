import numpy as np
import cv2
from back_logic.anchor import *
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
class EDseg():
    def __init__(self):
        self.anchorlist = anchorList()
        return
    def segmentation(self, image, max_arg):
        temp_image = (image-6*0.05-0.1)*10
        temp_image = np.squeeze(temp_image, axis=2)

        arr = []
        anchorlist = anchorList()
        for y in range(temp_image.shape[0], 0,  -max_arg): # 180
            count=0
            # print("Y = {}".format(y))
            for x in range(0,temp_image.shape[1], max_arg): # 300
                new_arr = temp_image[y-max_arg:y, x:x+max_arg]
                max_idx = np.argmax(new_arr)
                max = np.max(new_arr)

                max_x_idx = x + max_idx%max_arg
                max_y_idx = y - max_arg + max_idx//max_arg

                temp_image[y-max_arg:y, x:x+max_arg] = -100
                temp_image[max_y_idx, max_x_idx ]=max

                if max > -1.5:
                    anchorlist.addNode(max_x_idx,max_y_idx,max)
                    count +=1
        #     print("{} COUNT = {}".format(y,count))
        # print("DATA = {}".format(arr))

        self.anchorlist = anchorlist
        # anchorlist.printList()
        return anchorlist, image


    def getSegimage(self,img):
        re_img = img.copy()
        laneidx=0
        for idx, anchor in enumerate(self.anchorlist.list):
            if len(anchor.nodelist) < 0:
                continue
            for node in anchor.nodelist:
                    re_img = cv2.circle(img, (node.x,node.y), 1, color_list[laneidx])
            laneidx+=1
        # cv2.imshow("111",img)
        
        return re_img
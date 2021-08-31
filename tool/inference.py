from numpy.core.arrayprint import format_float_positional
import torch
import cv2
import numpy as np
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
import glob
import os

import math
from back_logic.segmentation import EDseg
from back_logic.evaluate import EDeval

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
            # self.save_image(output_image)
            self.save_image(output_image)

    def get_model(self):
        if self.cfg.backbone == "VGG16":
            model = myModel()
        elif self.cfg.backbone == "VGG16_rf20":
            model = VGG16_rf20()
        elif self.cfg.backbone == "ResNet34":
            model = ResNet34()
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

        e = -0.1
        e = 0.003
        for idx in range(10):
            cv2.imshow("THRESHOLD "+str(idx),(output_image-idx*0.08+e)*60)
        print(output_image)
        seg = EDseg()
        anchorlist = seg.segmentation(output_image,8)
        seg.showSegimage(img)



        anchor_image = img.copy()
        anchorlist.printList()
        e=EDeval()
        e.getH_sample(anchorlist, 200, 660, 20)

        cv2.imshow("333",cv2.resize(anchor_image,(1280,720)))
        cv2.waitKey()


    def scoreSave(self, image):
        seg = EDseg()
        anchorlist = seg.segmentation(image,20)


    def inference_dir(self):
        
        
        model = self.get_model()

        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))
        # folder_list= glob.glob(os.path.join(path,"0530"))
        anchor_tensor = []
        pathlist = []
        # print("folder_list = {}".format(folder_list))
        for folder_path in folder_list:       
            sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            # sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            # print("sub_folder_list = {}".format(sub_folder_list))
            # idx =0
            for folder in sub_folder_list:
                # if idx==10:
                #     break
                # idx+=1
                filepath = os.path.join(folder,"20.jpg")
                # print("PATH = {}".format(filepath))

                 #----------------------- Get Model ---------------------------------------------


                #----------------------- Get Image ---------------------------------------------
                
                img = cv2.imread(filepath)
                img = cv2.resize(img, (304, 176))
                input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()

                #----------------------- Inference ---------------------------------------------

                output_tensor = model(input_tensor)
                output_image = output_tensor[0].permute(1,2,0).detach().numpy()
                seg = EDseg()
                anchorlist = seg.segmentation(output_image, 8)
                anchor_tensor.append(anchorlist)
                pathlist.append(filepath)
                # seg.showSegimage(img)
            print("Inference Finished!")
        return anchor_tensor, pathlist

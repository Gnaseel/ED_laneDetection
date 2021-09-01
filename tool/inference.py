from numpy.core.arrayprint import format_float_positional
import torch
import cv2
import numpy as np
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
import glob
import os

from back_logic.segmentation import EDseg
from back_logic.evaluate import EDeval

class Inference():
    def __init__(self, args):
        self.cfg = args
        self.model_path = self.cfg.model_path
        self.image_path = self.cfg.image_path
        self.image_save_path = self.cfg.save_path
        self.max_arg = 0


    def inference(self):

        #----------------------- Get Model ---------------------------------------------

        model = self.get_model()
        self.max_arg = model.maxArg
        #----------------------- Get Image ---------------------------------------------

        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (model.output_size[1], model.output_size[0]))
        input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()


        
        #----------------------- Inference ---------------------------------------------

        output_tensor = model(input_tensor)
        output_image = output_tensor[0].permute(1,2,0).detach().numpy()

        #----------------------- Show Image ---------------------------------------------
        if self.cfg.show:
            self.show_image(img, output_image, self.max_arg)
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
        # model.load_state_dict(torch.load(self.model_path))
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()
        return model

    def save_image(self, image, fileName):
        # --------------------------Save segmented map
        # print("SAVE PATH = {}".format(self.image_save_path))
        # dir = self.image_save_path.split(os.path.sep)

        # print("DIR= {}".format(self.image_save_path))
        fir_dir = os.path.join(self.image_save_path,str(fileName)+".jpg")
        # print("Path : {}".format(fir_dir))

        # fir_dir = "D:/temp/abc.jpg"
        # fir_dir =self.image_save_path+"\_inference.jpg"



        # --------------------------Save GroundTruth ____ only train, test data


        # --------------------------Save SegImg ____ 
        image = cv2.resize(image, (300,180))
        cv2.imwrite(fir_dir, image)
        # cv2.imshow("Seg_image", image)
        # cv2.waitKey()
        # cv2.imwrite(os.listdir(self.image_save_path)[:-1])
        return

    def show_image(self, img, output_image, maxArg):
        cv2.imshow("ori",img)
        cv2.imshow("output",output_image)

        e = -0.1
        e = 0.003
        for idx in range(10):
            cv2.imshow("THRESHOLD "+str(idx),(output_image-idx*0.08+e)*60)
        print(output_image)
        seg = EDseg()
        anchorlist, N = seg.segmentation(output_image,maxArg)
        seg.getSegimage(img)

        anchor_image = img.copy()
        anchorlist.printList()
        e=EDeval()
        e.getH_sample(anchorlist, 200, 660, 20)

        cv2.imshow("333",cv2.resize(anchor_image,(1280,720)))
        cv2.waitKey()


    # def scoreSave(self, image):
    #     seg = EDseg()
    #     anchorlist = seg.segmentation(image,20)


    def inference_dir(self):
        
        
        model = self.get_model()

        path=self.cfg.image_path
        folder_list= glob.glob(os.path.join(path,"*"))
        anchor_tensor = []
        pathlist = []
        idx =0
        for folder_path in folder_list:       
            sub_folder_list = glob.glob(os.path.join(folder_path, "*"))
            for folder in sub_folder_list:
                
                #----------------------- Get Image ---------------------------------------------
                filepath = os.path.join(folder,"20.jpg")
                # print("File Path {}".format(filepath))
                path_list = filepath.split('\\')
                # print("LIST {}".format(path_list))
                re_path = os.path.join(*path_list[-4:])
                # print("RE PATH {}".format(re_path))

                img = cv2.imread(filepath)
                img = cv2.resize(img, (model.output_size[1], model.output_size[0]))
                input_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0,3,1,2).float()
                
                

                #----------------------- Inference ---------------------------------------------

                output_tensor = model(input_tensor)
                output_image = output_tensor[0].permute(1,2,0).detach().numpy()
                
                #----------------------- Get Model ---------------------------------------------
                seg = EDseg()
                anchorlist, key_image = seg.segmentation(output_image, model.maxArg)
                # anchorlist, key_image = seg.segmentation(output_image, 16)
                anchor_tensor.append(anchorlist)
                pathlist.append(re_path)
                segImg = seg.getSegimage(img)


                #----------------------- Save data ---------------------------------------------
                fileName = folder
                self.save_image(segImg, idx)
                self.save_image(key_image*3000, str(idx)+"_key")
                self.save_image(output_image*200, str(idx)+"_output")
                if idx%200==0:
                    print("Idx {}".format(idx))
                    # break
                # if idx==200:
                #     return

                idx+=1
                
            print("Inference Finished!")
        return anchor_tensor, pathlist

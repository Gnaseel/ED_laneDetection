import torch
import cv2
import numpy as np
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
import glob
import os
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
        cv2.waitKey()
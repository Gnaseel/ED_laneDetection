from tool.trainer import Trainer
from tool.inference import Inference
from back_logic.evaluate import EDeval
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
from model.ResNet34_lin import ResNet34_lin
from model.ResNet34_seg import ResNet34_seg
from model.ResNet34_delta import ResNet34_delta
from model.ResNet50 import ResNet50
from torchsummary import summary
from torchvision import models
import os
from evaluator.lane import LaneEval
from evaluator.lane import Eval_Cfg
from evaluator.lane import Eval_data
from back_logic.image_saver import ImgSaver
from tool.scoring import Scoring
import glob
import cv2

import torch
class EngineTheRun():
    def __init__(self, args):
        self.cfg= args
        self.device = 'cpu'
        if args.device!='-1':
            self.device='cuda:'+args.device
        print("My Device is {}".format(self.device))

        return
    def train(self):
        trainer = Trainer(self.cfg)
        trainer.model = self.getModel().to(self.device)
        # trainer.dataset_path = "D:\\lane_dataset\\img_lane_640.npy"
        trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/img_lane_640.npy"

        trainer.device = self.device
        if self.cfg.backbone=="ResNet34_seg":
            trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/segmented_img_1027.npy"
            trainer.train_seg()
        if self.cfg.backbone=="ResNet34_delta":
            trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/segmented_img_1027.npy"
            trainer.train_delta()
        elif self.cfg.backbone=="ResNet34_deg":
            trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/segmented_img_1027.npy"
            trainer.train_deg()
        elif self.cfg.backbone=="ResNet50":
            trainer.train_lane_lin()
        else:
            trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/img_lane_640.npy"
            trainer.train_lane_lin()

    def inference(self):
        inferencer = Inference(self.cfg)
        inferencer.model = self.getModel().to(self.device)
        inferencer.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        inferencer.model.eval()
        if self.cfg.backbone=="ResNet34_deg":
            print("SET Model 2")
            inferencer.model2 = ResNet34_seg()
            inferencer.model2.to(self.device)
            inferencer.model2.eval()
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/11_06_14_14_device_cuda:2/epoch_50_index_339.pth", map_location='cpu'))
            inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/lane_segmentation/epoch_200_index_339.pth", map_location='cpu'))
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/lane_segmentation/epoch_100_index_339.pth", map_location='cpu'))
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/lane_segmentation/epoch_70_index_339.pth", map_location='cpu'))

        inferencer.device = self.device
        # inferencer.model.to(self.device)
        
        if self.cfg.showAll:
            inferencer.inference_all()
        else:
            inferencer.inference()


    def scoring(self):

        
        inferencer = Inference(self.cfg)
        inferencer.model = self.getModel()
        inferencer.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        inferencer.model.to(self.device)
        inferencer.model.eval()
        inferencer.device = self.device
        os.makedirs(inferencer.image_save_path, exist_ok=True)
        

        score = Scoring()
        score.device = self.device     

        if self.cfg.backbone=="ResNet34_deg":
            inferencer.model2 = ResNet34_seg()
            inferencer.model2.to(self.device)
            inferencer.model2.eval()
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/11_06_14_14_device_cuda:2/epoch_50_index_339.pth", map_location='cpu'))
            inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/lane_segmentation/epoch_100_index_339.pth", map_location='cpu'))

            # inferencer.inference_dir_deg()
            lane_tensor, path_list = inferencer.inference_dir_deg()
            print("DEG FINIASHED")
        elif self.cfg.backbone=="ResNet34_seg":
            img_list, seg_list = score.get_validation_set(self.cfg.image_path)
            loss = score.get_segmantation_CE(inferencer.model, img_list, seg_list, threshold = -0.2)
            print("Total loss = {}".format(loss))
            # for i1, i2 in zip(img_list, seg_list):
            #     # print(i)
            #     # seg_path = os.path.join(os.sep, *(self.cfg.image_path.split(os.sep)[:-2]), "seg_label", *(i.split(os.sep)[-3:-1]), "20.png")

            #     # seg_img = cv2.imread(seg_path)
            #     print(i1)
            #     print(i2)
                # print(seg_img.shape)
            return
        else:
            lane_tensor, path_list = inferencer.inference_dir()
        
        
        imgSaver = ImgSaver(self.cfg)
        imgSaver.device = self.device
        filepaths=[]
        if len(lane_tensor) > 2700:
            evaluator = EDeval()
            for idx, lane in enumerate(lane_tensor):
                if len(lane)>5:
                    lane_tensor[idx] = lane[0:5]
            evaluator.save_JSON(lane_tensor, path_list)
            bench = LaneEval()
            eval_cfg = Eval_Cfg()
            print("BENCH1")
            eval_cfg = bench.bench_one_submit("./back_logic/result_li.json","./evaluator/gt.json")
            eval_cfg.sort_list()
            #--------------------- Save Good Image ---------------
            idx =0
            save_image_num=20
            for i in eval_cfg.eval_list:
                idx+=1
                added_path = os.path.join(self.cfg.image_path, *i.filePath.split(os.sep)[1:])
                filepaths.append(added_path) #+ "/0531/1492729085263099246/20.jpg")
# /home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple/test_set/clips/0601/1495058651589498298/20.jpg
                if idx > save_image_num:
                    break
            #--------------------- Save Bad Image ---------------
            idx =0
            for i in reversed(eval_cfg.eval_list):
                idx+=1
                added_path = os.path.join(self.cfg.image_path, *i.filePath.split(os.sep)[1:])
                filepaths.append(added_path) #+ "/0531/1492729085263099246/20.jpg")

                if idx > save_image_num:
                    break
            if self.cfg.backbone=="ResNet34_deg":
                imgSaver.save_image_dir_deg(inferencer.model, inferencer.model2, filepaths, save_image_num)
            # else:
                # imgSaver.save_image_dir(filepaths)
        else:
            evaluator = EDeval()
            # evaluator.save_JSON(lane_tensor, path_list)
            bench = LaneEval()
            eval_cfg = Eval_Cfg()
            print("BENCH1")
            acc, fp, fn = 0,0,0
            for lane, path in zip(lane_tensor, path_list):
                # print("LANE {}".format(type(lane)))
                # print("LANE {}".format(lane))
                # print("PATH {}".format(path))
                if len(lane)>5:
                    lane = lane[0:5]
                a, p, n = bench.bench_one_instance(lane, path,"./evaluator/gt.json")
            
                acc += a
                fp += p
                fn += n
                print("{} {} {}".format(a, p, n))
            acc /=len(lane_tensor)
            fp /=len(lane_tensor)
            fn /=len(lane_tensor)
            print("LANE : {} ACC : {: >5.4f}, FP : {: >0.3f}, FN : {: >0.3f}".format(len(lane_tensor), acc,fp,fn))

            # eval_cfg.sort_list()

            for path in path_list:
                filepaths.append(os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple", path))
            imgSaver.save_image_dir_deg(inferencer.model, inferencer.model2, filepaths)
        return
    def getModel(self):
        model = None
        if self.cfg.backbone == "VGG16":
            model = myModel()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "VGG16_rf20":
            model = VGG16_rf20()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34":
            model = ResNet34()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_lin":
            model = ResNet34_lin()
            summary(model, (3, 176, 304),device='cpu')
        elif self.cfg.backbone == "ResNet34_delta":
            model = ResNet34_delta()
            summary(model, (3, 176, 304),device='cpu')
        elif self.cfg.backbone == "ResNet50":
            model = ResNet50()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_seg":
            model = ResNet34_seg()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_deg":
            model = ResNet34_delta()
            summary(model, (3, 368, 640),device='cpu')
        return model

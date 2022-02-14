from tool.trainer import Trainer
from tool.inference import Inference
from back_logic.evaluate import EDeval
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
from model.ResNet34_seg import ResNet34_seg
from model.ResNet34_seg_SCNN import ResNet34_seg_SCNN
from model.ResNet34_delta_SCNN import ResNet34_delta_SCNN
from model.ResNet18_delta_SCNN import ResNet18_delta_SCNN
from model.ResNet18_total import ResNet18_total
from model.ResNet18_seg_SCNN import ResNet18_seg_SCNN
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
# import config.config_window as CFG
import config.config_RIPPER as CFG
import torch
class EngineTheRun():
    def __init__(self, args):
        self.cfg= args
        # self.cfg.device = 'cpu'
        # self.cfg.device
        self.cfg.model_path = os.path.join(CFG.weight_file_path, CFG.delta_weight_file)
        self.cfg.heat_model_path = os.path.join(CFG.weight_file_path, CFG.heat_weight_file)
        self.cfg.dataset_path =CFG.dataset_path
        self.cfg.image_path =CFG.image_path
        self.cfg.model = self.getModel()
        print("MODEL PATH {}".format(self.cfg.model_path))
        if args.device!='-1':
            # self.device='cuda:'+args.device
            self.cfg.device='cuda:'+args.device
        else:
            self.cfg.device='cpu'
        print("My Device is {}".format(self.cfg.device))

        return
    def train(self):
        trainer = Trainer(self.cfg)
        trainer.model = self.getModel().to(self.device)
        # trainer.dataset_path = "D:\\lane_dataset\\img_lane_640.npy"
        trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/segmented_img_1027.npy"

        # tuSimple
        trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Dataset/segmented_img_1027.npy")
        # cuLane
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_1.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_2.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_3.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_4.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_5.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_6.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_7.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_8.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_9.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_10.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_11.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_12.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_13.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_14.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_15.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_16.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_17.npy")
        # trainer.datasets_path.append("/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_1.npy")

        trainer.device = self.device
        if self.cfg.backbone=="ResNet34_seg" or self.cfg.backbone=="ResNet18_seg_SCNN":
            trainer.train_seg()
        if self.cfg.backbone=="ResNet34_delta":
            trainer.train_delta()
        elif self.cfg.backbone=="ResNet34_deg":
            trainer.train_deg()
        elif self.cfg.backbone=="ResNet18_total":
            trainer.train_total()
        elif self.cfg.backbone=="ResNet34_delta_SCNN":
            trainer.train_deg()
        elif self.cfg.backbone=="ResNet18_delta_SCNN":
            trainer.train_deg()
        elif self.cfg.backbone=="ResNet50":
            trainer.train_lane_lin()
        else:
            print("[engine] Model Not Founded!!")

    def inference(self):
        inferencer = Inference(self.cfg)
        if self.cfg.showAll:
            inferencer.inference_all()
        else:
            inferencer.inference()
    def scoring(self):
        inferencer = Inference(self.cfg)
        # inferencer.model = self.getModel().to(self.device)
        # inferencer.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        # inferencer.model.eval()
        # inferencer.device = self.device
        os.makedirs(inferencer.image_save_path, exist_ok=True)
        

        score = Scoring()
        score.device = self.cfg.device     

        # if self.cfg.backbone=="ResNet34_deg":
        if self.cfg.backbone=="ResNet34_deg"  or self.cfg.backbone=="ResNet18_delta_SCNN" or self.cfg.backbone=="ResNet34_delta_SCNN":

            # inferencer.model2 = ResNet18_seg_SCNN()
            # inferencer.model2.to(self.device)
            # inferencer.model2.eval()
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/SCNN_18/epoch_600_index_339.pth", map_location='cpu'))
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/12_27_17_41_device_cuda:0/epoch_100_index_339.pth", map_location='cpu'))
            # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/lane_segmentation/epoch_70_index_339.pth", map_location='cpu'))


            # inferencer.inference_dir_deg()
            lane_tensor, path_list = inferencer.inference_dir_deg()
            print("DEG FINIASHED")

        elif self.cfg.backbone=="ResNet34_seg" or self.cfg.backbone=="ResNet18_seg_SCNN"  or self.cfg.backbone=="ResNet34_seg_SCNN" :
            img_list, seg_list = score.get_validation_set(self.cfg.image_path)
            loss, FP, FN = score.get_segmantation_CE(inferencer.model, img_list, seg_list, threshold = 0.5)
            print("Total loss = {} . FP {}    FN {}".format(loss, FP, FN))
            # return
        else:
            lane_tensor, path_list = inferencer.inference_dir()
        
        # for idx, lane in enumerate(lane_tensor):
        #     print("LANE = {}".format(len(lane)))
        #     print("LANE = {}".format(lane))
        #     if len(lane)>5:
        #         print("to = {}".format(len(lane_tensor[idx][0:5])))
        #         lane_tensor[idx] =  lane_tensor[idx][0:5]
        imgSaver = ImgSaver(self.cfg)
        imgSaver.device = self.cfg.device
        filepaths=[]
        save_image_num=20

        evaluator = EDeval()
        bench = LaneEval()


        if len(lane_tensor) > 2700:
            evaluator.save_JSON(lane_tensor, path_list)
            print("BENCH1")
            evaluator.eval_list = bench.bench_one_submit("./back_logic/result_li.json","./evaluator/gt.json")
            lane_heatmap = evaluator.get_lane_table(evaluator.eval_list)
            evaluator.sort_list()
            #--------------------- Save Good Image ---------------
            for idx, list in enumerate(evaluator.eval_list):
                # print("ADDED {}".format(self.cfg.image_path))
                # added_path = os.path.join(self.cfg.image_path, *list.filePath.split(os.sep)[1:])
                added_path = os.path.join(self.cfg.dataset_path, *list.filePath.split(os.sep))
                print("Dataset_path = {}".format(self.cfg.dataset_path))
                print("ADDED22 {}".format(*list.filePath.split(os.sep)))
                imgSaver.save_image_dir_deg(inferencer, added_path, "bad")
                if idx > save_image_num:
                    break
            #--------------------- Save Bad Image ---------------
            for idx, list in enumerate(reversed(evaluator.eval_list)):
                added_path = os.path.join(self.cfg.dataset_path, *list.filePath.split(os.sep))   #"/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple" + (need)"/0531/1492729085263099246/20.jpg")
                imgSaver.save_image_dir_deg(inferencer, added_path, "good")
                if idx > save_image_num:
                    break
            evaluator.eval_list = bench.bench_one_submit("./back_logic/result_li.json","./evaluator/gt.json")
            
        else:
            acc, fp, fn = 0,0,0
            for lane, path in zip(lane_tensor, path_list):
                # if len(lane)>5:
                #     lane = lane[0:5]
                a, p, n = bench.bench_one_instance(lane, path,"./evaluator/gt.json")
                acc += a
                fp += p
                fn += n
                print("{} {} {}".format(a, p, n))
                print("Dataset_path = {}".format(self.cfg.dataset_path))
                print("file_path = {}".format(path))
                final_path = os.path.join(self.cfg.dataset_path, path)
                # print("ADDED22 {}".format(*list.filePath.split(os.sep)))
                # imgSaver.save_image_dir_deg(inferencer, self.cfg.image_path, "bad")
                # file_path = os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple", path)
                imgSaver.save_image_dir_deg(inferencer, final_path, "SOME")
            acc /=len(lane_tensor)
            fp /=len(lane_tensor)
            fn /=len(lane_tensor)
            print("LANE : {} ACC : {: >5.4f}, FP : {: >0.3f}, FN : {: >0.3f}".format(len(lane_tensor), acc,fp,fn))
            

            
            # for path in path_list:
            #     filepaths.append(os.path.join("/home/ubuntu/Hgnaseel_SHL/Dataset/tuSimple", path))
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
        elif self.cfg.backbone == "ResNet34_delta":
            model = ResNet34_delta()
            summary(model, (3, 176, 304),device='cpu')
        elif self.cfg.backbone == "ResNet50":
            model = ResNet50()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_seg":
            model = ResNet34_seg()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_seg_SCNN":
            model = ResNet34_seg_SCNN()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_delta_SCNN":
            model = ResNet34_delta_SCNN()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet18_delta_SCNN":
            model = ResNet18_delta_SCNN()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet18_seg_SCNN":
            model = ResNet18_seg_SCNN()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_deg":
            model = ResNet34_delta()
            summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet18_total":
            model = ResNet18_total()
            summary(model, (3, 368, 640),device='cpu')
        else:
            print("[Engine] Model Not Defined!!!")
        return model

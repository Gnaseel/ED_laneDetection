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
        # print(torch.__version__)
        # print(torch.__version__)
        # print(torch.cuda.is_available())
        # print(torch.cuda.get_arch_list())
        # return
        # self.cfg.device = 'cpu'

        
        # self.cfg.device
        self.cfg.dataset = CFG.dataset
        self.cfg.model_path = os.path.join(CFG.weight_file_path, CFG.delta_weight_file)
        self.cfg.output_path = CFG.output_path
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
        trainer.model = self.getModel().to(self.cfg.device)
        # trainer.dataset_path = "D:\\lane_dataset\\img_lane_640.npy"
        trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/segmented_img_1027.npy"

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
        keyPointMode = False
        # keyPointMode = True
        inferencer = Inference(self.cfg)

        if keyPointMode:
            sum_dist=0
            sum_key_count=0
            path_list = inferencer.get_test_list_tuSimple()
            for file_path in path_list:

                evaluator = EDeval()
                path = os.path.join(self.cfg.dataset_path, file_path)

                img = cv2.imread(path)
                pred_lane, a, b = inferencer.inference_instance(img, file_path)
                gt_lane = evaluator.getKeypoint(file_path)
                
                mean_dist, key_count = evaluator.key_eval_v2(gt_lane, pred_lane)
                sum_dist += mean_dist
                sum_key_count +=key_count
                print("dist {}".format(mean_dist))

            sum_dist = sum_dist/len(path_list)
            sum_key_count = sum_key_count/len(path_list)
            print("Key Point Mean square = {}, {}".format(sum_dist, sum_key_count))
            return

        os.makedirs(inferencer.image_save_path, exist_ok=True)

        score = Scoring(self.cfg)
        score.device = self.cfg.device     

        # if self.cfg.backbone=="ResNet34_deg":
        if self.cfg.backbone=="ResNet34_deg"  or self.cfg.backbone=="ResNet18_delta_SCNN" or self.cfg.backbone=="ResNet34_delta_SCNN":

            lane_tensor, path_list = inferencer.inference_dir_deg()
            # lane_tensor, path_list = inferencer.inference_dir_deg_batch()
            print("DEG FINIASHED")

        elif self.cfg.backbone=="ResNet34_seg" or self.cfg.backbone=="ResNet18_seg_SCNN"  or self.cfg.backbone=="ResNet34_seg_SCNN" :
            img_list, seg_list = score.get_validation_set_tuSimple(self.cfg.image_path)
            loss, FP, FN = score.get_segmantation_CE(inferencer.model, img_list, seg_list, threshold = 0.5)
            print("Total loss = {} . FP {}    FN {}".format(loss, FP, FN))
            # return
        else:
            return

        if self.cfg.dataset=="tuSimple":
            score.result_save_tuSimple(lane_tensor, path_list)

        if self.cfg.dataset=="cuLane":
            score.result_save_cuLane(lane_tensor, path_list)

    def getModel(self):
        model = None
        if self.cfg.backbone == "VGG16":
            model = myModel()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "VGG16_rf20":
            model = VGG16_rf20()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34":
            model = ResNet34()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_delta":
            model = ResNet34_delta()
            # summary(model, (3, 176, 304),device='cpu')
        elif self.cfg.backbone == "ResNet50":
            model = ResNet50()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_seg":
            model = ResNet34_seg()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_seg_SCNN":
            model = ResNet34_seg_SCNN()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_delta_SCNN":
            model = ResNet34_delta_SCNN()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet18_delta_SCNN":
            model = ResNet18_delta_SCNN()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet18_seg_SCNN":
            model = ResNet18_seg_SCNN()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet34_deg":
            model = ResNet34_delta()
            # summary(model, (3, 368, 640),device='cpu')
        elif self.cfg.backbone == "ResNet18_total":
            model = ResNet18_total()
            # summary(model, (3, 368, 640),device='cpu')
        else:
            print("[Engine] Model Not Defined!!!")
        return model

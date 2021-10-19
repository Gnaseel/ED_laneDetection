from tool.trainer import Trainer
from tool.inference import Inference
from back_logic.evaluate import EDeval
from model.VGG16 import myModel
from model.VGG16_rf20 import VGG16_rf20
from model.ResNet34 import ResNet34
from model.ResNet50 import ResNet50
from model.ResNet34_lin import ResNet34_lin
from torchsummary import summary
import torch
class EngineTheRun():
    def __init__(self, args):
        self.cfg= args
        return
    def train(self):
        trainer = Trainer(self.cfg)
        trainer.model = self.getModel()
        trainer.dataset_path = "D:\\lane_dataset\\img_lane_640.npy"
        if self.cfg.backbone=="ResNet34_lin":
            trainer.dataset_path = "/home/ubuntu/Hgnaseel_SHL/Dataset/img_lane.npy"
#             trainer.dataset_path = "D:\\lane_dataset\\img_lane.npy"

            trainer.train_lane_lin()
        elif self.cfg.backbone=="ResNet50":
            trainer.dataset_path = "D:\\lane_dataset\\img_lane_640.npy"
            trainer.train_lane_lin()
        else:
            trainer.train_lane_lin()
    def inference(self):
        inferencer = Inference(self.cfg)
        inferencer.model = self.getModel()
        inferencer.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        inferencer.model.eval()
        
        
        if self.cfg.showAll:
            inferencer.inference_all()
        else:
            inferencer.inference()

        
    def scoring(self):
        inferencer = Inference(self.cfg)
        inferencer.model = self.getModel()
        inferencer.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        lane_tensor, path_list = inferencer.inference_dir()
        evaluator = EDeval()
        evaluator.save_JSON(lane_tensor, path_list)
        # print(lane_tensor[0])
        
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
        elif self.cfg.backbone == "ResNet50":
            model = ResNet50()
            summary(model, (3, 368, 640),device='cpu')
        return model

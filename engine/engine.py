from tool.trainer import Trainer
from tool.inference import Inference
from back_logic.evaluate import EDeval

class EngineTheRun():
    def __init__(self, args):
        self.cfg= args
        return
    def train(self):
        trainer = Trainer(self.cfg)
        trainer.train()
    def inference(self):
        inferencer = Inference(self.cfg)
        inferencer.inference()
    def scoring(self):
        inferencer = Inference(self.cfg)
        anchor_tensor, path_list = inferencer.inference_dir()
        evaluator = EDeval()
        lane_tensor = evaluator.getH_sample_all(anchor_tensor, 160, 720, 10)
        evaluator.save_JSON(lane_tensor, path_list)
        print(lane_tensor[0])
      
from tool.trainer import Trainer
from tool.inference import Inference


class EngineTheRun():
    def __init__(self, args):
        self.cfg= args
        return
    def train(self):
        trainer = Trainer(self.cfg)
        trainer.train()
    def inference(self):
        trainer = Inference(self.cfg)
        trainer.inference()
      
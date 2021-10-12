import torch
import os
import time
class Logger:
    def __init__(self):
        self.log_path=""
        return
    def saveTrainingLog(self, trainer):
        file_path = self.log_path +'/epoch_'+str(trainer.epoch) + '_index_'+str(trainer.index)+'.pth'
        torch.save(trainer.model.state_dict(),file_path)
        return
    def saveTrainingtxt(self, trainer):
        f=open("train_log.txt",'a')
        f.write("loss of {} epoch, {} index : {}".format(trainer.epoch, trainer.index, trainer.loss))
        f.close()
        return
    def printTrainingLog(self, trainer):
        print("IDX {}".format(trainer.index))
        print("loss of {} epoch, {} index : {}".format(trainer.epoch, trainer.index, trainer.loss))
    
    def logging(self, trainer):
        self.saveTrainingLog(trainer)
        self.saveTrainingtxt(trainer)
        self.printTrainingLog(trainer)
        return
    def makeLogDir(self):
        str_time = time.strftime('%Y_%m_%d_%H_%M_%Z', time.localtime(time.time()))
        self.log_path = '../weight_file/'+str_time
        os.makedirs(self.log_path, exist_ok=True)
        return
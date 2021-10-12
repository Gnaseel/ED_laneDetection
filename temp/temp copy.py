
from tool.scoring import Scoring


python main.py --mode train --backbone ResNet34
python main.py --mode train --backbone ResNet34_lin
# python main.py --mode train --backbone VGG16_rf20

# python main.py --mode i --model_path D:\\Lane\\weight_file\\2021_08_24\\epoch_15_index_0.pth --image_path D:\\lane_dataset\\train_set\\clips\\0313-1\\660\\20.jpg
# python main.py --mode inference --model_path D:\Lane\weight_file\2021_08_24\epoch_15_index_0.pth --image_path D:\lane_dataset\train_set\clips\0313-1\660\20.jpg --show
# python main.py --mode inference --model_path .\sample\data\vgg16_rf_epoch_29_index_20.pth --image_path .\sample\data\2.jpg --show --backbone VGG16_rf20
# python main.py --mode inference --model_path D:\Lane\weight_file\2021_08_24\epoch_15_index_0.pth --image_path D:\lane_dataset\train_set\clips\0313-1\660\20.jpg --show


# python main.py --mode inference --model_path .\sample\data\temp2.pth --image_path .\sample\data\2.jpg --backbone ResNet34 --show
# python main.py --mode inference --model_path .\sample\data\temp2.pth --image_path .\sample\data\2.jpg --backbone ResNet34 --show


python main.py --mode inference --model_path .\sample\data\lane.pth --image_path .\sample\data\2.jpg --backbone ResNet34_lin --show


# python main.py --mode score --model_path .\sample\data\vgg16_rf_epoch_29_index_20.pth --image_path C:\Users\aiste\Downloads\test_set\clips --backbone VGG16_rf20
# python main.py --mode score --model_path .\sample\data\res_epoch_69_index_20.pth --image_path C:\Users\aiste\Downloads\test_set\clips --backbone ResNet34
python main.py --mode score --model_path .\sample\data\epoch_1500.pth --image_path C:\Users\aiste\Downloads\test_set\clips --backbone ResNet34

# python .\evaluator\lane.py .\evaluator\result_res.json .\evaluator\gt.json 
python .\evaluator\lane.py .\evaluator\result_li.json .\evaluator\gt.json 

# python main.py --mode inference --model_path .\sample\data\res_epoch_69_index_20.pth --image_path .\sample\data\2.jpg --show --backbone ResNet34

python main.py --mode inference --model_path .\sample\data\lane_t_2.pth --image_path  .\sample\data\2.jpg --backbone ResNet34_lin --show
python main.py --mode inference --image_path  C:\Users\aiste\Downloads\test_set\clips --backbone ResNet34_lin --showAll --model_path .\sample\data\l5.pth 
python main.py --mode score --model_path .\sample\data\l3.pth --image_path  C:\Users\aiste\Downloads\test_set\clips --backbone ResNet34_lin
## ED-lane

### Introduction

**This network is being prepared!!!**  
  
ED-lane is open source lane segmentation network based on encoder-decoder structure.  
It has a very simple structure, so it can be easily used by people who just start deep learning.  
Developer can easily change and test parameters of ED-lane according to the purpose.  


### Contents


### Quick Start

### Usage

**Install**  

Create a virtual environment by conda (option)

```
conda create -n edlane python=3.7 -y
conda activate edlane
```

Install dependencies
```
pip install -r requirements.txt
```


**Data preprocess**  

This network supports CULane, TuSimple dataset.  
* [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)  
* [CULane](https://xingangpan.github.io/projects/CULane.html)  
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)  

Get segmentation annotation from TuSimple dataset and save it.
```
python tools/data_preprocess.py -tusimple --root $TUSIMPLEROOT
```

**Train**  

If you don't use save path argment, model is saved in ../weight_file/year_month_day/epock_n_index_n.pth
```
python main.py --mode train

```

**Test**  

```
python main.py --mode inference --model_path [YOUR MODEL PATH] --image_path [YOUR IMAGE PATH]

```

### Supported Network

**supported backbone**

 - [x] VGG  
 - [x] ResNet  

### Contributing

All contributions for ED-lane are welcomed.  
We are accepting any issue, pull requeast.  

### Contack

* email : hgnaseel@gmail.com

<!-- ### Licenses
### Reference -->
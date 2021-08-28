# ED-lane

## Introduction

**This network is being prepared!!!**  
  
ED-lane is open source lane segmentation network based on encoder-decoder structure.  
It has a very simple structure, so it can be easily used by people who just start deep learning.  
Developer can easily change and test parameters of ED-lane according to the purpose.  



  
---
### Preview  

**Input image, output segmentation map** 
<p></p>  
<image width="600" height="240" src="sample/readme_image/file2.png">  
  
**Instance segmentation**  
<p></p>  
<image width="800" height="200" src="sample/readme_image/file3.png">  
  
  
   

<p></p>  

**This network consists largely of two processes.**  
  
(a) A module that acquire a sementic segmentation map - with Enc-dec network  
(b) A module that sperate sementic data to each lane - with hand-crafted feature using derivative  
## Contents  

## Install  

Create a virtual environment by conda (option)

```
conda create -n edlane python=3.7 -y
conda activate edlane
```

Install dependencies
```
pip install -r requirements.txt
```

## Quick Start

You can use pre-trained model with sample data
```
python main.py --mode inference --model_path .\sample\data\epoch_18_index_10.pth --image_path .\sample\data\1.jpg --show

```
## Usage


### **Data preprocess**  

This network supports CULane, TuSimple dataset.  
* [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)  
* [CULane](https://xingangpan.github.io/projects/CULane.html)  
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)  

Get segmentation annotation from TuSimple dataset and save it.
```
python tools/data_preprocess.py -tusimple --root $TUSIMPLEROOT
```

### **Train**  

Train the network. default backbone network is VGG16 model.  
You can check the all arguments information by -help
```
python main.py --mode train  

python main.py -help
```
*If you don't use save path argment, model is saved in ../weight_file/year_month_day/epock_n_index_n.pth*

### **Test**  

```
python main.py --mode inference --model_path [YOUR MODEL PATH] --image_path [YOUR IMAGE PATH]

```

## Supported Network

**supported backbone**

 - [x] VGG  
 - [x] ResNet  
 - [ ] MobileNet  

## Contributing

All contributions for ED-lane are welcomed.  
We are accepting any issue, pull requeast.  

## Contact

* email : hgnaseel@gmail.com

## Licenses  
  
<image width="500" height="244" src="sample/readme_image/Apache_Software_Foundation_Logo_.png">  
  
This package is released under the Apache 2.0 license.  
  
<!-- ## Reference -->
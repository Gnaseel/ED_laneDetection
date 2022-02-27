import os
import json
import csv
import glob
import argparse
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def getImageFromPath(path):
    img = cv2.imread(path)
    img = cv2.resize(img, dsize = (640, 368), interpolation=cv2.INTER_NEAREST)
    return img

def getLabelFromPath(path):
    img = cv2.imread(path)
    if img is None:
        print("EMPTY IMAGE")
        print("PATH = {}".format(path))
        return None
    img = cv2.resize(img, dsize = (640, 368), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_tensor = torch.Tensor(img).to('cuda:1')
    img_tensor = torch.where(img_tensor>0, torch.tensor(1).to('cuda:1'), torch.tensor(0).to('cuda:1'))
    img = img_tensor.cpu().detach().numpy()
    img = np.array(img, dtype=np.uint8)
    return img    

def draw_lines(img, lanes, height, instancewise=False):
    for i, lane in enumerate(lanes):
        pts = [[x,y] for x, y in zip(lane, height) if (x!=-2 and y!=-2)]
        pts = np.array([pts])
        if not instancewise:
            cv2.polylines(img, pts, False,255, thickness=7)
        else:
            cv2.polylines(img, pts, False,50*i+20, thickness=7)

def readJSON(dir, json_string):
    print(os.path.join(dir, json_string))
    json_paths = glob.glob(os.path.join(dir, json_string))
    print(json_paths)
    data=[]
    print("------------------")
    for path in json_paths:
        with open(path) as f:
            d = (line.strip() for line in f)
            d_str = "[{0}]".format(','.join(d))
            data.append(json.loads(d_str))
    return data

def convert_label(output_dir, data, instancewise=True):
    counter = 0

    for i in range(len(data)):
        for j in range(len(data[i])):
            img = np.zeros([720, 1280], dtype=np.uint8)
            lanes = data[i][j]['lanes']
            height = data[i][j]['h_samples']
            draw_lines(img, lanes, height, instancewise)
            output_name = data[i][j]['raw_file'].split('/')[1:]
            
            output_path = os.path.join(output_dir,*output_name)[:-3]+"png"
            output__file_path = os.path.join(output_dir,*data[i][j]['raw_file'].split('/')[1:-1])

            print(output_path)
            if not os.path.exists(output__file_path):
                print("[NO DIR]    DIR = "+str(output__file_path))
                os.makedirs(output__file_path)
            cv2.imwrite(output_path, img)
            # return
            # cv2.imshow("asdf",img)
            # cv2.waitKey(0)
            counter += 1
    
    print("COUNTER = "+str(counter))

# lane_data FileNum * LaneNum * (height=, raw=, lane=,..)
# Have to change gt label_*_*
def tuSimple_gt2seg():
    jsonPath = '/workspace/data/tuSimple'
    outputPath = jsonPath+"/seged_gt/clips"
    json_string = 'label_*_*.json'
    lane_data = readJSON(jsonPath, json_string)
    convert_label(outputPath, lane_data)

def get_npFromtuSimple(dataPath, savePath):
   
    imageSubpath = ""
    gtSubpath = "seged_gt"
    
    json_string = 'label_data_*.json'
    input_path_list = glob.glob(os.path.join(dataPath, json_string))
    print(os.path.join(dataPath, imageSubpath, json_string))
    print(input_path_list)
    input_data_set=[]
    label_data_set=[]
    idx = 0
    for path in input_path_list:
        print(path)
        lane_txt_file = open(path)
        lane_data = lane_txt_file.readlines()
        for line in lane_data:
            data = json.loads(line)
            # print("------------")
            raw_path = os.path.join(dataPath, data["raw_file"])
            gt_path = os.path.join(dataPath, gtSubpath, data["raw_file"])[:-3]+"png"
            # print(raw_path)
            # print(gt_path)
            img = cv2.imread(raw_path)
            img2 = cv2.imread(gt_path)
            input_data_set.append(getImageFromPath(raw_path))
            label_data_set.append(getLabelFromPath(gt_path))
            # print(img.shape)
            # print(img2.shape)
            if idx%100==0:
                print("{} / ?".format(idx))
            idx+=1

    print("INPUT DATASET SIZE = "+str(len(input_data_set)))
    print("LABEL DATASET SIZE = "+str(len(label_data_set)))

    input_data_set = np.array(input_data_set)
    label_data_set = np.array(label_data_set)

    x_train, x_test, y_train, y_test = train_test_split(input_data_set, label_data_set)
    xy = (x_train, x_test, y_train, y_test)
    np.save(savePath, xy)
    return

def get_npFromcuLANE(dataPath, savePath, data_fileName):
    txt_path =os.path.join(dataPath, 'list', data_fileName+'.txt')
    txt_file = open(txt_path, 'r')
    data_list = []
    idx = 0
    train_image_list = []
    seged_image_list = []
    error_count = 0
    while True:
        line = txt_file.readline()
        if not line: 
            print("Brake {}".format(idx))
            break
        if idx%40!=0:
            idx +=1
            continue
        if idx % 1000 == 0:
            print("{} / 88880 ".format(idx))

        # 0 = img_path, 1 = txt_path, 2345 = lane+\n
        line_data = line.split(' ')
        train_path = dataPath+line_data[0]
        seged_path = dataPath+line_data[1]

        train_img = getImageFromPath(train_path)
        seged_img = getLabelFromPath(seged_path)
        if seged_img is None:
            error_count+=1
            print("error_count = {}".format(error_count))
            continue
        # print("APPEND")
        train_image_list.append(train_img)
        seged_image_list.append(seged_img)
        idx +=1
    print(len(train_image_list))
    print(len(seged_image_list))
    x_train, x_test, y_train, y_test = train_test_split(np.array(train_image_list), np.array(seged_image_list))
    xy = (x_train, x_test, y_train, y_test)
    save_path = os.path.join(savePath, "img_culane_0215_40.npy")
    print("SAVING......")
    np.save(save_path, xy)
    print("save_path SAVED!")

        # if idx % 5000 ==0 and idx!=0:
        #     train_image_list_np = np.array(train_image_list)
        #     seged_image_list_np = np.array(seged_image_list)
        #     x_train, x_test, y_train, y_test = train_test_split(train_image_list_np, seged_image_list_np)
        #     xy = (x_train, x_test, y_train, y_test)
        #     save_path = os.path.join(savePath, "img_culane_0215_"+str(int(idx/5000))+".npy")
        #     print("SAVING......")
        #     np.save(save_path, xy)
        #     print("save_path SAVED!")
        #     train_image_list=[]
        #     seged_image_list=[]
    return




if __name__=="__main__":
    # getDatasets()
    # show_np()

    # 1. GT text file to Segmented image (tuSimple)
    # tuSimple_gt2seg()

    # 2. Get npy file from raw+segmented image (tuSimple)
    get_npFromtuSimple(dataPath="/workspace/data/tuSimple", savePath="/workspace/src/ED_laneDetection/data/img_tuSimple_0215")
    # 3. Get npy file from raw+segmented image (cuLane)
    # get_npFromcuLANE(dataPath = "/workspace/data/cuLane", savePath = "/workspace/src/ED_laneDetection/data", fileName="train_gt")


    #temp
    # img = cv2.imread("/workspace/data/cuLane/laneseg_label_w16/driver_23_30frame/05160753_0455.MP4/00015.png")
    # img = img*100
    # print(img.sum())
    # cv2.imwrite("aaaaaa.png", img)
# def show_np(np_path="/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_1.npy"):
#     x,y,z,c = np.load(np_path, allow_pickle=True)
#     print("{} ".format(z.shape))
#     idx = 999
#     cv2.imwrite("temp_raw.jpg", x[idx])
#     cv2.imwrite("temp_gt.jpg", z[idx]*100)
#     return


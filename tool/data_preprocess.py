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
            # return

    return data

def draw_lines(img, lanes, height, instancewise=False):
    for i, lane in enumerate(lanes):
        pts = [[x,y] for x, y in zip(lane, height) if (x!=-2 and y!=-2)]
        pts = np.array([pts])
        if not instancewise:
            cv2.polylines(img, pts, False,255, thickness=7)
        else:
            cv2.polylines(img, pts, False,50*i+20, thickness=7)

def save_label_images(output_dir, data, instancewise=True):
    counter = 0

    for i in range(len(data)):
        for j in range(len(data[i])):
            img = np.zeros([720, 1280], dtype=np.uint8)
            lanes = data[i][j]['lanes']
            height = data[i][j]['h_samples']
            draw_lines(img, lanes, height, instancewise)
            output_name = data[i][j]['raw_file'].split('/')[1:]

            
            output_path = os.path.join(output_dir,*output_name)
            output__file_path = os.path.join(output_dir,*data[i][j]['raw_file'].split('/')[1:-1])

            print(output_path)
            if not os.path.exists(output__file_path):
                print("[NO DIR]    DIR = "+str(output__file_path))
                os.makedirs(output__file_path)

            # cv2.imwrite(output_path, img)
            # cv2.imshow("asdf",img)
            # cv2.waitKey(0)
            counter += 1
    
    print("COUNTER = "+str(counter))

def convert_label(output_dir, data):
    save_label_images(output_dir, data)
    return

def getImageFromPath(path, extension='jpg'):
    imgPath = path+"\\20."+extension
    img = cv2.imread(imgPath)
    img = cv2.resize(img, dsize = (300, 180))
    # img = img/256
    return img

def getLabelFromPath(path, extension='jpg'):
    imgPath = path+"\\20."+extension
    img = cv2.imread(imgPath)
    img = cv2.resize(img, dsize = (300, 180))
    # cv2.imshow("RGB", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] > 10:
                img[x,y]=1
                # img[x,y]=img[x,y]
            else:
                img[x,y]=0
    # cv2.imshow("GRAY", img*100)
    # cv2.waitKey(0)
    # img = img/256
    return img    

def getDatasets(dataPath="D:\\lane_dataset\\"):
   
    imageSubpath = "train_set\\clips\\"
    gtSubpath = "converted_gt\\"
    image_dir = dataPath + imageSubpath
    gt_dir = dataPath + gtSubpath
    input_path_list = glob.glob(os.path.join(image_dir, '*'))
    gt_path_list = glob.glob(os.path.join(gt_dir, '*'))

    input_data_set = []
    label_data_set = []

    data_set_size = 0
    for path in input_path_list:
        sub_path_list = glob.glob(path+'\\*')
        data_set_size += len(sub_path_list)

    idx = 0
    for path in input_path_list:
        sub_path_list = glob.glob(path+'\\*')
        for subPath in sub_path_list:
            idx += 1
            if idx % 100 == 0:
                print("Input Data Set "+str(idx)+"  /  "+str(data_set_size))
            input_data_set.append(getImageFromPath(subPath, extension='jpg'))
        
    idx = 0
    for path in gt_path_list:
        sub_path_list = glob.glob(path+'\\*')
        for subPath in sub_path_list:
            idx += 1
            if idx % 100 == 0:
                print("LABEL Data Set "+str(idx)+"  /  "+str(data_set_size))
            label_data_set.append(getLabelFromPath(subPath, extension='jpg'))

        

    print("INPUT DATASET SIZE = "+str(len(input_data_set)))
    print("LABEL DATASET SIZE = "+str(len(label_data_set)))

    # for i in 10:
    #     cv2.imshow("I IMG", input_data_set[i])
    #     cv2.imshow("L IMG", label_data_set[i])
    #     cv2.waitKey(0)

    input_data_set = np.array(input_data_set)
    label_data_set = np.array(label_data_set)

    x_train, x_test, y_train, y_test = train_test_split(input_data_set, label_data_set)
    xy = (x_train, x_test, y_train, y_test)
    np.save("D:\\\lane_dataset\\image_data_0816.npy", xy)
    print(glob.glob(os.path.join(gt_dir, '*')))


    return

if __name__=="__main__":
    getDatasets()
    
    # outputPath = "D:\\lane_dataset\\converted_gt"
    # jsonPath = 'D:\\lane_dataset\\train_set'
    # json_string = 'label_data_*.json'
    # data = readJSON(jsonPath, json_string)
    # convert_label(outputPath, data)

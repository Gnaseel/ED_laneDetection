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

def getImageList(path, dataset):
    txt_path =os.path.join(path, 'list', dataset+'.txt')
    txt_file = open(txt_path, 'r')
    data_list = []
    while True:
        line = txt_file.readline()
        if not line: break
        data_list.append(line[:-1]) # -1 FOR delete \n
    print(len(data_list))
    # print(data_list)
    # print(txt_path)
    return data_list

def getImageFromPath(path):
    img = cv2.imread(path)
    img = cv2.resize(img, dsize = (640, 368), interpolation=cv2.INTER_NEAREST)
    return img
none_path_list=[]
def getLabelFromPath(path):
    img = cv2.imread(path)
    if img is None:
        print("EMPTY IMAGE")
        print("PATH = {}".format(path))
        none_path_list.append(path)
        return None
    img = cv2.resize(img, dsize = (640, 368), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(type(img))
    # print(img.dtype)
    img_tensor = torch.Tensor(img).to('cuda:0')
    img_tensor = torch.where(img_tensor>0, 1, 0)
    img = img_tensor.cpu().detach().numpy()
    img = np.array(img, dtype=np.uint8)
    # print(type(img))
    # print(img.dtype)

    # for x in range(img.shape[0]):
    #     for y in range(img.shape[1]):
    #         if img[x,y] > 0:
    #             img[x,y]=1
    #         else:
    #             img[x,y]=0
    return img    

def get_npFromcuLANE(dataPath):
    # image_category_list = os.path.join(dataPath, list, test_split)
    image_path_list = getImageList(dataPath, dataset="train")
    train_image_list = []
    seged_image_list = []
    error_count = 0
    for idx, path in enumerate(image_path_list):
        # if idx==100:
        #     break
        train_path = os.path.join(dataPath,*path.split(os.sep))
        train_img = getImageFromPath(train_path)

        seged_path = os.path.join(dataPath, 'laneseg_label', *path.split(os.sep))
        seged_path_png = seged_path[:-3]+'png'
        seged_img = getLabelFromPath(seged_path_png)
        if seged_img is None:
            error_count+=1
            print("error_count = {}".format(error_count))
            continue
        train_image_list.append(train_img)
        seged_image_list.append(seged_img)
        if idx % 100 == 0:
            print("{} / {}".format(idx, len(image_path_list)))

        if idx % 5000 ==0 and idx!=0:
            train_image_list_np = np.array(train_image_list)
            seged_image_list_np = np.array(seged_image_list)
            x_train, x_test, y_train, y_test = train_test_split(train_image_list_np, seged_image_list_np)
            xy = (x_train, x_test, y_train, y_test)
            save_path = "img_culane_0111_"+str(int(idx/5000))+".npy"
            print("SAVING......")
            np.save(save_path, xy)
            print("save_path SAVED!")
            train_image_list=[]
            seged_image_list=[]

    print(none_path_list)
    return

def show_np(np_path="/home/ubuntu/Hgnaseel_SHL/Network/edLane/tool/img_culane_0111_1.npy"):
    x,y,z,c = np.load(np_path, allow_pickle=True)
    print("{} ".format(z.shape))
    idx = 999
    cv2.imwrite("temp_raw.jpg", x[idx])
    cv2.imwrite("temp_gt.jpg", z[idx]*100)


    # seg_img = getLabelFromPath("/home/ubuntu/Hgnaseel_SHL/Dataset/CULane/laneseg_label/driver_23_30frame/05160820_0464.MP4/01755.png")
    # print("{} ".format(seg_img.shape))
    # cv2.imwrite("temptemp.jpg", seg_img*100)
    return

if __name__=="__main__":
    # getDatasets()
    show_np()
    # get_npFromcuLANE(dataPath = "/home/ubuntu/Hgnaseel_SHL/Dataset/CULane")

    # outputPath = "D:\\lane_dataset\\converted_gt"
    # jsonPath = 'D:\\lane_dataset\\train_set'
    # json_string = 'label_data_*.json'
    # data = readJSON(jsonPath, json_string)
    # convert_label(outputPath, data)

# def getDatasets(dataPath="D:\\lane_dataset\\"):
   
#     imageSubpath = "train_set\\clips\\"
#     gtSubpath = "converted_gt\\"
#     image_dir = dataPath + imageSubpath
#     gt_dir = dataPath + gtSubpath
#     input_path_list = glob.glob(os.path.join(image_dir, '*'))
#     gt_path_list = glob.glob(os.path.join(gt_dir, '*'))

#     input_data_set = []
#     label_data_set = []

#     data_set_size = 0
#     for path in input_path_list:
#         sub_path_list = glob.glob(path+'\\*')
#         data_set_size += len(sub_path_list)

#     idx = 0
#     for path in input_path_list:
#         sub_path_list = glob.glob(path+'\\*')
#         for subPath in sub_path_list:
#             idx += 1
#             if idx % 100 == 0:
#                 print("Input Data Set "+str(idx)+"  /  "+str(data_set_size))
#             input_data_set.append(getImageFromPath(subPath, extension='jpg'))
        
#     idx = 0
#     for path in gt_path_list:
#         sub_path_list = glob.glob(path+'\\*')
#         for subPath in sub_path_list:
#             idx += 1
#             if idx % 100 == 0:
#                 print("LABEL Data Set "+str(idx)+"  /  "+str(data_set_size))
#             label_data_set.append(getLabelFromPath(subPath, extension='jpg'))

        

#     print("INPUT DATASET SIZE = "+str(len(input_data_set)))
#     print("LABEL DATASET SIZE = "+str(len(label_data_set)))

#     # for i in 10:
#     #     cv2.imshow("I IMG", input_data_set[i])
#     #     cv2.imshow("L IMG", label_data_set[i])
#     #     cv2.waitKey(0)

#     input_data_set = np.array(input_data_set)
#     label_data_set = np.array(label_data_set)

#     x_train, x_test, y_train, y_test = train_test_split(input_data_set, label_data_set)
#     xy = (x_train, x_test, y_train, y_test)
#     np.save("D:\\\lane_dataset\\image_data_0816.npy", xy)
#     print(glob.glob(os.path.join(gt_dir, '*')))


#     return


# def get_5line(dataPath="D:\\lane_dataset\\"):
#     imageSubpath = "train_set\\clips\\"
#     gtSubpath = "train_set\\seg_label\\"

#     image_dir = dataPath + imageSubpath
#     gt_dir = dataPath + gtSubpath

#     input_path_list = glob.glob(os.path.join(image_dir, '*'))
#     gt_path_list = glob.glob(os.path.join(gt_dir, '*'))

#     input_data_set = []
#     label_data_set = []

#     data_set_size = 0

#     for path in input_path_list:
#         sub_path_list = glob.glob(path+'\\*')
#         data_set_size += len(sub_path_list)

#     idx = 0
#     for path in input_path_list:
#         sub_path_list = glob.glob(path+'\\*')
#         for subPath in sub_path_list:
#             idx = idx + 1
#             if idx % 100 == 0:
#                 print("Input Data Set "+str(idx)+"  /  "+str(data_set_size))
            
#             paths=subPath.split('\\')
#             new_path = os.path.join(gt_dir, paths[-2], paths[-1])

#             np_img=getImageFromPath(subPath, extension='jpg')
#             np_ar=getLabelFromPath(new_path, extension='png')
#             input_data_set.append(np_img)
#             label_data_set.append(np_ar)
        
#     print("INPUT DATASET SIZE = "+str(len(input_data_set)))
#     print("LABEL DATASET SIZE = "+str(len(label_data_set)))

#     input_data_set = np.array(input_data_set)
#     label_data_set = np.array(label_data_set)

#     print("INPUT DATASET SHAPE = {}".format(input_data_set.shape))
#     print("LABEL DATASET SHAPE = {}".format(label_data_set.shape))


#     x_train, x_test, y_train, y_test = train_test_split(input_data_set, label_data_set)
#     xy = (x_train, x_test, y_train, y_test)
#     np.save("D:\\\lane_dataset\\img_lane.npy", xy)
#     print(glob.glob(os.path.join(gt_dir, '*')))



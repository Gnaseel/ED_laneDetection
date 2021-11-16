import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class anchorList():
    def __init__(self):
        self.list=[]
        return
    
    def getDist(self, pre_anc, new_anc):
        subx = (pre_anc.x - new_anc.x)*2
        suby = (pre_anc.y - new_anc.y)

        return math.sqrt(subx*subx + suby*suby)

    def getTilt(self, node1, node2):
        return math.atan2(node2.y-node1.y, node2.x - node1.x)*180/math.pi

    def nor_tilt(self, deg):
        while deg>180:
            deg -=360
        while deg<-180:
            deg +=180
        return deg
        
    def addNode(self,posx, posy, val):
        new_node = node()
        new_node.x=posx
        new_node.y=posy
        new_node.val=val

        
        min_dist = 300
        min_idx=-1
        # Get Min Dist
        for idx, anc in enumerate( self.list):
            dist = self.getDist(anc.nodelist[-1], new_node)
            if min_dist > dist:
                min_dist=dist
                min_idx=idx
            
        
        if min_dist > 100:
            new_anchor = anchor()
            new_anchor.nodelist.append(new_node)
            self.list.append(new_anchor)
            return



        if len(self.list[min_idx].nodelist) != 0:
            dist = self.getDist(self.list[min_idx].nodelist[-1], new_node)
            tilt = self.getTilt(self.list[min_idx].nodelist[-1], new_node)
            # print("DIST = {}".format(dist))
            # print("TILT = {}".format(tilt))
            if tilt >= 0:
                return
            if len(self.list[min_idx].nodelist)>4 and abs( tilt- self.list[min_idx].tilt_avg[-1])  >=60:
                return
            elif len(self.list[min_idx].nodelist)>1 and abs( tilt- self.list[min_idx].tilt_avg[-1])  >=70:
                return
        

        self.list[min_idx].nodelist.append(new_node)
        self.list[min_idx].dist.append(dist)
        self.list[min_idx].tilt.append(tilt)
        if len(self.list[min_idx].nodelist) <3:
            self.list[min_idx].tilt_avg.append(tilt)
        else:
            # print("TILT {} AVG {} ".format(tilt, self.list[min_idx].tilt[-2]*0.3 + tilt*0.7))
            self.list[min_idx].tilt_avg.append(self.list[min_idx].tilt[-2]*0.3 + tilt*0.7)
    

    # delete the node what is short
    def filtering(self):
        newList = []
        for node in self.list:
            if len(node.nodelist) < 2:
                continue
            # if self.getDist(node.nodelist[0], node.nodelist[-1]) < 30:
            #     continue
            newList.append(node)
        return newList
    
    #interpolate Edge of node
    def intpEdge(self, height_size):
        newList = []
        for lane in self.list: # lane is anchor
            # x =
            # y = 
            nodeNum = len(lane.nodelist)
            if nodeNum<3:
                continue
            # size = 3
            # if  <3:
            x = np.array([lane.x for lane in lane.nodelist])
            y = np.expand_dims(np.array([lane.y for lane in lane.nodelist]), axis=1)

            x_end = np.array([lane.x for lane in lane.nodelist[0:3]])
            y_end = np.expand_dims(np.array([lane.y for lane in lane.nodelist[0:3]]), axis=1)

            poly_reg = PolynomialFeatures(degree = 2)
            Y_poly = poly_reg.fit_transform(y)
            poly_reg.fit(Y_poly, x)
            lin_reg_2 = LinearRegression() 
            lin_reg_2.fit(Y_poly, x)

            poly_reg_end = PolynomialFeatures(degree = 1)
            Y_poly_end = poly_reg_end.fit_transform(y_end)
            poly_reg_end.fit(Y_poly_end, x_end)
            lin_reg_2_end = LinearRegression() 
            lin_reg_2_end.fit(Y_poly_end, x_end)


            # Y_zero = np.arange(min(y), height_size, 3)
            # X=lin_reg_2.predict(poly_reg.fit_transform(Y_grid))
            


            Y_grid = np.arange(min(y), max(y), 3)
            Y_grid = Y_grid.reshape((len(Y_grid), 1))
            if len(Y_grid)<3:
                continue
   
            X=lin_reg_2.predict(poly_reg.fit_transform(Y_grid))

            Y_grid_end = np.arange(max(y_end), height_size, 3)
            Y_grid_end = Y_grid_end.reshape((len(Y_grid_end), 1))
            X_end=lin_reg_2_end.predict(poly_reg_end.fit_transform(Y_grid_end))
            lane.nodelist = []

            for idx, val in enumerate(Y_grid):
                newNode = node()
                newNode.x = int(X[idx])
                newNode.y = Y_grid[idx]
                # if prex * int(X[idx])
                lane.nodelist.append(newNode)
            for idx, val in enumerate(Y_grid_end):
                newNode = node()
                newNode.x = int(X_end[idx])
                newNode.y = Y_grid_end[idx]
                # if prex * int(X[idx])
                lane.nodelist.append(newNode)


            newList.append(lane)

            
            # print(Y_grid)
            # print(X)
            # plt.scatter(x, y, color = 'green')
            # plt.plot(Y_grid, X, color = 'blue')
            # plt.scatter(Y_grid, X, color = 'red')
            # plt.title('Truth or Bluff (Polynomial Regression)')
            # plt.xlabel('Position level')
            # plt.ylabel('Salary')
            # plt.show()
        return newList

    def printList(self):
        for idx, anchor in enumerate(self.list):
            print("{} Anchor Count = {}".format(idx, len(anchor.nodelist)))
            # if len(anchor.nodelist) >5:
                # for i in range(4):
                #     print("     TILT = {}".format(anchor.tilt[-4+i]))


    
class anchor():
    def __init__(self):
        self.nodelist=[]
        self.firstRun = True
        self.tilt = []
        self.dist = []
        self.tilt_avg=[]
        return
    def printAnchor(self):
        for idx, node in enumerate(self.nodelist):
            if idx==len(self.nodelist)-1:
                    break
            print("--------------{}------------".format(idx))
            print("{} {}".format(node.x, node.y))
            print("Tilt = {}".format(self.tilt[idx]))
            print("Tilt avg = {}".format(self.tilt_avg[idx]))
            print("Tilt sub = {}".format(self.tilt_avg[idx] - self.tilt[idx]))
            print("Dist = {}".format(self.dist[idx]))
            
class node():
    def __init__(self):
        self.x=0    # Height    left 2 right
        self.y=0    # Height    top 2 bottom
        self.val = 0

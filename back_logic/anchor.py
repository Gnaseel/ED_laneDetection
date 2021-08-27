import math

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
            print("DIST = {}".format(dist))
            print("TILT = {}".format(tilt))
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
            print("TILT {} AVG {} ".format(tilt, self.list[min_idx].tilt[-2]*0.3 + tilt*0.7))
            self.list[min_idx].tilt_avg.append(self.list[min_idx].tilt[-2]*0.3 + tilt*0.7)

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
        self.x=0
        self.y=0
        self.val = 0
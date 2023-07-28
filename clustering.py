import math
import numpy as np
from datacleanup import *


class Point :
    
    def __init__(self, coords) :
        self.coords = coords
        self.currCluster = None
        
    @property
    def dim(self) :
        return len(self.coords)
        
    def distFrom(self, other) :
        if self.dim != other.dim :
            raise Exception("dimension mismatch: self has dim {} and other has dim {}".format(self.dim, other.dim))
        sum_diff = 0
        for i in range(self.dim):
            sum_diff += (self.coords[i] - other.coords[i])**2
        dist = math.sqrt(sum_diff)
        return dist
        
    def moveToCluster(self, dest) :
        if (self.currCluster is dest) :
            return False
        else :
            if (self.currCluster) :
                self.currCluster.removePoint(self)                
            dest.addPoint(self)
            self.currCluster = dest
            return True
            
    def closest(self, listOfPoints) :
        minDist = self.distFrom(listOfPoints[0])
        minPt = listOfPoints[0]
        for p in listOfPoints :
            if (self.distFrom(p) < minDist) :
                minDist = self.distFrom(p)
                minPt = p
        return minPt
        
    def __getitem__(self, i) :
        return self.coords[i]
        
    def __str__(self) :
        return str(self.coords)
        
    def __repr__(self) :
        return "Point: " + self.__str__()


class Cluster :
    
    def __init__(self, center = Point([0, 0])) :
        self.center = center
        self.points = set()
    
    @property    
    def coords(self) :
        return self.center.coords
        
    @property
    def dim(self) :
        return self.center.dim
        
    def addPoint(self, p) :
        self.points.add(p)
        
    def removePoint(self, p) :
        self.points.remove(p)
    
    @property
    def avgDistance(self) :
        sum_dists = 0
        no_points = len(self.points)
        for point in self.points:
            sum_dists += (self.center).distFrom(point)
        return sum_dists / no_points
    
    def updateCenter(self) :
        if len(self.points) == 0:
            return self
        else:
            average = [0] * self.center.dim
            for point in self.points:
                for i in range(self.center.dim):
                    average[i] += point[i]
            average = [i / len(self.points) for i in average]
            self.center = Point(average)
            return average
            
    def printAllPoints(self) :
        print (str(self))
        for p in self.points :
            print ("   {}".format(p))
        
    def __str__(self) :
        return "Cluster: {} points and center = {}".format(len(self.points), self.center)
        
    def __repr__(self) :
        return self.__str__()

#create a list of points
#input: data, a p-by-k numpy array
#output: a list of p k-dimensional points, with each point's coordinates
#        coming from one row of data
def makePointList(data) :
    #fill in
    points = []
    for pair in data:
        point = Point(pair)
        points.append(point)
    return points
        
#create a list of clusters with centers initialized using data
#input: data: a k-by-d numpy array
#output: a list of k Clusters, with each cluster having a d-dimensional
#        initial center, each center coming from a row
#hint: you may find makePointList useful for this
def createClusters(data) :
    centers = makePointList(data)
    return [Cluster(c) for c in centers]
    
    
def kmeans(pointdata, clusterdata) :
    
    points = makePointList(pointdata)
    clusters = createClusters(clusterdata)
    points.reverse()
    
    flag = True
    count = 0

    while(flag):
        for point in points:
            close_cluster = point.closest(clusters)
            flag = point.moveToCluster(close_cluster)
            
            for cluster in clusters:
                cluster.updateCenter()
            
            if flag == True:
                count += 1
            
            if count == len(points):
                flag = False
    
    return clusters
    
    
    
if __name__ == '__main__' :

    data = getFinalData1()
    #centers = np.array([[0,0,0,0,0,0,0], [0.3, 0.25, 0.1, 1, 0.5, 1, 1], [0.7, 0.5, 5, 2, 1, 2, 2], [1.2, 1, 30, 5, 1.5, 5, 5]], dtype=float)
    centers = np.array([[0.9, 0.6, 1, 2, 1, 0.5, 0.5], [1, 0.7, 15, 3, 1.05, 2, 2], [1.1, 0.8, 35, 5, 1.1, 6, 4]], dtype=float)
    
    clusters = kmeans(data, centers)
    for c in clusters :
        c.printAllPoints()
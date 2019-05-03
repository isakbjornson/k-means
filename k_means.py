'''
Team Member 1: Isak Bjornson
Team Member 2: Gage Guttmann
GU Username: ibjornson
File Name: k_means.py
'''

#Based on Peter Harrington, Machine Learning in Action

import numpy as np
import matplotlib.pyplot as plt

'''
Data comes in like this:
0.xxx ...  0.yyyy
where the values are floating point numbers representing points in a space
my example is m x 2, but the code would work for n features
Reads these into an m x 2 numpy matrix, where m is the number of points
'''
def loadData(file_name):
    with open(file_name) as fin:
        rows = (line.strip().split('\t') for line in fin)
        dataMat = [map(float,row) for row in rows]
    return np.mat(dataMat)

'''
Constructs a k x n matrix of randomly generated points as the
initial centroids. The points have to be in the range of the points
in the data set
'''
def randCent(dataMat,k):
    numCol = np.shape(dataMat)[1]  #notice the number of cols is not fixed.
    centroids = np.mat(np.zeros((k,numCol))) #kxnumCol matrix of zeros
    for col in range(numCol):
        minCol = np.min(dataMat[:,col]) #minimum from each column
        maxCol = np.max(dataMat[:,col]) #maximum from each column
        rangeCol = float(maxCol - minCol)
        centroids[:,col] = minCol + rangeCol * np.random.rand(k,1)
    return centroids

'''
Computes the Euclidean distance between two points
Each point is vector, composed of n values, idicating a point in n space 
'''
def distEucl(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

    

def kMeans(dataMat, k, distMeas=distEucl, createCent=randCent):
    m = np.shape(dataMat)[0]  #how many items in data set
    
    #create an mX2 natrix filled with zeros
    #each row stores centroi information for a point
    #col 0 stores the centroid index to which the point belongs
    #col 1 stores the distance from the point to the centroid
    clusterAssment = np.mat(np.zeros((m,2)))
    #80x2 matrix

    #create k randomly placed centroids
    centroids = createCent(dataMat,k)
    
    '''
    k means:
    
	do:
		for every point in the data:
			for every centroid:
				find the distance from the point to the centroid
            assign the point to the cluster whose centroid it is closest to
		for every cluster:
			calculate the mean of the points in that cluster
			move the centroid to the mean
	while (any point has shifted cluster assignment)
    '''
    
    newCents = centroids.copy()
    dist = [0, 0, 0, 0] #distances between point and centroids 
    clusterShift = True
    
    while (clusterShift):
        clusterShift = False
        dInd = 0 #index in the dataMat
        for pt in dataMat:
            cInd = 0 #index of the centroid in dist[]
            smDist = distMeas(pt, centroids[0])
            smInd = 0
            
            for ct in centroids:
                dist[cInd] = (distMeas(pt, ct))
                if dist[cInd] < smDist:
                    smDist = dist[cInd] #shortest point-centroid distance
                    smInd = cInd        #index of centroid
                    
                cInd += 1
                
            #if point shifts to a different centroid, loop again
            if (clusterAssment.item(dInd, 0) != smInd):
                clusterShift = True
            
            clusterAssment.itemset((dInd,0), smInd)
            clusterAssment.itemset((dInd, 1), smDist)
            dInd += 1
            
            clusterInd = 0
            c1X =c1Y= c2X= c2Y= c3X= c3Y= c4X= c4Y = 0; #X and Y values for centroids
            c1c= c2c= c3c= c4c = 1;     #count for number of points in cluster
            dataPoints = dataMat.tolist()
            
            while clusterInd < 80:
                if clusterAssment.item(clusterInd, 0) == 0:
                    c1X += (dataPoints[clusterInd][0])
                    c1Y += (dataPoints[clusterInd][1])
                    c1c += 1
                elif clusterAssment.item(clusterInd, 0) == 1:
                    c2X += (dataPoints[clusterInd][0])
                    c2Y += (dataPoints[clusterInd][1])
                    c2c += 1
                elif clusterAssment.item(clusterInd, 0) == 2:
                    c3X += (dataPoints[clusterInd][0])
                    c3Y += (dataPoints[clusterInd][1])
                    c3c += 1
                elif clusterAssment.item(clusterInd, 0) == 3:
                    c4X += (dataPoints[clusterInd][0])
                    c4Y += (dataPoints[clusterInd][1])
                    c4c += 1
                
                clusterInd += 1
                
            #set x and y values of the centroid to the mean of the points for each cluster
            centroids.itemset((0,0), c1X/c1c)
            centroids.itemset((0,1), c1Y/c1c)
            centroids.itemset((1,0), c2X/c2c)
            centroids.itemset((1,1), c2Y/c2c)
            centroids.itemset((2,0), c3X/c3c)
            centroids.itemset((2,1), c3Y/c3c)
            centroids.itemset((3,0), c4X/c4c)
            centroids.itemset((3,1), c4Y/c4c)
            
        #plot_results(dataMat, centroids)
    
    iter = 0    
    return centroids, iter #is the number of iterations required

#plot the centroids and points using matplotlib
def plot_results(dataMat, centroids):
    dataPoints = np.asarray(dataMat)
    
    xPoints = dataPoints[:,0] #Slice the matrix for x
    yPoints = dataPoints[:,1] #Slice the matrix for y

    centPoints = np.asarray(centroids)
    xCens = centPoints[:,0]
    yCens = centPoints[:,1]
    
    plt.scatter(xPoints,yPoints, color= 'g') #points are green
    plt.scatter(xCens, yCens, color= 'r')   #centroids are red
    plt.title('k-Means Clustering')
    plt.show()
    

def main():
    k = 4   #4 centroids
    dataMat = loadData("testSet.txt")

    centroids, iterations = kMeans(dataMat, k, distEucl, randCent)

    plot_results(dataMat, centroids)

main()

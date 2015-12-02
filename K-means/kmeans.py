from __future__ import division
import sys
import numpy as np
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import csv
import math
#############################################################################
#
# Function definitions go here
#
#############################################################################

#calculate euclidean distance
def euclidean(A, B):
	return scipy.spatial.distance.euclidean(A, B)

def sqeuclidean(A, B):
	return ((scipy.spatial.distance.euclidean(A,B)) ** 2)	

#calculate cosine distance
def cosine(A, B):
	return scipy.spatial.distance.cosine(A,B)

def predictCluster(data, centroid):
	distances = np.zeros((len(data),2))
	datamod = np.delete(data, 0, 1)
#	for row in datamod

# Utility function which calculates distances - can be changed to cosine		
def calculateDistance(data, centroid):
	#there needs to  be two  distances - first cluster and second
	cl = len(centroid)
	distances = np.zeros((len(data),cl))
	for k in range(0, len(centroid)):
		for i in range(0, len(data)):
			distances[i][k] = euclidean(data[i,1:3], centroid[k])#Change distance type here
	return distances #I am trying to calculate the distance to each cluster. and depending on which one it is closest to, the cluster value gets assigned.


#assign the cluster based on distance
def assignClusters(distances):
	pClusters = np.zeros((len(distances), 1), int)
	for i in range(0, len(distances)):
		minDist = 999
		index = 0
		for j in range(0, len(distances[i])):
			if distances[i][j] < minDist:
				minDist = distances[i][j]
				index = j+1
		pClusters[i] = index
	return pClusters

#recomputing the centroids
def getRelevantData(data, ckeys, k, rCentroids):
	cdata = np.zeros((len(ckeys[k]),2)) #this will change for wine dataset - more attributes
	for i in range(0, len(ckeys[k])):
		cdata[i] = data[i, 1:3] #this will change for wine dataset - more attributes there
#	print cdata
	rCentroids[k-1] = np.mean(cdata, axis = 0)
	#print rCentroids[k]
	#print np.mean(cdata, axis=0)

#Main function to recompute cluster centroids
def recomputeCentroids(data, predictedCluster, k, initialCentroids):
	#from the data, just get the average of each attribute and this would be new centroid
	#None of the intermediate data structures are needed. Only the end result
	numAttributes = len(data[0])-1 #Store the number of attributes -1 to discard id
	recomputedCentroids = initialCentroids
	dictOfClusters = {}
	for i in range(0, k):
		dictOfClusters[i+1] = []
	for i in range(0, k):
		for j in range(0, len(predictedCluster)):
			if predictedCluster[j] == i+1:
				dictOfClusters[i+1].append(j) #Dictofclusters is used to hold a hashmap where value is list of ids where cluster is k
		getRelevantData(data, dictOfClusters, i+1, recomputedCentroids)
#	print recomputedCentroids
	return recomputedCentroids

def getTrueSSE(data, ckeys, currk):
	cData = np.zeros((len(ckeys[currk]), 2))
	i1 = 0
	dist = []
	sumDistance = 0
	for ids in ckeys[currk]:
		cData[i1] = data[ids,1:3]
		i1 += 1
	tCentroid = np.mean(cData, 0)
	for i in range(0, len(cData)):
		dist.append(sqeuclidean(cData[i], tCentroid))
	sumDistance = sum(dist)
	return sumDistance

def computeTrueSSE(data, numClusters):
	dictClusters = {}
	TrueSSE = 0
	for i in range(1, numClusters+1):
		dictClusters[i] = []
	for i in range(1, numClusters+1):
		for j in range(0, len(data)):
			if data[j][-1] == i:
				dictClusters[i].append(j)
		TrueSSE += getTrueSSE(data, dictClusters, i)
	print 'Value of True SSE is - ' + str(TrueSSE)

def getSSEforCentroids(data, ckeys, k, fCentroids):
	pcdata = np.zeros((len(ckeys[k]),2))
	dist = []
	sumOfDistances = 0
	#pcdata should contain specific data of cluster only
	iter1 = 0
	for ids in ckeys[k]:
		pcdata[iter1] = data[ids,1:3]
		iter1 += 1
	currCentroid = np.mean(pcdata, 0)
	#for i in range(0, len(ckeys[k])):
	#	pcdata[i] = data[i,1:3]
	for i in range(0, len(pcdata)):
		dist.append(sqeuclidean(pcdata[i], currCentroid))
	#for ids in ckeys[k]:
	#	dist.append(sqeuclidean(pcdata[ids-1], fCentroids[k-1]))
	#for i in range(0, len(ckeys[k])):
	#	dist[i] = sqeuclidean(pcdata[ckeys[k][i]],fCentroids[k-1])
	sumOfDistances = sum(dist)
	return sumOfDistances

def computeSSE(data, predictedCluster, finalCentroids, k):
	dictOfClusters = {}
	SSE = 0
	#dictionaru keeps a track of all ids of a particular cluster
	for i in range(0, k):
		dictOfClusters[i+1] = []
	for i in range(1, k+1):
		for j in range(0, len(predictedCluster)):
			if predictedCluster[j] == i:
				dictOfClusters[i].append(j)
	for i in range(1, k+1):
		SSE += getSSEforCentroids(data, dictOfClusters, i, finalCentroids)
#	for i in range(0, k):
#		for j in range(0, len(predictedCluster)):
#			if predictedCluster[j] == i+1:
#				dictOfClusters[i+1].append(j)
#		SSE += getSSEforCentroid(data, dictOfClusters, i+1, finalCentroids)
	print 'Sum of Squared Errors(SSE) is - ' + str(SSE) 

def allDataDistance(allData, cpoint):
	distance = np.zeros((len(allData),1))
	for i in range(0, len(allData)):
		distance[i] = sqeuclidean(allData[i], cpoint)
	return (np.sum(distance))

def computeTrueSSB(data, numClusters):
	means = np.zeros((1,2))
	means = getMeansForAll(data)
	SSB = 0
	dictOfClusters = {}
	for i in range(1, numClusters+1):
		dictOfClusters[i] = []
	for i in range(1, numClusters+1):
		for j in range(0, len(data)):
			if data[j][-1] == i:
				dictOfClusters[i].append(j)
	for i in range(1, numClusters+1):
		clusterData = np.zeros((len(dictOfClusters[i]),2))
		i2 = 0
		for ids in dictOfClusters[i]:
			clusterData[i2] = data[ids, 1:3]
			i2 += 1
		currCentroid = np.mean(clusterData, 0)
		SSB += len(dictOfClusters[i]) * sqeuclidean(currCentroid, means)
	print 'Value of True SSB is - ' + str(SSB)

def getMeansForAll(data):
	modData = np.zeros((len(data),2))
	for i in range(0, len(data)):
		modData[i] = data[i,1:3]
	return np.mean(modData, 0)

def computeSSB(data, finalCentroids, k, predictedCluster):
	means = np.zeros((1,2))
	means = getMeansForAll(data)
	#allData = np.zeros((len(data),2))
	SSB = 0
	dictOfClusters = {}
	for i in range(0, k):
		dictOfClusters[i+1] = []
#	for i in range(0, len(data)):
#		allData[i] = data[i,1:3]
	for i in range(0, k):
		for j in range(0, len(predictedCluster)):
			if predictedCluster[j] == i+1:
				dictOfClusters[i+1].append(j)
	for i in range(1, k+1):
		clusterData = np.zeros((len(dictOfClusters[i]),2))
		i2 = 0
		for ids in dictOfClusters[i]:
			clusterData[i2] = data[ids, 1:3]
			i2 += 1
		currCentroid = np.mean(clusterData, 0)
		SSB += len(dictOfClusters[i])*sqeuclidean(currCentroid, means)
#	for i in range(0, k):
#		SSB += (len(dictOfClusters[i+1])*allDataDistance(allData, finalCentroids[i]))
	print 'Between cluster distance SSB is - ' + str(SSB)

def computenewSSB(finalCentroids):
	newMean = np.mean(finalCentroids, 0)
	SSB = 0
	for i in range(0, len(finalCentroids)):
		SSB += euclidean(finalCentroids[i], newMean)
	print 'SSB is - ' + str(SSB)
	

def getSC(instance, fullData, ckeys, currk, predictedCluster, k):
	#distance 'a' is average of intra cluster distance
	sum = 0
	for ids in ckeys[currk]:
		sum += euclidean(instance, fullData[ids])
	a = sum/len(ckeys[currk]) #Gives average distance
	minDist = 99999
	for i in range(1, k+1):
		if i != currk:
			#now we need to find inter cluster average distance
			for j in range(0, len(predictedCluster)):
				if predictedCluster[j] == i:
					ckeys[i].append(j)
			sumintra = 0
			for ids in ckeys[i]:
				sumintra += euclidean(instance, fullData[ids])
			sumintra = sumintra/len(ckeys[i])
			if sumintra < minDist:
				minDist = sumintra
	b = minDist # minimum among average distance to other cluster
	scInstance = float(b-a)/max(a,b)
	return scInstance
#	nckeys = {}
#	nckeys[nc] = []
#	for i in range(0, len(predictedCluster)):
#		if predictedCluster[i] != currk:
#			nckeys[nc].append(i+1)
#	for ids in nckeys[nc]:
		

def computeSilhouette(data, predictedCluster, finalCentroids, currk, k):
#	print predictedCluster[5]
	rData = np.zeros((len(data), 2))
	sWidth = 0
	for i in range(0, len(data)):
		rData[i] = data[i,1:3]
	dictOfClusters = {}
	for i in range(0, k):
		dictOfClusters[i+1] = []
	for i in range(0, len(predictedCluster)):
		if predictedCluster[i] == currk:
			dictOfClusters[currk].append(i)
	for ids in dictOfClusters[currk]:#This is only for the current centroid - each point in current centroid
		sWidth += getSC(rData[ids-1], rData, dictOfClusters, currk, predictedCluster, k)
	sWidth = sWidth/len(dictOfClusters[currk])
	return sWidth
	

#############################################################################
#
# The main program contents goes here.
#
#############################################################################
print 'Enter the number of clusters:'
k = input()
#print k k contains the number of clusters (correct)
#now, there are two files - easy and hard. first let us try to finish coding for easy case
inputFile = file(sys.argv[1],'r')
reader = csv.reader(inputFile, delimiter=',')
header = reader.next()
data = []
for row in reader:
	data.append(row)

dType = [int, float, float, int]

for i in range(0, len(data)):
    data[i] = [t(x) for t,x in zip(dType, data[i])]


actualCluster = []
for i in range(0, len(data)):
	actualCluster.insert(i, data[i][-1])

#print actualCluster
# actualCluster will be holding the values of true cluster. Data will not have that particular value but we need to add the predicted cluster	

data = np.asarray(data)
data = np.delete(data, -1,1)

#Data now contains everthing - id, x1, x2 - given x1 and x2 for 2 records, we need to find the euclidean distance

#we need to choose two random rows as the centroid for the k means
import random
centroids = []
for i in range(0, k):
	centroids.insert(i,random.randint(1, len(data)))

#print centroids

initialPoints = np.zeros((k,2))
for i in range(0, len(centroids)):
	initialPoints[i,:] = data[centroids[i]-1,1:3]

#for entry in centroids:
#	points = 
#	initialPoints.append(data[entry-1][1], data[entry-1][2])

import math
#print initialPoints

#if math.abs(updatedCentroids - initialPoints) > 0.2
#	print 'hello'

#TODO - need to compare initial centroid with new centroid before next iteration begins
#Can make it go up to 50/100 iterations.
initialCentroids = np.copy(initialPoints)
nextIteration = True
counter = 0
print 'initial values: '
print initialCentroids
finalCentroids = None
while nextIteration is True:
#for i in range(0, 10):
	clusterDistance = calculateDistance(data, initialCentroids)
#	predictCluster(data, initialPoints)
#	print clusterDistance[centroids[0]-1]
	predictedClusters = assignClusters(clusterDistance)
#	print predictedClusters	
	#Initial centroids picked, distance found, cluster predicted. Now need to recompute centroid 
	#Steps to re-compute centroid - find number of initial centroids. make that many buckets
	#add all ids belonging to a particular bucket. take average values for attributes of those ids -  this is new centroid	if counter == 0:
	oldCentroids = np.copy(initialCentroids)
	print 'old values:'
	print oldCentroids
	newCentroids = np.copy(recomputeCentroids(data, predictedClusters, k, initialCentroids))
#	else:
#		newCentroids = np.copy(recomputeCentroids(data, predictedClusters, k, newCentroids))
	initialCentroids = np.copy(newCentroids)
	print 'recomputed value'
	print initialCentroids
	print 'difference between' 
	print oldCentroids
	print initialCentroids
	print 'result'
	difference = np.abs(oldCentroids - initialCentroids)
	finalCentroids = np.copy(initialCentroids)
	count2 = 0
	for row in difference:
		for col in row:
#			print col ,
			if col < 0.005 or col == 0.0:
				count2 +=1
				if count2 >= 2:
					nextIteration = False
#	difference = np.abs(oldCentroids,newCentroids)
#	counter += 1
#	for row in difference:
#		for col in row:
#			if col < 0.01:
#				nextIteration = False

#print predictedClusters
resultFile = file('predictedeasy.csv', 'w')
headerLine = 'ID,PredictedCluster\n'
resultFile.writelines(headerLine)
for i in range(0, len(predictedClusters)):
	line = "" + str(i+1) + "," + str(predictedClusters[i]) + '\n'
	resultFile.writelines(line)
resultFile.close()


#finalCentroids = np.copy(recomputeCentroids(data, predictedClusters, k, initialCentroids))

backupFile = file(sys.argv[1], 'r')
inputFile.seek(0)
read2 = csv.reader(inputFile, delimiter=',')
head2 = read2.next()

data2 = []
for row in read2:
	data2.append(row)

dType2 = [int, float, float, int]

for i in range(0, len(data2)):
    data2[i] = [t(x) for t,x in zip(dType2, data2[i])]

data2 = np.asarray(data2)

computeTrueSSE(data2, 2)
computeTrueSSB(data2, 2)

computeSSE(data, predictedClusters, finalCentroids, k)
#def computeSSB(data, finalCentroids, k, predictedCluster):
computeSSB(data, finalCentroids, k, predictedClusters)
#computenewSSB(finalCentroids)
sumSilhouette = 0
for i in range(0, k):
	#for each cluster, calculate silhouette co-efficient
	avgSCWidth = computeSilhouette(data, predictedClusters, finalCentroids, i+1, k)
	print 'Average Silhouette width for cluster ' + str(i+1) + ' - ' + str(avgSCWidth)
	sumSilhouette += avgSCWidth

print 'Average Silhouette width for entire clustering is ' + str(float(sumSilhouette)/k)
#computeSilhouette(predictedClusters)
#computeCMatrix(data, predictedClusters) #Use scikit.learn.metrics
Actual = []
Predicted = []
for i in range(0, len(predictedClusters)):
	Predicted.append(predictedClusters[i])
	Actual.append(actualCluster[i])

from sklearn.metrics import confusion_matrix

resArray = confusion_matrix(Actual, Predicted)

print resArray
fileForSPlot = file('sploteasy.csv', 'w')
headline1 = 'ID,X1,X2,Actual,Predicted\n'
fileForSPlot.writelines(headline1)
for i in range(0, len(predictedClusters)):
	line = "" + str(i+1) + "," + str(data[i][1]) + ',' + str(data[i][2]) + ',' + str(actualCluster[i])+','+ str(predictedClusters[i][0]) + '\n'
	fileForSPlot.writelines(line)
fileForSPlot.close()

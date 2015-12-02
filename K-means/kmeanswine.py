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

def normalizeAcidity(data):
	results = np.amax(data, 0)
	maxAcid = results[1]
	results = np.amin(data, 0)
	minAcid = results[1]
	for i in range(0, len(data)):
		data[i][1] = (data[i][1] - minAcid)/float(maxAcid - minAcid)

def normalizeSugars(data):
	results = np.amax(data, 0)
	maxAcid = results[4]
	results = np.amin(data, 0)
	minAcid = results[4]
	for i in range(0, len(data)):
		data[i][4] = (data[i][4] - minAcid)/float(maxAcid - minAcid)

def normalizeFreeSulfur(data):
	results = np.amax(data, 0)
	maxAcid = results[6]
	results = np.amin(data, 0)
	minAcid = results[6]
	for i in range(0, len(data)):
		data[i][6] = (data[i][6] - minAcid)/float(maxAcid - minAcid)

def normalizeTotalSulfur(data):
	results = np.amax(data, 0)
	maxAcid = results[7]
	results = np.amin(data, 0)
	minAcid = results[7]
	for i in range(0, len(data)):
		data[i][7] = (data[i][7] - minAcid)/float(maxAcid - minAcid)

def normalizePH(data):
	results = np.amax(data, 0)
	maxAcid = results[9]
	results = np.amin(data, 0)
	minAcid = results[9]
	for i in range(0, len(data)):
		data[i][9] = (data[i][9] - minAcid)/float(maxAcid - minAcid)

def normalizeAlcohol(data):
	results = np.amax(data, 0)
	maxAcid = results[11]
	results = np.amin(data, 0)
	minAcid = results[11]
	for i in range(0, len(data)):
		data[i][11] = (data[i][11] - minAcid)/float(maxAcid - minAcid)

# Utility function which calculates distances - can be changed to cosine        
def calculateDistance(data, centroid):
	#there needs to  be two  distances - first cluster and second
	cl = len(centroid)
	distances = np.zeros((len(data),cl))
	for k in range(0, len(centroid)):
		for i in range(0, len(data)):
			distances[i][k] = euclidean(data[i,1:12], centroid[k])#Change distance type here
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
	cdata = np.zeros((len(ckeys[k]),11)) #this will change for wine dataset - more attributes
	for i in range(0, len(ckeys[k])):
		cdata[i] = data[i, 1:12] #this will change for wine dataset - more attributes there
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
	cData = np.zeros((len(ckeys[currk]), 11))
	i1 = 0
	dist = []
	sumDistance = 0
	for ids in ckeys[currk]:
		cData[i1] = data[ids,1:12]
		i1 += 1
	tCentroid = np.mean(cData, 0)
	for i in range(0, len(cData)):
		dist.append(sqeuclidean(cData[i], tCentroid))
	sumDistance = sum(dist)
	return sumDistance


def computeTrueSSE(data, cluster, k):
	dictClusters = {}
	TrueSSE = 0
	for i in range(1, k+1):
		dictClusters[i] = []
	for i in range(1, k+1):
		for j in range(0, len(cluster)):
			if cluster[j] == i:
				dictClusters[i].append(j)
		TrueSSE += getTrueSSE(data, dictClusters, i)
	print 'Value of True SSE is - ' + str(TrueSSE)

def getSSEforCentroid(data, ckeys, k, fCentroids):
	pcdata = np.zeros((len(ckeys[k]),11))
	dist = []
	sumDistance = 0
	iter1 = 0
	for ids in ckeys[k]:
		pcdata[iter1] = data[ids, 1:12]
		iter1 += 1
	currCentroid = np.mean(pcdata, 0)
#	for i in range(0, len(ckeys[k])):
#		pcdata[i] = data[i,1:12]
	for i in range(0, len(pcdata)):
		dist.append(sqeuclidean(pcdata[i],currCentroid))
	sumOfDistances = sum(dist)
	return sumOfDistances

def computeSSE(data, predictedCluster, finalCentroids):
	dictOfClusters = {}
	SSE = 0
	for i in range(0, k):
		dictOfClusters[i+1] = []
	for i in range(0, k):
		for j in range(0, len(predictedCluster)):
			if predictedCluster[j] == i+1:
				dictOfClusters[i+1].append(j)
		SSE += getSSEforCentroid(data, dictOfClusters, i+1, finalCentroids)
	print 'Sum of Squared Errors(SSE) is - ' + str(SSE)

def allDataDistance(allData, cpoint):
	distance = np.zeros((len(allData),1))
	for i in range(0, len(allData)):
		distance[i] = sqeuclidean(allData[i], cpoint)
	return (np.sum(distance))

def computeTrueSSB(data, cluster, k):
	means = np.zeros((1,11))
	means = getAllMeans(data)
	SSB = 0
	dictOfClusters = {}
	for i in range(1, k+1):
		dictOfClusters[i] = []
	for i in range(1, k+1):
		for j in range(0, len(cluster)):
			if cluster[j] == i:
				dictOfClusters[i].append(j)
	for i in range(1, k+1):
		clusterData = np.zeros((len(dictOfClusters[i]),11))
		i2 = 0
		for ids in dictOfClusters[i]:
			clusterData[i2] = data[ids, 1:12]
			i2 += 1
		currCentroid = np.mean(clusterData, 0)
		SSB += len(dictOfClusters[i]) * sqeuclidean(currCentroid, means)
	print 'Value of True SSB is - ' + str(SSB)

def getAllMeans(data):
	moData = np.zeros((len(data), 11))
	for i in range(0, len(data)):
		moData[i] = data[i, 1:12]
	return np.mean(moData, 0)

def computeSSB(data, finalCentroids, k, predictedCluster):
	means = np.zeros((1, 11))
	means = getAllMeans(data)
#	allData = np.zeros((len(data),11))
	SSB = 0
	dictOfClusters = {}
	for i in range(0, k):
		dictOfClusters[i+1] = []
#	for i in range(0, len(data)):
#		allData[i] = data[i,1:12]
	for i in range(0, k):
		for j in range(0, len(predictedCluster)):
			if predictedCluster[j] == i+1:
				dictOfClusters[i+1].append(j)
	for i in range(1, k+1):
		clusterData = np.zeros((len(dictOfClusters[i]),11))
		i2 = 0
		for ids in dictOfClusters[i]:
			clusterData[i2] = data[ids, 1:12]
			i2 += 1
		currCentroid = np.mean(clusterData, 0)
		SSB += len(dictOfClusters[i])*sqeuclidean(currCentroid, means)
#	for i in range(0, k):
#		SSB += (len(dictOfClusters[i+1])*allDataDistance(allData, finalCentroids[i]))
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
#   nckeys = {}
#   nckeys[nc] = []
#   for i in range(0, len(predictedCluster)):
#       if predictedCluster[i] != currk:
#           nckeys[nc].append(i+1)
#   for ids in nckeys[nc]:

def computeSilhouette(data, predictedCluster, finalCentroids, currk, k):
	print currk
#	print predictedCluster[5]
	rData = np.zeros((len(data), 11))
	sWidth = 0
	for i in range(0, len(data)):
		rData[i] = data[i,1:12]
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

def computeValidation(confusion, k):
	#Pick the majority in each row for cluster purity
	mLists = []
	confusion2 = np.transpose(confusion)
	for eid in np.max(confusion2, 1):
		if eid != 0:
			mLists.append(eid)
	totalSum = []
	for totals in np.sum(confusion2, 1):
		if totals != 0:
			totalSum.append(totals)
	purity = []
	for i in range(0, len(totalSum)):
		purity.append(float(mLists[i])/totalSum[i])
	for i in range(0, len(purity)):
		print 'Purity of cluster ' + str(i+1) + ' is - ' + str(purity[i])
	

#############################################################################
#
# Main program goes here
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

dType = [int, float, float, float, float, float, float, float, float, float, float, float, int, str]

for i in range(0, len(data)):
	data[i] = [t(x) for t,x in zip(dType, data[i])]

for i in range(0, len(data)):
	if data[i][-1] == 'Low':
		data[i][-1] = 1
	else:
		data[i][-1] = 2

twoCluster = []
for i in range(0, len(data)):
	twoCluster.append(data[i][-1])

multiCluster = []
for i  in range(0, len(data)):
	if data[i][-2] == 3:
		multiCluster.append(1)
	elif data[i][-2] == 4:
		multiCluster.append(2)
	elif data[i][-2] == 5:
		multiCluster.append(3)
	elif data[i][-2] == 6:
		multiCluster.append(4)
	elif data[i][-2] == 7:
		multiCluster.append(5)
	else:
		multiCluster.append(6)

data = np.asarray(data)
data = np.delete(data, -1, 1)
data = np.delete(data, -1, 1)

print data[3]

#need to normalize the data
normalizeAcidity(data)
normalizeSugars(data)
normalizeFreeSulfur(data)
normalizeTotalSulfur(data)
normalizePH(data)
normalizeAlcohol(data)

#print data[3]
#Now that the data is normalized, we can carry out clustering 
#we need to choose two random rows as the centroid for the k means
import random
centroids = []
for i in range(0, k):
	centroids.insert(i,random.randint(1, len(data)))

#print centroids

initialPoints = np.zeros((k,11))#Changed from 2. Now 11 attributes
for i in range(0, len(centroids)):
	initialPoints[i,:] = data[centroids[i]-1,1:12]

print initialPoints

initialCentroids = np.copy(initialPoints)
nextIteration = True
counter = 0
print 'initial values: '
print initialCentroids
finalCentroids = None
counter1 = 0
while nextIteration is True:
#for i in range(0, 10):
	clusterDistance = calculateDistance(data, initialCentroids)
#   predictCluster(data, initialPoints)
#   print clusterDistance[centroids[0]-1]
	predictedClusters = assignClusters(clusterDistance)
#   print predictedClusters 
    #Initial centroids picked, distance found, cluster predicted. Now need to recompute centroid 
    #Steps to re-compute centroid - find number of initial centroids. make that many buckets
    #add all ids belonging to a particular bucket. take average values for attributes of those ids -  this is new centroid  if counter == 0:
	oldCentroids = np.copy(initialCentroids)
	print 'old values:'
	print oldCentroids
	newCentroids = np.copy(recomputeCentroids(data, predictedClusters, k, initialCentroids))
#   else:
#       newCentroids = np.copy(recomputeCentroids(data, predictedClusters, k, newCentroids))
	initialCentroids = np.copy(newCentroids)
	print 'recomputed value'
	print initialCentroids
	print 'difference between'
	print oldCentroids, initialCentroids
	print 'result'
	difference = np.abs(oldCentroids - initialCentroids)
	finalCentroids = np.copy(initialCentroids)
	counter += 1
	for row in difference:
		for col in row:
#           print col ,
			if col < 0.000001 or col == 0.0 or math.isnan(col) or counter>20:
				nextIteration = False
#   difference = np.abs(oldCentroids,newCentroids)
#   counter += 1
#   for row in difference:
#       for col in row:
#           if col < 0.01:
#               nextIteration = False

print predictedClusters

resultFile = file('predictedwine.csv', 'w')
headerLine = 'ID,PredictedCluster\n'
resultFile.writelines(headerLine)
for i in range(0, len(predictedClusters)):
	line = "" + str(i+1) + "," + str(predictedClusters[i]) + '\n'
	resultFile.writelines(line)
resultFile.close()


#finalCentroids = np.copy(recomputeCentroids(data, predictedClusters, k, initialCentroids))

computeTrueSSE(data, multiCluster, 6)
computeTrueSSB(data, multiCluster, 6)
computeSSE(data, predictedClusters, finalCentroids)
#def computeSSB(data, finalCentroids, k, predictedCluster):
computeSSB(data, finalCentroids, k, predictedClusters)
sumSilhouette = 0
#TODO uncomment for silhouette
for i in range(0, k):
	#for each cluster, calculate silhouette co-efficient
	avgSCWidth = computeSilhouette(data, predictedClusters, finalCentroids, i+1, k)
	print 'Average Silhouette width for cluster ' + str(i+1) + ' - ' + str(avgSCWidth)
	sumSilhouette += avgSCWidth

print 'Average Silhouette width for entire clustering is - ' + str(float(sumSilhouette)/k)
#TODO uncomment till previous line
#computeSilhouette(predictedClusters)
#computeCMatrix(data, predictedClusters) #Use scikit.learn.metrics
Actual = []
Predicted = []
for i in range(0, len(predictedClusters)):
	Predicted.append(predictedClusters[i])
	Actual.append(multiCluster[i])

from sklearn.metrics import confusion_matrix

resArray = confusion_matrix(Actual, Predicted)

print resArray
computeValidation(resArray, k)
fileforSPlot = file('splotwine.csv', 'w')
headline1 = 'ID, fx_acidity,vol_acidity, citric_acid, residual_sugar, chlorides, free sulphates, total sulphates, density, ph, sulphur, alcohol, actualCluster, predictedCluster\n'
fileforSPlot.writelines(headline1)
for i in range(0, len(predictedClusters)):
	line = "" + str(i+1) + "," + str(data[i][1]) + ',' + str(data[i][2]) + ',' + str(data[i][3]) + ',' + str(data[i][4]) + ',' +str(data[i][5]) + ',' + str(data[i][6]) + ',' + str(data[i][7]) + ',' + str(data[i][8]) + ',' + str(data[i][9]) + ',' + str(data[i][10]) + ',' + str(data[i][11]) +  ',' + str(multiCluster[i])+','+ str(predictedClusters[i][0]) + '\n'
	fileforSPlot.writelines(line)
fileforSPlot.close()

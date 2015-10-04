import numpy as np
import matplotlib as pl
import sys
import csv

filename = sys.argv[1]
testfile = sys.argv[2]

#Use euclidean distance first
#let us calculate euclidean distance for iris data set first

def preprocessclass(data):
	classInfo = len(data[0])-1
	for i in range(0, len(data)):
		if(data[i][classInfo] == 'Iris-setosa'):
			data[i][classInfo] = 0
		elif(data[i][classInfo] == 'Iris-versicolor'):
			data[i][classInfo] = 1
		elif(data[i][classInfo] == 'Iris-virginica'):
			data[i][classInfo] = 2 

def recomputeClass(res):
        for i in range(0, len(res)):
                if res[i][1] == 0:
                        res[i][1] = 'Iris-setosa'
                elif res[i][1] == 1:
                        res[i][1] = 'Iris-versicolor'
		elif res[i][1] == 2:
			res[i][1] = 'Iris-virginica'
                if res[i][2] == 0:
                        res[i][2] = 'Iris-setosa'
                elif res[i][2] == 1:
                        res[i][2] = 'Iris-versicolor'
		elif res[i][2] == 2:
			res[i][2] = 'Iris-virginica'
        return res

def findManhattan(record, data, result):
	transidsfortrain = data[:, [0]]
	data = np.delete(data, 0, 1)
	numAttributes = len(data[0])
	trainClasses = data[:, [numAttributes-1]]
	data = np.delete(data, (numAttributes-1), 1)
	counter = 0
	for row in record:
		currentRecord = row[1:-1]
		print currentRecord
		tempResult = np.abs(data - currentRecord)
		tempResult = np.sum(tempResult, axis = 1).reshape(len(tempResult),1)
#		tempResult = np.sqrt(tempResult)
		tempResult = np.hstack((tempResult,trainClasses))
		tempResult = tempResult[np.argsort(tempResult[:,0])]
		result[counter][1] = tempResult[0][0]
		result[counter][2] = tempResult[0][1]
		result[counter][3] = tempResult[1][0]
		result[counter][4] = tempResult[1][1]
		result[counter][5] = tempResult[2][0]
		result[counter][6] = tempResult[2][1]
		
		result[counter][7] = tempResult[3][0]
		result[counter][8] = tempResult[3][1]
		result[counter][9] = tempResult[4][0]
		result[counter][10] = tempResult[4][1]
		#TODO more things will be appended to result if value of n changes
		result[counter][11] = tempResult[5][0]
		result[counter][12] = tempResult[5][1]
		counter += 1

def findDistance(record, data, result):
	transidsfortrain = data[:, [0]]
	data = np.delete(data, 0, 1)
	numAttributes = len(data[0])
	trainClasses = data[:, [numAttributes-1]]
	data = np.delete(data, (numAttributes-1), 1)
	counter = 0
	for row in record:
		currentRecord = row[1:-1]
		print currentRecord
		tempResult = (data - currentRecord)**2
		tempResult = np.sum(tempResult, axis = 1).reshape(len(tempResult),1)
		tempResult = np.sqrt(tempResult)
		tempResult = np.hstack((tempResult,trainClasses))
		tempResult = tempResult[np.argsort(tempResult[:,0])]
		result[counter][1] = tempResult[0][0]
		result[counter][2] = tempResult[0][1]
		result[counter][3] = tempResult[1][0]
		result[counter][4] = tempResult[1][1]
		result[counter][5] = tempResult[2][0]
		result[counter][6] = tempResult[2][1]
		result[counter][7] = tempResult[3][0]
		result[counter][8] = tempResult[3][1]
		result[counter][9] = tempResult[4][0]
		result[counter][10] = tempResult[4][1]
		#TODO more things will be appended to result if value of n changes
		result[counter][11] = tempResult[5][0]
		result[counter][12] = tempResult[5][1]
		counter += 1
		
	

def knnpredictor(data1, data2):
	#data1 is training data
	#data2 is test data
	#print data1
	#print data2
	
	#initialize an empty array where results are stored
	#TODO if we change the number of closest distances, this needs to change
	result = np.zeros((len(data2),14)) #Change '12' to higher value for increased distances
	for i in range(0, len(result)):
		result[i][0] = i+1
	transidsfortest = data2[:, [0]]
	closestDistances = 6 #TODO will change if N changes
	#print data1
	#print numAttributes
	#store the class variables separately
	#finalTrainingData is ready to be processed.
	#find closest points along with classes for each of the test data
	#findManhattan(data2, data1, result) # This is manhattan distance
	findDistance(data2, data1, result) # This is Euclid distance
	for i in range(0, len(data2)):
		result[i][-1] = data2[i][-1]
	#result contains the test record and their closest points along with actual class variable
	result = result.tolist()
	print result
	finalResult = [[] for i in range(len(result))]
	#the class for closest points is in params 3,5,7,9,11. Now we need to make a prediction of which class the test record belongs to
	for index in range(0, len(result)):
		counterA = 0
		counterB = 0
		counterC = 0  #for the different classes available for iris dataset
		finalResult[index].append(result[index][0]) #Append the transaction id
		finalResult[index].append(result[index][-1]) #Append the actual class
		#now, we need to see the class of the closest points and then predict the class for this record
		for i in range(2,13,2): #TODO will change if n changes
			print i
			if(result[index][i] == 0):
				counterA += 1
			elif(result[index][i] == 1):
				counterB += 1
			elif(result[index][i] == 2):
				counterC += 1
			#i = i+2

		predictedClass = 0
		postProb = 0.0
		print counterA, counterB, counterC
		if counterA > counterB:
			if counterA > counterC:
				predictedClass = 0
				postProb = float(counterA)/closestDistances
			else:
				predictedClass = 2
				postProb = float(counterC)/closestDistances
		elif counterB > counterC:
			predictedClass = 1
			postProb = float(counterB)/closestDistances
		elif counterC > counterB:
			predictedClass = 2
			postProb = float(counterC)/closestDistances
		elif counterA == counterB:
			predictedClass = 0
			postProb = float(counterA)/closestDistances
		elif counterA == counterC:
			predictedClass = 2
			postProb = float(counterC)/closestDistances
		elif counterB == counterC:
			predictedClass = 2
			postProb = float(counterC)/closestDistances
		finalResult[index].append(predictedClass)
		finalResult[index].append(postProb)
		print predictedClass
		print postProb
		print finalResult
	
	dTypeforFR = [int, int, int, float]
	for i in range(0, len(finalResult)):
        	finalResult[i] = [t(x) for t,x in zip(dTypeforFR, finalResult[i])]
	print finalResult
	finalResult = recomputeClass(finalResult)
	resultFile = open('predictedClass.csv', 'w')
	headerLine = "Trans_ID,ActualClass,PredictedClass,PostProb\n"
	resultFile.writelines(headerLine)
	for i in range(0, len(finalResult)):
		line = "" + str(finalResult[i][0]) + ',' + str(finalResult[i][1]) + ',' + str(finalResult[i][2]) + ',' + str(finalResult[i][3]) + '\n'
		resultFile.writelines(line)
	resultFile.close()

f1 = open(filename, 'r')
readData = csv.reader(f1, delimiter=',')
data  = []
for row in readData:
        data.append(row)
labels = data[0]
data = data[1:]
#print data
#pre-process data so that final column contains proper numerals for class variables
preprocessclass(data)
#preprocessed data now available
dType = [float, float, float, float, int]
for i in range(0, len(data)):
        data[i] = [t(x) for t,x in zip(dType, data[i])]

for i in range(1, len(data) + 1):
	data[i-1].insert(0, i)

f2 = open(testfile, 'r')
readData2 = csv.reader(f2, delimiter=',')
data_test  = []
for row in readData2:
        data_test.append(row)
#print data_test
data_test = data_test[1:]
preprocessclass(data_test)
for i in range(0, len(data_test)):
	data_test[i] = [t(x) for t,x in zip(dType, data_test[i])]
#data form now
#print data_test
#print data_test
data = np.array(data)

#for each element in data_test, we need to calculate the closest 5 points in training data
#for i in range(0, len(data)):
#	print data[i][len(data[0])-1]

#need to put transaction id for iris data
for i in range(1, len(data_test)+1):
	data_test[i-1].insert(0,i)

data_test = np.array(data_test)
print data_test
#calculate distance and predict class
knnpredictor(data, data_test)

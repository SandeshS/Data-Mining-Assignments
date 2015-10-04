import numpy as np
import matplotlib as pl
import sys
import csv
import scipy.stats as st

datafile = sys.argv[1]
datafileTest = sys.argv[2]

def handleCategorical(res):
	for i in range(0, len(res)):
		if (res[i][1] < 0 or res[i][1] > 1):
			res[i][1] = 1
		if (res[i][3] < 0 or res[i][3] > 1):
			res[i][3] = 1
		if (res[i][4] < 0 or res[i][4] > 1):
			res[i][4] = 1
		if (res[i][5] < 0 or res[i][5] > 1):
			res[i][5] = 1
		if (res[i][6] < 0 or res[i][6] > 1):
			res[i][6] = 1
		if (res[i][7] < 0 or res[i][7] > 1):
			res[i][7] = 1
		if (res[i][12] < 0 or res[i][12] > 1):
			res[i][12] = 1
	return res

def recomputeClass(res):
	for i in range(0, len(res)):
		if res[i][1] == 0:
			res[i][1] = '<=50K'
		elif res[i][1] == 1:
			res[i][1] = '>50K'
		if res[i][2] == 0:
			res[i][2] = '<=50K'
		elif res[i][2] == 1:
			res[i][2] = '>50K'
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
#		print currentRecord
		tempResult = np.abs(data - currentRecord)
		tempResult = handleCategorical(tempResult)
		tempResult = np.sum(tempResult, axis = 1).reshape(len(tempResult),1)
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
		#Extra distances added
                result[counter][13] = tempResult[6][0]
                result[counter][14] = tempResult[6][1]
                result[counter][15] = tempResult[7][0]
                result[counter][16] = tempResult[7][1]
                result[counter][17] = tempResult[8][0]
                result[counter][18] = tempResult[8][1]
                result[counter][19] = tempResult[9][0]
                result[counter][20] = tempResult[9][1]
                result[counter][21] = tempResult[10][0]
                result[counter][22] = tempResult[10][1]
                result[counter][23] = tempResult[11][0]
                result[counter][24] = tempResult[11][1]
                result[counter][25] = tempResult[12][0]
                result[counter][26] = tempResult[12][1]
                result[counter][27] = tempResult[13][0]
                result[counter][28] = tempResult[13][1]
                result[counter][29] = tempResult[14][0]
                result[counter][30] = tempResult[14][1]
                result[counter][31] = tempResult[15][0]
                result[counter][32] = tempResult[15][1]
                result[counter][33] = tempResult[16][0]
                result[counter][34] = tempResult[16][1]
               # result[counter][35] = tempResult[17][0]
               # result[counter][36] = tempResult[17][1]
		counter+=1

def findDistance(record, data, result):
	transidsfortrain = data[:, [0]]
	data = np.delete(data, 0, 1)
	numAttributes = len(data[0])
	trainClasses = data[:, [numAttributes-1]]
	data = np.delete(data, (numAttributes-1), 1)
	counter = 0
	for row in record:
		currentRecord = row[1:-1]
#		print currentRecord
		tempResult = (data - currentRecord)
		tempResult = handleCategorical(tempResult)
		tempResult = tempResult**2
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
		#Extra distances added
                result[counter][13] = tempResult[6][0]
                result[counter][14] = tempResult[6][1]
                result[counter][15] = tempResult[7][0]
                result[counter][16] = tempResult[7][1]
                result[counter][17] = tempResult[8][0]
                result[counter][18] = tempResult[8][1]
                result[counter][19] = tempResult[9][0]
                result[counter][20] = tempResult[9][1]
		#extra additions 10 -20
		
                result[counter][21] = tempResult[10][0]
                result[counter][22] = tempResult[10][1]
                result[counter][23] = tempResult[11][0]
                result[counter][24] = tempResult[11][1]
                result[counter][25] = tempResult[12][0]
                result[counter][26] = tempResult[12][1]
                result[counter][27] = tempResult[13][0]
                result[counter][28] = tempResult[13][1]
                result[counter][29] = tempResult[14][0]
                result[counter][30] = tempResult[14][1]
                result[counter][31] = tempResult[15][0]
                result[counter][32] = tempResult[15][1]
                result[counter][33] = tempResult[16][0]
                result[counter][34] = tempResult[16][1]
#                result[counter][35] = tempResult[17][0]
#                result[counter][36] = tempResult[17][1]
#                result[counter][37] = tempResult[18][0]
#                result[counter][38] = tempResult[18][1]
#                result[counter][39] = tempResult[19][0]
#                result[counter][40] = tempResult[19][1]
		#extra additions 20 - 30
#
#                result[counter][41] = tempResult[20][0]
#                result[counter][42] = tempResult[20][1]
#                result[counter][43] = tempResult[21][0]
#                result[counter][44] = tempResult[21][1]
#                result[counter][45] = tempResult[22][0]
#                result[counter][46] = tempResult[22][1]
#                result[counter][47] = tempResult[23][0]
#                result[counter][48] = tempResult[23][1]
#                result[counter][49] = tempResult[24][0]
#                result[counter][50] = tempResult[24][1]
#                result[counter][51] = tempResult[25][0]
#                result[counter][52] = tempResult[25][1]
#                result[counter][53] = tempResult[26][0]
#                result[counter][54] = tempResult[26][1]
#                result[counter][55] = tempResult[27][0]
#                result[counter][56] = tempResult[27][1]
#                result[counter][57] = tempResult[28][0]
#                result[counter][58] = tempResult[28][1]
#                result[counter][59] = tempResult[29][0]
#                result[counter][60] = tempResult[29][1]
		counter +=1


def knnpredictor(indata1,indata2):
	
	#initialize a result array
	result = np.zeros((len(indata2),36)) #TODO change '14' if nearest neighbors are increased  -- Changed from 14 to 22 6 -10. Changed from 22 to 42 10 - 20. Changed from 42 to 62 20 - 30
	for i in range(0, len(result)):
		result[i][0] = indata2[i][0]
	#print result
	transidsfortest = indata2[:, [0]]
	closestDistances = 17 #TODO will change if nearest neighbour changes - Changed from 6 - 10 and 10 - 20 20 - 30
	findDistance(indata2, indata1, result)  # This is Euclid distance
	#findManhattan(indata2, indata1, result)
	for i in range(0, len(indata2)):
		result[i][-1] = indata2[i][-1]
	result = result.tolist()
	print result
	finalResult = [[] for i in range(len(result))]
	for index in range(0, len(result)):
		below50k = 0
		above50k = 0
		finalResult[index].append(result[index][0])
		finalResult[index].append(result[index][-1])
		for i in range(2, 35, 2): #TODO will change if n changes - changed from 13 to 21 to 41 41 to 61
			if result[index][i] == 0.0:
				below50k += 1
			elif result[index][i] == 1.0:
				above50k += 1
		predictedClass = 0
		postProb = 0.0
		print below50k, above50k
		if below50k > above50k:
			predictedClass = 0
			postProb = float(below50k)/closestDistances
		elif above50k > below50k:
			predictedClass = 1
			postProb = float(above50k)/closestDistances
		else:
			#Default class is 0
			predictedClass = 0
			postProb = float(below50k)/closestDistances
		finalResult[index].append(predictedClass)
		finalResult[index].append(postProb)
		print predictedClass, postProb
	
	dTypeforFR = [int, int, int, float]
        for i in range(0, len(finalResult)):
                finalResult[i] = [t(x) for t,x in zip(dTypeforFR, finalResult[i])]
        print finalResult
	finalResult = recomputeClass(finalResult)
	resultFile = open('predictedIncomeClass.csv', 'w')
	headerLine = "Trans_ID,ActualClass,PredictedClass,PostProb\n"
	resultFile.writelines(headerLine)
        for i in range(0, len(finalResult)):
                line = "" + str(finalResult[i][0]) + ',' + str(finalResult[i][1]) + ',' + str(finalResult[i][2]) + ',' + str(finalResult[i][3]) + '\n'
                resultFile.writelines(line)
        resultFile.close()



def preprocess_workclass(datum):
        for i in range(0, len(datum)):
		if datum[i][2] == " Private":
			datum[i][2] = 1
		elif datum[i][2] == " Federal-gov":
			datum[i][2] = 2
		elif datum[i][2] == " Local-gov":
			datum[i][2] = 3
		elif datum[i][2] == " Never-worked":
			datum[i][2] = 4
		elif datum[i][2] == " Self-emp-inc":
			datum[i][2] = 5
		elif datum[i][2] == " Self-emp-not-inc":
			datum[i][2] = 6
		elif datum[i][2] == " State-gov":
			datum[i][2] = 7
		elif datum[i][2] == " Without-pay":
			datum[i][2] = 8
		elif datum[i][2] == " ?":
			datum[i][2] = -1
	return datum

def preprocess_maritalstatus(datum):
        for i in range(0, len(datum)):
                if datum[i][5] == " Divorced":
                        datum[i][5] = 1
                elif datum[i][5] == " Married-AF-spouse":
                        datum[i][5] = 2
                elif datum[i][5] == " Married-civ-spouse":
                        datum[i][5] = 3
                elif datum[i][5] == " Married-spouse-absent":
                        datum[i][5] = 4
                elif datum[i][5] == " Never-married":
                        datum[i][5] = 5
                elif datum[i][5] == " Separated":
                        datum[i][5] = 6
                elif datum[i][5] == " Widowed":
                        datum[i][5] = 7
        return datum

def preprocess_occupation(datum):
        for i in range(0, len(datum)):
                if datum[i][6] == " Adm-clerical":
                        datum[i][6] = 1
                elif datum[i][6] == " Armed-Forces":
                        datum[i][6] = 2
                elif datum[i][6] == " Craft-repair":
                        datum[i][6] = 3
                elif datum[i][6] == " Exec-managerial":
                        datum[i][6] = 4
                elif datum[i][6] == " Farming-fishing":
                        datum[i][6] = 5
                elif datum[i][6] == " Handlers-cleaners":
                        datum[i][6] = 6
                elif datum[i][6] == " Machine-op-inspct":
                        datum[i][6] = 7
                elif datum[i][6] == " Other-service":
                        datum[i][6] = 8
                elif datum[i][6] == " Priv-house-serv":
                        datum[i][6] = 9
                elif datum[i][6] == " Prof-specialty":
                        datum[i][6] = 10
                elif datum[i][6] == " Protective-serv":
                        datum[i][6] = 11
                elif datum[i][6] == " Sales":
                        datum[i][6] = 12
                elif datum[i][6] == " Tech-support":
                        datum[i][6] = 13
                elif datum[i][6] == " Transport-moving":
                        datum[i][6] = 14
                elif datum[i][6] == " ?":
                        datum[i][6] = -1
        return datum

def preprocess_relations(datum):
        for i in range(0, len(datum)):
                if datum[i][7] == " Husband":
                        datum[i][7] = 1
                elif datum[i][7] == " Not-in-family":
                        datum[i][7] = 2
                elif datum[i][7] == " Other-relative":
                        datum[i][7] = 3
                elif datum[i][7] == " Own-child":
                        datum[i][7] = 4
                elif datum[i][7] == " Unmarried":
                        datum[i][7] = 5
                elif datum[i][7] == " Wife":
                        datum[i][7] = 6
        return datum

def preprocess_race(datum):
        for i in range(0, len(datum)):
                if datum[i][8] == " Black":
                        datum[i][8] = 1
                elif datum[i][8] == " Amer-Indian-Eskimo":
                        datum[i][8] = 2
                elif datum[i][8] == " Asian-Pac-Islander":
                        datum[i][8] = 3
                elif datum[i][8] == " Other":
                        datum[i][8] = 4
                elif datum[i][8] == " White":
                        datum[i][8] = 5
        return datum

def preprocess_gender(datum):
        for i in range(0, len(datum)):
                if datum[i][9] == " Female":
                        datum[i][9] = 0
                elif datum[i][9] == " Male":
                        datum[i][9] = 1
        return datum

def preprocess_country(datum):
        for i in range(0, len(datum)):
                if datum[i][13] == " Cambodia":
                        datum[i][13] = 1
                elif datum[i][13] == " Canada":
                        datum[i][13] = 2
                elif datum[i][13] == " China":
                        datum[i][13] = 3
                elif datum[i][13] == " Columbia":
                        datum[i][13] = 4
                elif datum[i][13] == " Cuba":
                        datum[i][13] = 5
                elif datum[i][13] == " Dominican-Republic":
                        datum[i][13] = 6
                elif datum[i][13] == " Ecuador":
                        datum[i][13] = 7
                elif datum[i][13] == " El-Salvador":
                        datum[i][13] = 8
                elif datum[i][13] == " England":
                        datum[i][13] = 9
                elif datum[i][13] == " France":
                        datum[i][13] = 10
                elif datum[i][13] == " Germany":
                        datum[i][13] = 11
                elif datum[i][13] == " Greece":
                        datum[i][13] = 12
                elif datum[i][13] == " Guatemala":
                        datum[i][13] = 13
                elif datum[i][13] == " Haiti":
                        datum[i][13] = 14
                elif datum[i][13] == " Honduras":
                        datum[i][13] = 15
                elif datum[i][13] == " Hong":
                        datum[i][13] = 16
                elif datum[i][13] == " India":
                        datum[i][13] = 17
                elif datum[i][13] == " Iran":
                        datum[i][13] = 18
                elif datum[i][13] == " Ireland":
                        datum[i][13] = 19
                elif datum[i][13] == " Italy":
                        datum[i][13] = 20
                elif datum[i][13] == " Jamaica":
                        datum[i][13] = 21
                elif datum[i][13] == " Japan":
                        datum[i][13] = 22
                elif datum[i][13] == " Laos":
                        datum[i][13] = 23
                elif datum[i][13] == " Mexico":
                        datum[i][13] = 24
                elif datum[i][13] == " Nicaragua":
                        datum[i][13] = 25
                elif datum[i][13] == " Peru":
                        datum[i][13] = 26
                elif datum[i][13] == " Philippines":
                        datum[i][13] = 27
                elif datum[i][13] == " Poland":
                        datum[i][13] = 28
                elif datum[i][13] == " Portugal":
                        datum[i][13] = 29
                elif datum[i][13] == " Puerto-Rico":
                        datum[i][13] = 30
                elif datum[i][13] == " Scotland":
                        datum[i][13] = 31
                elif datum[i][13] == " South":
                        datum[i][13] = 32
                elif datum[i][13] == " Taiwan":
                        datum[i][13] = 33
                elif datum[i][13] == " Thailand":
                        datum[i][13] = 34
                elif datum[i][13] == " Trinadad&Tobago":
                        datum[i][13] = 35
                elif datum[i][13] == " United-States":
                        datum[i][13] = 36
                elif datum[i][13] == " Vietnam":
                        datum[i][13] = 37
                elif datum[i][13] == " Yugoslavia":
                        datum[i][13] = 38
                elif datum[i][13] == " ?":
                        datum[i][13] = -1
        return datum

def preprocess_class(datum):
	for i in range(0, len(datum)):
		if datum[i][14] == " <=50K":
			datum[i][14] = 0
		elif datum[i][14] == " >50K":
			datum[i][14] = 1
	return datum

def normalize_age(npdata):
	max_age = np.amax(npdata[:,[1]])
	min_age = np.amin(npdata[:, [1]])
#	print max_age, min_age
	for i in range(0, len(npdata)):
		npdata[i][1] = (npdata[i][1] - min_age)/(max_age - min_age)
	return npdata

def plugMissingClass(npdata, value):
	for i in range(0, len(npdata)):
		if npdata[i][2] == -1:
			npdata[i][2] = value
	return npdata

def plugMissingOccupation(npdata, value):
	for i in range(0, len(npdata)):
		if npdata[i][6] == -1:
			npdata[i][6] = value
	return npdata

def plugMissingCountry(npdata, value):
	for i in range(0, len(npdata)):
		if npdata[i][13] == -1:
			npdata[i][13] = value
	return npdata

def normalizeFuncWt(npdata):
	max_wt = np.amax(npdata[:,[3]])
	min_wt = np.amin(npdata[:, [3]])
	for i in range(0, len(npdata)):
		npdata[i][3] = (npdata[i][3] - min_wt)/(max_wt - min_wt)
	return npdata

def normalizeCptGainandLoss(npdata):
	max_gain = np.amax(npdata[:, [10]])
	min_gain = np.amin(npdata[:, [10]])
	max_loss = np.amax(npdata[:, [11]])
	min_loss = np.amin(npdata[:, [11]])
#	print max_gain, min_gain, max_loss, min_loss
	for i in range(0, len(npdata)):
		npdata[i][10] = (npdata[i][10] - min_gain)/(max_gain - min_gain)
		npdata[i][11] = (npdata[i][11] - min_loss)/(max_loss - min_loss)
	return npdata

def normalizeHrsPerWeek(npdata):
	max_hrs = np.amax(npdata[:, [12]])
	min_hrs = np.amin(npdata[:, [12]])
	for i in range(0, len(npdata)):
		npdata[i][12] = (npdata[i][12] - min_hrs)/(max_hrs - min_hrs)
	return npdata


#Read the data into a list. List manipulation is easy since data is of mixed type.
f1 = open(datafile, 'r')
readData = csv.reader(f1, delimiter=',')
data  = []
for row in readData:
        data.append(row)

data = data[1:]
#zip the data to the required formats
dType = [int, float, str, float, str, float, str, str, str, str, str, float, float, float, str, str]
for i in range(0, len(data)):
        data[i] = [t(x) for t,x in zip(dType, data[i])]
#we can remove one attribute - Education which is same as education category
for i in range(0, len(data)):
	result = data[i].pop(4)
#attribute is removed. Data is now ready to undergo further processing
data = preprocess_workclass(data)
data = preprocess_maritalstatus(data)
data = preprocess_occupation(data)
data = preprocess_relations(data)
data = preprocess_race(data)
data = preprocess_gender(data)
data = preprocess_country(data)
data = preprocess_class(data)
#TODO At this point, pre-processed data can be written to the file
npDataTrain = np.array(data, dtype = "float_")

#before normalizing age, calculate the mean value
#meanAgeForTrain = np.mean(npDataTrain[:,[1]])
npDataTrain = normalize_age(npDataTrain)
#After analyzing the data, we can see that only workclass, occupation and country have missing values. We need to compute the mean for these columns and plug the missing values
#### Store the max and min age for training data. Also calculate mean TODO Also - Age value is never missing - so no need to plug in
meanWorkClassForTrain = np.mean(npDataTrain[:,[2]])
meanOccupationForTrain = np.mean(npDataTrain[:, [6]])
meanCountryForTrain = np.mean(npDataTrain[:,[13]])

# Now to  plug the missing values
npDataTrain = plugMissingClass(npDataTrain, meanWorkClassForTrain)
npDataTrain = plugMissingOccupation(npDataTrain, meanOccupationForTrain)
npDataTrain = plugMissingCountry(npDataTrain, meanCountryForTrain)

#now to normalzie some of the attributes
npDataTrain = normalizeFuncWt(npDataTrain)
npDataTrain = normalizeCptGainandLoss(npDataTrain)
npDataTrain = normalizeHrsPerWeek(npDataTrain)
print npDataTrain
print len(npDataTrain)

#
###### Note that the same set of operations is performed for the test data set
#
#
f2 = open(datafileTest, 'r')
readData2 = csv.reader(f2, delimiter=',')
data2  = []
for row in readData2:
        data2.append(row)
data2 = data2[1:]

#TODO take care of label row for both iris and income
#zip the data to the required formats
dType2 = [int, float, str, float, str, float, str, str, str, str, str, float, float, float, str, str]
for i in range(0, len(data2)):
        data2[i] = [t(x) for t,x in zip(dType2, data2[i])]

#we can remove one attribute - Education which is same as education category
for i in range(0, len(data2)):
        result = data2[i].pop(4)
#attribute is removed. Data is now ready to undergo further processing
data2 = preprocess_workclass(data2)
data2 = preprocess_maritalstatus(data2)
data2 = preprocess_occupation(data2)
data2 = preprocess_relations(data2)
data2 = preprocess_race(data2)
data2 = preprocess_gender(data2)
data2 = preprocess_country(data2)
data2 = preprocess_class(data2)
#TODO At this point, pre-processed data can be written to the file
print data2[0]
npDataTest = np.array(data2, dtype="float_")

#plug missing values and then normalize
npDataTest = plugMissingClass(npDataTest, meanWorkClassForTrain)
npDataTest = plugMissingOccupation(npDataTest, meanOccupationForTrain)
npDataTest = plugMissingCountry(npDataTest, meanCountryForTrain)

npDataTest = normalize_age(npDataTest)
npDataTest = normalizeFuncWt(npDataTest)
npDataTest = normalizeCptGainandLoss(npDataTest)
npDataTest = normalizeHrsPerWeek(npDataTest)
print npDataTest
print len(npDataTest)
knnpredictor(npDataTrain, npDataTest)

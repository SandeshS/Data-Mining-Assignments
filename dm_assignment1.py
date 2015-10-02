#from __future__ import division
import numpy as np
import sys
import scipy as sp
import math
import matplotlib.pyplot as plt

dataset = sys.argv[1]
#type of distance can be taken from command line
distance_type = sys.argv[2]
K = 5
#Number of closest points to be caculated

#get the data loaded from the file in the form of a matrix
#iris_data = np.genfromtxt(dataset, delimiter=',')
##print iris_data.shape
#print len(iris_data)
#print iris_data
#cleaning up the data which included NaN
#iris_data = np.delete(iris_data, iris_data.shape[1]-1, axis = 1)
#Adding the index column to the data - might not be needed
#indexes = arange(1,len(iris_data)).reshape(len(iris_data),1)
#iris_data = np.hstack((indexes, iris_data))
#print iris_data

#def manhattan_distances(data):
#	distance_matrix = np.zeros((iris_data.shape[0],iris_data.shape[1]))
#	for i in range(0, data.shape[0]):
		#get the matrix by removing the specific row
#		remaining_matrix = data[range(0,i)+range(i+1,data.shape[1])]
		

#iris_data_filter= iris_data[~np.isnan(iris_data)]
#iris_data_filter.dtype
#iris_data_filter.reshape(len(iris_data), iris_data.shape[1]-1)
#print iris_data_filter

#def categorical_distance(row, data):
#	num_attributes = data.shape[1]
#	spot = data[row]
#	remains = data[range(0, row) + range(row+1, len(data))]
#	for i in range(0, len(remains)):
#		counter = 0
#		for k in range(0, num_attributes):
#			if(data[row][k] == data[i][k]):
#				counter++
#		distance = 1 - (counter/num_attributes)
	

def remove_unwanted_data(data, name):
	names = list(data.dtype.names)
	if name in names:
		names.remove(name)
	new_data = data[names]
	return new_data

def manhattan_distance(data):
	#this distance is just subtracting the values of the attributes and summing them up.
	#for each row, calculate distance with other rows and sum them up. so you would get 149 distances for each row.
	for row in range(0, len(data)):
		#TODO need to remove transids when computing distance
		remaining_data = data[range(0, row) + range(row+1,len(data))]
		#removing and preserving trans_ids
		trans_data = remaining_data[:,[0]]
		#now remove the trans_ids before computing the distance
		remaining_data = np.delete(remaining_data, 0, 1)
		#Correctly prints results -- print remaining_data
		result = abs(remaining_data - data[row,[range(1, iris_data.shape[1])]]);
		#Calculation is correctprint result
		result_new = np.zeros((len(result),1))
		for rowi in range(0, len(result)):
			result_new[rowi] = result[rowi].sum();
		#Append the trans_data so that we know the trans_id
		result_new = np.hstack((trans_data, result_new))	
		#Correctly prints the result here -- print result_new
		#now that we got result, let us sort it to easily get closest five elements
		result_new = result_new[np.argsort(result_new[:,1])]
		#Correctly prints results -- print result_new
		#now write the contents to file for each row
		file_d1 = open('distance_manhattan.csv','a')
		result_line_in_file = '' + str(row+1) + '\t'
		for i in range(0,K):
			result_line_in_file += str(int(result_new[i][0])) + '\t' + str(result_new[i][1]) + '\t'
		result_line_in_file += '\n'
		file_d1.writelines(result_line_in_file)
		file_d1.close()
		#[plt.plot(row, result_new[x][1]) for x in range(0, len(result_new))]
#		print len(result_new)
		
def manhattan_for_income(data):
	for row in range(0, len(data)):
		spotlight = data[range(row,row+1)]
		#save current trans id
		current_id = spotlight[0][0]
		#remove transid from spotlight
		spotlight = remove_unwanted_data(spotlight, 'f0')
		remains = data[range(0,row) + range(row+1,10)]
		trans_id = remains['f0']
#		trans_id.reshape((len(trans_id),1))
		new_ids = np.zeros((len(trans_id),1), dtype='|S10')
		for i in range(0, len(trans_id)):
			new_ids[i] = trans_id[i]
		new_remains = remove_unwanted_data(remains, 'f0')
		results = np.zeros((len(new_remains),1), dtype = '<f8')
		for row2 in range(0, len(new_remains)):
			for attr in range(0,13):
				if attr == 1:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 3:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 4:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 6:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 7:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 12:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				else:
					results[row2] += abs(new_remains[row2][attr] - spotlight[0][attr])
		dist_calc = np.hstack((new_ids,results))
		dist_calc = dist_calc[np.argsort(dist_calc[:,1])]
#		print dist_calc
		file_d2 = open('distance_manhattan_income.csv','a')
		result_in_file = '' + str(current_id) + '\t'
		for j in range(0,K+1):
			result_in_file += str(dist_calc[j][0]) + '\t' + str(dist_calc[j][1]) + '\t'
		result_in_file += '\n'
		file_d2.writelines(result_in_file)
		file_d2.close()
		print '*',
		#results and trans id can be stacked
		#new_remains - spotlight should give result1
		#print abs(new_remains - spotlight)

def euclidean_for_income(data):
	for row in range(0, len(data)):
		spotlight = data[range(row, row+1)]
		#save transaction id
		current_id = spotlight[0][0]
		#remove trans id from spotlight
		spotlight = remove_unwanted_data(spotlight, 'f0')
		remains = data[range(0, row) + range(row+1, len(data))] #len(data)
		trans_id = remains['f0']
		new_ids = np.zeros((len(trans_id),1), dtype='|S10')
		for i in range(0, len(trans_id)):
			new_ids[i] = trans_id[i]
		new_remains = remove_unwanted_data(remains, 'f0')
		results = np.zeros((len(new_remains), 1), dtype='<f8')
		for row2 in range(0, len(new_remains)):
			for attr in range(0,13):
				if attr == 1:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 3:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 4:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 6:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 7:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				elif attr == 12:
					if new_remains[row2][attr] == spotlight[0][attr]:
						results[row2] += 0
					else:
						results[row2] += 1
				else:
					results[row2] += (new_remains[row2][attr] - spotlight[0][attr])**2
		for row2 in range(0, len(new_remains)):
			results[row2] = np.sqrt(results[row2])
		dist_calc = np.hstack((new_ids,results))
		#print dist_calc.dtype.descr
		#dist_num = dist_calc[:,1]
		#dist_num = dist_num.astype(np.float)
		#dist_num = dist_num.reshape((len(dist_num),1))
		#print dist_num
		#preserved_ids = dist_calc[:,0]
		#preserved_ids = preserved_ids.reshape((len(dist_num),1))
		#consolidated_val = np.hstack((preserved_ids, dist_num))
		#print dist_calc[:,1]
		#d1 = consolidated_val.dtype.descr
		#d1[0] = (d1[0][0], '<f8')
		#consolidated_val = consolidated_val.astype(d1)
		#d1[0][1] = (d1[0],'<f8')
		#consolidated_val = consolidated_val.astype(d1)
		#print consolidated_val
		dist_calc = dist_calc[np.argsort(dist_calc[:,1])]
		file_d3 = open('distance_euclidean_income.csv', 'a')
		r_in_file = ''+current_id + '\t'
		for j in range(0, K+1):
			r_in_file += str(dist_calc[j][0]) + '\t' + str(dist_calc[j][1]) + '\t'
		r_in_file += '\n'
		file_d3.writelines(r_in_file)
		file_d3.close()
		print '*',

def euclid_distance(data):
	#this distance is calculated by subtracting the values, taking the square, summing it up and then taking the root of the answer
	for row in range(0, len(data)): #TODO change to len(data) in the end
		#TODO need to remove transids when computing distance
		#remaining_data is going to have all the rows other than the trans_id under consideration
		remaining_data = data[range(0, row) + range(row+1,len(data))]
		#removing and preserving trans_ids
		trans_data = remaining_data[:,[0]]
		#now remove the trans_ids before computing the distance
		remaining_data = np.delete(remaining_data, 0, 1)
		# Can be removed - verified print remaining_data
		result = (remaining_data - data[row,[range(1, iris_data.shape[1])]])**2;
		#print result
		#Trying to do a column-wise sum and reshaping the array
		result = result.sum(1).reshape(len(result), 1)
		result = np.sqrt(result)
		result = np.hstack((trans_data, result))
		result = result[np.argsort(result[:,1])]
		#Correctly prints result -- print result
		file_d2 = open('distance_euclid.csv', 'a')
		result_line_euclid = '' + str(row+1) + '\t'
		for i in range(0,K):
			result_line_euclid += str(int(result[i][0])) + '\t' + str(result[i][1]) + '\t'
		result_line_euclid += '\n'
		file_d2.writelines(result_line_euclid)
		file_d2.close()

def preprocess_age(data):
	max_age = np.amax(data)
	min_age = np.amin(data)
	for i in range(0, len(data)):
		data[i] = (data[i] - min_age)/(float)(max_age - min_age)
		#print data[i]
	#print data

def preprocess_fnlwgt(data):
	max_value = np.amax(data)
	min_value = np.amin(data)
	for i in range(0, len(data)):
		data[i] = (data[i] - min_value)/(float)(max_value - min_value)
	#print data

def preprocess_educat(data):
	max_educat = np.amax(data)
	min_educat = np.amin(data)
	for i in range(0, len(data)):
		data[i] = (data[i] - min_educat)/(max_educat - min_educat)
	#print data

def preprocess_capital_gain(data):
	max_gain = np.amax(data)
	min_gain = np.amin(data)
	for i in range(0, len(data)):
		data[i] = (data[i] - min_gain)/(max_gain - min_gain)
	#print data

def preprocess_capital_loss(data):
	max_loss = np.amax(data)
	min_loss = np.amin(data)
	for i in range(0, len(data)):
		data[i] = (data[i] - min_loss)/(max_loss - min_loss)
	#print data

def preprocess_weekly_hours(data):
	max_hours = np.amax(data)
	min_hours = np.amin(data)
	for i in range(0, len(data)):
		data[i] = (data[i] - min_hours)/(max_hours - min_hours)
	#print data

def preprocess_workclass(data, descr):
	for i in range(0, len(data['f2'])):
		if data['f2'][i] == '" Private"':
			data['f2'][i] = 1
		elif data['f2'][i] == '" Federal-gov"':
			data['f2'][i] = 2
		elif data['f2'][i] == '" Local-gov"':
			data['f2'][i] = 3
		elif data['f2'][i] == '" Never-worked"':
			data['f2'][i] = 4
		elif data['f2'][i] == '" Self-emp-inc"':
			data['f2'][i] = 5
		elif data['f2'][i] == '" Self-emp-not-inc"':
			data['f2'][i] = 6
		elif data['f2'][i] == '" State-gov"':
			data['f2'][i] = 7
		elif data['f2'][i] == '" Without-pay"':
			data['f2'][i] = 8
		elif data['f2'][i] == '" ?"':
			data['f2'][i] = 0.0
	descr[2] = (descr[2][0], '<f8')
	data = data.astype(descr)
	#print data
	return data

def preprocess_maritalstatus(data, descr):
	for i in range(0, len(data['f6'])):
		if data['f6'][i] == '" Divorced"':
			data['f6'][i] = 1
		elif data['f6'][i] == '" Married-AF-spouse"':
			data['f6'][i] = 2
		elif data['f6'][i] == '" Married-civ-spouse"':
			data['f6'][i] = 3
		elif data['f6'][i] == '" Married-spouse-absen':
			data['f6'][i] = 4
		elif data['f6'][i] == '" Never-married"':
			data['f6'][i] = 5
		elif data['f6'][i] == '" Separated"':
			data['f6'][i] = 6
		elif data['f6'][i] == '" Widowed"':
			data['f6'][i] = 7
	descr[5] = (descr[5][0], '<f8')
	data = data.astype(descr)
	#print data
	return data

def preprocess_occupation(data, descr):
	for i in range(0, len(data['f7'])):
		if data['f7'][i] == '" Adm-clerical"':
			data['f7'][i] = 1
		elif data['f7'][i] == '" Armed-Forces"':
			data['f7'][i] = 2
		elif data['f7'][i] == '" Craft-repair"':
			data['f7'][i] = 3
		elif data['f7'][i] == '" Exec-managerial"':
			data['f7'][i] = 4
		elif data['f7'][i] == '" Farming-fishing"':
			data['f7'][i] = 5
		elif data['f7'][i] == '" Handlers-cleaners"':
			data['f7'][i] = 6
		elif data['f7'][i] == '" Machine-op-inspct"':
			data['f7'][i] = 7
		elif data['f7'][i] == '" Other-service"':
			data['f7'][i] = 8
		elif data['f7'][i] == '" Priv-house-serv"':
			data['f7'][i] = 9
		elif data['f7'][i] == '" Prof-specialty"':
			data['f7'][i] = 10
		elif data['f7'][i] == '" Protective-serv"':
			data['f7'][i] = 11
		elif data['f7'][i] == '" Sales"':
			data['f7'][i] = 12
		elif data['f7'][i] == '" Tech-support"':
			data['f7'][i] = 13
		elif data['f7'][i] == '" Transport-moving"':
			data['f7'][i] = 14
		elif data['f7'][i] == '" ?"':
			data['f7'][i] = 0.0
	descr[6] = (descr[6][0], '<f8')
	data = data.astype(descr)
	#print data
	return data

def preprocess_relations(data, descr):
	for i in range(0, len(data['f8'])):
		if data['f8'][i] == '" Husband"':
			data['f8'][i] = 1
		elif data['f8'][i] == '" Not-in-family"':
			data['f8'][i] = 2
		elif data['f8'][i] == '" Other-relative"':
			data['f8'][i] = 3
		elif data['f8'][i] == '" Own-child"':
			data['f8'][i] = 4
		elif data['f8'][i] == '" Unmarried"':
			data['f8'][i] = 5
		elif data['f8'][i] == '" Wife"':
			data['f8'][i] = 6
	descr[7] = (descr[7][0], '<f8')
	data = data.astype(descr)
	#print data
	return data

def preprocess_race(data, descr):
	for i in range(0, len(data['f9'])):
		if data['f9'][i] == '" Black"':
			data['f9'][i] = 1
		elif data['f9'][i] == '" Amer-Indian-Eskimo"':
			data['f9'][i] = 2
		elif data['f9'][i] == '" Asian-Pac-Islander"':
			data['f9'][i] = 3
		elif data['f9'][i] == '" Other"':
			data['f9'][i] = 4
		elif data['f9'][i] == '" White"':
			data['f9'][i] = 5
	descr[8] = (descr[8][0], '<f8')
	data = data.astype(descr)
	#print data
	return data

def preprocess_gender(data, descr):
	for i in range(0, len(data['f10'])):
		if data['f10'][i] == '" Female"':
			data['f10'][i] = 0
		elif data['f10'][i] == '" Male"':
			data['f10'][i] = 1
	descr[9] = (descr[9][0], '<f8')
	data = data.astype(descr)
	return data

def preprocess_country(data, descr):
	for i in range(0, len(data['f14'])):
		if data['f14'][i] == '" Cambodia"':
			data['f14'][i] = 1
		elif data['f14'][i] == '" Canada"':
			data['f14'][i] = 2
		elif data['f14'][i] == '" China"':
			data['f14'][i] = 3
		elif data['f14'][i] == '" Columbia"':
			data['f14'][i] = 4
		elif data['f14'][i] == '" Cuba"':
			data['f14'][i] = 5
		elif data['f14'][i] == '" Dominican-Republic"':
			data['f14'][i] = 6
		elif data['f14'][i] == '" Ecuador"':
			data['f14'][i] = 7
		elif data['f14'][i] == '" El-Salvador"':
			data['f14'][i] = 8
		elif data['f14'][i] == '" England"':
			data['f14'][i] = 9
		elif data['f14'][i] == '" France"':
			data['f14'][i] = 10
		elif data['f14'][i] == '" Germany"':
			data['f14'][i] = 11
		elif data['f14'][i] == '" Greece"':
			data['f14'][i] = 12
		elif data['f14'][i] == '" Guatemala"':
			data['f14'][i] = 13
		elif data['f14'][i] == '" Haiti"':
			data['f14'][i] = 14
		elif data['f14'][i] == '" Honduras"':
			data['f14'][i] = 15
		elif data['f14'][i] == '" Hong"':
			data['f14'][i] = 16
		elif data['f14'][i] == '" India"':
			data['f14'][i] = 17
		elif data['f14'][i] == '" Iran"':
			data['f14'][i] = 18
		elif data['f14'][i] == '" Ireland"':
			data['f14'][i] = 19
		elif data['f14'][i] == '" Italy"':
			data['f14'][i] = 20
		elif data['f14'][i] == '" Jamaica"':
			data['f14'][i] = 21
		elif data['f14'][i] == '" Japan"':
			data['f14'][i] = 22
		elif data['f14'][i] == '" Laos"':
			data['f14'][i] = 23
		elif data['f14'][i] == '" Mexico"':
			data['f14'][i] = 24
		elif data['f14'][i] == '" Nicaragua"':
			data['f14'][i] = 25
		elif data['f14'][i] == '" Peru"':
			data['f14'][i] = 26
		elif data['f14'][i] == '" Philippines"':
			data['f14'][i] = 27
		elif data['f14'][i] == '" Poland"':
			data['f14'][i] = 28
		elif data['f14'][i] == '" Portugal"':
			data['f14'][i] = 29
		elif data['f14'][i] == '" Puerto-Rico"':
			data['f14'][i] = 30
		elif data['f14'][i] == '" Scotland"':
			data['f14'][i] = 31
		elif data['f14'][i] == '" South"':
			data['f14'][i] = 32
		elif data['f14'][i] == '" Taiwan"':
			data['f14'][i] = 33
		elif data['f14'][i] == '" Thailand"':
			data['f14'][i] = 34
		elif data['f14'][i] == '" Trinadad&Tobago"':
			data['f14'][i] = 35
		elif data['f14'][i] == '" United-States"':
			data['f14'][i] = 36
		elif data['f14'][i] == '" Vietnam"':
			data['f14'][i] = 37
		elif data['f14'][i] == '" Yugoslavia"':
			data['f14'][i] = 38
		elif data['f14'][i] == '" ?"':
			data['f14'][i] = 0.0
	descr[13] = (descr[13][0], '<f8')
	data = data.astype(descr)
	#print data
	return data

def plugin_workclass(data):
	mean_wc = math.ceil((sum(data['f2']))/len(data['f2']))
	for i in range(0, len(data['f2'])):
		if data['f2'][i] == 0.0:
			data['f2'][i] = mean_wc
		
def plugin_occupation(data):
	mean_occup = math.ceil((sum(data['f7']))/len(data['f7']))
	for i in range(0, len(data['f7'])):
		if data['f7'][i] == 0.0:
			data['f7'][i] = mean_occup

def plugin_country(data):
	mean_ctry = math.ceil((sum(data['f14']))/len(data['f14']))
	for i in range(0, len(data['f14'])):
		if data['f14'][i] == 0.0:
			data['f14'][i] = mean_ctry


def normalize_workclass(data):
	for i in range(0, len(data['f2'])):
		data['f2'][i] = (data['f2'][i] - 1)/(8-1)

def normalize_occupation(data):
	for i in range(0, len(data['f7'])):
		data['f7'][i] = (data['f7'][i] - 1)/(14-1)

def normalize_country(data):
	for i in range(0, len(data['f14'])):
		data['f14'][i] = (data['f14'][i] - 1)/(38-1)

def normalize_marital(data):
	for i in range(0, len(data['f6'])):
		data['f6'][i] = (data['f6'][i] - 1)/(7-1)

def normalize_relations(data):
	for i in range(0, len(data['f8'])):
		data['f8'][i] = (data['f8'][i] - 1)/(6-1)

def normalize_race(data):
	for i in range(0, len(data['f9'])):
		data['f9'][i] = (data['f9'][i] - 1)/(5-1)

if (dataset == "iris.data"):
	#get data
	iris_data = np.genfromtxt(dataset, delimiter=',')
	#clean data - removes the class variable which is NaN
	iris_data = np.delete(iris_data, iris_data.shape[1]-1, axis = 1)
	#TODO remove this when needed -- print iris_data
	#we need to add a column for transaction id
	trans_ids = np.arange(1,iris_data.shape[0]+1, dtype=int).reshape(iris_data.shape[0],1)
	#TODO remove the comment when needed print trans_ids
	iris_data = np.hstack((trans_ids,iris_data))
	#TODO check if trans id is printed correctly
	#remove comment when needed -- print iris_data
	#clean data is now obtained. Now we need to compute distance and find closest matches upto 5 of them

	#prepocess(data)
	#Write header for manhattan distance file
	if distance_type == 'manhattan':
		headerline = "ID" + '\t' + '1st' + '\t' + 'dist\t2nd\tdist\t3rd\tdist\t4th\tdist\t5th\tdist\n'
		file1 = open('distance_manhattan.csv', 'w')
		file1.writelines(headerline)
		file1.close()
		#calculate manhattan distance
		manhattan_distance(iris_data)
		#plt.show()

	elif distance_type == 'euclid':
		#Write header for euclid distance file
		headerline = "ID" + '\t' + '1st' + '\t' + '1st dist\t2nd\t2nd dist\t3rd\t3rd dist\t4th\t4th dist\t5th\t5th dist\n'
		file1 = open('distance_euclid.csv', 'w')
		file1.writelines(headerline)
		file1.close()
		euclid_distance(iris_data)

elif (dataset == "income_NEW.csv"):
	#get data
	income_data = np.genfromtxt('income_NEW.csv', delimiter=',', dtype = "S22, f8, S22, f8, S22, f8, S22, S22, S22, S22, S22, f8, f8, f8, S22")
	#delete header
	income_data = np.delete(income_data, 0, 0)
	#prepare the data for processing
	#normalize the age
	preprocess_age(income_data['f1'])
	#normalize fnlwgt
	preprocess_fnlwgt(income_data['f3'])
	#since the education and education_cat columns provide the same data, we can collapse one of the columns
	income_data = remove_unwanted_data(income_data, 'f4')
	#now that unwanted data is removed, we will normalize the ordinal categorical attribute which is edu_cat
	preprocess_educat(income_data['f5'])
	descriptor = income_data.dtype.descr
	#normalize capital gain, capital loss and hours_per_week values
	preprocess_capital_gain(income_data['f11'])
	preprocess_capital_loss(income_data['f12'])
	preprocess_weekly_hours(income_data['f13'])
	#preprocess the occupation data and categorize it
	income_data = preprocess_workclass(income_data, descriptor)
	#TODO occupation will have negative values for missing data. need to plug it in with mean value of the attribute.
	income_data = preprocess_maritalstatus(income_data, descriptor)
#	new_descriptor = descriptor
#	new_descriptor[2] = (new_descriptor[2][0], '<f8')
#	income_data = income_data.astype(new_descriptor)
	#marital status does not have missing values - no plugging in needed
	income_data = preprocess_occupation(income_data, descriptor)
	#TODO occupation has missing values - need to plug in mean
	income_data = preprocess_relations(income_data, descriptor)
	#No missing values from here on. no further plugging needed 
	income_data = preprocess_race(income_data, descriptor)
	income_data = preprocess_gender(income_data, descriptor)
	income_data = preprocess_country(income_data, descriptor)
	#TODO country has missing values. need to plug in 
	#first steps - normalize new data
	plugin_workclass(income_data)
	plugin_occupation(income_data)
	plugin_country(income_data)
#	normalize_workclass(income_data)
#	normalize_marital(income_data)
#	normalize_occupation(income_data)
#	normalize_relations(income_data)
#	normalize_race(income_data)
#	normalize_country(income_data)
	if distance_type == 'manhattan':
		headerline = "ID" + '\t' + '1st' + '\t' + 'dist\t2nd\tdist\t3rd\tdist\t4th\tdist\t5th\tdist\t6th\tdist\n'
		file1 = open('distance_manhattan_income.csv', 'w')
		file1.writelines(headerline)
		file1.close()
		#calculate manhattan distance
		manhattan_for_income(income_data)	

	elif distance_type == 'euclid':
		#Write header for euclid distance file
		headerline = "ID" + '\t' + '1st' + '\t' + '1st dist\t2nd\t2nd dist\t3rd\t3rd dist\t4th\t4th dist\t5th\t5th dist\t6th\t6thdist\n'
		file1 = open('distance_euclidean_income.csv', 'w')
		file1.writelines(headerline)
		file1.close()
		euclidean_for_income(income_data)

#def manhattan_distance(data):
#this distance is just subtracting the values of the attributes and summing them up.

#def euclidean_distance(data):
#this distance follows from pythagorean theorem. given two points, take square of the difference. then sum up.


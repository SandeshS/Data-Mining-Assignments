# Data-Mining-Assignments
Some code up on GitHub!

This is a simple distance metric calculator for dataset containing income data.
The code is not the most efficient. I was adamant on using Python for this particular assignment as it has a great amount of flexibility.

Another point of interest is that, the code is not super efficient. I have used numpy arrays to store data since it is super fast to carry out operations on numpy arrays. However, I did mess up while reading from the csv file. This will be fixed in the next set of code.

The code is now fixed in Assignment2 to run much faster on datasets. Some changes that were made -
i) Use of lists to initially read from csv file - Since the data is of different types, reading directly to a numpy array makes it harder to handle the data. By changing to lists, which are - by default - a data type whhich support different data types, the program is easier to write and cases are easy to handle.
ii) Convert to a numpy array only once all the normalization and plugging of missing values has taken place. It is relatively easy if numpy array - as the name suggests - has numerical form of data. 

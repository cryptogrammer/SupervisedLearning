from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np
from pybrain.utilities import percentError
from sklearn.neighbors import KNeighborsClassifier
import sys
import matplotlib.pyplot as plt
import math
import time

## calculate the root mean square error
def rmsError(actual_y, predicted_y):
    net = 0
    for k in range(len(actual_y)):
        net = net + (actual_y[k] - predicted_y[k])*(actual_y[k] - predicted_y[k])
    return math.sqrt(net/len(actual_y))

def binaryError(actual_y, predicted_y):
    net = 0
    for k in range(len(actual_y)):
        if(actual_y[k] != predicted_y[k]):
		net = net+1
    net = net*1.0
    return net/len(actual_y)

######## GENERATE INSTANCES ########
f = open('splice.train', 'r')
set_one_train = [ [n for n in line.rstrip().split(' ')] for line in f.readlines() ]
f.close()

f = open('splice.test', 'r')
set_one_test = [ [n for n in line.rstrip().split(' ')] for line in f.readlines() ]
f.close()

for x in set_one_train:
    for y in range(1, len(x)):
        x[y] = (y+1)*(ord(x[y])-64)

one_train_x = [ a[1:] for a in set_one_train ]
one_train_y = [ a[0] for a in set_one_train ]

for m in range(len(one_train_y)):
    if(one_train_y[m]=="IE"):
        one_train_y[m] = 0
    elif(one_train_y[m]=="EI"):
        one_train_y[m] = 1
    else:
        one_train_y[m] = 2

#for m in range(len(one_train_x)):
#   one_train_x[m] = sum(one_train_x[m])

for x in set_one_test:
    for y in range(1, len(x)):
        x[y] = (y+1)*(ord(x[y])-64)


one_test_x = [ a[1:] for a in set_one_test ]
one_test_y = [ a[0] for a in set_one_test ]

for m in range(len(one_test_y)):
    if(one_train_y[m]=="IE"):
        one_test_y[m] = 0
    elif(one_train_y[m]=="EI"):
        one_test_y[m] = 1
    else:
        one_test_y[m] = 2

#for m in range(len(one_test_x)):
#   one_test_x[m] = sum(one_test_x[m])

one_train_x = one_train_x[:len(one_train_x)/2]
one_train_y = one_train_y[:len(one_train_y)/2]
one_test_x = one_test_x[:len(one_test_x)]
one_test_y = one_test_y[:len(one_test_y)]


def createScatterPlot(xLabel, yLabel, xData, yData, filename):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, yData, 'o')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')

def createComparisonPlot(xLabel, yLabel, xData, y1Data, y2Data, filename, linename):
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(xData, y1Data, color = 'blue')
    plt.plot(xData, y2Data, color = 'red')
    plt.legend(linename)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')



######## SUPPORT VECTOR MACHINES ########
print "SUPPORT VECTOR MACHINES"
svc = svm.SVC(kernel='linear')
#print one_train_x
#print one_train_y
svc.fit(one_train_x, one_train_y)
print "FIT"
start = time.time()
one_pred_y = [ svc.predict(x)[0] for x in one_test_x ]
print "PREDICT"
rms_test = binaryError(one_test_y, one_pred_y)
timeTaken = time.time() - start
one_pred_train = [ svc.predict(x)[0] for x in one_train_x ]
print "TRAIN PREDICT"
rms_train = binaryError(one_train_y, one_pred_train)
#createScatterPlot('X', 'Actual Y', one_test_x, error, 'scattersvm.pdf')
print "*"*50

######## kNN ########
# print "k-NEAREST NEIGHBORS"
# neighbors = 0
# neighborArray = np.zeros([50])
# for m in range(50):
#     neighborArray[m] = m+1
# rms_test = np.zeros([50])
# rms_train = np.zeros([50])
# timeTaken = np.zeros([50])
# for x in range(50):
#     neighbors = neighbors + 1
#     neigh = KNeighborsClassifier(n_neighbors=neighbors)
#     start = time.time()
#     neigh.fit(one_train_x, one_train_y)
#     one_pred_y = [neigh.predict(x)[0] for x in one_test_x]
#     timeTaken[neighbors-1] = time.time()-start
#     one_pred_train = [neigh.predict(x)[0] for x in one_train_x]
#     rms_test[neighbors-1] = rmsError(one_test_y, one_pred_y)
# #    rms_train[neighbors-1] = rmsError(one_train_y, one_pred_train)
#     print neighbors
#createComparisonPlot('K value', 'RMS Error', neighborArray, rms_test, rms_train, 'comparison.pdf', ['Out-of-Sample Data', 'In-Sample Data'])
print "TRAINING ERROR"
print rms_train
print "TESTING ERROR"
print rms_test

print "TRAINING SET SIZE"
print len(one_train_y)

print "TIME TAKEN"
print timeTaken

#createScatterPlot('K values', 'RMS Error', neighborArray, rms_test, 'scatterknn.pdf')

if __name__ == '__main__':
	print sys.argv
	if len(sys.argv) > 1:
		classifier = sys.argv[2]

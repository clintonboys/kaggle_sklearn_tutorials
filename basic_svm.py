__author__ = 'cboys'
import numpy as np
from sklearn import svm
import csv

train = np.loadtxt(open('train.csv', 'rb'), delimiter=",", skiprows=0)
trainLabels = np.loadtxt(open('trainLabels.csv', 'rb'), delimiter=',', skiprows=0)
test=np.loadtxt(open('test.csv','rb'), delimiter=',', skiprows=0)

clf = svm.LinearSVC()
clf.fit(train, trainLabels)
predictions = clf.predict(test)

with open('basic_svm_submission.csv', 'wb') as prediction_file:
    writer=csv.writer(prediction_file, delimiter=',')
    writer.writerow(['Id','Solution'])
    for i in range(0,len(predictions)):
        writer.writerow([i+1,int(predictions[i])])

# scores around 80%
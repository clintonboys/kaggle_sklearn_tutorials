__author__ = 'cboys'
##
## svm_model.py
##

import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA # package for principal
                                          # component analysis
from sklearn import svm
import csv



X_train = pd.read_csv('train.csv', header=None).as_matrix()
X_test = pd.read_csv('test.csv', header=None).as_matrix()
trainLabels = np.loadtxt(open('trainLabels.csv', 'rb'), delimiter=',', skiprows=0)

pca=PCA(n_components=12, whiten=True)
#pca.fit(np.r_[X_train, X_test],trainLabels)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf = svm.SVC(C=3, gamma=0.6)
clf.fit(X_train_pca,trainLabels)

predictions = clf.predict(X_test_pca)

with open('svm_model_submission.csv', 'wb') as prediction_file:
    writer=csv.writer(prediction_file, delimiter=',')
    writer.writerow(['Id','Solution'])
    for i in range(0,len(predictions)):
        writer.writerow([i+1,int(predictions[i])])

# scores around 92%, could maybe get a bit better tweaking parameters for SVC
# -- use GridSearch to do this? Need a way of testing "goodness" of model
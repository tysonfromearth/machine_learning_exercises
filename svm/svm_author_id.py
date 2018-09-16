#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### my code and explanatory text ###

# import
from sklearn import svm
from sklearn.metrics import accuracy_score

# cut training data down to 1% original size (for optimizing C parameter only)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

# define classifier as clf
clf = svm.SVC(C=10000.0, kernel = 'rbf')

# fit the classifier, time the fit, and print results of timing test
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# make predictions using the fitted classifer, assign to pred, time, and print
# results of timing test
t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

# test the accuracy_score, define as accuracy, and print results
accuracy = accuracy_score(pred, labels_test)
print "accuracy score", accuracy

# pull and print predictions for observations 10, 26, and 50
answer_10 = pred[10]
answer_26 = pred[26]
answer_50 = pred[50]

print "Prediction for observation 10:", answer_10
print "Prediction for observation 26:", answer_26
print "Prediction for observation 50:", answer_50

# how many test events predicted to be Chris(1)?
print "There are", sum(pred), "emails predicted to be Chris."

#########################################################

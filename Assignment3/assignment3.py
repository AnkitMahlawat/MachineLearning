from oct2py import Oct2Py
from sklearn.svm import SVR
from scipy.stats import pearsonr
from pandas import DataFrame
import cv2
import random 
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

X = np.load('Assignment2/feature.npy')
list1=np.load('Assignment2/SS.npy')
lis=np.load('roundoff.npy')
y=lis+list1
sum_LG=0
sum_MLP=0
sum_SVR=0
for j in range(10):
    sum1=0
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train=np.round(y_train)/2
        y_test-=np.round(y_test)/2
        clf = LogisticRegression().fit(X_train, y_train)
        predict=clf.predict(X_test)
        sum1+=pearsonr(y_test,predict)[0]
    sum_LG+=sum1
    print('avg plcc logistic regression:',sum1/5)
    sum1=0
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train=np.round(y_train)/2
        y_test-=np.round(y_test)/2
        clf = MLPClassifier().fit(X_train, y_train)
        predict=clf.predict(X_test)
        sum1+=pearsonr(y_test,predict)[0]
    print('avg plcc neural network:',sum1/5)
    sum_MLP+=sum1
    sum1=0
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, list1, test_size=0.2)
        clf = SVR(C=1.0e3).fit(X_train, y_train)
        predict=clf.predict(X_test)
        sum1+=pearsonr(y_test,predict)[0]
    print('avg plcc SVR:',sum1/5)
    sum_SVR+=sum1
    print(j)
print(sum_LG/50,sum_MLP/50,sum_SVR/50)

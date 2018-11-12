from oct2py import Oct2Py
from sklearn.svm import SVR
from scipy.stats import pearsonr
import numpy as np
from pandas import DataFrame
import cv2
import random 
di='databaserelease2/'
ref_dir=di+'refimgs/'
arr_folder=[di+'jp2k/',di+'jpeg/',di+'wn/',di+'gblur/',di+'fastfading/']
file=open(di+'SS.txt','r')
list1=file.read().split()
file.close()
list1 = [float(i) for i in list1]
matrix = open('feature-mat.txt').read().split()
x1 = [float(i) for i in matrix]
X = np.reshape(x1, (int(len(x1)/128), 128))
info = open('info.txt').read().split()
for j in range(10):
    X_tr=np.zeros((1,128))
    y=[]
    string=[]
    ran=random.sample(range(982), 982)
#     print(ran)
    ran.sort()
#     print(ran)
    for i in ran:
        X_tr=np.append(X_tr,[X[i]],axis=0)
        y.append(list1[i])
    X_tr = np.delete(X_tr,[0],0)
    clf = SVR()
    clf.fit(X_tr,y)
    ran=random.sample(range(982), 200)
    ran.sort()
    X_test=np.zeros((1,128))
    y1=[]
    for i in ran:
        X_test=np.append(X_test,[X[i]],axis=0)
        y1.append(list1[i])
        if(i<227):
            string.append(arr_folder[0]+info[3*i+1])
        elif(i<460):
            string.append(arr_folder[1]+info[3*i+1])
        elif(i<634):
            string.append(arr_folder[2]+info[3*i+1])
        elif(i<808):
            string.append(arr_folder[3]+info[3*i+1])
        else:
            string.append(arr_folder[4]+info[3*i+1])
    print(string)
    X_test = np.delete(X_test,[0],0)
    lst=clf.predict(X_test)
    plcc=pearsonr(y1,lst)
    if(plcc[0]>0.94):
        df = DataFrame({'File_Name': string,'DMOS value': y1, 'Predicted Score': lst})
        df.to_excel('Subjective_Score.xlsx', sheet_name='sheet1', index=False)
    print(plcc)

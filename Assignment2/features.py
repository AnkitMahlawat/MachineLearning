from oct2py import Oct2Py
from sklearn.svm import SVR
from scipy.stats import pearsonr
import numpy as np
import cv2
di='databaserelease2/'
ref_dir=di+'refimgs/'
arr_folder=[di+'jp2k/',di+'jpeg/',di+'wn/',di+'gblur/',di+'fastfading/']
X=np.zeros((1,256))
with Oct2Py() as oc:
    oc.eval('pkg load signal')
    for folder in arr_folder:
        print(folder)
        file = open(folder+'info.txt','r')
        string = file.read().split()
        length=len(string)/3
        print(length)
        i=0
        while i<length:
            img_o = cv2.imread(ref_dir+string[3*i],0)    
            img_d = cv2.imread(folder+string[3*i+1],0)    
            arr = oc.feature_smc1(img_o,img_d)
            X=np.append(X,arr, axis=0)
            print(i,end="")    
            i+=1
        file.close()
X=np.delete(X,[0],0)
with open('feature-mat.txt','wb') as f:
    for line in X:
        np.savetxt(f, line, fmt='%.2f')
f.close()
print(X,np.shape(X))
np.save('feature.npv',X)





import numpy as np
import cv2
import math as mt
import glob

def psnr(img):
    altrow=img[::2]
    arr = np.arange(512)
    altcol=arr[1::2]
    finalmat=np.delete(altrow, altcol,1)
    finalmat=finalmat/255
    sixteen=16
    for a in range(0,256,sixteen):
        for b in range(0,256,sixteen):
            if a!=0 and b!=0 and a!=256-sixteen and b!=256-sixteen:
                for i in range(a,sixteen+a):
                    for j in range(b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a==0 and b!=0 and a!=256-sixteen and b!=256-sixteen:
                for i in range(1+a,sixteen+a):
                    for j in range(b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a!=0 and b==0 and a!=256-sixteen and b!=256-sixteen:
                for i in range(a,sixteen+a):
                    for j in range(1+b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a!=0 and b!=0 and a==256-sixteen and b!=256-sixteen:
                for i in range(a,sixteen-1+a):
                    for j in range(b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a!=0 and b!=0 and a!=256-sixteen and b==256-sixteen:
                for i in range(a,sixteen+a):
                    for j in range(b,sixteen-1+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a==0 and b==0:
                for i in range(1+a,sixteen+a):
                    for j in range(1+b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a==256-sixteen and b==0:
                for i in range(a,sixteen-1+a):
                    for j in range(1+b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a==0 and b==256-sixteen:
                for i in range(1+a,sixteen+a):
                    for j in range(b,sixteen-1+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            elif a==256-sixteen and b==256-sixteen:
                for i in range(a,sixteen-1+a):
                    for j in range(b,sixteen-1+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j-1],finalmat[i-1][j+1],finalmat[i+1][j-1],finalmat[i+1][j+1]]],0)
            train_arrT=np.transpose(train_arr)
            try:
                beta=np.matmul(np.linalg.inv(np.matmul(train_arrT,train_arr)),np.matmul(train_arrT,Y_mat))
            except:
                beta = np.array([[0.25],[0.25],[0.25],[0.25]])
            # beta=abs(beta)
            if a==0 and b==0:
                img1=img.copy()
            for i in range(1+2*a,2*sixteen+2*a,2):
                for j in range(1+2*b,2*sixteen+2*b,2):
                    if i!=511 and j !=511:
                        img1[i][j]=(beta[0][0]*img1[i-1][j-1]+beta[1][0]*img1[i-1][j+1]+beta[2][0]*img1[i+1][j-1]+beta[3][0]*img1[i+1][j+1])/(beta[0][0]+beta[1][0]+beta[2][0]+beta[3][0])
            #2nd round
            if a!=0 and b!=0 and a!=256-sixteen and b!=256-sixteen:
                for i in range(a,sixteen+a):
                    for j in range(b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a==0 and b!=0 and a!=256-sixteen and b!=256-sixteen:
                for i in range(1+a,sixteen+a):
                    for j in range(b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a!=0 and b==0 and a!=256-sixteen and b!=256-sixteen:
                for i in range(a,sixteen+a):
                    for j in range(1+b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a!=0 and b!=0 and a==256-sixteen and b!=256-sixteen:
                for i in range(a,sixteen-1+a):
                    for j in range(b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a!=0 and b!=0 and a!=256-sixteen and b==256-sixteen:
                for i in range(a,sixteen+a):
                    for j in range(b,sixteen-1+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a==0 and b==0:
                for i in range(1+a,sixteen+a):
                    for j in range(1+b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a==256-sixteen and b==0:
                for i in range(a,sixteen-1+a):
                    for j in range(1+b,sixteen+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a==0 and b==256-sixteen:
                for i in range(1+a,sixteen+a):
                    for j in range(b,sixteen-1+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            elif a==256-sixteen and b==256-sixteen:
                for i in range(a,sixteen-1+a):
                    for j in range(b,sixteen-1+b):
                        if i==1+a and j==1+a:
                            Y_mat = np.array([[finalmat[i][j]]])
                            train_arr=np.array([[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]])
                        else:
                            Y_mat=np.append(Y_mat,[[finalmat[i][j]]],0)
                            train_arr=np.append(train_arr,[[finalmat[i-1][j],finalmat[i][j+1],finalmat[i][j-1],finalmat[i+1][j]]],0)
            train_arrT=np.transpose(train_arr)
            try:
                beta = np.array([[0.5],[0.5],[0.5],[0.5]])
            except:
                beta = np.array([[0.25],[0.25],[0.25],[0.25]])
            # beta=abs(beta)
            for i in range(2*a,2*sixteen+2*a,2):
                for j in range(1+2*b,2*sixteen+2*b,2):
                    if i!=0 and j!=511:
                        img1[i][j]=(beta[1][0]*img1[i][j+1]+beta[2][0]*img1[i][j-1])/(beta[1][0]+beta[2][0])
            for i in range(1+2*a,2*sixteen+2*a,2):
                for j in range(2*b,2*sixteen+2*b,2):
                    if i!=511 and j!=0:
                        img1[i][j]=(beta[0][0]*img1[i-1][j]+beta[3][0]*img1[i+1][j])/(beta[0][0]+beta[3][0])
    imgz=img-img
    for i in range(0,512):
        for j in range(0,512):
            if (img[i][j])>=(img1[i][j]):
                imgz[i][j]=img[i][j]-img1[i][j]
            else:
                imgz[i][j]=img1[i][j]-img[i][j]
    sumz=0
    for i in range(0,512):
        for j in range(0,512):
            sumz+=int(imgz[i][j])*int(imgz[i][j])        
    sumz=int(sumz/(512*512))
    psnr=10*(mt.log10(255*255/sumz))
    print(psnr)
    # cv2.imshow('image',img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return psnr
images=[]
avg=0
for file in glob.glob("standard_test_images/*.tif"):
    #print(file)
    avg+=psnr( cv2.imread(file,0) )    
print("Average psnr of all images ",avg/12)

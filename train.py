import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

#add your folder name here and place it with main.py
cl = ['Cat','Dog']

#Extracting features and labels from images
def read_image():
    i=0
    imgf=[]
    lab=[]
    for ct in cl:
        for img in os.listdir(ct):
            cim = os.path.join(ct,img)
            #print(cim)
            img = cv2.imread(cim, cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(64,64))

            #if hog used then no need for reshape hog helps in feature extraction from images
            hog_features = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False)
            imgf.append(hog_features)


            lab.append(ct)
            '''#cv2.imshow('image',img)
            # #cv2.waitKey(0)
            # #print(i)
            # #i+=1
            imgf.append(img)
            lab.append(ct)'''
    return np.array(imgf),np.array(lab)

imgn,lab = read_image()
#print(imgn.shape)

'''#Reshapping Only use if you are not using HOG
smpl_n,height,width = imgn.shape
imgn = imgn.reshape(smpl_n,height*width)'''



#80%-20% train test split ratio 
xtrain,xtest,ytrain,ytest = train_test_split(imgn,lab,test_size=0.2)

#training model change C to adjust margin to get more accurate result, same with scaling and kernal change it also 
reg = SVC(kernel='rbf', C=1, gamma='scale')
reg.fit(xtrain,ytrain)

#predicting
pr = reg.predict(xtest)
accuracy = accuracy_score(ytest, pr)
print(f"Accuracy: {accuracy}")
# DeepNet-KaplanMeier
#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image


# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc

#from PIL import Image
from numpy import *
from keras import backend as K



#input image dimensions
OG_size = 150
img_rows, img_cols = 50, 50 # 50, 50 #normally. make sure this is an even number
Center = OG_size/2
x1, y1, x2, y2 = Center-img_rows/2,Center-img_rows/2,Center+img_cols/2,Center+img_cols/2 # 50, 50, 100, 100

# number of channels
img_channels = 1

# data

Outcomes_file = '/home/chintan/Desktop/tsne_python/Stage1/Stage1and2.csv' #define the outcomes file, sorted according to PID
path1 = '/home/chintan/Desktop/tsne_python/Stage1/Stage1and2' #change this to only test images    
path2 = '/home/chintan/Desktop/tsne_python/Stage1/TestImageCrops'  #DELETE  

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples


for file in listing:
    im = Image.open(path1 + '/' + file) 
    img = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
    #img = im.resize((img_rows,img_cols))
    gray = img.convert('RGB')
                #need to do some more processing here           
    gray.save(path2 +'/' +  file, "PNG")

imlist = os.listdir(path2)
imlist.sort() #make sure to SORT the images to match 

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes

Outcomes = pd.read_csv(Outcomes_file) #create seperate outcomes file for TEST data
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'SURV2']) #pick the column with the labels of interest
Stage = pd.Series.as_matrix(Outcomes.loc[:,'STAGE12'])

label = Outcome_of_interest

data,Label = immatrix,label
train_data = [data,Label]

#pick image index for demo. Not the same as im_index for layer visualization
#im_index1 = 1
#img=immatrix[im_index1].reshape(img_rows,img_cols,3)
#plt.title('Image Preprocessing example')
#plt.figure(1)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')
#plt.show()
print (train_data[0].shape)
print (train_data[1].shape)

#MODEL#

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 30

from keras import backend as K #this just makes stuff work
K.set_image_dim_ordering('th')
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X, y) = (train_data[0],train_data[1])

X = X.reshape(X.shape[0], img_rows, img_cols,3)

X= X.astype('float32')

X /= 255

print('X shape:', X.shape)
print(X.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y = np_utils.to_categorical(y, nb_classes)

#load pretrained model

predDir = '/home/chintan/Desktop/'
modelFile = (os.path.join(predDir,'TheSurvivalModel_VGG16.h5'))  #make sure the model is saved here. 
model = load_model(modelFile)

#predictions using model

score = model.evaluate(X, Y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from sklearn.metrics import roc_curve, auc

def AUC(test_labels,test_prediction,nb):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 

Y_pred = model.predict(X)

ROC2 = AUC(Y, Y_pred, nb_classes)

print(ROC2[1])

#Plot ROC curve
def AUCalt( test_labels , test_prediction):
    # convert to non-categorial
    test_prediction = np.array( [ x[1] for x in test_prediction   ])
    test_labels = np.array( [ 0 if x[0]==1 else 1 for x in test_labels   ])
    # get rates
    fpr, tpr, thresholds = roc_curve(test_labels, test_prediction, pos_label=1)
    # get auc
    myAuc = plt.plot(fpr, tpr)
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(['VGG-16','SVM','RFC','MVR'],loc=4)
    plt.show()
    #myAuc = auc(fpr, tpr)
    return myAuc
   
plt.figure(1)    
ROC_VGG = AUCalt(Y, Y_pred)

#T = np.linspace(0,24, X.shape[0])
#E = Y_pred[:,1]
#E[E>=mean(E)] = 1
#E[E<mean(E)] = 0


#from lifelines import KaplanMeierFitter
#kmf = KaplanMeierFitter()
#kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)

#plt.figure()
#kmf.survival_function_
#kmf.median_
#kmf.plot()

#groups = Stage
#ix = (groups == 'A')

#plt.figure()
#kmf.fit(T[~ix], E[~ix], label='Stage 1')
#ax = kmf.plot()

#kmf.fit(T[ix], E[ix], label='Stage 2')
#kmf.plot(ax=ax)
#plt.xlabel('Months elapsed')
#plt.ylabel('Survivorship')
#plt.title('KM Plot from Model Predictions')


#####

T = np.linspace(0,24, X.shape[0])
E = label
g = Y_pred[:,1]
g[g>=median(g)] = 1
g[g<median(g)] = 0

#g[g>= 0.95] = 1
#g[g< 0.95] = 0


from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)

#plt.figure()
kmf.survival_function_
kmf.median_
kmf.plot()

groups = g
ix = (groups == 0)

#plt.figure()
kmf.fit(T[~ix], E[~ix], label='Low risk')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='High risk')
kmf.plot(ax=ax)
plt.xlabel('Months elapsed')
plt.ylabel('Survivorship')
plt.title('KM Plot from Model Predictions')

from lifelines.statistics import logrank_test

results = logrank_test(T[ix], T[~ix], E[ix], E[~ix], alpha=.99)
print results

         
